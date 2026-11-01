"""Cooperative cancellation of /api/relax when the HTTP client disconnects.

Covers the per-run ``cancel_event`` (watcher, run_relaxation, elastic), the
outermost pure-ASGI disconnect detector, and a real-uvicorn end-to-end check
that a closed socket frees the global relax semaphore within seconds.
"""
import asyncio
import json
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from pymatgen.core import Structure

import main
from asgi_disconnect import RelaxDisconnectMiddleware, STATE_KEY
from main import RelaxationAborted, RelaxationWatcher, run_relaxation, session_manager
from mlip import elastic as elastic_mod, executor, registry as mlip_registry
from mlip.elastic import compute_elastic
from tests.conftest import FakeBackend


# -- cancel_event in the compute layers ------------------------------------

class _FakeAtoms:
    def get_forces(self):
        return np.array([[0.01, 0.0, 0.0]])

    def get_potential_energy(self):
        return -1.0


def _cu():
    s = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    s.make_supercell((2, 2, 2))
    return s


def test_is_aborted_and_reason_precedence():
    ev = threading.Event()
    assert not executor.is_aborted() and not executor.is_aborted(ev)
    assert executor.abort_reason(ev) == "timeout"
    ev.set()
    assert executor.is_aborted(ev) and not executor.abort_requested()
    assert executor.abort_reason(ev) == "client_disconnected"
    executor.request_abort()
    try:
        assert executor.abort_reason(ev) == "shutdown"
    finally:
        executor.clear_abort()


def test_watcher_aborts_on_cancel_event_without_touching_global_flag():
    ev = threading.Event()
    watcher = RelaxationWatcher(_FakeAtoms(), fmax=0.1, max_steps=10,
                                deadline=time.monotonic() + 100, cancel_event=ev)
    watcher()  # not set: fine
    ev.set()
    with pytest.raises(RelaxationAborted):
        watcher()
    assert not executor.abort_requested()


def test_run_relaxation_reports_client_disconnected():
    ev = threading.Event()
    ev.set()
    result = run_relaxation(EMT(), _cu(), fmax=0.01, steps=50, optimizer_name="FIRE",
                            timeout_s=100, cancel_event=ev)
    assert result["aborted"] is True
    assert result["abort_reason"] == "client_disconnected"
    assert not executor.abort_requested()


def test_run_relaxation_without_cancel_event_unchanged():
    result = run_relaxation(EMT(), _cu(), fmax=0.5, steps=5, optimizer_name="FIRE",
                            timeout_s=100)
    assert result["aborted"] is False and result["abort_reason"] is None


def test_compute_elastic_stops_before_first_cell_on_cancel_event():
    ev = threading.Event()
    ev.set()
    with pytest.raises(elastic_mod.ElasticTimeout, match="aborted before"):
        compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT(), cancel_event=ev)
    assert not executor.abort_requested()


def test_compute_elastic_observer_stops_mid_cell_on_cancel_event():
    ev = threading.Event()

    class Calc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            ev.set()  # client leaves while the first cell is evaluated
            self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3)),
                            "stress": np.zeros(6)}

    with pytest.raises(elastic_mod.ElasticTimeout, match="aborted during"):
        compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), Calc(), cancel_event=ev)


# -- the pure-ASGI wrapper ---------------------------------------------------

def _drive(app, path="/api/relax", method="POST", chunks=(b"{}",), disconnect_after=None):
    """Run an ASGI app by hand. Returns (events, scope).

    ``disconnect_after``: seconds after which the fake client disconnects
    (None = receive() blocks forever after the body).
    """
    scope = {"type": "http", "method": method, "path": path, "headers": [],
             "state": {}}
    queue = asyncio.Queue()
    for i, c in enumerate(chunks):
        queue.put_nowait({"type": "http.request", "body": c,
                          "more_body": i < len(chunks) - 1})
    sent = []

    async def receive():
        return await queue.get()

    async def send(msg):
        sent.append(msg)

    async def main_():
        if disconnect_after is not None:
            asyncio.get_running_loop().call_later(
                disconnect_after, queue.put_nowait, {"type": "http.disconnect"})
        await asyncio.wait_for(app(scope, receive, send), timeout=5)

    asyncio.run(main_())
    return sent, scope


def test_wrapper_replays_multichunk_body_then_disconnect():
    seen = {}

    async def inner(scope, receive, send):
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body"):
                break
        seen["body"] = body
        seen["event"] = scope["state"][STATE_KEY]
        seen["before"] = seen["event"].is_set()
        seen["next"] = await receive()  # blocks until the real disconnect
        seen["after"] = seen["event"].is_set()

    _drive(RelaxDisconnectMiddleware(inner), chunks=(b'{"a"', b':1}'), disconnect_after=0.1)
    assert seen["body"] == b'{"a":1}'
    assert seen["before"] is False
    assert seen["next"]["type"] == "http.disconnect"
    assert seen["after"] is True


def test_wrapper_no_disconnect_never_sets_event():
    seen = {}

    async def inner(scope, receive, send):
        await receive()
        await asyncio.sleep(0.2)
        seen["set"] = scope["state"][STATE_KEY].is_set()

    _drive(RelaxDisconnectMiddleware(inner))
    assert seen["set"] is False


def test_wrapper_ignores_other_routes_and_methods():
    calls = []

    async def inner(scope, receive, send):
        calls.append((scope["path"], receive, STATE_KEY in scope.get("state", {})))

    mw = RelaxDisconnectMiddleware(inner)
    for path, method in (("/api/predict", "POST"), ("/api/relax", "GET"),
                         ("/api/relax/x", "POST")):
        _drive(mw, path=path, method=method)
    assert [c[2] for c in calls] == [False, False, False]


def test_wrapper_is_outermost_in_main_app():
    assert main.app.user_middleware[0].cls is RelaxDisconnectMiddleware


def test_rate_limit_still_applies_to_relax(client, monkeypatch):
    """Rate limiting still applies in front of /api/relax (wrapper is transparent)."""
    monkeypatch.setattr(main, "_rate_limit_check", lambda host: False)
    r = client.post("/api/relax", json={"session_id": "x"})
    assert r.status_code == 429


def test_handler_without_wrapper_state_warns_once_and_uses_fresh_event(caplog):
    """Direct call (no middleware state): handler still runs, and logs a visible
    one-time warning that disconnect detection is inactive."""
    from types import SimpleNamespace
    from fastapi import HTTPException
    main._warned_no_disconnect_state = False
    req = SimpleNamespace(scope={})
    with caplog.at_level("WARNING", logger=main.logger.name):
        for _ in range(2):
            with pytest.raises(HTTPException) as ei:
                asyncio.run(main.relax_structure({"session_id": "nope"}, req))
            assert ei.value.status_code == 404  # got past the cancel_event setup
    warnings = [r for r in caplog.records if "disconnect detection is inactive" in r.getMessage()]
    assert len(warnings) == 1


def test_deadline_abort_stays_timeout_even_if_cancel_set_afterwards():
    ev = threading.Event()
    watcher = RelaxationWatcher(_FakeAtoms(), fmax=0.1, max_steps=10,
                                deadline=time.monotonic() - 1, cancel_event=ev)
    with pytest.raises(RelaxationAborted) as ei:
        watcher()
    ev.set()  # client leaves only after the deadline abort
    assert ei.value.reason == "timeout"


def test_run_relaxation_timeout_reason_not_rewritten_by_later_flags():
    ev = threading.Event()
    result = run_relaxation(EMT(), _cu(), fmax=0.01, steps=50, optimizer_name="FIRE",
                            timeout_s=-1, cancel_event=ev)
    ev.set()  # only after the run ended
    assert result["abort_reason"] == "timeout"


def test_elastic_timeout_reason_recorded_at_raise():
    ev = threading.Event()
    with pytest.raises(elastic_mod.ElasticTimeout) as ei:
        compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT(),
                        deadline=time.monotonic() - 1, cancel_event=ev)
    ev.set()
    assert ei.value.reason == "timeout"
    with pytest.raises(elastic_mod.ElasticTimeout) as ei:
        compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT(), cancel_event=ev)
    assert ei.value.reason == "client_disconnected"


def test_wrapper_matches_with_root_path():
    seen = {}

    async def inner(scope, receive, send):
        seen["state"] = STATE_KEY in scope.get("state", {})

    mw = RelaxDisconnectMiddleware(inner)

    async def shim(scope, receive, send):
        scope["root_path"] = "/proxy"
        await mw(scope, receive, send)

    for path in ("/proxy/api/relax", "/api/relax"):
        seen.clear()
        _drive(shim, path=path)
        assert seen["state"] is True, path
    seen.clear()
    _drive(shim, path="/proxy/api/other")
    assert seen["state"] is False


def test_wrapper_rejects_oversized_content_length_and_chunked_body():
    called = []

    async def inner(scope, receive, send):
        called.append(1)

    mw = RelaxDisconnectMiddleware(inner, max_body=10)

    async def shim(scope, receive, send):
        scope["headers"] = [(b"content-length", b"1000")]
        await mw(scope, receive, send)

    sent, _ = _drive(shim, chunks=(b"x" * 5,))
    assert sent[0]["status"] == 413 and not called
    sent, _ = _drive(mw, chunks=(b"x" * 6, b"x" * 6))
    assert sent[0]["status"] == 413 and not called
    _drive(mw, chunks=(b"x" * 10,))
    assert called == [1]


def test_oversized_relax_body_gets_413_via_main_app(client):
    r = client.post("/api/relax", content=b"x" * (2 * 1024 * 1024),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_relax_still_logged_by_analytics_middleware(client):
    r = client.post("/api/relax", json={"session_id": "does-not-exist"})
    assert r.status_code == 404
    assert any(row for row in _access_rows() if row[0] == "/api/relax")


def _access_rows():
    with main.analytics_db._get_connection() as conn:
        return conn.execute("SELECT path, status_code FROM access_logs").fetchall()


def test_malformed_and_oversized_body_errors_unchanged(client):
    bad = client.post("/api/relax", content=b"{not json",
                      headers={"content-type": "application/json"})
    assert bad.status_code == 422
    notdict = client.post("/api/relax", json=[1, 2])
    assert notdict.status_code == 422
    missing = client.post("/api/relax", json={})
    assert missing.status_code == 400


# -- real uvicorn: disconnect frees the semaphore ----------------------------

class _SlowEMT(EMT):
    """EMT with constant extra force and a per-evaluation delay: never converges."""
    implemented_properties = EMT.implemented_properties

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        time.sleep(0.05)
        self.results["forces"] = self.results["forces"] + np.array([0.5, 0.0, 0.0])


class _SlowBackend(FakeBackend):
    id = "fake-slow"
    display_name = "Fake Slow"
    predict_calls = 0

    async def _build_calculator(self):
        return _SlowEMT()

    async def predict_one(self, structure, properties):
        type(self).predict_calls += 1
        return await super().predict_one(structure, properties)


@pytest.fixture
def live_url(monkeypatch):
    import uvicorn
    monkeypatch.setitem(mlip_registry.MODEL_REGISTRY, "fake-slow", _SlowBackend)
    _SlowBackend.predict_calls = 0
    monkeypatch.setattr(main, "_relax_semaphore", asyncio.Semaphore(1))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    t0 = time.time()
    while not server.started:
        assert time.time() - t0 < 20 and t.is_alive()
        time.sleep(0.05)
    yield port
    server.should_exit = True
    t.join(timeout=15)


def _session():
    s = _cu()
    s.perturb(0.1)
    return session_manager.create_session(str(uuid.uuid4()), "Cu.cif", s)


def _raw_post(port, payload):
    body = json.dumps(payload).encode()
    c = socket.create_connection(("127.0.0.1", port))
    c.sendall(b"POST /api/relax HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % len(body) + body)
    return c


def _recv_http(c, timeout=60):
    c.settimeout(timeout)
    data = b""
    while b"\r\n\r\n" not in data:
        data += c.recv(65536)
    head, _, rest = data.partition(b"\r\n\r\n")
    length = int([h.split(b":")[1] for h in head.split(b"\r\n")
                  if h.lower().startswith(b"content-length")][0])
    while len(rest) < length:
        rest += c.recv(65536)
    return int(head.split()[1]), json.loads(rest)


def _wait_until(pred, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return time.monotonic() - t0
        time.sleep(0.05)
    return None


def _slow_payload(sid, **kw):
    return {"session_id": sid, "model": "fake-slow", "fmax": 0.01, "max_steps": 500,
            "optimizer": "FIRE", **kw}


def _steps_done(sid):
    return len(main._relax_progress.get(sid, {}).get("history", []))


def _elastic_started(sid):
    msg = (main._relax_progress.get(sid, {}).get("state") or {}).get("message", "")
    return msg.startswith("Elastic")


def test_disconnect_mid_relax_frees_semaphore_and_skips_rest(live_url):
    sid = _session()
    c = _raw_post(live_url, _slow_payload(sid))
    assert _wait_until(lambda: _steps_done(sid) >= 3, 60) is not None  # optimizer running
    assert main._relax_semaphore.locked()
    c.close()
    freed_after = _wait_until(lambda: not main._relax_semaphore.locked(), 60)
    assert freed_after is not None, "semaphore not released after client disconnect"
    assert not executor.abort_requested()
    # Only the initial prediction ran; no final prediction.
    assert _SlowBackend.predict_calls == 1
    # No write-back of a relaxed structure.
    assert session_manager.get_session_info(sid).get("relaxed_structure") is None
    # A second relax is accepted (no 429) and, given a quick run, completes.
    sid2 = _session()
    c2 = _raw_post(live_url, _slow_payload(sid2, max_steps=3))
    status, body = _recv_http(c2)
    c2.close()
    assert status == 200 and body["status"] == "success"


def test_no_disconnect_completes_normally(live_url):
    sid = _session()
    c = _raw_post(live_url, _slow_payload(sid, max_steps=4))
    status, body = _recv_http(c)
    c.close()
    assert status == 200
    assert body["status"] == "success"
    assert body["relaxation_info"]["abort_reason"] is None
    assert _SlowBackend.predict_calls == 2
    assert session_manager.get_session_info(sid).get("relaxed_structure") is not None


def test_disconnect_during_elastic_stage_cancels_and_frees_semaphore(live_url):
    sid = _session()
    c = _raw_post(live_url, _slow_payload(sid, max_steps=3, elastic=True))
    assert _wait_until(lambda: _elastic_started(sid), 120) is not None
    assert main._relax_semaphore.locked()
    c.close()
    assert _wait_until(lambda: not main._relax_semaphore.locked(), 60) is not None
    assert not executor.abort_requested()
    assert _SlowBackend.predict_calls == 2  # initial + final; elastic then cancelled
    assert session_manager.get_session_info(sid).get("relaxed_structure") is None
    c2 = _raw_post(live_url, _slow_payload(_session(), max_steps=3))
    status, body = _recv_http(c2)
    c2.close()
    assert status == 200 and body["status"] == "success"
