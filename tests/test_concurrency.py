"""Concurrency tests for the CPU-bound-work offloading fix (T1-T6):

- /api/apply-atomic-operations, /api/reset-session-structure,
  /api/generate-modified-structure-cif, /api/chgnet-predict, and
  /api/generate-relaxed-structure-cif now run their pymatgen work in a
  worker thread via asyncio.to_thread instead of blocking the single
  FastAPI event loop (see the private *_sync/_load_*/_label_and_* helpers
  next to each endpoint in main.py).
- /api/apply-atomic-operations and /api/reset-session-structure also gained
  a per-session asyncio.Lock (main._get_session_lock) plus a 409 response
  when the session was replaced wholesale (e.g. by /api/create-supercell)
  while the heavy work was in flight.
- /api/generate-relaxed-structure-cif now labels a private .copy() instead
  of mutating the session's stored relaxed_structure in place.
- The four remaining CifParser.get_structures(primitive=False) call sites
  were migrated to parse_structures(primitive=False).

These tests use httpx.AsyncClient(transport=ASGITransport(app=main.app))
instead of the synchronous TestClient wherever genuine concurrency must be
observed: only a real event loop with two requests actually in flight at
once can demonstrate non-blocking behavior or exercise the session lock --
TestClient's blocking portal serializes everything through a single worker
and cannot.
"""
import asyncio
import threading
import time
import uuid
import warnings
from pathlib import Path

from httpx import AsyncClient, ASGITransport
from pymatgen.core import Structure

import main
from main import session_manager


def _cu_structure(supercell=None):
    structure = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    if supercell is not None:
        structure.make_supercell(supercell)
    return structure


def _create_session(structure, filename="Metals/Cu.cif"):
    # session_manager.create_session() mints its own id rather than trusting
    # this one (see its docstring); a fresh uuid4 is always accepted as-is
    # since it can never already be a key in session_manager.sessions.
    session_id = str(uuid.uuid4())
    return session_manager.create_session(session_id, filename, structure)


def _capture_thread(monkeypatch, obj, attr_name):
    """
    Wrap obj.<attr_name> so that calling it records the thread it actually
    ran on (into the returned dict, under "thread") before calling through
    to the real implementation. Used to prove a sync helper executes off
    whatever thread is driving the calling coroutine.
    """
    captured = {}
    original = getattr(obj, attr_name)

    def wrapper(*args, **kwargs):
        captured["thread"] = threading.current_thread()
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, attr_name, wrapper)
    return captured


# ---------------------------------------------------------------------------
# 1. The event loop is not blocked while the heavy work runs
# ---------------------------------------------------------------------------

def test_health_not_blocked_by_slow_apply_atomic_operations(monkeypatch):
    """
    /api/apply-atomic-operations must not block /health (or any other
    concurrent request) for the duration of its pymatgen work: with the T1
    sync helper artificially slowed down, a concurrent /health request must
    still complete almost immediately instead of waiting behind it.
    """
    structure = _cu_structure()
    session_id = _create_session(structure)

    original = main._apply_atomic_operations_sync

    def slow(*args, **kwargs):
        time.sleep(0.5)
        return original(*args, **kwargs)

    monkeypatch.setattr(main, "_apply_atomic_operations_sync", slow)

    async def run():
        transport = ASGITransport(app=main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            t0 = time.monotonic()

            async def do_apply():
                r = await ac.post("/api/apply-atomic-operations", json={
                    "session_id": session_id, "operations": [],
                })
                return r, time.monotonic() - t0

            async def do_health():
                r = await ac.get("/health")
                return r, time.monotonic() - t0

            return await asyncio.gather(do_apply(), do_health())

    (apply_resp, apply_elapsed), (health_resp, health_elapsed) = asyncio.run(run())

    assert apply_resp.status_code == 200
    assert health_resp.status_code == 200
    assert apply_elapsed >= 0.5
    assert health_elapsed < 0.3, (
        f"/health took {health_elapsed:.3f}s -- it was blocked behind the "
        f"slow apply-atomic-operations request"
    )


# ---------------------------------------------------------------------------
# 2. Each new sync helper actually runs off the event-loop thread
# ---------------------------------------------------------------------------

def test_apply_atomic_operations_helper_runs_off_event_loop_thread(client, monkeypatch):
    structure = _cu_structure()
    session_id = _create_session(structure)

    loop_thread = _capture_thread(monkeypatch, session_manager, "get_session_info")
    helper_thread = _capture_thread(monkeypatch, main, "_apply_atomic_operations_sync")

    response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id, "operations": [],
    })

    assert response.status_code == 200
    assert "thread" in loop_thread and "thread" in helper_thread
    assert helper_thread["thread"] != loop_thread["thread"]


def test_reset_session_structure_helper_runs_off_event_loop_thread(client, monkeypatch):
    structure = _cu_structure()
    session_id = _create_session(structure)

    loop_thread = _capture_thread(monkeypatch, session_manager, "get_session_info")
    helper_thread = _capture_thread(monkeypatch, main, "_reset_session_structure_sync")

    response = client.post("/api/reset-session-structure", json={"session_id": session_id})

    assert response.status_code == 200
    assert "thread" in loop_thread and "thread" in helper_thread
    assert helper_thread["thread"] != loop_thread["thread"]


def test_generate_modified_structure_cif_helper_runs_off_event_loop_thread(client, monkeypatch):
    loop_thread = _capture_thread(monkeypatch, main, "validate_supercell_size")
    helper_thread = _capture_thread(monkeypatch, main, "_generate_modified_structure_cif_sync")

    response = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1], "operations": [],
    })

    assert response.status_code == 200
    assert "thread" in loop_thread and "thread" in helper_thread
    assert helper_thread["thread"] != loop_thread["thread"]


def test_chgnet_predict_helper_runs_off_event_loop_thread(client, monkeypatch):
    loop_thread = _capture_thread(monkeypatch, main, "validate_supercell_size")
    helper_thread = _capture_thread(monkeypatch, main, "_load_structure_for_chgnet_predict")

    # Structure resolution (the offloaded part) always runs first regardless
    # of whether CHGNet itself is available, so don't assert on the final
    # status code (200, or 503 if the model can't load) -- only that the
    # helper ran, and ran off the event-loop thread.
    client.post("/api/chgnet-predict", json={
        "filename": "Metals/Cu.cif", "operations": [], "supercell_size": [1, 1, 1],
    })

    assert "thread" in loop_thread and "thread" in helper_thread
    assert helper_thread["thread"] != loop_thread["thread"]


def test_generate_relaxed_cif_helper_runs_off_event_loop_thread(client, monkeypatch):
    structure = _cu_structure()
    session_id = _create_session(structure)
    session_info = session_manager.get_session_info(session_id)
    session_info["relaxed_structure"] = structure.copy()
    session_info["chgnet_result"] = {"fmax": 0.1, "converged": True, "steps": 1, "optimizer": "FIRE"}

    loop_thread = _capture_thread(monkeypatch, session_manager, "get_session_info")
    helper_thread = _capture_thread(monkeypatch, main, "_label_and_write_relaxed_cif")

    response = client.post("/api/generate-relaxed-structure-cif", json={"session_id": session_id})

    assert response.status_code == 200
    assert "thread" in loop_thread and "thread" in helper_thread
    assert helper_thread["thread"] != loop_thread["thread"]


# ---------------------------------------------------------------------------
# 3. Two concurrent apply-atomic-operations requests on the same session
#    don't corrupt state (the per-session lock serializes them)
# ---------------------------------------------------------------------------

def test_apply_atomic_operations_concurrent_same_session_no_corruption(monkeypatch):
    structure = _cu_structure()
    session_id = _create_session(structure)

    original = main._apply_atomic_operations_sync

    def slow(*args, **kwargs):
        time.sleep(0.2)
        return original(*args, **kwargs)

    monkeypatch.setattr(main, "_apply_atomic_operations_sync", slow)

    ops_a = [{"action": "substitute", "index": 0, "to": "Ni"}]
    ops_b = [{"action": "substitute", "index": 0, "to": "Fe"}]

    async def run():
        transport = ASGITransport(app=main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await asyncio.gather(
                ac.post("/api/apply-atomic-operations", json={
                    "session_id": session_id, "operations": ops_a,
                }),
                ac.post("/api/apply-atomic-operations", json={
                    "session_id": session_id, "operations": ops_b,
                }),
            )

    resp_a, resp_b = asyncio.run(run())

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    # The lock (main._get_session_lock) serializes the two requests instead
    # of letting them interleave; whichever landed last, session_manager.
    # update_structure() must have been applied cleanly -- the final
    # operations list must be exactly one of the two, never a mix/corruption.
    final_ops = session_manager.sessions[session_id]["operations"]
    assert final_ops in (ops_a, ops_b)


# ---------------------------------------------------------------------------
# 4. A concurrent session replacement (create-supercell) makes an in-flight
#    apply-atomic-operations request fail with 409 instead of clobbering it
# ---------------------------------------------------------------------------

def test_apply_atomic_operations_409_on_concurrent_session_replacement(monkeypatch):
    structure = _cu_structure()
    session_id = _create_session(structure)

    original = main._apply_atomic_operations_sync

    def slow(*args, **kwargs):
        time.sleep(0.4)
        return original(*args, **kwargs)

    monkeypatch.setattr(main, "_apply_atomic_operations_sync", slow)

    ops = [{"action": "substitute", "index": 0, "to": "Ni"}]

    async def run():
        transport = ASGITransport(app=main.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            apply_task = asyncio.create_task(ac.post(
                "/api/apply-atomic-operations",
                json={"session_id": session_id, "operations": ops},
            ))
            # Give the apply request time to acquire the session lock and
            # enter the (slowed) asyncio.to_thread call before replacing the
            # session out from under it.
            await asyncio.sleep(0.1)

            replace_resp = await ac.post("/api/create-supercell", json={
                "crystal_data": {
                    "filename": "Metals/Cu.cif", "volume": 1.0,
                    "num_sites": 1, "formula": "Cu",
                },
                "supercell_size": [1, 1, 1],
                "session_id": session_id,
            })
            apply_resp = await apply_task
            return apply_resp, replace_resp

    apply_resp, replace_resp = asyncio.run(run())

    assert replace_resp.status_code == 200
    assert replace_resp.json()["session_id"] == session_id
    assert apply_resp.status_code == 409

    # The session's current structure must reflect create-supercell's
    # replacement, not the stale apply-atomic-operations result: no Ni
    # substitution, and it must match what create-supercell actually built.
    expected_structure = Structure.from_dict(replace_resp.json()["structure_dict"])
    current = session_manager.get_current_structure(session_id)
    assert "Ni" not in str(current.formula)
    assert str(current.formula) == str(expected_structure.formula)
    assert len(current.sites) == len(expected_structure.sites)


# ---------------------------------------------------------------------------
# 5. generate-relaxed-structure-cif must not mutate the session's stored
#    relaxed_structure object (T5)
# ---------------------------------------------------------------------------

def test_generate_relaxed_cif_does_not_mutate_session_structure(client):
    structure = _cu_structure()
    session_id = _create_session(structure)
    session_info = session_manager.get_session_info(session_id)
    relaxed = structure.copy()
    session_info["relaxed_structure"] = relaxed
    session_info["chgnet_result"] = {"fmax": 0.1, "converged": True, "steps": 1, "optimizer": "FIRE"}

    original_labels = [site.label for site in relaxed.sites]

    response = client.post("/api/generate-relaxed-structure-cif", json={"session_id": session_id})
    assert response.status_code == 200

    # apply_element_labels_to_structure mutates in place; if the endpoint
    # had run it on the session's own object (instead of a private .copy())
    # the labels below would have changed from their pre-request values.
    assert session_info["relaxed_structure"] is relaxed
    after_labels = [site.label for site in relaxed.sites]
    assert after_labels == original_labels


# ---------------------------------------------------------------------------
# 6. T6: the four migrated main.py call sites no longer trigger the
#    deprecated CifParser.get_structures() FutureWarning
# ---------------------------------------------------------------------------

def test_no_get_structures_deprecation_warning(client):
    """
    Regression for T6: load_base_supercell, analyze_cif_file_sync,
    _load_and_build_supercell, and _resolve_and_expand_supercell_direct all
    used to call the deprecated CifParser.get_structures(); they now call
    parse_structures() instead. Exercise all four code paths and assert none
    of them emits the 'get_structures is deprecated' FutureWarning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # load_base_supercell's sample-file fallback branch (no session_id).
        r1 = client.post("/api/generate-modified-structure-cif", json={
            "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1], "operations": [],
        })
        assert r1.status_code == 200

        # analyze_cif_file_sync's CifParser fallback branch.
        r2 = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
        assert r2.status_code == 200

        # _load_and_build_supercell's sample-file branch (no structure_data).
        r3 = client.post("/api/create-supercell", json={
            "crystal_data": {
                "filename": "Metals/Cu.cif", "volume": 1.0,
                "num_sites": 1, "formula": "Cu",
            },
            "supercell_size": [1, 1, 1],
        })
        assert r3.status_code == 200

        # _resolve_and_expand_supercell_direct's sample-file branch.
        r4 = client.post("/api/generate-supercell-cif-direct", json={
            "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1],
        })
        assert r4.status_code == 200

    deprecated = [w for w in caught if "get_structures is deprecated" in str(w.message)]
    assert not deprecated, [str(w.message) for w in deprecated]
