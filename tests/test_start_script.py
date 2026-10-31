"""Tests for start_crystalnexus.py's pure LAN-mode helper functions.

start_crystalnexus.py guards its re-exec-into-venv behavior
(_relaunch_in_venv()) behind `if __name__ == "__main__":`, so importing the
module from pytest is safe -- it never spawns a subprocess or execs itself.
These tests exercise the pure helpers (host/token resolution, LAN IPv4
detection, the read-only Windows Firewall check, child-process environment
construction, and CLI argument parsing) in isolation, with no real server
process ever started.
"""
import ipaddress
import platform
from types import SimpleNamespace

import pytest

import start_crystalnexus as sx


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------

def test_import_does_not_relaunch_or_hang():
    """The mere fact that `import start_crystalnexus as sx` above succeeded
    (without hanging or spawning a child process) already proves
    _relaunch_in_venv() did not run. This test just documents/asserts the
    module loaded cleanly and exposes a sane default HOST."""
    assert sx.HOST  # some default value was computed at import time
    assert isinstance(sx.HOST, str)
    assert isinstance(sx.PORT, int)


def test_default_port_is_8090(monkeypatch):
    """PORT defaults to 8090 when CRYSTALNEXUS_PORT is unset, independent
    of whatever the developer's shell happens to have set (regression:
    the module reads the env var once at import time, so a stray
    CRYSTALNEXUS_PORT left over from a previous run silently changed what
    this assertion saw)."""
    import importlib
    monkeypatch.delenv("CRYSTALNEXUS_PORT", raising=False)
    importlib.reload(sx)
    try:
        assert sx.PORT == 8090
    finally:
        importlib.reload(sx)  # restore module state for tests that follow


# ---------------------------------------------------------------------------
# resolve_host
# ---------------------------------------------------------------------------

def test_resolve_host_default_loopback():
    assert sx.resolve_host(False, None) == "127.0.0.1"


def test_resolve_host_env_overrides_default():
    assert sx.resolve_host(False, "0.0.0.0") == "0.0.0.0"


def test_resolve_host_lan_flag_beats_env():
    assert sx.resolve_host(True, "127.0.0.1") == "0.0.0.0"


def test_resolve_host_lan_flag_with_no_env():
    assert sx.resolve_host(True, None) == "0.0.0.0"


def test_resolve_host_empty_env_treated_as_unset():
    assert sx.resolve_host(False, "") == "127.0.0.1"


# ---------------------------------------------------------------------------
# is_lan_binding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5"])
def test_is_lan_binding_true_for_lan_addresses(host):
    assert sx.is_lan_binding(host) is True


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", ""])
def test_is_lan_binding_false_for_loopback_addresses(host):
    assert sx.is_lan_binding(host) is False


# ---------------------------------------------------------------------------
# health_host
# ---------------------------------------------------------------------------

def test_health_host_wildcard_ipv4_maps_to_loopback():
    assert sx.health_host("0.0.0.0") == "127.0.0.1"


def test_health_host_loopback_stays_loopback():
    assert sx.health_host("127.0.0.1") == "127.0.0.1"


def test_health_host_wildcard_ipv6_maps_to_bracketed_loopback():
    assert sx.health_host("::") == "[::1]"


def test_health_host_explicit_interface_ip_passes_through():
    assert sx.health_host("192.168.1.5") == "192.168.1.5"


# ---------------------------------------------------------------------------
# resolve_analytics_token
# ---------------------------------------------------------------------------

def test_resolve_analytics_token_unset_and_not_lan_mode_returns_none(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    assert sx.resolve_analytics_token(False) is None


def test_resolve_analytics_token_unset_and_lan_mode_generates_token(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    token = sx.resolve_analytics_token(True)
    assert token is not None
    assert len(token) >= 32


def test_resolve_analytics_token_explicit_value_used_verbatim(monkeypatch):
    monkeypatch.setenv("CRYSTALNEXUS_ANALYTICS_TOKEN", "abc")
    assert sx.resolve_analytics_token(False) == "abc"
    assert sx.resolve_analytics_token(True) == "abc"


def test_resolve_analytics_token_empty_string_is_opt_out(monkeypatch, capsys):
    monkeypatch.setenv("CRYSTALNEXUS_ANALYTICS_TOKEN", "")
    assert sx.resolve_analytics_token(True) is None  # must not raise
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


# ---------------------------------------------------------------------------
# _child_env
# ---------------------------------------------------------------------------

def test_child_env_sets_host_port_and_token(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    env = sx._child_env("0.0.0.0", 9000, "mytoken")
    assert env["CRYSTALNEXUS_HOST"] == "0.0.0.0"
    assert env["CRYSTALNEXUS_PORT"] == "9000"
    assert env["CRYSTALNEXUS_ANALYTICS_TOKEN"] == "mytoken"


def test_child_env_inherits_path(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    env = sx._child_env("127.0.0.1", 8090, None)
    assert "PATH" in env or "Path" in env


def test_child_env_does_not_mutate_real_os_environ(monkeypatch):
    import os

    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    sx._child_env("0.0.0.0", 9000, "mytoken")
    assert "CRYSTALNEXUS_ANALYTICS_TOKEN" not in os.environ


def test_child_env_none_token_omits_key(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    env = sx._child_env("127.0.0.1", 8090, None)
    assert "CRYSTALNEXUS_ANALYTICS_TOKEN" not in env


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = sx.parse_args([])
    assert args.lan is False
    assert args.port is None


def test_parse_args_lan_flag():
    assert sx.parse_args(["--lan"]).lan is True


def test_parse_args_port():
    assert sx.parse_args(["--port", "9000"]).port == 9000


def test_parse_args_no_firewall_check():
    assert sx.parse_args(["--no-firewall-check"]).no_firewall_check is True


# ---------------------------------------------------------------------------
# detect_lan_ipv4
# ---------------------------------------------------------------------------

def test_detect_lan_ipv4_returns_two_tuple():
    result = sx.detect_lan_ipv4()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_detect_lan_ipv4_never_raises():
    # Simply calling it is the test: it must not raise even without
    # network access.
    sx.detect_lan_ipv4()


def test_detect_lan_ipv4_primary_is_valid_ipv4_when_present():
    primary, _others = sx.detect_lan_ipv4()
    if primary is not None:
        ipaddress.IPv4Address(primary)  # must not raise


def test_detect_lan_ipv4_others_exclude_loopback_and_link_local():
    _primary, others = sx.detect_lan_ipv4()
    for ip in others:
        addr = ipaddress.IPv4Address(ip)
        assert not addr.is_loopback
        assert not addr.is_link_local


# ---------------------------------------------------------------------------
# check_firewall_rule
# ---------------------------------------------------------------------------

@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_none_on_subprocess_error(monkeypatch):
    def fake_run(*args, **kwargs):
        raise OSError("powershell not found")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8090) is None


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_none_on_nonzero_returncode(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8090) is None


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_true_on_match(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="MATCH:CrystalNexus 8090\n")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8090) is True


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_false_on_no_match(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8090) is False


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_is_read_only():
    """Static guard: check_firewall_rule() and the PowerShell snippet it
    runs must never mutate firewall state. This protects against someone
    later turning the read-only inspection into a mutating one."""
    import inspect

    source = inspect.getsource(sx.check_firewall_rule)
    for forbidden in ("New-NetFirewallRule", "Set-NetFirewallRule", "Remove-NetFirewallRule",
                       "netsh advfirewall firewall add"):
        assert forbidden not in source

    ps_snippet = sx._FIREWALL_CHECK_PS
    for forbidden in ("New-", "Set-", "Remove-", "add rule", "delete rule"):
        assert forbidden not in ps_snippet


# ---------------------------------------------------------------------------
# Child output draining (launcher hang regression)
# ---------------------------------------------------------------------------

import collections
import io
import queue
import subprocess
import sys
import threading


def _fake_proc(raw: bytes):
    """Fake process whose stdout is a utf-8/replace text stream over raw bytes."""
    stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace")
    return SimpleNamespace(stdout=stream)


def test_child_env_forces_utf8(monkeypatch):
    env = sx._child_env("127.0.0.1", 8090, None)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUTF8"] == "1"


def test_reader_survives_invalid_bytes():
    proc = _fake_proc(b"ok\n\x81\xff bad\n\xe3\x81\x82 good\nlast\n")
    tail = collections.deque(maxlen=10)
    sx._stream_child_output(proc, tail, queue.Queue())
    assert len(tail) == 4
    assert tail[-1] == "last\n"


def test_reader_survives_per_line_decode_exception():
    class Flaky:
        def __init__(self):
            self.lines = ["a\n", UnicodeDecodeError("cp932", b"\xff", 0, 1, "bad"), "b\n", ""]

        def readline(self):
            item = self.lines.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    tail = collections.deque(maxlen=10)
    sx._stream_child_output(SimpleNamespace(stdout=Flaky()), tail, queue.Queue())
    assert list(tail) == ["a\n", "b\n"]


def test_reader_does_not_block_when_display_queue_full():
    n = 5000
    proc = _fake_proc(b"".join(b"line %d\n" % i for i in range(n)))
    tail = collections.deque(maxlen=200)
    q = queue.Queue(maxsize=10)  # nobody consumes
    t = threading.Thread(target=sx._stream_child_output, args=(proc, tail, q), daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive()
    assert tail[-1] == f"line {n - 1}\n"
    assert q.qsize() <= 10


def test_display_failure_does_not_stop_draining(monkeypatch):
    def broken_print(*a, **k):
        raise UnicodeEncodeError("cp932", "x", 0, 1, "boom")

    monkeypatch.setattr("builtins.print", broken_print)
    n = 3000
    proc = _fake_proc(b"".join(b"line %d\n" % i for i in range(n)))
    tail = collections.deque(maxlen=200)
    q = queue.Queue(maxsize=sx._DISPLAY_QUEUE_MAX)
    disp = threading.Thread(target=sx._display_child_output, args=(q,), daemon=True)
    disp.start()
    sx._stream_child_output(proc, tail, q)
    q.put(None)
    disp.join(timeout=10)
    assert not disp.is_alive()
    assert tail[-1] == f"line {n - 1}\n"


def test_display_oserror_is_tolerated(monkeypatch):
    def broken_print(*a, **k):
        raise OSError("closed")

    monkeypatch.setattr("builtins.print", broken_print)
    q = queue.Queue()
    for i in range(50):
        q.put(f"l{i}\n")
    q.put(None)
    sx._display_child_output(q)  # must return without raising


def test_real_subprocess_with_invalid_bytes_and_volume(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'start\\n\\x81\\xff bad\\n'); sys.stdout.flush()\n"
        "for i in range(100000): print('line', i)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    tail = collections.deque(maxlen=50)
    t = threading.Thread(target=sx._stream_child_output, args=(proc, tail, queue.Queue(maxsize=5)), daemon=True)
    t.start()
    t.join(timeout=60)
    assert not t.is_alive()
    assert proc.wait(timeout=10) == 0, "".join(tail)
    assert tail[-1] == "line 99999\n"


def test_monitor_warns_when_reader_dead(monkeypatch):
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    monkeypatch.setattr(sx, "_output_threads", [dead])
    monkeypatch.setattr(sx, "shutdown_requested", False)
    monkeypatch.setattr(sx.requests, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    calls = {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] >= 1:
            monkeypatch.setattr(sx, "shutdown_requested", True)

    monkeypatch.setattr(sx.time, "sleep", fake_sleep)
    q = queue.Queue()
    proc = SimpleNamespace(poll=lambda: None, returncode=None)
    sx.monitor_process_health(proc, q)
    msgs = []
    while not q.empty():
        msgs.append(q.get())
    assert any("reader thread" in m[1] for m in msgs)


# ---------------------------------------------------------------------------
# Launcher robustness: pid-checked health, tree kill, cross-platform stop
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def test_check_backend_status_without_expected_pids(monkeypatch):
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "CrystalNexus", "pid": 1}))
    assert sx.check_backend_status() is True


def test_check_backend_status_rejects_foreign_pid(monkeypatch):
    """An OLD server still holding the port must not count as the new one."""
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "CrystalNexus", "pid": 111}))
    assert sx.check_backend_status({222, 333}) is False
    assert sx.check_backend_status({111}) is True


def test_check_backend_status_rejects_other_service_and_bad_json(monkeypatch):
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "Other", "pid": 1}))
    assert sx.check_backend_status({1}) is False
    monkeypatch.setattr(sx.requests, "get", lambda *a, **k: _Resp(200, None))
    assert sx.check_backend_status({1}) is False


def test_health_endpoint_reports_pid():
    import os
    import asyncio
    import main
    def req(host):
        return SimpleNamespace(client=SimpleNamespace(host=host))

    for host in ("127.0.0.1", "::1"):
        data = asyncio.run(main.health_check(req(host)))
        assert data["service"] == "CrystalNexus"
        assert data["pid"] == os.getpid()
    # Non-loopback (LAN) clients must not see the pid
    assert "pid" not in asyncio.run(main.health_check(req("192.168.1.5")))
    assert "pid" not in asyncio.run(main.health_check(SimpleNamespace(client=None)))


def test_stop_existing_server_handles_all_pids_and_skips_non_python(monkeypatch):
    monkeypatch.setattr(sx, "_listening_pids", lambda port: [10, 20, 30])

    repo = r"C:\Users\x\python\CrystalNexus"

    class P:
        def __init__(self, pid):
            self.pid = pid

        def name(self):
            return {10: "python.exe", 20: "nginx.exe", 30: "python3"}[self.pid]

        def cmdline(self):
            return {10: ["python.exe", "-m", "uvicorn", "main:app"],
                    20: ["nginx.exe", "-c", repo + r"\nginx.conf"],
                    30: ["python3", repo + r"\start_crystalnexus.py"]}[self.pid]

        def parents(self):
            return []

    monkeypatch.setattr(sx.psutil, "Process", P)
    killed = []
    monkeypatch.setattr(sx, "_kill_pid_tree", lambda pid, timeout=5, wait_root=True: killed.append(pid))
    assert sx.stop_existing_server() is True
    assert killed == [10, 30]


def test_is_crystalnexus_server_ignores_repo_path_substring():
    """The repo lives under ...\\python\\CrystalNexus; that path alone must not match."""
    class P:
        def name(self):
            return "node.exe"

        def cmdline(self):
            return [r"C:\Users\x\python\CrystalNexus\tool\node.exe", "app.js"]

        def parents(self):
            return []

    assert sx._is_crystalnexus_server(P()) is False

    class Py(P):
        def name(self):
            return "python.exe"

        def cmdline(self):
            return ["python.exe", r"C:\Users\x\python\other\server.py"]

    assert sx._is_crystalnexus_server(Py()) is False


def test_is_crystalnexus_server_accepts_reload_worker_via_parent():
    class Parent:
        def name(self):
            return "python.exe"

        def cmdline(self):
            return ["python.exe", "-m", "uvicorn", "main:app", "--reload"]

    class Worker:
        def name(self):
            return "python.exe"

        def cmdline(self):
            return ["python.exe", "-c", "from multiprocessing.spawn import spawn_main"]

        def parents(self):
            return [Parent()]

    assert sx._is_crystalnexus_server(Worker()) is True


def test_stop_existing_server_reports_kill_failure(monkeypatch, capsys):
    monkeypatch.setattr(sx, "_listening_pids", lambda port: [10])
    monkeypatch.setattr(sx.psutil, "Process", lambda pid: SimpleNamespace(pid=pid))
    monkeypatch.setattr(sx, "_is_crystalnexus_server", lambda p: True)
    monkeypatch.setattr(sx, "_kill_pid_tree", lambda pid, timeout=5, wait_root=True: "Access is denied.")
    assert sx.stop_existing_server() is False
    out = capsys.readouterr().out
    assert "Access is denied." in out
    assert "OK Existing server" not in out


def test_listening_pids_falls_back_when_access_denied(monkeypatch):
    def denied(kind=None):
        raise sx.psutil.AccessDenied()

    monkeypatch.setattr(sx.psutil, "net_connections", denied)
    conn = SimpleNamespace(status=sx.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=sx.PORT))
    other = SimpleNamespace(status=sx.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=1))
    procs = [SimpleNamespace(pid=7, net_connections=lambda kind=None: [conn]),
             SimpleNamespace(pid=8, net_connections=lambda kind=None: [other]),
             SimpleNamespace(pid=9, net_connections=lambda kind=None: (_ for _ in ()).throw(sx.psutil.AccessDenied()))]
    monkeypatch.setattr(sx.psutil, "process_iter", lambda: procs)
    assert sx._listening_pids(sx.PORT) == [7]


def test_wait_for_port_free_honours_shutdown(monkeypatch):
    monkeypatch.setattr(sx, "_port_in_use", lambda: False)
    monkeypatch.setattr(sx, "shutdown_requested", True)
    assert sx.wait_for_port_free(5) is False


def test_start_backend_does_not_spawn_when_shutdown_requested(monkeypatch):
    monkeypatch.setattr(sx, "wait_for_port_free", lambda t=0: True)
    monkeypatch.setattr(sx, "shutdown_requested", True)
    monkeypatch.setattr(sx.os, "chdir", lambda p: None)
    spawned = []
    monkeypatch.setattr(sx.subprocess, "Popen", lambda *a, **k: spawned.append(1))
    assert sx.start_backend() is None
    assert not spawned


def test_kill_process_tree_preserves_child_exit_status_posix():
    import subprocess
    import sys
    if sx.platform.system() == "Windows":
        pytest.skip("POSIX reaping semantics")
    proc = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    sx.kill_process_tree(proc)
    assert proc.returncode is not None and proc.returncode != 0


def test_health_json_helper():
    assert sx._health_json(_Resp(200, {"service": "CrystalNexus"})) == {"service": "CrystalNexus"}
    assert sx._health_json(_Resp(500, {"service": "CrystalNexus"})) is None
    assert sx._health_json(_Resp(200, {"service": "X"})) is None
    assert sx._health_json(_Resp(200, None)) is None
    assert sx._health_json(_Resp(200, [1])) is None


def test_stop_existing_server_none_found(monkeypatch):
    monkeypatch.setattr(sx, "_listening_pids", lambda port: [])
    assert sx.stop_existing_server() is False


def test_kill_process_tree_kills_tree_then_waits(monkeypatch):
    calls = []
    monkeypatch.setattr(sx, "_kill_pid_tree",
                        lambda pid, timeout=5, wait_root=True: calls.append(("kill", pid, wait_root)))
    proc = SimpleNamespace(pid=42, wait=lambda timeout=None: calls.append(("wait", timeout)))
    sx.kill_process_tree(proc, timeout=3)
    # root is our own child: reaped by Popen.wait only, not psutil.wait_procs
    assert calls == [("kill", 42, False), ("wait", 3)]


def test_kill_pid_tree_windows_uses_taskkill_tree(monkeypatch):
    monkeypatch.setattr(sx.platform, "system", lambda: "Windows")
    seen = []
    monkeypatch.setattr(sx.subprocess, "run",
                        lambda cmd, **k: seen.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(sx.psutil, "pid_exists", lambda pid: False)
    assert sx._kill_pid_tree(99) is None
    assert seen == [["taskkill", "/PID", "99", "/T", "/F"]]


def test_kill_pid_tree_windows_reports_failure(monkeypatch):
    monkeypatch.setattr(sx.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sx.subprocess, "run",
                        lambda cmd, **k: SimpleNamespace(returncode=5, stdout="", stderr="Access is denied."))
    monkeypatch.setattr(sx.psutil, "pid_exists", lambda pid: True)
    err = sx._kill_pid_tree(99, timeout=0)
    assert "Access is denied" in err


def test_kill_pid_tree_real_children_psutil_path(monkeypatch):
    """Spawn parent+child python processes and verify both die (psutil path)."""
    import subprocess
    import sys
    import time
    monkeypatch.setattr(sx.platform, "system", lambda: "Linux")
    code = ("import subprocess,sys,time;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        kids = set()
        for _ in range(50):
            kids = sx._process_tree_pids(proc.pid) - {proc.pid}
            if kids:
                break
            time.sleep(0.1)
        assert kids
        sx.kill_process_tree(proc)
        assert proc.poll() is not None
        time.sleep(0.2)
        assert not any(sx.psutil.pid_exists(k) for k in kids)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_wait_for_port_free_times_out(monkeypatch):
    monkeypatch.setattr(sx, "_port_in_use", lambda: True)
    monkeypatch.setattr(sx.time, "sleep", lambda s: None)
    t = iter([0, 1, 2, 100, 100, 100])
    monkeypatch.setattr(sx.time, "time", lambda: next(t))
    assert sx.wait_for_port_free(5) is False


def test_wait_for_port_free_true_when_free(monkeypatch):
    monkeypatch.setattr(sx, "_port_in_use", lambda: False)
    assert sx.wait_for_port_free(5) is True


def test_start_backend_errors_when_port_never_frees(monkeypatch):
    monkeypatch.setattr(sx, "wait_for_port_free", lambda t=0: False)
    monkeypatch.setattr(sx.os, "chdir", lambda p: None)
    popen_called = []
    monkeypatch.setattr(sx.subprocess, "Popen", lambda *a, **k: popen_called.append(1))
    assert sx.start_backend() is None
    assert not popen_called


def _fake_popen_setup(monkeypatch):
    proc = SimpleNamespace(pid=500, poll=lambda: None, stdout=None)
    monkeypatch.setattr(sx, "wait_for_port_free", lambda t=0: True)
    monkeypatch.setattr(sx.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(sx, "_start_output_threads",
                        lambda p, tail: (SimpleNamespace(join=lambda timeout=None: None), None))
    monkeypatch.setattr(sx.time, "sleep", lambda s: None)
    monkeypatch.setattr(sx, "_process_tree_pids", lambda pid: {500, 501})
    monkeypatch.setattr(sx.os, "chdir", lambda p: None)
    monkeypatch.setattr(sx, "server_process", None)
    return proc


def test_start_backend_ignores_old_server_pid_then_kills_on_timeout(monkeypatch):
    proc = _fake_popen_setup(monkeypatch)
    monkeypatch.setattr(sx, "MAX_STARTUP_WAIT", 3)
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "CrystalNexus", "pid": 999}))
    killed = []
    monkeypatch.setattr(sx, "kill_process_tree", lambda p, timeout=5: killed.append(p))
    assert sx.start_backend() is None
    assert killed == [proc]


def test_start_backend_accepts_worker_descendant_pid(monkeypatch):
    proc = _fake_popen_setup(monkeypatch)
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "CrystalNexus", "pid": 501}))
    assert sx.start_backend() is proc


def test_start_backend_aborts_on_shutdown_request(monkeypatch):
    proc = _fake_popen_setup(monkeypatch)
    monkeypatch.setattr(sx, "shutdown_requested", False)

    def sleep_then_ctrl_c(s):
        sx.shutdown_requested = True  # Ctrl+C arrives during the startup wait
    monkeypatch.setattr(sx.time, "sleep", sleep_then_ctrl_c)
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "CrystalNexus", "pid": 500}))
    killed = []
    monkeypatch.setattr(sx, "kill_process_tree", lambda p, timeout=5: killed.append(p))
    assert sx.start_backend() is None
    assert killed == [proc]


def test_signal_handler_kills_tree(monkeypatch):
    killed = []
    monkeypatch.setattr(sx, "kill_process_tree", lambda p, timeout=5: killed.append(p))
    monkeypatch.setattr(sx, "server_process", "PROC")
    monkeypatch.setattr(sx, "shutdown_requested", False)
    sx.signal_handler(2, None)
    assert killed == ["PROC"]


def test_monitor_flags_wrong_service(monkeypatch):
    import queue
    monkeypatch.setattr(sx, "shutdown_requested", False)
    monkeypatch.setattr(sx, "_output_threads", [])
    monkeypatch.setattr(sx.requests, "get",
                        lambda *a, **k: _Resp(200, {"service": "Other"}))

    def stop(s):
        sx.shutdown_requested = True
    monkeypatch.setattr(sx.time, "sleep", stop)
    q = queue.Queue()
    sx.monitor_process_health(SimpleNamespace(poll=lambda: None), q)
    kind, _ = q.get_nowait()
    assert kind == "health_check_failed"
    sx.shutdown_requested = False
