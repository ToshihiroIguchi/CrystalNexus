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
