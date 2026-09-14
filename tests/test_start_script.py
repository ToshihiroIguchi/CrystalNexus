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
    env = sx._child_env("127.0.0.1", 8080, None)
    assert "PATH" in env or "Path" in env


def test_child_env_does_not_mutate_real_os_environ(monkeypatch):
    import os

    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    sx._child_env("0.0.0.0", 9000, "mytoken")
    assert "CRYSTALNEXUS_ANALYTICS_TOKEN" not in os.environ


def test_child_env_none_token_omits_key(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_ANALYTICS_TOKEN", raising=False)
    env = sx._child_env("127.0.0.1", 8080, None)
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
    assert sx.check_firewall_rule(8080) is None


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_none_on_nonzero_returncode(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8080) is None


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_true_on_match(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="MATCH:CrystalNexus 8080\n")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8080) is True


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only firewall check")
def test_check_firewall_rule_returns_false_on_no_match(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(sx.subprocess, "run", fake_run)
    assert sx.check_firewall_rule(8080) is False


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
