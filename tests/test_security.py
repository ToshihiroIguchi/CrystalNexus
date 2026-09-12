"""Tests for the security/validation functions defined in main.py.

These tests import the real implementations from main (no local copies),
so any regression in the production validation logic is caught here.
"""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pymatgen.core import Structure

import main
from main import (
    ALLOWED_ELEMENTS,
    MAX_TOTAL_SITES,
    SessionManager,
    _reject_msonable_payload,
    check_supercell_site_limit,
    require_analytics_access,
    safe_filename,
    safe_path,
    sanitize_display_filename,
    validate_element,
    validate_supercell_size,
)


# ---------------------------------------------------------------------------
# validate_element
# ---------------------------------------------------------------------------

def test_validate_element_accepts_valid_element():
    assert validate_element("Fe") == "Fe"


def test_validate_element_strips_whitespace():
    assert validate_element(" Cu ") == "Cu"


def test_validate_element_rejects_unknown_element():
    with pytest.raises(ValueError):
        validate_element("InvalidElement")


def test_validate_element_rejects_empty_string():
    with pytest.raises(ValueError):
        validate_element("")


def test_validate_element_rejects_non_string():
    with pytest.raises(ValueError):
        validate_element(42)


def test_allowed_elements_contains_common_elements():
    for element in ["H", "C", "N", "O", "Fe", "Cu", "Zn", "Al", "Si"]:
        assert element in ALLOWED_ELEMENTS, f"Element {element} should be supported"


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------

def test_safe_filename_accepts_valid_cif():
    assert safe_filename("test.cif") == "test.cif"


def test_safe_filename_strips_directory_components():
    # basename is applied, so a path collapses to its final component
    assert safe_filename("some/dir/structure.cif") == "structure.cif"


def test_safe_filename_rejects_empty():
    with pytest.raises(ValueError):
        safe_filename("")


def test_safe_filename_rejects_non_cif_extension():
    with pytest.raises(ValueError):
        safe_filename("malicious.exe")


def test_safe_filename_rejects_traversal_to_non_cif():
    with pytest.raises(ValueError):
        safe_filename("../../../etc/passwd")


def test_safe_filename_rejects_embedded_crlf():
    """Regression for S-9: a name like this used to sail through (no '/'
    or '\\', ends with .cif) and later got echoed into a
    Content-Disposition header and a CIF comment line."""
    with pytest.raises(ValueError):
        safe_filename("a\r\nX-Injected: 1.cif")


# ---------------------------------------------------------------------------
# sanitize_display_filename -- regression test for S-9
# ---------------------------------------------------------------------------

def test_sanitize_display_filename_passes_through_normal_names():
    assert sanitize_display_filename("Metals/Cu.cif") == "Metals/Cu.cif"


def test_sanitize_display_filename_strips_control_characters():
    assert sanitize_display_filename("a\r\nX-Injected: 1.cif") == "aX-Injected: 1.cif"
    assert "\n" not in sanitize_display_filename("a\nb.cif")
    assert "\r" not in sanitize_display_filename("a\rb.cif")


def test_generate_relaxed_structure_cif_sanitizes_stored_filename(client):
    """End-to-end: session_info['filename'] is echoed into both a response
    header and a CIF comment line without ever going through
    safe_path()/safe_filename() (the structure came from the session, not
    a filesystem lookup) -- sanitize_display_filename is what protects
    this path."""
    structure = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    session_id = main.session_manager.create_session(None, "evil\r\nX-Injected: 1.cif", structure)

    response = client.post("/api/generate-relaxed-structure-cif", json={"session_id": session_id})
    assert response.status_code == 200
    assert "\r" not in response.headers["content-disposition"]
    assert "\n" not in response.headers["content-disposition"]
    first_line = response.text.split("\n", 1)[0]
    assert "\r" not in first_line


# ---------------------------------------------------------------------------
# safe_path (subdirectory-aware path validation)
# ---------------------------------------------------------------------------

def test_safe_path_accepts_valid_subdirectory_path():
    assert safe_path("Metals/Cu.cif") == "Metals/Cu.cif"


def test_safe_path_normalizes_backslashes():
    assert safe_path("Metals\\Cu.cif") == "Metals/Cu.cif"


def test_safe_path_rejects_empty():
    with pytest.raises(ValueError):
        safe_path("")


def test_safe_path_rejects_simple_traversal():
    with pytest.raises(ValueError):
        safe_path("../x.cif")


def test_safe_path_rejects_nested_traversal():
    with pytest.raises(ValueError):
        safe_path("Metals/../../x.cif")


def test_safe_path_rejects_posix_absolute_path():
    with pytest.raises(ValueError):
        safe_path("/etc/x.cif")


def test_safe_path_rejects_backslash_absolute_path():
    with pytest.raises(ValueError):
        safe_path("\\Windows\\x.cif")


def test_safe_path_rejects_windows_drive_absolute_path():
    with pytest.raises(ValueError):
        safe_path("C:/Windows/x.cif")


def test_safe_path_rejects_non_cif_extension():
    with pytest.raises(ValueError):
        safe_path("Metals/notes.txt")


# ---------------------------------------------------------------------------
# validate_supercell_size
# ---------------------------------------------------------------------------

def test_validate_supercell_size_accepts_valid_size():
    assert validate_supercell_size([2, 2, 2]) == [2, 2, 2]


def test_validate_supercell_size_rejects_zero_dimension():
    with pytest.raises(ValueError):
        validate_supercell_size([0, 2, 2])


def test_validate_supercell_size_rejects_oversized_dimension():
    with pytest.raises(ValueError):
        validate_supercell_size([100, 100, 100])


def test_validate_supercell_size_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_supercell_size([1, 2])


def test_validate_supercell_size_rejects_non_integer():
    with pytest.raises(ValueError):
        validate_supercell_size([1.5, 2, 2])


# ---------------------------------------------------------------------------
# check_supercell_site_limit
#
# validate_supercell_size() bounds each dimension independently but never
# their product, so a base structure with enough sites can still blow up
# to an enormous supercell within MAX_SUPERCELL_DIM. This is the guard
# that catches the multiplicative case.
# ---------------------------------------------------------------------------

def test_check_supercell_site_limit_accepts_within_limit():
    check_supercell_site_limit(20, [10, 10, 10])  # exactly MAX_TOTAL_SITES


def test_check_supercell_site_limit_rejects_over_limit():
    with pytest.raises(HTTPException) as excinfo:
        check_supercell_site_limit(21, [10, 10, 10])  # 21000 > 20000
    assert excinfo.value.status_code == 413


def test_check_supercell_site_limit_message_reports_computed_total():
    with pytest.raises(HTTPException) as excinfo:
        check_supercell_site_limit(68, [10, 10, 10])
    assert "68000" in str(excinfo.value.detail)
    assert str(MAX_TOTAL_SITES) in str(excinfo.value.detail)


# ---------------------------------------------------------------------------
# _reject_msonable_payload
#
# Regression test for the deserialization-guard bypass: Structure.from_dict()
# round-trips each site's `properties` dict through monty.json.MontyDecoder,
# which performs a dynamic __import__(modname) + getattr + cls.from_dict()
# on any nested {"@module": ..., "@class": ...} dict. A shallow check of
# only the outer Structure dict does not see markers nested inside
# sites[*].properties.
# ---------------------------------------------------------------------------

def test_reject_msonable_payload_allows_plain_structure_dict():
    payload = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "lattice": {"matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
        "sites": [
            {"species": [{"element": "Cu", "occu": 1}], "abc": [0, 0, 0], "label": "Cu"},
        ],
    }
    _reject_msonable_payload(payload)  # must not raise


def test_reject_msonable_payload_rejects_marker_nested_in_site_properties():
    payload = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "lattice": {"matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
        "sites": [
            {
                "species": [{"element": "Cu", "occu": 1}],
                "abc": [0, 0, 0],
                "properties": {"evil": {"@module": "os", "@class": "system"}},
            },
        ],
    }
    with pytest.raises(HTTPException) as excinfo:
        _reject_msonable_payload(payload)
    assert excinfo.value.status_code == 400


def test_reject_msonable_payload_rejects_callable_marker_deeply_nested():
    payload = {"a": {"b": [{"c": {"@module": "os", "@callable": "system"}}]}}
    with pytest.raises(HTTPException):
        _reject_msonable_payload(payload)


def test_reject_msonable_payload_rejects_excessive_nesting():
    payload = {}
    node = payload
    for _ in range(40):
        node["child"] = {}
        node = node["child"]
    with pytest.raises(HTTPException) as excinfo:
        _reject_msonable_payload(payload)
    assert excinfo.value.status_code == 400


# ---------------------------------------------------------------------------
# /api/create-supercell -- end-to-end deserialization-guard regression test
# ---------------------------------------------------------------------------

def test_create_supercell_rejects_nested_msonable_payload_without_importing_it(client):
    """
    A structure_data payload with a MontyDecoder marker nested inside
    sites[*].properties must be rejected with 400, and the referenced
    module must never actually be imported.
    """
    calls = []
    real_import = builtins.__import__

    def spy_import(name, *args, **kwargs):
        calls.append(name)
        return real_import(name, *args, **kwargs)

    crystal_data = {
        "filename": "evil.cif",
        "volume": 27.0,
        "num_sites": 1,
        "formula": "Cu1",
        "structure_data": {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "lattice": {
                "matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                "pbc": [True, True, True],
                "a": 3, "b": 3, "c": 3, "alpha": 90, "beta": 90, "gamma": 90, "volume": 27,
            },
            "sites": [
                {
                    "species": [{"element": "Cu", "occu": 1}],
                    "abc": [0, 0, 0],
                    "xyz": [0, 0, 0],
                    "label": "Cu",
                    "properties": {"evil": {"@module": "antigravity", "@class": "DoesNotMatter"}},
                },
            ],
        },
    }

    builtins.__import__ = spy_import
    try:
        response = client.post("/api/create-supercell", json={
            "crystal_data": crystal_data,
            "supercell_size": [1, 1, 1],
            "session_id": "test-msonable-guard",
        })
    finally:
        builtins.__import__ = real_import

    assert response.status_code == 400
    assert "antigravity" not in calls


# ---------------------------------------------------------------------------
# /api/generate-supercell-cif-direct -- regression test for S-4a
# (this endpoint previously never called validate_supercell_size(), so an
# oversized supercell_size ran make_supercell() unbounded and unthreaded)
# ---------------------------------------------------------------------------

def test_generate_supercell_cif_direct_rejects_oversized_dimensions(client):
    response = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Metals/Cu.cif",
        "supercell_size": [500, 500, 500],
    })
    assert response.status_code == 400


def test_generate_supercell_cif_direct_rejects_malformed_size(client):
    response = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Metals/Cu.cif",
        "supercell_size": "not-a-list",
    })
    assert response.status_code == 400


def test_generate_supercell_cif_direct_enforces_total_site_cap(client):
    # Nd2Fe14B.cif has 68 base sites; 68 * 10*10*10 = 68000 > MAX_TOTAL_SITES,
    # even though [10, 10, 10] passes the per-dimension check.
    response = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Metals/Nd2Fe14B.cif",
        "supercell_size": [10, 10, 10],
    })
    assert response.status_code == 413


def test_generate_supercell_cif_direct_accepts_valid_request(client):
    response = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Metals/Cu.cif",
        "supercell_size": [2, 2, 2],
    })
    assert response.status_code == 200
    assert "data_" in response.text or "_cell_length" in response.text


# ---------------------------------------------------------------------------
# Session id issuance -- regression test for S-2 (session fixation / IDOR)
#
# Session ids are the sole authorization check on every session-scoped
# endpoint. Before this fix, the client picked the id (Date.now() +
# Math.random() in the browser) and the server trusted it outright, so
# anyone who guessed or observed another user's id could read or mutate
# their session. The server must now mint its own id and ignore a
# caller-supplied one that it has not already issued.
# ---------------------------------------------------------------------------

def test_create_supercell_ignores_attacker_chosen_session_id(client):
    analyze_response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    assert analyze_response.status_code == 200
    crystal_data = analyze_response.json()

    attacker_chosen_id = "victim-guessed-session-id"
    response = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [1, 1, 1],
        "session_id": attacker_chosen_id,
    })
    assert response.status_code == 200
    body = response.json()

    server_session_id = body["session_id"]
    assert server_session_id is not None
    # The server must never adopt an id it did not already recognize.
    assert server_session_id != attacker_chosen_id

    # The attacker-chosen id must not have been created as a side effect.
    lookup = client.post("/api/apply-atomic-operations", json={
        "session_id": attacker_chosen_id,
        "operations": [],
    })
    assert lookup.status_code == 404

    # The server-issued id, in contrast, is a live, usable session.
    lookup_ok = client.post("/api/apply-atomic-operations", json={
        "session_id": server_session_id,
        "operations": [],
    })
    assert lookup_ok.status_code == 200


def test_create_supercell_continues_a_previously_issued_session(client):
    """Re-running create-supercell with the id the server already handed
    back (e.g. switching sample files within the same browser tab) must
    keep updating that same session, not mint a new one every time."""
    analyze_response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    crystal_data = analyze_response.json()

    first = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [1, 1, 1],
        "session_id": None,
    })
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert session_id

    second = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [2, 2, 2],
        "session_id": session_id,
    })
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id


def test_session_ids_are_high_entropy(client):
    """Guards against a regression back to a short/predictable id scheme
    (the original bug was Date.now() + 9 chars of Math.random())."""
    analyze_response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    crystal_data = analyze_response.json()

    response = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [1, 1, 1],
        "session_id": None,
    })
    session_id = response.json()["session_id"]
    assert len(session_id) >= 32


# ---------------------------------------------------------------------------
# require_analytics_access -- regression test for S-3
#
# /analytics and /api/analytics/* previously had no gate at all and
# exposed every visitor's IP/User-Agent history plus uploaded
# filenames/formulas to anyone who could reach the server.
# ---------------------------------------------------------------------------

def _fake_request(client_host=None, headers=None, query_params=None):
    """Minimal stand-in for fastapi.Request -- require_analytics_access
    only touches .client.host, .headers.get(), and .query_params.get()."""
    return SimpleNamespace(
        client=SimpleNamespace(host=client_host) if client_host is not None else None,
        headers=(headers or {}),
        query_params=(query_params or {}),
    )


def test_require_analytics_access_allows_loopback_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", None)
    require_analytics_access(_fake_request(client_host="127.0.0.1"))  # must not raise


def test_require_analytics_access_rejects_non_loopback_when_no_token_configured(monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", None)
    with pytest.raises(HTTPException) as excinfo:
        require_analytics_access(_fake_request(client_host="203.0.113.5"))
    assert excinfo.value.status_code == 401


def test_require_analytics_access_rejects_missing_token_when_configured(monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "s3cr3t")
    with pytest.raises(HTTPException) as excinfo:
        require_analytics_access(_fake_request(client_host="203.0.113.5"))
    assert excinfo.value.status_code == 401


def test_require_analytics_access_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "s3cr3t")
    with pytest.raises(HTTPException) as excinfo:
        require_analytics_access(_fake_request(
            client_host="203.0.113.5", headers={"x-analytics-token": "wrong"}
        ))
    assert excinfo.value.status_code == 401


def test_require_analytics_access_accepts_correct_header_token(monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "s3cr3t")
    require_analytics_access(_fake_request(
        client_host="203.0.113.5", headers={"x-analytics-token": "s3cr3t"}
    ))  # must not raise


def test_require_analytics_access_accepts_correct_query_token(monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "s3cr3t")
    require_analytics_access(_fake_request(
        client_host="203.0.113.5", query_params={"token": "s3cr3t"}
    ))  # must not raise


# End-to-end: the dependency actually wired onto the routes.
# TestClient's synthetic client address ("testclient") is not loopback, so
# hitting these with no token configured exercises the same rejection a
# real non-local visitor would hit.

def test_analytics_summary_rejects_non_loopback_without_token(client):
    response = client.get("/api/analytics/summary")
    assert response.status_code == 401


def test_analytics_recent_rejects_non_loopback_without_token(client):
    response = client.get("/api/analytics/recent")
    assert response.status_code == 401


def test_analytics_dashboard_page_rejects_non_loopback_without_token(client):
    response = client.get("/analytics")
    assert response.status_code == 401


def test_analytics_summary_accepts_correct_header_token(client, monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "test-token-123")
    response = client.get("/api/analytics/summary", headers={"X-Analytics-Token": "test-token-123"})
    assert response.status_code == 200


def test_analytics_summary_rejects_wrong_header_token(client, monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "test-token-123")
    response = client.get("/api/analytics/summary", headers={"X-Analytics-Token": "wrong"})
    assert response.status_code == 401


def test_analytics_ranking_accepts_correct_query_token(client, monkeypatch):
    monkeypatch.setattr(main, "CRYSTALNEXUS_ANALYTICS_TOKEN", "test-token-123")
    response = client.get("/api/analytics/ranking?token=test-token-123")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Resource limits -- regression tests for S-4b/c (unbounded upload buffering
# and unbounded session storage)
# ---------------------------------------------------------------------------

def test_analyze_cif_upload_rejects_oversized_file(client, sample_cif_dir, monkeypatch):
    """MAX_FILE_SIZE must be enforced while streaming the upload, not only
    after buffering the whole body (main.py reads in bounded chunks)."""
    monkeypatch.setattr(main, "MAX_FILE_SIZE", 100)  # bytes
    cif_bytes = (sample_cif_dir / "Metals" / "Cu.cif").read_bytes()
    assert len(cif_bytes) > 100

    response = client.post(
        "/api/analyze-cif-upload",
        files={"file": ("Cu.cif", cif_bytes, "chemical/x-cif")},
    )
    assert response.status_code == 413


def test_session_manager_evicts_least_recently_accessed_when_at_capacity(monkeypatch):
    """Without a cap, an anonymous caller could grow session_manager's
    dict without limit (cleanup_old_sessions only runs on a timer and only
    evicts sessions older than SESSION_CLEANUP_HOURS)."""
    monkeypatch.setattr(main, "MAX_SESSIONS", 3)
    manager = SessionManager()
    structure = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")

    ids = [manager.create_session(None, "Cu.cif", structure) for _ in range(3)]
    assert manager.get_session_count() == 3

    # Force a deterministic recency order instead of relying on wall-clock
    # resolution between calls: ids[0] is least-recently-accessed.
    for i, sid in enumerate(ids):
        manager.sessions[sid]['last_accessed'] = i

    new_id = manager.create_session(None, "Cu.cif", structure)

    assert manager.get_session_count() == 3
    assert ids[0] not in manager.sessions  # evicted
    assert ids[1] in manager.sessions
    assert ids[2] in manager.sessions
    assert new_id in manager.sessions


def test_session_manager_updating_existing_session_never_evicts(monkeypatch):
    """Continuing an already-known session (e.g. loading a different file
    into the same browser-tab session) must not count as growth."""
    monkeypatch.setattr(main, "MAX_SESSIONS", 2)
    manager = SessionManager()
    structure = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")

    id_a = manager.create_session(None, "Cu.cif", structure)
    id_b = manager.create_session(None, "Cu.cif", structure)
    assert manager.get_session_count() == 2

    # Re-using an existing id must not evict anything, even at capacity.
    returned = manager.create_session(id_a, "Ni.cif", structure)
    assert returned == id_a
    assert manager.get_session_count() == 2
    assert id_b in manager.sessions


# ---------------------------------------------------------------------------
# rate_limit_middleware -- regression test for S-4d (no request-volume
# protection anywhere in the app; see conftest.py's autouse fixture that
# raises this limit for every other test in the suite)
# ---------------------------------------------------------------------------

def test_rate_limit_returns_429_after_burst_exceeded(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_BURST", 3)
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS_PER_MINUTE", 3)  # slow refill
    main._rate_limit_buckets.clear()

    statuses = [
        client.post("/api/apply-atomic-operations", json={}).status_code
        for _ in range(5)
    ]

    # The first requests reach the handler (400: missing session_id); once
    # the bucket is exhausted, the middleware itself returns 429 without
    # ever calling the handler.
    assert statuses[0] == 400
    assert 429 in statuses


def test_rate_limit_is_per_client_and_only_applies_to_post_api(client, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT_BURST", 2)
    monkeypatch.setattr(main, "RATE_LIMIT_REQUESTS_PER_MINUTE", 2)
    main._rate_limit_buckets.clear()

    # Exhaust the bucket for this client.
    client.post("/api/apply-atomic-operations", json={})
    client.post("/api/apply-atomic-operations", json={})
    limited = client.post("/api/apply-atomic-operations", json={})
    assert limited.status_code == 429

    # GET requests (even under /api/) are not gated by this middleware.
    response = client.get("/api/chgnet-elements")
    assert response.status_code != 429

    # A different client IP gets its own bucket.
    other_bucket = main._TokenBucket(tokens=main.RATE_LIMIT_BURST, last_refill=0)
    main._rate_limit_buckets["203.0.113.7"] = other_bucket
    assert main._rate_limit_check("203.0.113.7") is True


# ---------------------------------------------------------------------------
# Error detail genericization -- regression test for S-6
#
# pymatgen/ASE exceptions routinely embed absolute filesystem paths,
# usernames, and library versions; a 500 response must never echo str(e)
# straight to the client (it's still logged server-side for debugging).
# ---------------------------------------------------------------------------

def test_no_500_handler_echoes_raw_exception_text():
    """Static guard across all of main.py, since triggering every 500
    path individually is impractical. Covers both detail=f"...{e}" and
    the bare detail=str(e) form."""
    import re
    source = Path("main.py").read_text(encoding="utf-8")
    f_string_form = re.findall(r'status_code=500,\s*detail=f"[^"]*\{(?:str\(e\)|e)\}', source)
    bare_form = re.findall(r'status_code=500,\s*detail=str\(e\)', source)
    assert f_string_form == []
    assert bare_form == []


def test_responses_carry_security_headers(client):
    """Regression test for S-8: no security headers were set anywhere."""
    response = client.get("/health")
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Content-Security-Policy"] == "frame-ancestors 'none'"
    assert response.headers["Referrer-Policy"] == "same-origin"


def test_create_supercell_500_does_not_leak_internal_exception_details(client, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError(r"leak: C:\Users\toshi\python\CrystalNexus\uploads\secret.cif")

    monkeypatch.setattr(main, "calculate_supercell_formula", _boom)

    analyze_response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    crystal_data = analyze_response.json()

    response = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [1, 1, 1],
        "session_id": None,
    })
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret" not in response.text
    assert "uploads" not in response.text
