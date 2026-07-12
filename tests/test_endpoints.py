"""Endpoint contract tests for CIF upload, atomic operations, and CIF generation.

These include regression tests for the HTTPException re-raise fix
(400/404 must not be masked as 500) and the safe_path traversal fix.
"""
import uuid
from pathlib import Path

import pytest


def _analyze_sample(client, filename="Metals/Cu.cif"):
    response = client.post("/api/analyze-cif-sample", json={"filename": filename})
    assert response.status_code == 200
    return response.json()


def _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2)):
    response = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": list(size),
        "session_id": session_id,
    })
    assert response.status_code == 200
    assert response.json()["status"] == "supercell_created"


# ---------------------------------------------------------------------------
# /api/analyze-cif-upload
# ---------------------------------------------------------------------------

def test_analyze_cif_upload_valid_file(client, sample_cif_dir):
    """Uploading a valid CIF file returns its analysis"""
    cif_bytes = (sample_cif_dir / "Metals" / "Cu.cif").read_bytes()

    uploads_dir = Path("uploads")
    before = set(uploads_dir.glob("*")) if uploads_dir.exists() else set()
    try:
        response = client.post(
            "/api/analyze-cif-upload",
            files={"file": ("Cu.cif", cif_bytes, "chemical/x-cif")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "Cu.cif"
        assert "Cu" in data["formula"]
        assert data["num_atoms"] > 0
    finally:
        # Remove the file the endpoint stored in uploads/ during this test
        if uploads_dir.exists():
            for leftover in set(uploads_dir.glob("*")) - before:
                leftover.unlink(missing_ok=True)


def test_analyze_cif_upload_windows_illegal_filename(client, sample_cif_dir):
    """Filenames with Windows-illegal characters upload fine (regression: temp path fix)"""
    cif_bytes = (sample_cif_dir / "Metals" / "Cu.cif").read_bytes()

    uploads_dir = Path("uploads")
    before = set(uploads_dir.glob("*")) if uploads_dir.exists() else set()
    try:
        response = client.post(
            "/api/analyze-cif-upload",
            files={"file": ("<img src=x onerror=alert(1)>.cif", cif_bytes, "chemical/x-cif")},
        )
        # Contract: 200 — the temp path is UUID-based, so the client filename never touches the filesystem
        assert response.status_code == 200
        data = response.json()
        assert "Cu" in data["formula"]
    finally:
        # Remove the file the endpoint stored in uploads/ during this test
        if uploads_dir.exists():
            for leftover in set(uploads_dir.glob("*")) - before:
                leftover.unlink(missing_ok=True)


def test_analyze_cif_upload_rejects_non_cif_file(client):
    """Uploading a non-CIF file is rejected with 400"""
    response = client.post(
        "/api/analyze-cif-upload",
        files={"file": ("evil.txt", b"this is not a CIF file", "text/plain")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /api/apply-atomic-operations
# ---------------------------------------------------------------------------

def test_apply_atomic_operations_flow(client):
    """Full flow: analyze -> create supercell -> substitute an atom"""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id)

    response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_sites"] > 0
    assert "Ni" in data["composition"]


def test_apply_atomic_operations_missing_session_id(client):
    """Request without session_id must return 400 (regression: HTTPException fix)"""
    response = client.post("/api/apply-atomic-operations", json={"operations": []})
    assert response.status_code == 400


def test_apply_atomic_operations_unknown_session_id(client):
    """Request with unknown session_id must return 404 (regression: HTTPException fix)"""
    response = client.post("/api/apply-atomic-operations", json={
        "session_id": str(uuid.uuid4()),
        "operations": [],
    })
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /api/generate-modified-structure-cif
# ---------------------------------------------------------------------------

def test_generate_modified_structure_cif_valid(client):
    """Valid request returns a CIF document"""
    response = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif",
        "operations": [],
        "supercell_size": [1, 1, 1],
    })
    assert response.status_code == 200
    assert "chemical/x-cif" in response.headers["content-type"]
    assert "data_" in response.text  # CIF data block present
    assert "Cu" in response.text


def test_generate_modified_structure_cif_missing_filename(client):
    """Request without filename must return 400"""
    response = client.post("/api/generate-modified-structure-cif", json={
        "operations": [],
        "supercell_size": [1, 1, 1],
    })
    assert response.status_code == 400


def test_generate_modified_structure_cif_rejects_traversal(client):
    """Traversal filename must return 400 (regression: safe_path fix)"""
    # '../x.cif' keeps the .cif extension so only the traversal check can trip
    response = client.post("/api/generate-modified-structure-cif", json={
        "filename": "../x.cif",
        "operations": [],
        "supercell_size": [1, 1, 1],
    })
    assert response.status_code == 400
