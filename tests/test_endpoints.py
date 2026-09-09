"""Endpoint contract tests for CIF upload, atomic operations, and CIF generation.

These include regression tests for the HTTPException re-raise fix
(400/404 must not be masked as 500) and the safe_path traversal fix.
"""
import uuid
from pathlib import Path

import pytest

from pymatgen.core import Structure

from main import find_interstitial_candidates


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


# ---------------------------------------------------------------------------
# find_interstitial_candidates (replaces pymatgen's VoronoiInterstitialGenerator)
#
# Regression coverage for the hang reported against atom insertion: the old
# VoronoiInterstitialGenerator's StructureMatcher-based symmetry grouping is
# O(C^2) full structure comparisons and took ~7 minutes on a 3x3x3 Cu supercell.
# find_interstitial_candidates must return the same sites in well under a second.
# ---------------------------------------------------------------------------

def _reference_void_distances(structure, element="Li", cutoff=5.0):
    """Reference implementation using pymatgen's own VoronoiInterstitialGenerator."""
    from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
    generator = VoronoiInterstitialGenerator()
    defects = generator.get_defects(structure, insert_species=[element])
    distances = []
    for defect in defects:
        neighbors = structure.get_neighbors(defect.site, cutoff)
        distances.append(round(min(n.nn_distance for n in neighbors), 3) if neighbors else 0.0)
    return sorted(distances)


def test_find_interstitial_candidates_matches_reference_small_cell(sample_cif_dir):
    """Fast path (<5-atom unit cell): must match pymatgen's own generator exactly."""
    structure = Structure.from_file(sample_cif_dir / "Metals" / "Cu.cif")
    fast_distances = sorted(c["min_dist"] for c in find_interstitial_candidates(structure))
    assert fast_distances == _reference_void_distances(structure)


@pytest.mark.slow
def test_find_interstitial_candidates_matches_reference_supercell(sample_cif_dir):
    """2x2x2 supercell: same octahedral/tetrahedral FCC sites as the reference generator."""
    structure = Structure.from_file(sample_cif_dir / "Metals" / "Cu.cif")
    structure.make_supercell((2, 2, 2))
    fast_distances = sorted(c["min_dist"] for c in find_interstitial_candidates(structure))
    assert fast_distances == _reference_void_distances(structure)


def test_find_interstitial_candidates_vacuum_structure_does_not_collapse(sample_cif_dir):
    """A molecule-in-a-box structure has no atoms within the fingerprint cutoff for
    most candidate sites; those must stay distinct instead of collapsing to one."""
    structure = Structure.from_file(sample_cif_dir / "Gases" / "CO2(gas).cif")
    candidates = find_interstitial_candidates(structure)
    assert len(candidates) > 1


def test_find_interstitial_candidates_respects_max_candidates(sample_cif_dir):
    structure = Structure.from_file(sample_cif_dir / "Metals" / "Nd2Fe14B.cif")
    candidates = find_interstitial_candidates(structure, max_candidates=5)
    assert len(candidates) <= 5
    # ids/labels stay 0-based and contiguous after truncation
    assert [c["id"] for c in candidates] == list(range(len(candidates)))


# ---------------------------------------------------------------------------
# /api/get-insertion-voids
# ---------------------------------------------------------------------------

def test_get_insertion_voids_missing_session_id(client):
    response = client.post("/api/get-insertion-voids", json={"element": "Li"})
    assert response.status_code == 400


def test_get_insertion_voids_unknown_session_id(client):
    response = client.post("/api/get-insertion-voids", json={
        "session_id": str(uuid.uuid4()),
        "element": "Li",
    })
    assert response.status_code == 404


@pytest.mark.slow
def test_get_insertion_voids_flow(client):
    """Full flow: analyze -> create supercell -> find insertion voids."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    response = client.post("/api/get-insertion-voids", json={
        "session_id": session_id,
        "element": "Li",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["voids"]) > 0
    assert {"id", "label", "frac_coords", "min_dist"} <= data["voids"][0].keys()


# ---------------------------------------------------------------------------
# /api/evaluate-insertion-energies
# ---------------------------------------------------------------------------

def test_evaluate_insertion_energies_missing_session_id(client):
    response = client.post("/api/evaluate-insertion-energies", json={
        "element": "Li",
        "sites": [{"id": 0, "frac_coords": [0.5, 0.5, 0.5]}],
    })
    assert response.status_code == 400


def test_evaluate_insertion_energies_unknown_session_id(client):
    response = client.post("/api/evaluate-insertion-energies", json={
        "session_id": str(uuid.uuid4()),
        "element": "Li",
        "sites": [{"id": 0, "frac_coords": [0.5, 0.5, 0.5]}],
    })
    assert response.status_code == 404


def test_evaluate_insertion_energies_exceeds_max_batch(client):
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    from main import MAX_INSERTION_BATCH
    too_many_sites = [{"id": i, "frac_coords": [0.1, 0.1, 0.1]} for i in range(MAX_INSERTION_BATCH + 1)]
    response = client.post("/api/evaluate-insertion-energies", json={
        "session_id": session_id,
        "element": "Li",
        "sites": too_many_sites,
    })
    assert response.status_code == 400


@pytest.mark.slow
def test_evaluate_insertion_energies_flow(client):
    """Full flow: analyze -> create supercell -> batch-evaluate insertion energies."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    voids_response = client.post("/api/get-insertion-voids", json={
        "session_id": session_id,
        "element": "Li",
    })
    assert voids_response.status_code == 200
    voids = voids_response.json()["voids"]
    assert len(voids) > 0

    response = client.post("/api/evaluate-insertion-energies", json={
        "session_id": session_id,
        "element": "Li",
        "sites": [{"id": v["id"], "frac_coords": v["frac_coords"]} for v in voids],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["results"]) == len(voids)
    for result in data["results"]:
        assert result["error"] is None
        assert isinstance(result["energy"], float)
