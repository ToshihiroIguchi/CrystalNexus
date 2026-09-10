"""Endpoint contract tests for CIF upload, atomic operations, and CIF generation.

These include regression tests for the HTTPException re-raise fix
(400/404 must not be masked as 500) and the safe_path traversal fix.
"""
import uuid
from pathlib import Path

import pytest

from pymatgen.core import Structure

from main import find_interstitial_candidates, apply_operations_to_structure


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


def test_upload_cif_roundtrip_through_substitution(client, sample_cif_dir):
    """Regression: uploaded (non-sample) CIFs must support the full modify
    flow -- upload -> supercell -> substitute -> regenerate CIF -- without
    404s. Uploads are saved under a random UUID filename with no reliable
    mapping back to the original name, so any step that re-resolves the
    file by name from disk (instead of from the session) breaks silently
    the moment an operation is applied."""
    cif_bytes = (sample_cif_dir / "Metals" / "Cu.cif").read_bytes()
    session_id = str(uuid.uuid4())

    uploads_dir = Path("uploads")
    before = set(uploads_dir.glob("*")) if uploads_dir.exists() else set()
    try:
        upload_response = client.post(
            "/api/analyze-cif-upload",
            files={"file": ("Cu.cif", cif_bytes, "chemical/x-cif")},
        )
        assert upload_response.status_code == 200
        crystal_data = upload_response.json()

        _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2))

        ops_response = client.post("/api/apply-atomic-operations", json={
            "session_id": session_id,
            "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
        })
        assert ops_response.status_code == 200
        assert "Ni" in ops_response.json()["composition"]

        # Before the fix, this 404'd for uploaded files: the endpoint looked
        # for "*_Cu.cif" in uploads/, but uploads are saved as "<uuid>.cif".
        cif_response = client.post("/api/generate-modified-structure-cif", json={
            "session_id": session_id,
            "filename": crystal_data["filename"],
            "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
            "supercell_size": [2, 2, 2],
        })
        assert cif_response.status_code == 200
        assert "Ni" in cif_response.text
    finally:
        if uploads_dir.exists():
            for leftover in set(uploads_dir.glob("*")) - before:
                leftover.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# apply_operations_to_structure (atomic operation replay order)
# ---------------------------------------------------------------------------

def test_apply_operations_to_structure_preserves_recorded_frame(sample_cif_dir):
    """Regression: operations must replay in the exact order recorded, each
    index referring to the structure state produced by the operations
    before it -- the same "live array" frame the client's own operation
    history uses -- not reordered by descending index against the original
    structure.

    Reproduces the reported bug: delete site 2, then (using the client's
    now-shifted array) substitute what the client sees as index 5, which is
    original site 6 since deleting site 2 shifted everything after it down
    by one. The old descending-index-first replay applied substitute@5 to
    the *original*, unshifted structure and silently hit the wrong atom."""
    structure = Structure.from_file(sample_cif_dir / "Metals" / "Cu.cif")
    structure.make_supercell((2, 2, 2))  # 32 uniform Cu sites
    original_coords = [tuple(site.frac_coords) for site in structure.sites]

    operations = [
        {"action": "delete", "index": 2},
        {"action": "substitute", "index": 5, "to": "Ni"},
    ]
    applied, skipped, property_warnings = apply_operations_to_structure(structure, operations, strict_mode=True)

    assert applied == 2
    assert skipped == []
    assert property_warnings == []

    ni_sites = [site for site in structure.sites if str(site.specie) == "Ni"]
    assert len(ni_sites) == 1
    assert tuple(ni_sites[0].frac_coords) == original_coords[6]


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


def test_apply_atomic_operations_returns_structure_state(client):
    """The response must carry server-authoritative structure state
    (formula/volume/density/labels/unique_elements) so the client can stop
    recomputing these itself from a parsed formula string."""
    from pymatgen.core.composition import Composition

    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2))

    response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
    })
    assert response.status_code == 200
    data = response.json()

    assert data["num_sites"] == 32
    assert data["density"] > 0

    counts = {str(el): int(amt) for el, amt in Composition(data["formula"]).items()}
    assert counts == {"Cu": 31, "Ni": 1}

    labels = data["labels"]
    assert len(labels) == 32
    assert len(set(labels)) == 32  # no duplicate labels
    assert labels[0] == "Ni0"


def test_apply_atomic_operations_delete_preserves_volume(client):
    """Regression: deleting an atom must not change the lattice volume
    (only atom count and density change) -- a client-side
    `volume *= (n-1)/n` approximation this response now replaces did."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2))

    before = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [],
    }).json()

    after = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [{"action": "delete", "index": 0}],
    }).json()

    assert after["num_sites"] == 31
    assert after["volume"] == pytest.approx(before["volume"])
    assert after["density"] < before["density"]


def test_apply_atomic_operations_labels_match_get_element_labels(client):
    """Both endpoints must derive labels from the same helper and agree on
    the same session's current structure."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2))

    apply_response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
    })
    labels_response = client.post("/api/get-element-labels", json={"session_id": session_id})

    assert apply_response.json()["labels"] == labels_response.json()["labels"]


def test_build_element_labels_handles_species():
    """Species (oxidation-state-bearing) sites, not just plain Elements,
    must be labeled by their element symbol (Ba2+ -> Ba0)."""
    from pymatgen.core import Lattice, Species, Structure as PmgStructure
    from main import build_element_labels

    lattice = Lattice.cubic(4.0)
    structure = PmgStructure(
        lattice,
        [Species("Ba", 2), "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    labels, unique_elements = build_element_labels(structure)
    assert labels == ["Ba0", "O0"]
    assert unique_elements == ["Ba", "O"]


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


def test_apply_atomic_operations_rejects_out_of_range_index(client):
    """Strict path (the authoritative session endpoint) must 400 on an
    out-of-range index rather than silently skipping it."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [{"action": "substitute", "index": 9999, "to": "Ni"}],
    })
    assert response.status_code == 400


def test_insert_then_substitute_inserted_atom(client):
    """Regression (Phase 1-2): substituting an atom inserted earlier in the
    same operation list must resolve against the post-insert index instead
    of being rejected as out of range against the original supercell."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2))

    num_sites = crystal_data["num_atoms"] * 8  # 2x2x2 scaling
    response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [
            {"action": "insert", "to": "Li", "coords": [0.5, 0.5, 0.5]},
            {"action": "substitute", "index": num_sites, "to": "Na"},
        ],
    })
    assert response.status_code == 200
    data = response.json()
    assert "Na" in data["composition"]
    assert "Li" not in data["composition"]
    assert data["num_sites"] == num_sites + 1


def test_update_structure_clears_relaxed_structure():
    """Regression (data-integrity bug): applying atomic operations must
    invalidate any previously relaxed structure, otherwise
    /api/generate-relaxed-structure-cif silently returns a stale,
    pre-operation relaxed cell after Analyze -> substitute."""
    from main import session_manager
    from pymatgen.core import Structure

    session_id = str(uuid.uuid4())
    structure = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    session_manager.create_session(session_id, "Cu.cif", structure)
    session_info = session_manager.get_session_info(session_id)
    session_info['relaxed_structure'] = structure.copy()
    session_info['chgnet_result'] = {'fmax': 0.1, 'converged': True, 'steps': 5}

    session_manager.update_structure(session_id, structure.copy())

    session_info = session_manager.get_session_info(session_id)
    assert 'relaxed_structure' not in session_info
    assert 'chgnet_result' not in session_info


def test_reset_session_structure_formula_shape(client):
    """Regression: structure_info.formula must be in the same
    'Cu32'-shaped form the client compares against supercell_info.formula
    (str(Structure.formula)), not str(Structure.composition) (which
    includes an oxidation-state suffix like 'Cu0+32') -- the shape
    mismatch caused a spurious "Sync issue detected" warning on every
    Reset even though nothing was actually out of sync."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(2, 2, 2))

    response = client.post("/api/reset-session-structure", json={"session_id": session_id})
    assert response.status_code == 200
    formula = response.json()["structure_info"]["formula"]
    assert formula == "Cu32"


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
    assert response.headers["x-operations-skipped"] == "0"


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


def test_generate_modified_structure_cif_with_substitution(client):
    """The operations:[] -only gap: a substitution must actually apply and
    show up both in the CIF body and in the metadata header."""
    response = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif",
        "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
        "supercell_size": [2, 2, 2],
    })
    assert response.status_code == 200
    assert "Ni" in response.text
    assert "# Final formula:" in response.text
    assert response.headers["x-operations-skipped"] == "0"


def test_generate_modified_structure_cif_lenient_skips_and_warns(client):
    """The lenient path (partial-success CIF generation) must report a
    skipped out-of-range operation via both the CIF metadata comment and
    the X-Operations-Skipped header, instead of silently dropping it."""
    response = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif",
        "operations": [{"action": "substitute", "index": 9999, "to": "Ni"}],
        "supercell_size": [1, 1, 1],
    })
    assert response.status_code == 200
    assert "# WARNING:" in response.text
    assert "skipped" in response.text
    assert response.headers["x-operations-skipped"] == "1"


def test_generate_modified_structure_cif_validates_supercell_size(client):
    """supercell_size must be validated here too (previously only
    /api/create-supercell validated it; this endpoint passed the client
    value straight to Structure.make_supercell)."""
    too_small = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif",
        "operations": [],
        "supercell_size": [0, 1, 1],
    })
    assert too_small.status_code == 400

    too_large = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif",
        "operations": [],
        "supercell_size": [99, 1, 1],
    })
    assert too_large.status_code == 400


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


# ---------------------------------------------------------------------------
# /api/evaluate-candidate-energies
# ---------------------------------------------------------------------------

def test_evaluate_candidate_energies_missing_session_id(client):
    response = client.post("/api/evaluate-candidate-energies", json={
        "candidates": [{"id": 0, "action": "delete", "index": 0}],
    })
    assert response.status_code == 400


def test_evaluate_candidate_energies_unknown_session_id(client):
    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": str(uuid.uuid4()),
        "candidates": [{"id": 0, "action": "delete", "index": 0}],
    })
    assert response.status_code == 404


def test_evaluate_candidate_energies_exceeds_max_batch(client):
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    from main import MAX_CANDIDATE_BATCH
    too_many = [{"id": i, "action": "delete", "index": 0} for i in range(MAX_CANDIDATE_BATCH + 1)]
    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": session_id,
        "candidates": too_many,
    })
    assert response.status_code == 400


def test_evaluate_candidate_energies_rejects_out_of_range_index(client):
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": session_id,
        "candidates": [{"id": 0, "action": "delete", "index": 9999}],
    })
    assert response.status_code == 400


def test_evaluate_candidate_energies_rejects_unknown_element(client):
    """validate_element must be enforced on the batch screening path too
    (previously untested)."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": session_id,
        "candidates": [{"id": 0, "action": "substitute", "index": 0, "to": "Xx"}],
    })
    assert response.status_code == 400


@pytest.mark.slow
def test_evaluate_candidate_energies_substitute_flow(client):
    """Full flow: analyze -> create supercell -> batch-evaluate substitution
    energies for every site in one CHGNet call (Auto-mode sweep)."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    num_sites = crystal_data["num_atoms"]
    candidates = [{"id": i, "action": "substitute", "index": i, "to": "Ni"} for i in range(num_sites)]

    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": session_id,
        "candidates": candidates,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["results"]) == num_sites
    for result in data["results"]:
        assert result["error"] is None
        assert isinstance(result["energy"], float)


@pytest.mark.slow
def test_evaluate_candidate_energies_delete_flow(client):
    """Full flow: analyze -> create supercell -> batch-evaluate deletion
    energies for every site in one CHGNet call (Auto-mode sweep)."""
    session_id = str(uuid.uuid4())
    crystal_data = _analyze_sample(client)
    _create_supercell_session(client, crystal_data, session_id, size=(1, 1, 1))

    num_sites = crystal_data["num_atoms"]
    candidates = [{"id": i, "action": "delete", "index": i} for i in range(num_sites)]

    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": session_id,
        "candidates": candidates,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["results"]) == num_sites
    for result in data["results"]:
        assert result["error"] is None
        assert isinstance(result["energy"], float)
