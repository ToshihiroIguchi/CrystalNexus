"""Regression tests for backend validation fixes: junk CIF handling,
operations validation, session supercell fallback, insertion distance checks,
atom-count status codes, rate-limit eviction and off-loop analytics."""
import asyncio
import time
import uuid

import pytest
from pymatgen.core import Structure

import main
from main import find_interstitial_candidates
from tests.test_endpoints import _analyze_sample, _create_supercell_session


def _session(client, size=(1, 1, 1)):
    return _create_supercell_session(client, _analyze_sample(client), str(uuid.uuid4()), size=size)


# ---------------------------------------------------------------------------
# B1 / B11: CIF analysis errors and upload parsing
# ---------------------------------------------------------------------------

def test_analyze_cif_upload_junk_content_is_400(client):
    response = client.post(
        "/api/analyze-cif-upload",
        files={"file": ("junk.cif", b"hello", "chemical/x-cif")},
    )
    assert response.status_code == 400


def test_analyze_cif_upload_parses_structure_only_once(client, sample_cif_dir, monkeypatch):
    calls = []
    original = Structure.from_file.__func__

    def _counting(cls, *args, **kwargs):
        calls.append(args)
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(Structure, "from_file", classmethod(_counting))
    cif_bytes = (sample_cif_dir / "Metals" / "Cu.cif").read_bytes()
    response = client.post(
        "/api/analyze-cif-upload",
        files={"file": ("Cu.cif", cif_bytes, "chemical/x-cif")},
    )
    assert response.status_code == 200
    assert "structure_data" in response.json()
    assert len(calls) == 1


def test_analyze_cif_upload_rejects_invalid_filename_before_parsing(client, sample_cif_dir,
                                                                    monkeypatch):
    def _reject(name):
        raise ValueError("Invalid filename")

    def _must_not_parse(*args, **kwargs):
        raise AssertionError("file was parsed before the filename was validated")

    monkeypatch.setattr(main, "safe_filename", _reject)
    monkeypatch.setattr(main, "_analyze_cif_file_and_structure_sync", _must_not_parse)
    cif_bytes = (sample_cif_dir / "Metals" / "Cu.cif").read_bytes()
    response = client.post(
        "/api/analyze-cif-upload",
        files={"file": ("Cu.cif", cif_bytes, "chemical/x-cif")},
    )
    assert response.status_code == 400


def test_apply_atomic_operations_post_operation_site_limit_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_TOTAL_SITES", 5)  # Cu.cif has 4 atoms
    session_id = _session(client)
    response = client.post("/api/apply-atomic-operations", json={
        "session_id": session_id,
        "operations": [_FREE_INSERT, {**_FREE_INSERT, "coords": [0.75, 0.75, 0.75]}],
    })
    assert response.status_code == 413


def test_analyze_cif_sample_junk_content_is_400(client, monkeypatch, tmp_path):
    (tmp_path / "junk.cif").write_text("hello")
    monkeypatch.setattr(main, "SAMPLE_CIF_DIR", tmp_path)
    response = client.post("/api/analyze-cif-sample", json={"filename": "junk.cif"})
    assert response.status_code == 400


def test_analyze_cif_symmetry_failure_is_400(client, monkeypatch, sample_cif_dir):
    def _boom(*args, **kwargs):
        raise RuntimeError("spglib failed")

    monkeypatch.setattr(main, "SpacegroupAnalyzer", _boom)
    with pytest.raises(ValueError):
        main.analyze_cif_file_sync(sample_cif_dir / "Metals" / "Cu.cif")
    response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# B2: whitespace in substitution target
# ---------------------------------------------------------------------------

def test_evaluate_candidate_energies_strips_substitution_element(client, fake_backends):
    session_id = _session(client)
    response = client.post("/api/evaluate-candidate-energies", json={
        "session_id": session_id,
        "candidates": [{"id": 0, "action": "substitute", "index": 0, "to": " Al"}],
    })
    assert response.status_code == 200
    assert response.json()["results"][0]["energy"] is not None


# ---------------------------------------------------------------------------
# B3: operations validation for generate-modified-structure-cif and predict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["/api/generate-modified-structure-cif", "/api/predict"])
def test_operations_must_be_a_list(client, fake_backends, url):
    for bad in (None, "x", {"action": "delete"}):
        response = client.post(url, json={
            "filename": "Metals/Cu.cif", "operations": bad, "supercell_size": [1, 1, 1],
        })
        assert response.status_code == 400, bad
        assert "operations must be a list" in response.json()["detail"]


@pytest.mark.parametrize("url", ["/api/generate-modified-structure-cif", "/api/predict"])
def test_operations_capped_per_request(client, fake_backends, monkeypatch, url):
    monkeypatch.setattr(main, "MAX_OPERATIONS_PER_REQUEST", 2)
    response = client.post(url, json={
        "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1],
        "operations": [{"action": "delete", "index": 0}] * 3,
    })
    assert response.status_code == 400


_FREE_INSERT = {"action": "insert", "to": "Cu", "coords": [0.25, 0.25, 0.25]}


def test_generate_modified_structure_cif_rechecks_site_limit_after_inserts(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_TOTAL_SITES", 4)  # Cu.cif has 4 atoms
    response = client.post("/api/generate-modified-structure-cif", json={
        "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1],
        "operations": [_FREE_INSERT],
    })
    assert response.status_code == 413


def test_predict_rechecks_atom_limit_after_inserts(client, fake_backends, monkeypatch):
    monkeypatch.setattr(main, "MAX_PREDICT_ATOMS", 4)
    response = client.post("/api/predict", json={
        "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1],
        "operations": [_FREE_INSERT],
    })
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# B4: absent supercell_size falls back to the session's size
# ---------------------------------------------------------------------------

def test_generate_modified_structure_cif_uses_session_supercell_size(client, fake_backends):
    session_id = _session(client, size=(2, 2, 2))
    body = {"session_id": session_id, "filename": "Cu.cif", "operations": []}
    response = client.post("/api/generate-modified-structure-cif", json=body)
    assert response.status_code == 200
    assert "# Number of atoms: 32" in response.text
    assert "# Supercell size: 2x2x2" in response.text

    mismatch = client.post("/api/generate-modified-structure-cif",
                           json={**body, "supercell_size": [1, 1, 1]})
    assert mismatch.status_code == 400
    same = client.post("/api/generate-modified-structure-cif",
                       json={**body, "supercell_size": [2, 2, 2]})
    assert same.status_code == 200


def test_predict_uses_session_supercell_size(client, fake_backends):
    session_id = _session(client, size=(2, 2, 2))
    body = {"session_id": session_id, "filename": "Cu.cif", "operations": []}
    response = client.post("/api/predict", json=body)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction["num_sites"] == 32
    assert prediction["supercell_size"] == [2, 2, 2]
    assert client.post("/api/predict", json={**body, "supercell_size": [1, 1, 1]}).status_code == 400


# ---------------------------------------------------------------------------
# B5: minimum insertion distance on the insertion-energy endpoints
# ---------------------------------------------------------------------------

def test_evaluate_insertion_energy_rejects_overlapping_site(client, fake_backends):
    session_id = _session(client)
    response = client.post("/api/evaluate-insertion-energy", json={
        "session_id": session_id, "element": "Cu", "frac_coords": [0.0, 0.0, 0.0],
    })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error" and body["energy"] is None
    assert "closer than" in body["detail"]


def test_evaluate_insertion_energies_flags_overlapping_site(client, fake_backends):
    session_id = _session(client)
    response = client.post("/api/evaluate-insertion-energies", json={
        "session_id": session_id, "element": "Cu",
        "sites": [{"id": "bad", "frac_coords": [0.0, 0.0, 0.0]},
                  {"id": "ok", "frac_coords": [0.25, 0.25, 0.25]}],
    })
    assert response.status_code == 200
    results = {r["id"]: r for r in response.json()["results"]}
    assert results["bad"]["energy"] is None and "closer than" in results["bad"]["error"]
    assert results["ok"]["error"] is None and results["ok"]["energy"] is not None


# ---------------------------------------------------------------------------
# B6: generate-supercell-cif-direct
# ---------------------------------------------------------------------------

def test_generate_supercell_cif_direct_unknown_session_is_404(client):
    response = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Metals/Cu.cif", "supercell_size": [1, 1, 1],
        "session_id": str(uuid.uuid4()),
    })
    assert response.status_code == 404


def test_generate_supercell_cif_direct_session_and_sample(client, fake_backends):
    session_id = _session(client)
    from_session = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Cu.cif", "supercell_size": [2, 1, 1], "session_id": session_id})
    assert from_session.status_code == 200
    assert "# Number of atoms: 8" in from_session.text
    assert "# Supercell size: 2x1x1" in from_session.text

    from_sample = client.post("/api/generate-supercell-cif-direct", json={
        "filename": "Metals/Cu.cif", "supercell_size": [1, 2, 1]})
    assert from_sample.status_code == 200
    assert "# Number of atoms: 8" in from_sample.text


# ---------------------------------------------------------------------------
# B7 / B8
# ---------------------------------------------------------------------------

def test_create_supercell_null_filename_defaults(client, fake_backends):
    crystal_data = _analyze_sample(client)
    crystal_data["filename"] = None
    session_id = _create_supercell_session(client, crystal_data, str(uuid.uuid4()), size=(1, 1, 1))
    assert main.session_manager.get_session_info(session_id)["filename"] == "unknown.cif"


def test_find_interstitial_candidates_vacuum_sites_rank_first(sample_cif_dir):
    """Sites with no atom inside the fingerprint cutoff are the largest voids:
    they must report the cutoff as min_dist and not be ranked last."""
    structure = Structure.from_file(sample_cif_dir / "Gases" / "CO2(gas).cif")
    candidates = find_interstitial_candidates(structure)
    dists = [c["min_dist"] for c in candidates]
    assert dists == sorted(dists, reverse=True)
    assert dists[0] >= 6.0


# ---------------------------------------------------------------------------
# B10: atom-count limits answer 413
# ---------------------------------------------------------------------------

def test_atom_count_limits_use_413(client, fake_backends, monkeypatch):
    session_id = _session(client, size=(2, 2, 2))
    monkeypatch.setattr(main, "MAX_PREDICT_ATOMS", 1)
    for url, extra in [
        ("/api/get-insertion-voids", {"element": "Li"}),
        ("/api/evaluate-insertion-energy", {"element": "Cu", "frac_coords": [0.25] * 3}),
        ("/api/evaluate-insertion-energies",
         {"element": "Cu", "sites": [{"id": 0, "frac_coords": [0.25] * 3}]}),
        ("/api/evaluate-candidate-energies",
         {"candidates": [{"id": 0, "action": "delete", "index": 0}]}),
        ("/api/equivalent-sites", {"element": "Cu"}),
    ]:
        response = client.post(url, json={"session_id": session_id, **extra})
        assert response.status_code == 413, url


def test_relax_steps_reports_optimizer_steps(client, fake_backends):
    session_id = _session(client, size=(2, 2, 2))
    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.05, "max_steps": 20, "optimizer": "LBFGS"})
    assert response.status_code == 200
    info = response.json()["relaxation_info"]
    assert info["steps"] == info["optimizer_steps"]
    assert info["trajectory_frames"] >= info["optimizer_steps"]


# ---------------------------------------------------------------------------
# B14 / rate-limit eviction
# ---------------------------------------------------------------------------

def test_rate_limit_evicts_stale_buckets():
    main._rate_limit_buckets["stale-client"] = main._TokenBucket(
        tokens=0, last_refill=time.time() - 100_000)
    main._rate_limit_last_evict = 0.0
    assert main._rate_limit_check("fresh-client")
    assert "stale-client" not in main._rate_limit_buckets
    assert "fresh-client" in main._rate_limit_buckets
    main._rate_limit_buckets.pop("fresh-client", None)


def test_analytics_log_event_runs_off_the_event_loop(client, monkeypatch):
    on_loop = []

    class _Recorder:
        def log_event(self, **kwargs):
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)

        def log_access(self, **kwargs):
            pass

    monkeypatch.setattr(main, "analytics_db", _Recorder())
    response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    assert response.status_code == 200
    assert on_loop == [False]
