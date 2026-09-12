"""Tests for the CHGNet relaxation endpoint (/api/chgnet-relax) and its
supporting helpers: request validation (main._validate_relax_params),
convergence evaluation (main.evaluate_convergence), concurrency control
(main._relax_semaphore), the timeout/abort path, and the response contract
consumed by the frontend's analysis modal.

Tests that load the real CHGNet model or run an actual relaxation are marked
@pytest.mark.slow.
"""
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymatgen.core import Structure

import main
from main import session_manager, evaluate_convergence


def _cu_structure(supercell=None):
    structure = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    if supercell is not None:
        structure.make_supercell(supercell)
    return structure


def _create_session(structure, filename="Cu.cif"):
    # session_manager.create_session() mints its own id rather than
    # trusting this one (see its docstring), so the id actually usable
    # afterward is its return value, not the local uuid.
    session_id = str(uuid.uuid4())
    return session_manager.create_session(session_id, filename, structure)


# ---------------------------------------------------------------------------
# Request validation (fast: rejected before CHGNet is ever loaded)
# ---------------------------------------------------------------------------

def test_relax_rejects_missing_session_id(client):
    response = client.post("/api/chgnet-relax", json={"fmax": 0.1, "max_steps": 10})
    assert response.status_code == 400


def test_relax_rejects_invalid_fmax(client):
    session_id = _create_session(_cu_structure())

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0, "max_steps": 10,
    })
    assert response.status_code == 400
    assert "fmax" in response.json()["detail"]

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 5.0, "max_steps": 10,
    })
    assert response.status_code == 400
    assert "fmax" in response.json()["detail"]


def test_relax_rejects_invalid_max_steps(client):
    session_id = _create_session(_cu_structure())

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 999999,
    })
    assert response.status_code == 400
    assert "max_steps" in response.json()["detail"]

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": "abc",
    })
    assert response.status_code == 400
    assert "max_steps" in response.json()["detail"]


def test_relax_rejects_invalid_optimizer(client):
    session_id = _create_session(_cu_structure())

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 10, "optimizer": "NOPE",
    })
    assert response.status_code == 400
    assert "optimizer" in response.json()["detail"]


def test_relax_rejects_missing_session(client):
    response = client.post("/api/chgnet-relax", json={
        "session_id": "nonexistent-id", "fmax": 0.1, "max_steps": 10,
    })
    assert response.status_code == 404
    assert response.json()["detail"] == "No structure found for session nonexistent-id"


def test_relax_rejects_oversized_structure(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_RELAX_ATOMS", 1)
    session_id = _create_session(_cu_structure())  # Cu.cif has 4 atoms > 1

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 10,
    })
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# evaluate_convergence unit tests (no CHGNet needed)
# ---------------------------------------------------------------------------

def test_evaluate_convergence_below_threshold():
    trajectory = SimpleNamespace(forces=[np.array([[0.01, 0, 0]])])
    assert evaluate_convergence(trajectory, fmax=0.1) is True


def test_evaluate_convergence_above_threshold():
    trajectory = SimpleNamespace(forces=[np.array([[1.0, 0, 0]])])
    assert evaluate_convergence(trajectory, fmax=0.1) is False


def test_evaluate_convergence_no_trajectory():
    assert evaluate_convergence(None, fmax=0.1) is False
    assert evaluate_convergence(SimpleNamespace(forces=[]), fmax=0.1) is False
    assert evaluate_convergence(SimpleNamespace(), fmax=0.1) is False


# ---------------------------------------------------------------------------
# Slow tests: real CHGNet load + relaxation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_relax_response_contract(client):
    """Full relax of a Cu 2x2x2 supercell; check the response shape the
    frontend's analysis modal relies on."""
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 50, "optimizer": "LBFGS",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    relaxation_info = data["relaxation_info"]
    for key in ("converged", "converged_optimizer", "steps", "fmax", "max_steps",
                "energy_change_eV", "energy_change_eV_per_atom", "optimizer_steps",
                "trajectory_frames", "optimizer", "aborted", "abort_reason"):
        assert key in relaxation_info, f"missing relaxation_info.{key}"

    final_prediction = data["final_prediction"]
    for key in ("energy_eV_per_atom", "total_energy_eV", "forces_eV_per_A",
                "site_energies_eV", "formula", "num_sites", "volume", "density"):
        assert key in final_prediction, f"missing final_prediction.{key}"

    initial_prediction = data["initial_prediction"]
    for key in ("energy_eV_per_atom", "total_energy_eV"):
        assert key in initial_prediction, f"missing initial_prediction.{key}"

    model_info = data["model_info"]
    for key in ("version", "device"):
        assert key in model_info, f"missing model_info.{key}"

    trajectory_data = data["trajectory_data"]
    for key in ("steps", "energies", "force_magnitudes", "forces"):
        assert key in trajectory_data, f"missing trajectory_data.{key}"


@pytest.mark.slow
def test_relax_second_concurrent_request_returns_429(client):
    """A second relax request while one is in flight must be rejected with 429."""
    session_id_1 = _create_session(_cu_structure(supercell=(2, 2, 2)))
    session_id_2 = _create_session(_cu_structure(supercell=(2, 2, 2)))

    first_response = {}

    def _run_first():
        resp = client.post("/api/chgnet-relax", json={
            "session_id": session_id_1, "fmax": 0.01, "max_steps": 100, "optimizer": "LBFGS",
        })
        first_response["status_code"] = resp.status_code

    thread = threading.Thread(target=_run_first)
    thread.start()
    time.sleep(1)

    try:
        second_response = client.post("/api/chgnet-relax", json={
            "session_id": session_id_2, "fmax": 0.1, "max_steps": 10,
        })
        assert second_response.status_code == 429
    finally:
        thread.join(timeout=300)

    assert first_response.get("status_code") == 200


@pytest.mark.slow
def test_relax_cif_header_includes_optimizer(client):
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))

    relax_response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 50, "optimizer": "LBFGS",
    })
    assert relax_response.status_code == 200

    cif_response = client.post("/api/generate-relaxed-structure-cif", json={
        "session_id": session_id,
    })
    assert cif_response.status_code == 200
    lines = cif_response.text.splitlines()
    assert any("optimizer=LBFGS" in line for line in lines)


@pytest.mark.slow
def test_relax_timeout_returns_partial_result(client, monkeypatch):
    monkeypatch.setattr(main, "RELAX_TIMEOUT_SECONDS", 0.001)
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))

    response = client.post("/api/chgnet-relax", json={
        "session_id": session_id, "fmax": 0.01, "max_steps": 100, "optimizer": "LBFGS",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["relaxation_info"]["aborted"] is True
    assert data["relaxation_info"]["abort_reason"] == "timeout"

    session_info = session_manager.get_session_info(session_id)
    assert "relaxed_structure" in session_info
