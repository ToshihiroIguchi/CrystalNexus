"""Tests for the relaxation endpoint (/api/relax) and its
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
    response = client.post("/api/relax", json={"fmax": 0.1, "max_steps": 10})
    assert response.status_code == 400


def test_relax_rejects_invalid_fmax(client):
    session_id = _create_session(_cu_structure())

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0, "max_steps": 10,
    })
    assert response.status_code == 400
    assert "fmax" in response.json()["detail"]

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 5.0, "max_steps": 10,
    })
    assert response.status_code == 400
    assert "fmax" in response.json()["detail"]


def test_relax_rejects_invalid_max_steps(client):
    session_id = _create_session(_cu_structure())

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 999999,
    })
    assert response.status_code == 400
    assert "max_steps" in response.json()["detail"]

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": "abc",
    })
    assert response.status_code == 400
    assert "max_steps" in response.json()["detail"]


def test_relax_rejects_invalid_optimizer(client):
    session_id = _create_session(_cu_structure())

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 10, "optimizer": "NOPE",
    })
    assert response.status_code == 400
    assert "optimizer" in response.json()["detail"]


def test_relax_rejects_missing_session(client):
    response = client.post("/api/relax", json={
        "session_id": "nonexistent-id", "fmax": 0.1, "max_steps": 10,
    })
    assert response.status_code == 404
    assert response.json()["detail"] == "No structure found for session nonexistent-id"


def test_relax_rejects_oversized_structure(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_RELAX_ATOMS", 1)
    session_id = _create_session(_cu_structure())  # Cu.cif has 4 atoms > 1

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 10,
    })
    assert response.status_code == 413


def test_relax_cleans_up_progress_slot_when_calculator_load_fails(client, monkeypatch):
    """Regression (C5): _relax_progress[session_id] is created before the
    (potentially slow) CHGNet calculator load. If that load fails, the
    outer `finally` must still pop the slot -- otherwise it leaks forever
    and /api/relax-progress/{session_id} would wait on a session that will
    never produce a result."""
    session_id = _create_session(_cu_structure())

    async def _boom():
        raise RuntimeError("forced failure")

    monkeypatch.setattr(main.get_backend("chgnet"), "get_calculator", _boom)

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 10,
    })
    assert response.status_code == 503
    assert session_id not in main._relax_progress


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
# _sanitize_json_value -- regression test for B8 (a CHGNet numerical blow-up
# can produce NaN/Infinity energies or forces, which json.dumps() emits as
# bare, invalid-JSON literals, breaking the response and the SSE progress
# stream's JSON.parse() on the client)
# ---------------------------------------------------------------------------

def test_sanitize_json_value_replaces_non_finite_floats():
    from main import _sanitize_json_value

    assert _sanitize_json_value(float("nan")) is None
    assert _sanitize_json_value(float("inf")) is None
    assert _sanitize_json_value(float("-inf")) is None
    assert _sanitize_json_value(1.5) == 1.5


def test_sanitize_json_value_recurses_through_nested_structures():
    from main import _sanitize_json_value

    payload = {
        "energy_eV": float("nan"),
        "nested": {"max_force_eV_per_A": float("inf")},
        "list_of_values": [1.0, float("-inf"), {"a": float("nan")}],
        "unaffected": "text",
    }
    sanitized = _sanitize_json_value(payload)

    assert sanitized["energy_eV"] is None
    assert sanitized["nested"]["max_force_eV_per_A"] is None
    assert sanitized["list_of_values"] == [1.0, None, {"a": None}]
    assert sanitized["unaffected"] == "text"


def test_relaxation_watcher_sanitizes_progress_slot():
    """RelaxationWatcher must sanitize the state dict it publishes -- a
    diverging relaxation can produce a non-finite potential energy, which
    would otherwise reach the SSE stream as a bare `NaN` literal."""
    from main import RelaxationWatcher

    class _FakeAtoms:
        def get_forces(self):
            return np.array([[0.01, 0.0, 0.0]])

        def get_potential_energy(self):
            return float("nan")

    progress_slot = {}
    watcher = RelaxationWatcher(_FakeAtoms(), fmax=0.1, max_steps=10,
                                 deadline=time.monotonic() + 100, progress_slot=progress_slot)
    watcher()

    assert progress_slot["state"]["energy_eV"] is None


def test_relaxation_watcher_records_history_and_uses_force_source():
    """Each step appends to the slot's history, and max force comes from the
    force_source (the cell filter when relax_cell=True), not raw atoms."""
    from main import RelaxationWatcher, _new_history_points

    class _FakeAtoms:
        def get_forces(self):
            return np.array([[0.01, 0.0, 0.0]])

        def get_potential_energy(self):
            return -1.5

    class _FakeFilter:
        def get_forces(self):
            # extra cell-DOF row with a larger "force"
            return np.array([[0.01, 0.0, 0.0], [0.0, 0.5, 0.0]])

    slot = {}
    watcher = RelaxationWatcher(_FakeAtoms(), fmax=0.1, max_steps=10,
                                 deadline=time.monotonic() + 100, progress_slot=slot,
                                 force_source=_FakeFilter())
    watcher()
    watcher()

    assert [h["step"] for h in slot["history"]] == [0, 1]
    assert slot["history"][0] == {"step": 0, "energy_eV": -1.5, "max_force_eV_per_A": 0.5}
    assert slot["state"]["max_force_eV_per_A"] == 0.5

    # Per-connection diffing: only entries beyond what was already sent
    assert _new_history_points(slot, 0) == slot["history"]
    assert _new_history_points(slot, 1) == slot["history"][1:]
    assert _new_history_points(slot, 2) == []
    assert _new_history_points({}, 0) == []


# ---------------------------------------------------------------------------
# Slow tests: real CHGNet load + relaxation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_relax_response_contract(client):
    """Full relax of a Cu 2x2x2 supercell; check the response shape the
    frontend's analysis modal relies on."""
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))

    response = client.post("/api/relax", json={
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
    for key in ("id", "display_name", "version", "device"):
        assert key in model_info, f"missing model_info.{key}"
    assert model_info["id"] == "chgnet"
    assert model_info["display_name"] == "CHGNet"
    assert session_manager.get_session_info(session_id)["relax_result"]["model"] == "chgnet"

    trajectory_data = data["trajectory_data"]
    for key in ("steps", "energies", "force_magnitudes", "forces"):
        assert key in trajectory_data, f"missing trajectory_data.{key}"


@pytest.mark.slow
def test_relax_trajectory_energies_are_per_atom(client):
    """Regression (B3): trajectory.energies (TrajectoryRecorder) is TOTAL
    energy per step, but the response's trajectory_data.energies is
    documented/consumed as eV/atom (matching final_prediction's
    energy_eV_per_atom). The last trajectory energy must be close to
    energy_eV_per_atom, not off by a factor of num_atoms (the total)."""
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))  # 32 atoms

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 20, "optimizer": "LBFGS",
    })
    assert response.status_code == 200
    data = response.json()

    energy_per_atom = data["final_prediction"]["energy_eV_per_atom"]
    trajectory_energies = data["trajectory_data"]["energies"]
    assert len(trajectory_energies) > 0

    last_energy = trajectory_energies[-1]
    assert last_energy == pytest.approx(energy_per_atom, abs=0.05)
    # Sanity check that this isn't accidentally still the total energy
    # (which would be ~32x larger in magnitude for this 32-atom structure).
    assert abs(last_energy - energy_per_atom * 32) > 1.0


@pytest.mark.slow
def test_relax_second_concurrent_request_returns_429(client):
    """A second relax request while one is in flight must be rejected with 429."""
    session_id_1 = _create_session(_cu_structure(supercell=(2, 2, 2)))
    session_id_2 = _create_session(_cu_structure(supercell=(2, 2, 2)))

    first_response = {}

    def _run_first():
        resp = client.post("/api/relax", json={
            "session_id": session_id_1, "fmax": 0.01, "max_steps": 100, "optimizer": "LBFGS",
        })
        first_response["status_code"] = resp.status_code

    thread = threading.Thread(target=_run_first)
    thread.start()
    time.sleep(1)

    try:
        second_response = client.post("/api/relax", json={
            "session_id": session_id_2, "fmax": 0.1, "max_steps": 10,
        })
        assert second_response.status_code == 429
    finally:
        thread.join(timeout=300)

    assert first_response.get("status_code") == 200


@pytest.mark.slow
def test_relax_stale_result_not_saved_after_concurrent_edit(client):
    """If the session's structure is edited while a relaxation on it is still
    running, the relaxation's own completion must not resurrect
    relaxed_structure/relax_result for a structure that is no longer current."""
    structure = _cu_structure(supercell=(2, 2, 2))
    session_id = _create_session(structure)

    relax_thread_response = {}

    def _run_relax():
        resp = client.post("/api/relax", json={
            "session_id": session_id, "fmax": 0.01, "max_steps": 500, "optimizer": "LBFGS",
        })
        relax_thread_response["status_code"] = resp.status_code

    thread = threading.Thread(target=_run_relax)
    thread.start()
    time.sleep(1)  # let the relaxation start

    # Simulate a same-session structure edit (e.g. atom substitution) while
    # the relaxation above is still running.
    edited_structure = _cu_structure()  # different atom count from the 2x2x2 supercell
    session_manager.update_structure(session_id, edited_structure)

    thread.join(timeout=300)
    assert relax_thread_response.get("status_code") == 200

    session_info = session_manager.get_session_info(session_id)
    # The edit must win: no stale relaxed_structure/relax_result from the
    # relaxation that was running against the pre-edit structure.
    assert "relaxed_structure" not in session_info
    assert "relax_result" not in session_info
    assert session_info["current_structure"] is edited_structure


@pytest.mark.slow
def test_relax_cif_header_includes_optimizer(client):
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))

    relax_response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 50, "optimizer": "LBFGS",
    })
    assert relax_response.status_code == 200

    cif_response = client.post("/api/generate-relaxed-structure-cif", json={
        "session_id": session_id,
    })
    assert cif_response.status_code == 200
    lines = cif_response.text.splitlines()
    assert any("optimizer=LBFGS" in line for line in lines)
    assert any(line.startswith("# CHGNet relaxation: model=chgnet,") for line in lines)


@pytest.mark.slow
def test_relax_timeout_returns_partial_result(client, monkeypatch):
    monkeypatch.setattr(main, "RELAX_TIMEOUT_SECONDS", 0.001)
    session_id = _create_session(_cu_structure(supercell=(2, 2, 2)))

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.01, "max_steps": 100, "optimizer": "LBFGS",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["relaxation_info"]["aborted"] is True
    assert data["relaxation_info"]["abort_reason"] == "timeout"

    session_info = session_manager.get_session_info(session_id)
    assert "relaxed_structure" in session_info


# ---------------------------------------------------------------------------
# Symmetry-analysis failure after relaxation / deadline start
# ---------------------------------------------------------------------------

def test_relax_survives_spacegroup_analyzer_failure(client, monkeypatch):
    """A spglib failure on the relaxed structure must not discard the
    relaxation: the endpoint still returns 200 (symmetry fields None) and
    the relaxed structure is saved to the session."""
    structure = _cu_structure()
    session_id = _create_session(structure)

    async def _fake_calculator():
        return object()

    def _fake_run_relaxation(calculator, structure, **kwargs):
        n = len(structure)

        class _Trajectory(SimpleNamespace):
            def __len__(self):
                return len(self.energies)

        trajectory = _Trajectory(forces=[np.zeros((n, 3))], energies=[0.0],
                                 stresses=[], atom_positions=[], cells=[])
        return {"final_structure": structure.copy(), "trajectory": trajectory,
                "ase_converged": True, "optimizer_steps": 1, "aborted": False}

    def _boom(*args, **kwargs):
        raise RuntimeError("spglib failure")

    monkeypatch.setattr(main.get_backend("chgnet"), "get_calculator", _fake_calculator)
    monkeypatch.setattr(main, "run_relaxation", _fake_run_relaxation)
    monkeypatch.setattr("pymatgen.symmetry.analyzer.SpacegroupAnalyzer", _boom)

    response = client.post("/api/relax", json={
        "session_id": session_id, "fmax": 0.1, "max_steps": 10,
    })
    assert response.status_code == 200
    info = response.json()["relaxed_structure_info"]
    for key in ("space_group", "space_group_number", "point_group", "crystal_system"):
        assert info[key] is None
    assert "relaxed_structure" in session_manager.get_session_info(session_id)


def test_run_relaxation_aborts_when_timeout_already_exhausted():
    """The deadline is computed inside run_relaxation from timeout_s, so a
    non-positive timeout aborts at the first observer call."""
    from ase.calculators.emt import EMT

    class _EMTWithMagmoms(EMT):
        # TrajectoryRecorder reads magnetic moments, which plain EMT lacks.
        implemented_properties = EMT.implemented_properties + ["magmoms"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            super().calculate(atoms, properties, system_changes or [])
            self.results["magmoms"] = np.zeros(len(atoms))

    result = main.run_relaxation(_EMTWithMagmoms(), _cu_structure(supercell=(2, 2, 2)),
                                 fmax=0.001, steps=50, optimizer_name="FIRE",
                                 timeout_s=-1.0, relax_cell=False)
    assert result["aborted"] is True
    assert result["final_structure"] is not None


def test_evaluate_convergence_uses_recorded_max_force_with_cell_filter():
    # Atomic forces are below fmax but the cell-filter force (recorded
    # max_forces) is not: the reported flag must follow the latter.
    trajectory = SimpleNamespace(forces=[np.array([[0.01, 0, 0]])], max_forces=[0.5])
    assert evaluate_convergence(trajectory, fmax=0.1) is False
    trajectory = SimpleNamespace(forces=[np.array([[1.0, 0, 0]])], max_forces=[0.01])
    assert evaluate_convergence(trajectory, fmax=0.1) is True


def test_trajectory_recorder_records_force_source_max_force():
    from main import TrajectoryRecorder
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from ase.filters import FrechetCellFilter
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [0, 0, 4.0]], cell=[10, 10, 10], pbc=True)
    atoms.calc = LennardJones()
    atoms.get_magnetic_moments = lambda: np.zeros(len(atoms))  # LJ has no magmoms
    rec = TrajectoryRecorder(atoms, force_source=FrechetCellFilter(atoms))
    rec()
    expected = float(np.linalg.norm(FrechetCellFilter(atoms).get_forces(), axis=1).max())
    assert rec.max_forces == [expected]
