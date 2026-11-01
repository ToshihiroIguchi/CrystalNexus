"""Tests for elastic-constant calculation (mlip/elastic.py and /api/relax elastic=true).

Fast tests use the EMT-backed fake backend; real-model tests are marked
slow / mace.
"""
import uuid
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

import main
from main import session_manager
import inspect

from ase.build import bulk
from ase.calculators.emt import EMT

from mlip import elastic as elastic_mod
from mlip.elastic import ELASTIC_ION_FMAX, ELASTIC_ION_MAX_STEPS, compute_elastic, elastic_moduli

# Approximate experimental-like cubic constants for Cu (GPa).
CU_C11, CU_C12, CU_C44 = 170.0, 120.0, 75.0


def cubic_C(c11, c12, c44):
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            C[i, j] = c12
        C[i, i] = c11
    for i in range(3, 6):
        C[i, i] = c44
    return C


def _cu_conv(a=3.61):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Cu"], [[0, 0, 0]])


def _session(structure=None):
    return session_manager.create_session(
        str(uuid.uuid4()), "Cu.cif", structure if structure is not None else _cu_conv())


def _relax(client, sid, **extra):
    body = {"session_id": sid, "fmax": 0.01, "max_steps": 100, "optimizer": "LBFGS"}
    body.update(extra)
    return client.post("/api/relax", json=body)


# -- elastic_moduli (pure) ---------------------------------------------

def test_elastic_moduli_cubic_analytic():
    res = elastic_moduli(cubic_C(CU_C11, CU_C12, CU_C44))
    K = (CU_C11 + 2 * CU_C12) / 3
    G_v = (CU_C11 - CU_C12 + 3 * CU_C44) / 5
    G_r = 5 * (CU_C11 - CU_C12) * CU_C44 / (4 * CU_C44 + 3 * (CU_C11 - CU_C12))
    G_h = (G_v + G_r) / 2
    m = res["moduli"]
    assert res["stable"] is True
    assert m["K"]["voigt"] == pytest.approx(K)
    assert m["K"]["reuss"] == pytest.approx(K)  # identical for cubic crystals
    assert m["K"]["hill"] == pytest.approx(K)
    assert m["G"]["voigt"] == pytest.approx(G_v)
    assert m["G"]["reuss"] == pytest.approx(G_r)
    assert m["G"]["hill"] == pytest.approx(G_h)
    assert m["E"]["hill"] == pytest.approx(9 * K * G_h / (3 * K + G_h))
    assert m["nu"]["hill"] == pytest.approx((3 * K - 2 * G_h) / (2 * (3 * K + G_h)))
    assert res["min_eigenvalue_GPa"] == pytest.approx(
        min(CU_C11 - CU_C12, CU_C44, CU_C11 + 2 * CU_C12))
    assert np.asarray(res["C_GPa"]).shape == (6, 6)


def test_elastic_moduli_unstable():
    res = elastic_moduli(cubic_C(100.0, 130.0, 50.0))  # C11 < C12
    assert res["stable"] is False
    assert res["min_eigenvalue_GPa"] < 0
    assert res["moduli"]["K"]["reuss"] is None
    assert res["moduli"]["E"]["hill"] is None
    assert res["moduli"]["K"]["voigt"] is not None


def test_elastic_moduli_singular():
    res = elastic_moduli(np.zeros((6, 6)))
    assert res["stable"] is False
    assert res["moduli"]["G"]["reuss"] is None


def test_elastic_moduli_bad_shape():
    with pytest.raises(ValueError):
        elastic_moduli(np.eye(3))


# -- /api/relax with the fake EMT backend --------------------------------

def test_relax_elastic_emt_cu(client, fake_backends):
    data = _relax(client, _session(), elastic=True, elastic_strain=0.25).json()
    info = data["elastic_info"]
    assert "error" not in info
    C = np.asarray(info["C_GPa"])
    assert C.shape == (6, 6)
    assert C[0, 0] > C[0, 1] > 0
    assert C[3, 3] > 0
    assert info["stable"] is True
    assert info["moduli"]["E"]["hill"] > 0
    assert 0 < info["moduli"]["nu"]["hill"] < 0.5
    assert info["strain"] == pytest.approx(0.0025)
    assert info["n_strained_cells"] == 24
    assert info["consistency"]["consistent"] is True
    assert info["consistency"]["strain_half"] == pytest.approx(0.00125)
    assert set(info["consistency"]) == {
        "strain_half", "max_abs_diff_GPa", "consistent", "worst_component"}
    assert not any("strain magnitude" in w for w in info["warnings"])
    assert info["execution_time_s"] >= 0
    assert info["n_ion_relax_steps"] >= 0
    assert isinstance(info["warnings"], list)


def test_relax_without_elastic_has_no_elastic_info(client, fake_backends):
    data = _relax(client, _session()).json()
    assert "elastic_info" not in data
    data = _relax(client, _session(), elastic=False).json()
    assert "elastic_info" not in data


@pytest.mark.parametrize("strain", [0.05, 2.5, -1, "abc"])
def test_relax_rejects_bad_elastic_strain(client, fake_backends, strain):
    r = _relax(client, _session(), elastic=True, elastic_strain=strain)
    assert r.status_code == 400
    assert "elastic_strain" in r.json()["detail"]


def test_relax_rejects_non_bool_elastic(client, fake_backends):
    assert _relax(client, _session(), elastic="yes").status_code == 400


def test_relax_elastic_loose_fmax_warning(client, fake_backends):
    data = _relax(client, _session(), elastic=True, fmax=0.05).json()
    warns = data["elastic_info"]["warnings"]
    assert ("Relaxation force tolerance (0.05 eV/Å) is looser than 0.01 eV/Å; "
            "elastic constants may be inaccurate.") in warns


def test_relax_elastic_not_converged_warning(client, fake_backends):
    # Rattled cell with a single step cannot converge.
    s = _cu_conv()
    s = s.copy()
    s.make_supercell([2, 2, 2])
    s.perturb(0.1)
    data = _relax(client, _session(s), elastic=True, max_steps=1).json()
    assert ("Relaxation did not converge; elastic constants may be inaccurate."
            in data["elastic_info"]["warnings"])


def test_relax_elastic_failure_returns_error_not_500(client, fake_backends, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(main, "_run_elastic_sync", boom)
    r = _relax(client, _session(), elastic=True)
    assert r.status_code == 200
    info = r.json()["elastic_info"]
    assert "boom" in info["error"]
    assert isinstance(info["warnings"], list)


def test_relax_elastic_timeout_returns_error(client, fake_backends, monkeypatch):
    monkeypatch.setattr(main, "ELASTIC_TIMEOUT_SECONDS", 0.0)
    r = _relax(client, _session(), elastic=True)
    assert r.status_code == 200
    assert "time limit" in r.json()["elastic_info"]["error"]


def test_default_fmax_is_0_01():
    fmax, steps, opt, elastic, strain = main._validate_relax_params({})
    assert fmax == 0.01 and elastic is False and strain == 0.25


# -- ion-relaxation tolerance ---------------------------------------------

def test_compute_elastic_default_tolerances():
    sig = inspect.signature(compute_elastic)
    assert sig.parameters["fmax"].default == ELASTIC_ION_FMAX == 0.001
    assert sig.parameters["max_steps"].default == ELASTIC_ION_MAX_STEPS == 500


def test_compute_elastic_runs_fire_with_tight_fmax(monkeypatch):
    from ase.optimize import FIRE
    seen = []
    orig = FIRE.run

    def spy(self, fmax=0.05, steps=100000):
        seen.append((fmax, steps))
        return orig(self, fmax=fmax, steps=steps)

    monkeypatch.setattr(FIRE, "run", spy)
    compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT())
    assert len(seen) == 24
    assert all(f == ELASTIC_ION_FMAX and n == ELASTIC_ION_MAX_STEPS for f, n in seen)


def test_compute_elastic_reports_unconverged_cells():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
    atoms.rattle(0.1, seed=1)
    res = compute_elastic(atoms, EMT(), max_steps=1)
    assert res["n_unconverged_cells"] > 0
    ok = compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT())
    assert ok["n_unconverged_cells"] == 0


def test_relax_elastic_unconverged_ion_warning(client, fake_backends, monkeypatch):
    orig = elastic_mod.compute_elastic

    def fake(*a, **k):
        res = orig(*a, **k)
        res["n_unconverged_cells"] = 3
        return res

    monkeypatch.setattr(elastic_mod, "compute_elastic", fake)
    data = _relax(client, _session(), elastic=True).json()
    assert ("Ion relaxation in some strained cells did not converge; "
            "elastic constants may be inaccurate.") in data["elastic_info"]["warnings"]


# -- strain consistency ----------------------------------------------------

def _nonlinear_calc_for(atoms, c, big):
    """Fake calculator: stress (eV/A^3) = c*eps + big*eps^3, zero forces."""
    from ase.calculators.calculator import Calculator, all_changes

    class Calc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            h0 = np.array(base_cell)
            e = np.linalg.solve(h0, np.array(atoms.get_cell())) - np.eye(3)
            e = 0.5 * (e + e.T)
            sig = c * e + big * e ** 3
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
                "stress": np.array([sig[0, 0], sig[1, 1], sig[2, 2],
                                    sig[1, 2], sig[0, 2], sig[0, 1]]),
            }

    base_cell = atoms.get_cell()[:]
    return Calc()


def test_compute_elastic_consistent_emt_cu():
    res = compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT())
    assert res["n_strained_cells"] == 24
    assert res["consistency"]["consistent"] is True
    assert res["consistency"]["max_abs_diff_GPa"] < 3.0


def test_compute_elastic_without_consistency_check():
    calls = []
    res = compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT(),
                          check_consistency=False,
                          progress_cb=lambda k, n: calls.append((k, n)))
    assert res["n_strained_cells"] == 12
    assert res["consistency"] is None
    assert calls[0] == (1, 12) and calls[-1] == (12, 12)


def test_compute_elastic_progress_totals_with_check():
    calls = []
    compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT(),
                    progress_cb=lambda k, n: calls.append((k, n)))
    assert [k for k, _ in calls] == list(range(1, 25))
    assert {n for _, n in calls} == {24}


def test_compute_elastic_flags_strain_dependence():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    # Large cubic term: C(delta) = c + 3*big*delta^2 depends on the strain.
    calc = _nonlinear_calc_for(atoms, c=1.0, big=1.0e4)
    res = compute_elastic(atoms, calc, strain=0.01)
    assert res["consistency"]["consistent"] is False
    assert res["consistency"]["max_abs_diff_GPa"] > 3.0
    assert res["consistency"]["worst_component"].startswith("C")
    # A linear response is consistent.
    lin = compute_elastic(atoms, _nonlinear_calc_for(atoms, c=1.0, big=0.0), strain=0.01)
    assert lin["consistency"]["consistent"] is True


def test_strain_consistency_criterion():
    from mlip.elastic import strain_consistency
    C = cubic_C(170.0, 120.0, 75.0)
    assert strain_consistency(C, C + np.diag([3.0] * 6))[2] is True
    assert strain_consistency(np.diag([100.0] * 6), np.diag([104.0] * 6))[2] is True
    assert strain_consistency(np.diag([100.0] * 6), np.diag([115.0] * 6))[2] is False
    assert strain_consistency(np.diag([-214.0] * 6), np.diag([-32.0] * 6))[2] is False


def test_relax_elastic_inconsistent_warning(client, fake_backends, monkeypatch):
    orig = elastic_mod.compute_elastic

    def fake(*a, **k):
        res = orig(*a, **k)
        res["consistency"] = {"strain_half": 0.00125, "max_abs_diff_GPa": 182.0,
                              "consistent": False, "worst_component": "C44"}
        return res

    monkeypatch.setattr(elastic_mod, "compute_elastic", fake)
    data = _relax(client, _session(), elastic=True).json()
    warns = data["elastic_info"]["warnings"]
    assert any(w.startswith("Elastic constants depend on the strain magnitude "
                            "(C44 differs by 182.0 GPa between ±0.25% and ±0.125% strain)")
               for w in warns)


# -- real models ---------------------------------------------------------

@pytest.mark.slow
def test_relax_elastic_real_chgnet_cu(client):
    r = _relax(client, _session(), elastic=True, model="chgnet", max_steps=200)
    assert r.status_code == 200, r.text
    info = r.json()["elastic_info"]
    assert "error" not in info, info
    assert 100 < info["moduli"]["K"]["hill"] < 220


@pytest.mark.slow
@pytest.mark.mace
def test_relax_elastic_real_mace_cu(client):
    pytest.importorskip("mace")
    r = _relax(client, _session(), elastic=True, model="mace-mp-0-medium", max_steps=200)
    assert r.status_code == 200, r.text
    info = r.json()["elastic_info"]
    assert "error" not in info, info
    assert 100 < info["moduli"]["K"]["hill"] < 220


# -- shutdown abort flag -----------------------------------------------------

@pytest.fixture
def abort_flag():
    from mlip import executor
    executor.request_abort()
    yield
    executor.clear_abort()


def test_compute_elastic_stops_when_abort_requested(abort_flag):
    with pytest.raises(elastic_mod.ElasticTimeout, match="aborted"):
        compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), EMT())


def test_compute_elastic_observer_stops_on_abort_mid_cell():
    """The per-step observer (not just the per-cell pre-check) honours the flag."""
    from ase.calculators.calculator import Calculator, all_changes
    from mlip import executor

    class Calc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            executor.request_abort()  # flag raised while the first cell is evaluated
            self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3)),
                            "stress": np.zeros(6)}

    try:
        with pytest.raises(elastic_mod.ElasticTimeout, match="aborted during"):
            compute_elastic(bulk("Cu", "fcc", a=3.61, cubic=True), Calc())
    finally:
        executor.clear_abort()


# -- Voigt ordering regression guard ------------------------------------------

def test_compute_elastic_voigt_ordering_anisotropic_linear():
    """A linear fake with a full non-cubic stiffness (incl. C45/C16) must be
    recovered exactly, in ASE Voigt order xx, yy, zz, yz, xz, xy."""
    from ase import units
    from ase.calculators.calculator import Calculator, all_changes

    rng = np.random.default_rng(7)
    A = rng.uniform(-1.0, 1.0, (6, 6))
    C_true = 150.0 * np.eye(6) + 20.0 * (A + A.T)  # GPa, symmetric, all entries nonzero
    assert C_true[3, 4] != 0 and C_true[0, 5] != 0
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    base_cell = np.array(atoms.get_cell())

    class Calc(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e = np.linalg.solve(base_cell, np.array(atoms.get_cell())) - np.eye(3)
            e = 0.5 * (e + e.T)
            voigt = np.array([e[0, 0], e[1, 1], e[2, 2],
                              2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])  # engineering shear
            self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3)),
                            "stress": (C_true * units.GPa) @ voigt}

    res = compute_elastic(atoms, Calc(), check_consistency=False)
    np.testing.assert_allclose(np.array(res["C_GPa"]), C_true, atol=1e-6)


# -- shutdown handling in /api/relax -------------------------------------------

def test_relax_elastic_shutdown_abort_reported_distinctly(client, fake_backends, monkeypatch):
    from mlip import executor

    def aborted(*a, **k):
        executor.request_abort()
        raise elastic_mod.ElasticTimeout("aborted during strained cell 1/24", "shutdown")

    monkeypatch.setattr(main, "_run_elastic_sync", aborted)
    try:
        info = _relax(client, _session(), elastic=True).json()["elastic_info"]
    finally:
        executor.clear_abort()
    assert info["abort_reason"] == "shutdown"
    assert "shutting down" in info["error"]


def test_relax_returns_503_when_executor_already_shut_down(client, fake_backends):
    from mlip import executor

    executor.request_abort()
    executor.inference_executor.shutdown(wait=False)
    try:
        r = _relax(client, _session())
        assert r.status_code == 503
    finally:
        executor.clear_abort()
        executor.inference_executor.ensure_running()
    assert _relax(client, _session()).status_code == 200


def test_executor_can_be_restarted_after_shutdown():
    ex = executor_mod_restartable()
    ex.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        ex.submit(int)
    ex.ensure_running()
    assert ex.submit(int).result() == 0
    ex.shutdown(wait=True)


def executor_mod_restartable():
    from mlip.executor import _RestartableExecutor
    return _RestartableExecutor()
