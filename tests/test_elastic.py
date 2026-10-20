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
from mlip.elastic import elastic_moduli

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
    data = _relax(client, _session(), elastic=True, elastic_strain=0.5).json()
    info = data["elastic_info"]
    assert "error" not in info
    C = np.asarray(info["C_GPa"])
    assert C.shape == (6, 6)
    assert C[0, 0] > C[0, 1] > 0
    assert C[3, 3] > 0
    assert info["stable"] is True
    assert info["moduli"]["E"]["hill"] > 0
    assert 0 < info["moduli"]["nu"]["hill"] < 0.5
    assert info["strain"] == pytest.approx(0.005)
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
    assert fmax == 0.01 and elastic is False and strain == 0.5


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
