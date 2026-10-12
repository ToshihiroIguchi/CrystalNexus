"""Slow tests with the real MACE-MP-0 (medium) model (and CHGNet for comparison).

Run with:  pytest tests -m mace
Weights are downloaded to ~/.cache/mace on first use.
"""
import asyncio
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from ase import units
from pymatgen.core import Lattice, Structure

import main
from main import session_manager
from mlip import get_backend
from mlip.mace_backend import _STATIC_ATOMIC_NUMBERS

pytestmark = [pytest.mark.mace, pytest.mark.slow]

MACE_ID = "mace-mp-0-medium"
GOLDEN = json.loads((Path(__file__).parent / "data" / "chgnet_golden.json")
                    .read_text(encoding="utf-8"))

try:
    import mace  # noqa: F401
    _HAS_MACE = True
except ImportError:  # pragma: no cover
    _HAS_MACE = False

if not _HAS_MACE:
    pytestmark.append(pytest.mark.skip(reason="mace-torch not installed"))


@pytest.fixture(scope="module")
def mace_backend():
    return get_backend(MACE_ID)


@pytest.fixture(scope="module")
def chgnet_backend():
    return get_backend("chgnet")


def _run(coro):
    return asyncio.run(coro)


def _cu_conventional(a=3.65):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Cu"], [[0, 0, 0]])


def _si_primitive(a=5.43):
    lat = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def _nacl(a=5.64):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Na", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _session(structure, filename="X.cif"):
    return session_manager.create_session(str(uuid.uuid4()), filename, structure)


# ---------------------------------------------------------------------------
# Physics consistency (both models)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", ["chgnet", MACE_ID])
def test_forces_match_energy_finite_difference(model_id):
    backend = get_backend(model_id)
    s = _si_primitive()
    s.translate_sites([1], [0.012, -0.008, 0.015], frac_coords=False, to_unit_cell=False)

    forces = np.array(_run(backend.predict_one(s, ("energy", "forces")))["forces_eV_per_A"])
    h = 0.01
    # CHGNet evaluates in float32: energy noise (~1e-4 eV) divided by 2h limits
    # the finite-difference accuracy to a few 1e-3 eV/A (observed ~3e-3).
    # MACE runs in float64 and must agree to 1e-3.
    tol = 1e-3 if model_id == MACE_ID else 6e-3
    for axis in range(3):
        e = []
        for sign in (+1, -1):
            d = [0.0, 0.0, 0.0]
            d[axis] = sign * h
            t = s.copy()
            t.translate_sites([1], d, frac_coords=False, to_unit_cell=False)
            e.append(_run(backend.predict_one(t, ("energy",)))["total_energy_eV"])
        minus_grad = -(e[0] - e[1]) / (2 * h)
        assert minus_grad == pytest.approx(forces[1][axis], abs=tol), (model_id, axis)


@pytest.mark.parametrize("model_id", ["chgnet", MACE_ID])
def test_stress_matches_energy_finite_difference(model_id):
    """sigma_xx = (1/V) dE/d(eps_xx); stress_GPa carries the same sign (no flip)."""
    backend = get_backend(model_id)
    s = _cu_conventional(3.70)  # stretched so the stress is clearly non-zero
    stress = np.array(_run(backend.predict_one(s, ("energy", "stress")))["stress_GPa"])
    eps = 2e-3
    e = []
    for sign in (+1, -1):
        m = np.eye(3)
        m[0, 0] += sign * eps
        t = Structure(Lattice(s.lattice.matrix @ m), s.species, s.frac_coords)
        e.append(_run(backend.predict_one(t, ("energy",)))["total_energy_eV"])
    sigma_xx = (e[0] - e[1]) / (2 * eps) / s.volume / units.GPa
    assert abs(stress[0][0]) > 1.0, "test strain should give a clear stress"
    assert sigma_xx == pytest.approx(stress[0][0], rel=0.02), model_id


# ---------------------------------------------------------------------------
# Relaxed lattice constants
# ---------------------------------------------------------------------------

def _relaxed_a(backend, structure):
    calc = _run(backend.get_calculator())
    result = main.run_relaxation(calc, structure, fmax=0.02, steps=300,
                                 optimizer_name="LBFGS", timeout_s=900, relax_cell=True)
    return result["final_structure"].lattice.a


@pytest.mark.parametrize("name,builder,start,ref,tol", [
    ("Cu fcc", _cu_conventional, 3.70, 3.61, 0.02),
    ("NaCl", _nacl, 5.75, 5.64, 0.03),
])
def test_relaxed_lattice_constant(mace_backend, chgnet_backend, name, builder, start, ref, tol):
    a_mace = _relaxed_a(mace_backend, builder(start))
    a_chg = _relaxed_a(chgnet_backend, builder(start))
    print(f"\n[lattice] {name}: MACE a={a_mace:.4f}  CHGNet a={a_chg:.4f}  ref={ref}")
    assert a_mace == pytest.approx(ref, rel=tol)


def test_relaxed_lattice_constant_si(mace_backend, chgnet_backend):
    # Primitive diamond cell; conventional a = sqrt(2) * primitive-vector length.
    def conv_a(backend):
        calc = _run(backend.get_calculator())
        r = main.run_relaxation(calc, _si_primitive(5.50), fmax=0.02, steps=300,
                                optimizer_name="LBFGS", timeout_s=900, relax_cell=True)
        return r["final_structure"].lattice.a * np.sqrt(2)

    a_mace, a_chg = conv_a(mace_backend), conv_a(chgnet_backend)
    print(f"\n[lattice] Si: MACE a={a_mace:.4f}  CHGNet a={a_chg:.4f}  ref=5.43")
    assert a_mace == pytest.approx(5.43, rel=0.02)


# ---------------------------------------------------------------------------
# Isolation, element set, resources
# ---------------------------------------------------------------------------

def test_chgnet_unchanged_after_mace_load(chgnet_backend, mace_backend):
    golden = GOLDEN["predictions"]["Cu"]
    s = Structure.from_dict(golden["structure"])
    props = main._PROPS_PREDICT

    first = _run(chgnet_backend.predict_one(s, props))
    mace_pred = _run(mace_backend.predict_one(s, ("energy", "forces", "stress")))
    third = _run(chgnet_backend.predict_one(s, props))

    assert mace_pred["energy_eV_per_atom"] != first["energy_eV_per_atom"]
    for key in ("energy_eV_per_atom", "forces_eV_per_A", "stress_GPa",
                "magmoms_muB", "site_energies_eV"):
        assert np.array_equal(np.asarray(first[key]), np.asarray(third[key])), key
        assert np.asarray(third[key]) == pytest.approx(np.asarray(golden[key]), abs=1e-6)


def test_element_set_matches_model(mace_backend):
    calc = _run(mace_backend.get_calculator())
    z_model = sorted(int(z) for z in calc.models[0].atomic_numbers)
    assert z_model == sorted(_STATIC_ATOMIC_NUMBERS)
    syms = mace_backend.supported_elements()
    assert len(syms) == 89 and "Pu" in syms and "Po" not in syms
    assert mace_backend.parameters() == 4688656
    assert mace_backend.version() != "unknown"


def test_rss_with_both_models_loaded(chgnet_backend, mace_backend):
    psutil = pytest.importorskip("psutil")
    _run(chgnet_backend.load())
    _run(mace_backend.load())
    rss = psutil.Process().memory_info().rss / 1e6
    print(f"\n[memory] RSS with CHGNet + MACE loaded: {rss:.0f} MB")
    assert rss > 0


# ---------------------------------------------------------------------------
# End-to-end through the API
# ---------------------------------------------------------------------------

def test_predict_endpoint_mace(client):
    sid = _session(_cu_conventional(3.61), "Cu.cif")
    r = client.post("/api/predict", json={"session_id": sid, "filename": "Cu.cif",
                                          "model": MACE_ID})
    assert r.status_code == 200, r.text
    data = r.json()
    pred = data["prediction"]
    assert data["model_info"]["id"] == MACE_ID
    assert data["model_info"]["display_name"] == "MACE-MP-0 (medium)"
    assert data["model_info"]["parameters"] == 4688656
    assert not pred.get("magmoms_muB")  # absent / null
    assert "site_energies_eV" not in pred or pred["site_energies_eV"] is None
    for key in ("energy_eV_per_atom", "total_energy_eV", "forces_eV_per_A",
                "stress_GPa", "formula", "num_sites", "volume", "density"):
        assert key in pred


def test_relax_endpoint_mace_contract_and_cif(client):
    sid = _session(_cu_conventional(3.70), "Cu.cif")
    r = client.post("/api/relax", json={"session_id": sid, "fmax": 0.05, "max_steps": 50,
                                        "optimizer": "LBFGS", "model": MACE_ID})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model_info"]["id"] == MACE_ID
    for key in ("converged", "steps", "fmax", "optimizer_steps", "optimizer", "aborted"):
        assert key in data["relaxation_info"]
    for key in ("energy_eV_per_atom", "total_energy_eV", "forces_eV_per_A", "stress_GPa",
                "formula", "num_sites", "volume", "density"):
        assert key in data["final_prediction"], key
    assert "magmoms_muB" not in data["final_prediction"]
    for key in ("steps", "energies", "force_magnitudes", "forces"):
        assert key in data["trajectory_data"]
    assert session_manager.get_session_info(sid)["relax_result"]["model"] == MACE_ID

    cif = client.post("/api/generate-relaxed-structure-cif", json={"session_id": sid})
    assert cif.status_code == 200
    assert f"# MACE-MP-0 (medium) relaxation: model={MACE_ID}," in cif.text


def test_evaluate_endpoints_mace(client):
    sid = _session(_supercell(_cu_conventional(3.61), (2, 2, 2)))
    ins = client.post("/api/evaluate-insertion-energies", json={
        "session_id": sid, "element": "H", "model": MACE_ID,
        "sites": [{"id": 0, "frac_coords": [0.25, 0.25, 0.25]},
                  {"id": 1, "frac_coords": [0.125, 0.125, 0.125]}]})
    assert ins.status_code == 200, ins.text
    assert ins.json()["model"] == MACE_ID
    assert all(x["energy"] is not None for x in ins.json()["results"])

    cand = client.post("/api/evaluate-candidate-energies", json={
        "session_id": sid, "model": MACE_ID,
        "candidates": [{"id": 0, "action": "substitute", "index": 0, "to": "Ni"},
                       {"id": 1, "action": "delete", "index": 1}]})
    assert cand.status_code == 200, cand.text
    assert cand.json()["model"] == MACE_ID
    assert all(x["energy"] is not None for x in cand.json()["results"])


def _supercell(structure, size):
    s = structure.copy()
    s.make_supercell(size)
    return s


def test_unsupported_polonium_is_422_and_pu_listed(client):
    po = _cu_conventional(3.61)
    po.replace(0, "Po")
    sid = _session(po)
    r = client.post("/api/predict", json={"session_id": sid, "filename": "Cu.cif",
                                          "model": MACE_ID})
    assert r.status_code == 422 and "Po" in r.json()["detail"]
    assert MACE_ID in r.json()["detail"]
    r = client.post("/api/relax", json={"session_id": sid, "fmax": 0.1, "max_steps": 5,
                                        "model": MACE_ID})
    assert r.status_code == 422 and "Po" in r.json()["detail"]

    els = client.get("/api/elements", params={"model": MACE_ID}).json()
    assert els["available"] and "Pu" in els["elements"] and "Po" not in els["elements"]
    assert len(els["elements"]) == 89


def test_concurrent_mace_relax_and_chgnet_predict(client):
    relax_sid = _session(_supercell(_cu_conventional(3.70), (2, 2, 2)), "Cu.cif")
    pred_sid = _session(_cu_conventional(3.61), "Cu.cif")
    out = {}

    def _relax():
        out["relax"] = client.post("/api/relax", json={
            "session_id": relax_sid, "fmax": 0.02, "max_steps": 60,
            "optimizer": "LBFGS", "model": MACE_ID}).status_code

    t = threading.Thread(target=_relax)
    t.start()
    time.sleep(1.0)
    try:
        r = client.post("/api/predict", json={"session_id": pred_sid, "filename": "Cu.cif",
                                              "model": "chgnet"})
        out["predict"] = r.status_code
    finally:
        t.join(timeout=600)
    assert out == {"relax": 200, "predict": 200}
