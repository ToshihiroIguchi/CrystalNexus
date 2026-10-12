"""CHGNet numerical regression against tests/data/chgnet_golden.json.

The golden values were captured from the pre-abstraction main.py. Everything
here goes through the NEW registry/backend path, so these tests prove the
refactor did not change CHGNet's numbers.
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure

import main
from mlip import get_backend

GOLDEN_PATH = Path(__file__).parent / "data" / "chgnet_golden.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

PRED_ABS_TOL = 1e-6       # energies (bit-identical across torch 2.8 -> 2.14)
# Autograd-derived quantities (forces/stress) and float32 network outputs differ by up to ~1e-4
# between torch versions (kernel-level float32 rounding); energies stay identical.
DERIVED_ABS_TOL = 5e-4
ENERGY_ABS_TOL = 1e-5     # eV/atom after relaxation
LATTICE_ABS_TOL = 1e-3    # Angstrom / degrees after relaxation

pytestmark = pytest.mark.slow

_PREDICTION_KEYS = ("energy_eV_per_atom", "total_energy_eV", "forces_eV_per_A",
                    "stress_GPa", "magmoms_muB", "site_energies_eV")


@pytest.fixture(scope="module")
def backend():
    return get_backend("chgnet")


@pytest.mark.parametrize("name", sorted(GOLDEN["predictions"]))
def test_prediction_matches_golden(backend, name):
    golden = GOLDEN["predictions"][name]
    structure = Structure.from_dict(golden["structure"])

    pred = asyncio.run(backend.predict_one(structure, main._PROPS_PREDICT))

    for key in _PREDICTION_KEYS:
        assert pred[key] is not None, key
        assert np.asarray(pred[key]) == pytest.approx(
            np.asarray(golden[key]),
            abs=PRED_ABS_TOL if "energy" in key else DERIVED_ABS_TOL), f"{name}: {key}"
    # Features were not requested -> normalized to None.
    assert pred["crystal_features"] is None and pred["atom_features"] is None


@pytest.mark.parametrize("name", sorted(GOLDEN["predictions"]))
def test_batch_prediction_matches_single(backend, name):
    golden = GOLDEN["predictions"][name]
    structure = Structure.from_dict(golden["structure"])

    pred = asyncio.run(backend.predict([structure], main._PROPS_ENERGY))[0]

    assert pred["energy_eV_per_atom"] == pytest.approx(
        golden["energy_eV_per_atom"], abs=PRED_ABS_TOL)
    assert pred["forces_eV_per_A"] is None


@pytest.mark.parametrize("name", sorted(GOLDEN["relaxations"]))
def test_relaxation_matches_golden(backend, name):
    golden = GOLDEN["relaxations"][name]
    params = golden["params"]
    structure = Structure.from_dict(golden["input_structure"])

    calculator = asyncio.run(backend.get_calculator())
    result = main.run_relaxation(
        calculator, structure, fmax=params["fmax"], steps=params["steps"],
        optimizer_name=params["optimizer"], timeout_s=3600,
        relax_cell=params["relax_cell"])

    assert result["optimizer_steps"] == golden["optimizer_steps"]
    assert result["ase_converged"] == golden["ase_converged"]
    assert result["aborted"] == golden["aborted"]

    final = result["final_structure"]
    # CHGNet provides magmoms -> recorded as a site property.
    assert "magmom" in final.site_properties

    pred = asyncio.run(backend.predict_one(final, ("energy",)))
    assert pred["energy_eV_per_atom"] == pytest.approx(
        golden["final_energy_eV_per_atom"], abs=ENERGY_ABS_TOL)
    assert list(final.lattice.abc) == pytest.approx(
        golden["final_lattice_abc"], abs=LATTICE_ABS_TOL)
    assert list(final.lattice.angles) == pytest.approx(
        golden["final_lattice_angles"], abs=LATTICE_ABS_TOL)
    assert final.volume == pytest.approx(golden["final_volume"], rel=1e-4)


def test_stress_convention_matches_ase_calculator(backend):
    """predict() stress equals calc.get_stress(voigt=False)/GPa (no sign flip)."""
    from ase import units
    from pymatgen.io.ase import AseAtomsAdaptor

    from mlip import ase_stress_to_gpa

    conv = GOLDEN["stress_convention"]
    structure = Structure.from_dict(conv["structure"])
    pred = asyncio.run(backend.predict_one(structure, ("energy", "stress")))

    calculator = asyncio.run(backend.get_calculator())
    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = calculator
    via_ase = np.asarray(ase_stress_to_gpa(atoms.get_stress(voigt=False)))

    assert np.asarray(pred["stress_GPa"]) == pytest.approx(
        np.asarray(conv["predict_structure_s_GPa"]), abs=DERIVED_ABS_TOL)
    assert via_ase == pytest.approx(np.asarray(pred["stress_GPa"]), abs=1e-4)
    assert via_ase == pytest.approx(
        np.asarray(conv["calc_get_stress_over_units_GPa"]), abs=DERIVED_ABS_TOL)
