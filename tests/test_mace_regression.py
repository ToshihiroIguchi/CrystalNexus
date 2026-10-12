"""MACE-MP-0 (medium) numerical regression against tests/data/mace_golden.json.

The golden values were captured with the pinned dependency set (see "meta" in
the JSON) on the same structures as chgnet_golden.json. They detect numerical
drift after dependency upgrades (torch, numpy, e3nn, mace-torch, ...).
"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure

import main
from mlip import get_backend

pytest.importorskip("mace")

GOLDEN = json.loads((Path(__file__).parent / "data" / "mace_golden.json")
                    .read_text(encoding="utf-8"))

MACE_ID = "mace-mp-0-medium"
PRED_ABS_TOL = 1e-5       # MACE runs in float64 and was bit-reproducible locally
ENERGY_ABS_TOL = 1e-5     # eV/atom after relaxation
LATTICE_ABS_TOL = 1e-3    # Angstrom / degrees after relaxation

pytestmark = [pytest.mark.slow, pytest.mark.mace]


@pytest.fixture(scope="module")
def backend():
    return get_backend(MACE_ID)


@pytest.mark.parametrize("name", sorted(GOLDEN["predictions"]))
def test_prediction_matches_golden(backend, name):
    golden = GOLDEN["predictions"][name]
    structure = Structure.from_dict(golden["structure"])

    pred = asyncio.run(backend.predict_one(structure, main._PROPS_PREDICT))

    for key in ("energy_eV_per_atom", "total_energy_eV", "forces_eV_per_A",
                "stress_GPa"):
        assert pred[key] is not None, key
        assert np.asarray(pred[key]) == pytest.approx(
            np.asarray(golden[key]), abs=PRED_ABS_TOL), f"{name}: {key}"


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
    pred = asyncio.run(backend.predict_one(final, ("energy",)))
    assert pred["energy_eV_per_atom"] == pytest.approx(
        golden["final_energy_eV_per_atom"], abs=ENERGY_ABS_TOL)
    assert list(final.lattice.abc) == pytest.approx(
        golden["final_lattice_abc"], abs=LATTICE_ABS_TOL)
    assert list(final.lattice.angles) == pytest.approx(
        golden["final_lattice_angles"], abs=LATTICE_ABS_TOL)
    assert final.volume == pytest.approx(golden["final_volume"], rel=1e-4)
