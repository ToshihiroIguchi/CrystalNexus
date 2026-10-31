"""Fast tests for the mlip package (no real model is loaded)."""
import asyncio
from pathlib import Path

import numpy as np
import pytest
from ase import units
from fastapi import HTTPException
from pymatgen.core import Structure

import main
from mlip import (MODEL_REGISTRY, ModelUnavailableError, UnknownModelError,
                  ase_stress_to_gpa, default_model_id, get_backend, list_models,
                  make_prediction)
from mlip.base import RESULT_KEYS


def _cu(supercell=None):
    s = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    if supercell:
        s.make_supercell(supercell)
    return s


# -- registry -----------------------------------------------------------

def test_unknown_model_id_raises():
    with pytest.raises(UnknownModelError):
        get_backend("no-such-model")


def test_unknown_model_maps_to_http_400(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        main._resolve_backend("no-such-model")
    assert exc.value.status_code == 400


def test_unavailable_model_maps_to_http_503(monkeypatch, fake_backends):
    FakeBackend, _ = fake_backends
    monkeypatch.setattr(FakeBackend, "is_available", classmethod(lambda cls: False))
    with pytest.raises(ModelUnavailableError):
        get_backend("fake-emt")
    with pytest.raises(HTTPException) as exc:
        main._resolve_backend("fake-emt")
    assert exc.value.status_code == 503


def test_default_model_is_chgnet_without_env(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_DEFAULT_MODEL", raising=False)
    assert default_model_id() == "chgnet"
    assert "chgnet" in MODEL_REGISTRY


def test_default_model_selected_via_env(monkeypatch, fake_backends):
    FakeBackend, Magmom = fake_backends
    assert isinstance(get_backend(), FakeBackend)
    monkeypatch.setenv("CRYSTALNEXUS_DEFAULT_MODEL", "fake-emt-magmom")
    assert isinstance(get_backend(), Magmom)
    # An explicit id wins over the default.
    assert isinstance(get_backend("fake-emt"), FakeBackend)


def test_backend_is_singleton(fake_backends):
    FakeBackend, _ = fake_backends
    assert get_backend("fake-emt") is get_backend("fake-emt") is FakeBackend()


def test_list_models_schema(fake_backends):
    models = {m["id"]: m for m in list_models()}
    assert set(models["fake-emt"]) == {"id", "display_name", "available",
                                       "capabilities", "version", "is_default"}
    assert models["fake-emt"]["is_default"] is True
    assert models["fake-emt-magmom"]["is_default"] is False
    assert "magmom" in models["fake-emt-magmom"]["capabilities"]
    assert "chgnet" in models


# -- normalization ------------------------------------------------------

def test_ase_stress_to_gpa_value_and_sign():
    # 1 GPa expressed in ASE units (eV/A^3) maps back to exactly 1 GPa,
    # with no sign flip; shape is preserved.
    voigt = np.array([units.GPa, -2 * units.GPa, 0, 0, 0, 0])
    out = ase_stress_to_gpa(voigt)
    assert out == pytest.approx([1.0, -2.0, 0, 0, 0, 0])
    full = ase_stress_to_gpa(np.diag([units.GPa, 0.0, -units.GPa]))
    assert np.array(full) == pytest.approx(np.diag([1.0, 0.0, -1.0]))


def test_make_prediction_has_all_keys_and_none_for_missing():
    pred = make_prediction(4, energy_per_atom=-1.5)
    assert set(pred) == set(RESULT_KEYS)
    assert pred["total_energy_eV"] == pytest.approx(-6.0)
    for key in RESULT_KEYS:
        if key not in ("energy_eV_per_atom", "total_energy_eV"):
            assert pred[key] is None
    assert make_prediction(4)["total_energy_eV"] is None


def test_prediction_payload_drops_none_keys():
    payload = main._prediction_payload(make_prediction(2, energy_per_atom=1.0))
    assert set(payload) == {"energy_eV_per_atom", "total_energy_eV"}


# -- generic predict path via the FakeBackend (EMT) ----------------------

def test_fake_backend_predict_generic_path(fake_backends):
    backend = get_backend("fake-emt")
    structures = [_cu(), _cu((2, 1, 1))]
    props = ("energy", "forces", "stress", "magmom", "site_energies")
    results = asyncio.run(backend.predict(structures, props))
    assert len(results) == 2
    for s, r in zip(structures, results):
        assert set(r) == set(RESULT_KEYS)
        assert r["total_energy_eV"] == pytest.approx(r["energy_eV_per_atom"] * len(s))
        assert np.array(r["forces_eV_per_A"]).shape == (len(s), 3)
        assert np.array(r["stress_GPa"]).shape == (3, 3)
        # No magmom / site-energy / feature capability -> None, not missing.
        assert r["magmoms_muB"] is None
        assert r["site_energies_eV"] is None
        assert r["crystal_features"] is None and r["atom_features"] is None


def test_fake_backend_energy_only_skips_forces(fake_backends):
    backend = get_backend("fake-emt")
    r = asyncio.run(backend.predict_one(_cu(), ("energy",)))
    assert r["energy_eV_per_atom"] is not None
    assert r["forces_eV_per_A"] is None and r["stress_GPa"] is None


def test_magmom_backend_reports_magmoms(fake_backends):
    backend = get_backend("fake-emt-magmom")
    r = asyncio.run(backend.predict_one(_cu(), ("energy", "magmom")))
    assert r["magmoms_muB"] == [0.0] * len(_cu())


def test_predict_returns_none_for_failed_structure(fake_backends):
    backend = get_backend("fake-emt")
    bad = Structure.from_file(Path("sample_cif") / "Oxides" / "Fe3O4.cif")  # EMT lacks O/Fe? Fe/O unsupported
    results = asyncio.run(backend.predict([_cu(), bad], ("energy",)))
    assert results[0] is not None
    assert results[1] is None


# -- run_relaxation with / without magnetic moments ---------------------

def test_run_relaxation_without_magmoms(fake_backends):
    calc = asyncio.run(get_backend("fake-emt").get_calculator())
    # EMT *lists* magmoms but never computes them; run_relaxation must cope.
    result = main.run_relaxation(calc, _cu((2, 2, 2)), fmax=0.05, steps=20,
                                 optimizer_name="LBFGS", timeout_s=60, relax_cell=True)
    assert result["final_structure"] is not None
    assert "magmom" not in result["final_structure"].site_properties
    traj = result["trajectory"]
    assert len(traj) > 0
    assert all(m is None for m in traj.magmoms)


def test_run_relaxation_with_magmoms_records_site_property(fake_backends):
    calc = asyncio.run(get_backend("fake-emt-magmom").get_calculator())
    result = main.run_relaxation(calc, _cu((2, 2, 2)), fmax=0.05, steps=5,
                                 optimizer_name="LBFGS", timeout_s=60, relax_cell=False)
    assert "magmom" in result["final_structure"].site_properties
    assert all(m is not None for m in result["trajectory"].magmoms)


# -- shared executor ----------------------------------------------------

def test_single_shared_inference_executor():
    from mlip.executor import inference_executor
    assert main._inference_executor is inference_executor
    assert inference_executor._max_workers == 1


# -- invalid default model env ------------------------------------------------

def test_invalid_default_model_env_falls_back_to_chgnet(monkeypatch, caplog):
    from mlip import registry

    monkeypatch.setenv("CRYSTALNEXUS_DEFAULT_MODEL", "no-such-model")
    monkeypatch.setattr(registry, "_warned_invalid_defaults", set())
    with caplog.at_level("WARNING", logger="mlip.registry"):
        assert registry.default_model_id() == "chgnet"
        assert registry.default_model_id() == "chgnet"
    assert sum("no-such-model" in r.message for r in caplog.records) == 1  # logged once
    assert registry.get_backend().id == "chgnet"
    defaults = [m["id"] for m in registry.list_models() if m["is_default"]]
    assert defaults == ["chgnet"]


def test_api_models_default_is_valid_with_bad_env(client, monkeypatch):
    monkeypatch.setenv("CRYSTALNEXUS_DEFAULT_MODEL", "bogus")
    data = client.get("/api/models").json()
    assert data["default"] == "chgnet"
    assert [m["id"] for m in data["models"] if m["is_default"]] == ["chgnet"]
