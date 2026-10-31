"""Fast tests for CHGNetBackend batching/normalization/loading (mocked model)."""
import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Element, Structure

from mlip import chgnet_backend as cb
from mlip.base import MPTRJ_ATOMIC_NUMBERS, PROP_ENERGY, PROP_FORCES
from mlip.chgnet_backend import CHGNetBackend, normalize_chgnet_prediction
from mlip.mace_backend import _STATIC_ATOMIC_NUMBERS


def _cu():
    return Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")


def _raw(n_atoms, energy=-3.5):
    return {"e": np.float64(energy), "f": np.zeros((n_atoms, 3)),
            "s": np.zeros((3, 3)), "m": np.zeros(n_atoms)}


class FakeModel:
    """Stand-in for CHGNet with a scriptable predict_structure."""

    def __init__(self, fail_batches=False, short_batches=False):
        self.calls = []
        self.fail_batches = fail_batches
        self.short_batches = short_batches

    def predict_structure(self, structure, batch_size=None, **kwargs):
        is_batch = isinstance(structure, list)
        self.calls.append(len(structure) if is_batch else 1)
        if is_batch:
            if self.fail_batches:
                raise RuntimeError("batch boom")
            preds = [_raw(len(s)) for s in structure]
            return preds[:-1] if self.short_batches else preds
        return _raw(len(structure))


@pytest.fixture
def backend(monkeypatch):
    b = CHGNetBackend()
    monkeypatch.setattr(b, "_model", FakeModel())
    monkeypatch.setattr(CHGNetBackend, "batch_size", 2)
    return b


def _run(coro):
    return asyncio.run(coro)


def test_predict_chunks_by_batch_size(backend):
    out = _run(backend.predict([_cu() for _ in range(5)], {PROP_ENERGY}))
    assert backend._model.calls == [2, 2, 1]
    assert len(out) == 5 and all(o["energy_eV_per_atom"] == -3.5 for o in out)


def test_predict_empty_list(backend):
    assert _run(backend.predict([], {PROP_ENERGY})) == []
    assert backend._model.calls == []


def test_predict_batch_failure_falls_back_per_structure(backend):
    backend._model.fail_batches = True
    out = _run(backend.predict([_cu() for _ in range(3)], {PROP_ENERGY}))
    # failed chunk of 2, per-structure retries, then chunk of 1 (also a list call)
    assert len(out) == 3 and all(o is not None for o in out)
    assert backend._model.calls.count(1) >= 3


def test_predict_per_structure_failure_yields_none(backend):
    class Boom(FakeModel):
        def predict_structure(self, structure, batch_size=None, **kwargs):
            raise RuntimeError("always")

    backend._model = Boom()
    assert _run(backend.predict([_cu(), _cu()], {PROP_ENERGY})) == [None, None]


def test_predict_length_mismatch_falls_back(backend):
    backend._model.short_batches = True
    out = _run(backend.predict([_cu(), _cu()], {PROP_ENERGY}))
    assert len(out) == 2 and all(o is not None for o in out)
    assert backend._model.calls == [2, 1, 1]  # mismatching batch, then per-structure


def test_predict_bare_dict_for_single_structure_chunk(backend):
    class Bare(FakeModel):
        def predict_structure(self, structure, batch_size=None, **kwargs):
            return _raw(len(structure[0]))

    backend._model = Bare()
    out = _run(backend.predict([_cu()], {PROP_FORCES}))
    assert len(out) == 1 and out[0]["energy_eV_per_atom"] == -3.5


@pytest.mark.parametrize("keys", [
    {"energy": "e", "forces": "f", "stress": "s", "magmom": "m"},
    {"energy": "energy", "forces": "forces", "stress": "stress", "magmom": "magmom"},
    {"energy": "energy_per_atom", "forces": "force", "stress": "virial", "magmom": "magmoms"},
    {"energy": "e_per_atom", "forces": "f", "stress": "s", "magmom": "m"},
])
def test_normalize_accepts_alias_keys_without_leaking_raw_keys(keys):
    n = 2
    pred = {keys["energy"]: np.float64(-1.0), keys["forces"]: np.ones((n, 3)),
            keys["stress"]: np.eye(3), keys["magmom"]: np.zeros(n)}
    out = normalize_chgnet_prediction(pred, n)
    assert out["energy_eV_per_atom"] == -1.0 and out["total_energy_eV"] == -2.0
    assert out["forces_eV_per_A"] == [[1.0] * 3] * n
    assert out["stress_GPa"][0] == [1.0, 0.0, 0.0]
    assert out["magmoms_muB"] == [0.0, 0.0]
    for raw in keys.values():
        assert raw not in out  # the raw alias must not leak next to the normalized key


def test_normalize_alias_feature_and_site_keys_not_passed_through():
    pred = {"site_energy": np.zeros(2), "crystal_features": np.zeros(3),
            "atom_features": np.zeros((2, 3)), "unknown_extra": np.arange(2)}
    out = normalize_chgnet_prediction(pred, 2)
    assert out["site_energies_eV"] == [0.0, 0.0]
    assert out["crystal_features"] == [0.0] * 3
    assert out["atom_features"] == [[0.0] * 3] * 2
    assert "site_energy" not in out
    assert out["unknown_extra"] == [0, 1]  # unknown keys still pass through


# -- loading --------------------------------------------------------------

def test_model_load_runs_on_the_inference_executor_thread(monkeypatch):
    b = CHGNetBackend()
    monkeypatch.setattr(b, "_model", None)
    seen = {}

    class FakeCHGNet:
        @staticmethod
        def load(use_device=None, verbose=False):
            seen["thread"] = threading.current_thread()
            return object()

    import chgnet.model.model as chgnet_model_mod
    monkeypatch.setattr(chgnet_model_mod, "CHGNet", FakeCHGNet)

    async def go():
        await b.get_model()
        return threading.current_thread()

    loop_thread = _run(go())
    assert seen["thread"] is not loop_thread
    assert seen["thread"].name.startswith("mlip")  # single-worker inference executor
    assert b._model is not None


# -- supported elements -----------------------------------------------------

def test_chgnet_elements_match_mptrj_and_mace_set():
    syms = CHGNetBackend().supported_elements()
    assert [Element(s).Z for s in syms] == sorted(MPTRJ_ATOMIC_NUMBERS)
    assert set(MPTRJ_ATOMIC_NUMBERS) == set(_STATIC_ATOMIC_NUMBERS)
    for present in ("He", "Ne", "Ar", "Kr", "Xe", "Np", "Pu", "U"):
        assert present in syms
    for absent in ("Po", "At", "Rn", "Fr", "Ra", "Am"):
        assert absent not in syms


@pytest.mark.slow
def test_chgnet_elements_within_model_embedding():
    """Every supported element must index inside the installed model's
    element-wise tables (composition model has one weight per Z = 1..94)."""
    from chgnet.model import CHGNet
    model = CHGNet.load(verbose=False)
    n_elements = model.composition_model.state_dict()["fc.weight"].shape[-1]
    assert max(MPTRJ_ATOMIC_NUMBERS) <= n_elements
