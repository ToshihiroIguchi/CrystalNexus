"""Fast tests for per-request model selection (fake EMT backends, no real model)."""
import uuid
from pathlib import Path

import pytest
from pymatgen.core import Structure

import main
from main import session_manager
from mlip.mace_backend import MACEBackend, _STATIC_ATOMIC_NUMBERS


def _cu(supercell=(2, 2, 2)):
    s = Structure.from_file(Path("sample_cif") / "Metals" / "Cu.cif")
    s.make_supercell(supercell)
    return s


def _al_containing():
    s = _cu()
    s.replace(0, "Al")
    return s


def _session(structure=None):
    return session_manager.create_session(
        str(uuid.uuid4()), "Cu.cif", structure if structure is not None else _cu())


def _relax(client, sid, **extra):
    return client.post("/api/relax", json={
        "session_id": sid, "fmax": 0.05, "max_steps": 20, "optimizer": "LBFGS", **extra})


# -- /api/models and /api/elements -------------------------------------

def test_api_models_schema(client, fake_backends):
    data = client.get("/api/models").json()
    assert data["default"] == "fake-emt"
    models = {m["id"]: m for m in data["models"]}
    assert {"chgnet", "mace-mp-0-medium", "fake-emt"} <= set(models)
    for m in models.values():
        assert set(m) == {"id", "display_name", "available", "capabilities",
                          "version", "is_default"}
        assert m["capabilities"] == sorted(m["capabilities"])
        assert (m["version"] is None) == (not m["available"])
    assert models["fake-emt"]["is_default"] and not models["chgnet"]["is_default"]
    assert models["mace-mp-0-medium"]["display_name"] == "MACE-MP-0 (medium)"
    assert "magmom" not in models["mace-mp-0-medium"]["capabilities"]
    assert "batch" not in models["mace-mp-0-medium"]["capabilities"]


def test_api_models_exposes_atom_limit(client, fake_backends):
    # The start dialog uses this to cap the supercell size (F6).
    assert client.get("/api/models").json()["max_relax_atoms"] == main.MAX_RELAX_ATOMS


def test_api_models_marks_unavailable(client, fake_backends, monkeypatch):
    monkeypatch.setattr(MACEBackend, "is_available", classmethod(lambda cls: False))
    m = {x["id"]: x for x in client.get("/api/models").json()["models"]}["mace-mp-0-medium"]
    assert m["available"] is False and m["version"] is None


def test_api_elements_per_model(client, fake_backends):
    narrow = client.get("/api/elements", params={"model": "fake-narrow"}).json()
    assert narrow["elements"] == ["Cu"] and narrow["available"] is True
    assert narrow["model"] == "fake-narrow"
    emt = client.get("/api/elements", params={"model": "fake-emt"}).json()
    assert emt["elements"] == ["Al", "Ni", "Cu"]  # sorted by atomic number
    # No model -> registry default
    assert client.get("/api/elements").json()["model"] == "fake-emt"


def test_api_elements_unknown_400_and_unavailable_empty(client, fake_backends, monkeypatch):
    assert client.get("/api/elements", params={"model": "nope"}).status_code == 400
    monkeypatch.setattr(MACEBackend, "is_available", classmethod(lambda cls: False))
    r = client.get("/api/elements", params={"model": "mace-mp-0-medium"})
    assert r.status_code == 200
    assert r.json()["available"] is False and r.json()["elements"] == []


def test_static_mace_elements_exclude_po_include_pu():
    syms = MACEBackend().supported_elements()
    assert len(syms) == 89 == len(_STATIC_ATOMIC_NUMBERS)
    assert "Pu" in syms and "Po" not in syms and "He" in syms


def test_allowed_elements_is_union_of_available_models(monkeypatch):
    elements = main._initial_allowed_elements()
    assert "Pu" in elements and "Cu" in elements
    # CHGNet shares MACE's MPtrj element set, so only a registry without it
    # can shrink the union; unavailable models contribute nothing.
    monkeypatch.setattr(MACEBackend, "is_available", classmethod(lambda cls: False))
    assert "Pu" in main._initial_allowed_elements()
    monkeypatch.setattr(main, "MODEL_REGISTRY", {"fake": type("F", (), {
        "is_available": classmethod(lambda cls: True),
        "supported_elements": lambda self: ["Cu"]})})
    assert main._initial_allowed_elements() == {"Cu"}


# -- model routing and errors ------------------------------------------

def test_relax_unknown_model_400_unavailable_503(client, fake_backends, monkeypatch):
    sid = _session()
    assert _relax(client, sid, model="nope").status_code == 400
    monkeypatch.setattr(MACEBackend, "is_available", classmethod(lambda cls: False))
    r = _relax(client, sid, model="mace-mp-0-medium")
    assert r.status_code == 503 and "mace" in r.json()["detail"].lower()
    assert _relax(client, sid, model=5).status_code == 400


def test_relax_default_and_explicit_model_routing(client, fake_backends):
    sid = _session()
    r = _relax(client, sid)  # default from env -> fake-emt
    assert r.status_code == 200 and r.json()["model_info"]["id"] == "fake-emt"
    r = _relax(client, sid, model="fake-emt-magmom")
    assert r.json()["model_info"]["id"] == "fake-emt-magmom"
    assert session_manager.get_session_info(sid)["relax_result"]["model"] == "fake-emt-magmom"


def test_predict_unsupported_element_422(client, fake_backends):
    sid = _session(_al_containing())
    body = {"session_id": sid, "filename": "Cu.cif"}
    r = client.post("/api/predict", json={**body, "model": "fake-narrow"})
    assert r.status_code == 422
    assert "Al" in r.json()["detail"] and "fake-narrow" in r.json()["detail"]
    ok = client.post("/api/predict", json={**body, "model": "fake-emt"})
    assert ok.status_code == 200
    assert ok.json()["model_info"]["id"] == "fake-emt"


def test_predict_unknown_and_unavailable_model(client, fake_backends, monkeypatch):
    body = {"filename": "Metals/Cu.cif", "model": "nope"}
    assert client.post("/api/predict", json=body).status_code == 400
    monkeypatch.setattr(MACEBackend, "is_available", classmethod(lambda cls: False))
    body["model"] = "mace-mp-0-medium"
    assert client.post("/api/predict", json=body).status_code == 503


class _NoAcquireSemaphore:
    def __init__(self, locked):
        self._locked = locked

    def locked(self):
        return self._locked

    async def acquire(self):
        raise AssertionError("semaphore must not be acquired")

    def release(self):
        raise AssertionError("semaphore must not be released")


def test_relax_422_does_not_touch_semaphore(client, fake_backends, monkeypatch):
    monkeypatch.setattr(main, "_relax_semaphore", _NoAcquireSemaphore(False))
    r = _relax(client, _session(_al_containing()), model="fake-narrow")
    assert r.status_code == 422
    assert "Al" in r.json()["detail"] and "fake-narrow" in r.json()["detail"]


@pytest.mark.parametrize("model", ["fake-emt", "fake-narrow", "fake-emt-magmom"])
def test_relax_429_regardless_of_model(client, fake_backends, monkeypatch, model):
    monkeypatch.setattr(main, "_relax_semaphore", _NoAcquireSemaphore(True))
    assert _relax(client, _session(), model=model).status_code == 429


def test_relax_no_magmom_contract_matches_magmom_model(client, fake_backends):
    plain = _relax(client, _session(), model="fake-emt").json()
    mag = _relax(client, _session(), model="fake-emt-magmom").json()
    assert set(plain) == set(mag)
    for key in ("initial_prediction", "final_prediction", "relaxation_info",
                "relaxed_structure_info", "trajectory_data", "model_info"):
        extra = set(mag[key]) - set(plain[key])
        assert set(plain[key]) <= set(mag[key]), key
        assert extra <= {"magmoms_muB"}, (key, extra)
    assert "magmoms_muB" not in plain["final_prediction"]
    assert "magmoms_muB" in mag["final_prediction"]
    assert plain["model_info"]["display_name"] == "Fake EMT"


# -- evaluate endpoints --------------------------------------------------

def test_evaluate_insertion_energies_model_echo_and_422(client, fake_backends):
    sid = _session()
    body = {"session_id": sid, "element": "Al",
            "sites": [{"id": 0, "frac_coords": [0.25, 0.25, 0.25]}]}
    r = client.post("/api/evaluate-insertion-energies", json={**body, "model": "fake-emt"})
    assert r.status_code == 200 and r.json()["model"] == "fake-emt"
    assert r.json()["results"][0]["energy"] is not None
    assert client.post("/api/evaluate-insertion-energies", json=body).json()["model"] == "fake-emt"
    bad = client.post("/api/evaluate-insertion-energies", json={**body, "model": "fake-narrow"})
    assert bad.status_code == 422 and "Al" in bad.json()["detail"]
    assert client.post("/api/evaluate-insertion-energies",
                       json={**body, "model": "nope"}).status_code == 400


def test_evaluate_insertion_energy_single_echoes_model(client, fake_backends):
    r = client.post("/api/evaluate-insertion-energy", json={
        "session_id": _session(), "element": "Cu", "frac_coords": [0.25, 0.25, 0.25],
        "model": "fake-narrow"})
    assert r.status_code == 200 and r.json()["model"] == "fake-narrow"


def test_evaluate_candidate_energies_model_echo_and_422(client, fake_backends):
    sid = _session()
    body = {"session_id": sid, "candidates": [
        {"id": 1, "action": "substitute", "index": 0, "to": "Al"},
        {"id": 2, "action": "delete", "index": 1}]}
    r = client.post("/api/evaluate-candidate-energies", json={**body, "model": "fake-emt"})
    assert r.status_code == 200 and r.json()["model"] == "fake-emt"
    assert all(x["energy"] is not None for x in r.json()["results"])
    bad = client.post("/api/evaluate-candidate-energies", json={**body, "model": "fake-narrow"})
    assert bad.status_code == 422 and "Al" in bad.json()["detail"]


def test_get_insertable_elements_per_model(client, fake_backends):
    r = client.post("/api/get-insertable-elements", json={"model": "fake-narrow"})
    assert r.json()["insertable_elements"] == ["Cu"]
    allv = client.post("/api/get-insertable-elements", json={}).json()["insertable_elements"]
    assert "Cu" in allv and len(allv) > 3


def test_get_insertable_elements_echoes_model(client, fake_backends):
    r = client.post("/api/get-insertable-elements", json={"model": "fake-narrow"}).json()
    assert r["model"] == "fake-narrow"
    assert client.post("/api/get-insertable-elements", json={}).json()["model"] is None
    bad = client.post("/api/get-insertable-elements", json={"model": "nope"})
    assert bad.status_code == 400


def test_relaxed_cif_header_names_model(client, fake_backends):
    sid = _session()
    assert _relax(client, sid, model="fake-narrow").status_code == 200
    cif = client.post("/api/generate-relaxed-structure-cif", json={"session_id": sid})
    assert cif.status_code == 200
    assert "# Fake Narrow relaxation: model=fake-narrow," in cif.text


def test_relaxed_cif_model_mismatch_409(client, fake_backends):
    sid = _session()
    assert _relax(client, sid, model="fake-narrow").status_code == 200
    r = client.post("/api/generate-relaxed-structure-cif",
                    json={"session_id": sid, "model": "fake-emt"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "Fake Narrow" in detail and "Fake EMT" in detail


def test_relaxed_cif_matching_or_omitted_model_ok(client, fake_backends):
    sid = _session()
    assert _relax(client, sid, model="fake-narrow").status_code == 200
    url = "/api/generate-relaxed-structure-cif"
    omitted = client.post(url, json={"session_id": sid})
    matching = client.post(url, json={"session_id": sid, "model": "fake-narrow"})
    assert omitted.status_code == 200 and matching.status_code == 200
    header = "# Fake Narrow relaxation: model=fake-narrow,"
    assert header in omitted.text and header in matching.text


def test_relaxed_cif_unknown_or_bad_model_400(client, fake_backends):
    sid = _session()
    assert _relax(client, sid, model="fake-narrow").status_code == 200
    url = "/api/generate-relaxed-structure-cif"
    assert client.post(url, json={"session_id": sid, "model": "nope"}).status_code == 400
    assert client.post(url, json={"session_id": sid, "model": 123}).status_code == 400


# -- model chosen at load time (create-supercell) and fixed per session --

def _crystal(client, name="Cu"):
    r = client.post("/api/analyze-cif-sample", json={"filename": f"Metals/{name}.cif"})
    assert r.status_code == 200
    return r.json()


def _load(client, name="Cu", **extra):
    return client.post("/api/create-supercell", json={
        "crystal_data": _crystal(client, name), "supercell_size": [2, 2, 2], **extra})


def _loaded_sid(client, model=None, name="Cu"):
    extra = {"model": model} if model else {}
    r = _load(client, name, **extra)
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_create_supercell_stores_and_echoes_model(client, fake_backends):
    r = _load(client, model="fake-narrow")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-narrow"
    assert session_manager.sessions[body["session_id"]]["model"] == "fake-narrow"


def test_create_supercell_defaults_to_registry_default(client, fake_backends):
    body = _load(client).json()
    assert body["model"] == "fake-emt"
    assert session_manager.sessions[body["session_id"]]["model"] == "fake-emt"


def test_create_supercell_unknown_400_unavailable_503(client, fake_backends, monkeypatch):
    assert _load(client, model="nope").status_code == 400
    monkeypatch.setattr(MACEBackend, "is_available", classmethod(lambda cls: False))
    assert _load(client, model="mace-mp-0-medium").status_code == 503


def test_create_supercell_unsupported_element_422(client, fake_backends):
    # fake-narrow supports Cu only; Ni.cif is rejected up front
    r = _load(client, "Ni", model="fake-narrow")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Ni" in detail and "fake-narrow" in detail
    assert _load(client, "Ni", model="fake-emt").status_code == 200


def test_create_supercell_reuse_session_id_new_model_wins(client, fake_backends):
    first = _load(client, model="fake-narrow").json()
    again = _load(client, model="fake-emt", session_id=first["session_id"]).json()
    assert again["session_id"] == first["session_id"]
    assert session_manager.sessions[first["session_id"]]["model"] == "fake-emt"


def test_session_model_used_when_request_omits_model(client, fake_backends):
    sid = _loaded_sid(client, "fake-narrow")
    rel = _relax(client, sid)
    assert rel.status_code == 200 and rel.json()["model_info"]["id"] == "fake-narrow"
    pred = client.post("/api/predict", json={
        "session_id": sid, "filename": "Metals/Cu.cif", "supercell_size": [2, 2, 2]})
    assert pred.status_code == 200 and pred.json()["model_info"]["id"] == "fake-narrow"
    one = client.post("/api/evaluate-insertion-energy", json={
        "session_id": sid, "element": "Cu", "frac_coords": [0.25, 0.25, 0.25]})
    assert one.status_code == 200 and one.json()["model"] == "fake-narrow"
    many = client.post("/api/evaluate-insertion-energies", json={
        "session_id": sid, "element": "Cu",
        "sites": [{"id": 0, "frac_coords": [0.25, 0.25, 0.25]}]})
    assert many.status_code == 200 and many.json()["model"] == "fake-narrow"
    cand = client.post("/api/evaluate-candidate-energies", json={
        "session_id": sid, "candidates": [{"id": 1, "action": "delete", "index": 0}]})
    assert cand.status_code == 200 and cand.json()["model"] == "fake-narrow"
    ins = client.post("/api/get-insertable-elements", json={"session_id": sid})
    assert ins.json()["model"] == "fake-narrow" and ins.json()["insertable_elements"] == ["Cu"]


def test_different_model_than_session_409(client, fake_backends):
    sid = _loaded_sid(client, "fake-narrow")
    other = {"model": "fake-emt"}
    base = {"session_id": sid}
    calls = [
        ("/api/relax", {**base, **other}),
        ("/api/predict", {**base, **other, "filename": "Metals/Cu.cif"}),
        ("/api/evaluate-insertion-energy",
         {**base, **other, "element": "Cu", "frac_coords": [0.25, 0.25, 0.25]}),
        ("/api/evaluate-insertion-energies",
         {**base, **other, "element": "Cu",
          "sites": [{"id": 0, "frac_coords": [0.25, 0.25, 0.25]}]}),
        ("/api/evaluate-candidate-energies",
         {**base, **other, "candidates": [{"id": 1, "action": "delete", "index": 0}]}),
        ("/api/get-insertable-elements", {**base, **other}),
        ("/api/generate-relaxed-structure-cif", {**base, **other}),
    ]
    for url, body in calls:
        r = client.post(url, json=body)
        assert r.status_code == 409, (url, r.status_code, r.text)
        assert "fake-narrow" in r.json()["detail"] and "fake-emt" in r.json()["detail"]
    # Matching model explicitly is fine
    assert _relax(client, sid, model="fake-narrow").status_code == 200


def test_session_without_model_keeps_legacy_behavior(client, fake_backends):
    sid = _session()
    assert "model" not in session_manager.sessions[sid]
    assert _relax(client, sid).json()["model_info"]["id"] == "fake-emt"
    assert _relax(client, sid, model="fake-narrow").json()["model_info"]["id"] == "fake-narrow"


def test_helper_session_model_id():
    assert main._session_model_id({}, None) is None
    assert main._session_model_id({}, "x") == "x"
    assert main._session_model_id({"model": "a"}, None) == "a"
    assert main._session_model_id({"model": "a"}, "a") == "a"
    with pytest.raises(main.HTTPException) as exc:
        main._session_model_id({"model": "a"}, "b")
    assert exc.value.status_code == 409


def test_edit_validation_uses_session_model(client, fake_backends):
    sid = _loaded_sid(client, "fake-narrow")
    ops = [{"action": "substitute", "index": 0, "to": "Al"}]
    r = client.post("/api/apply-atomic-operations", json={"session_id": sid, "operations": ops})
    assert r.status_code == 400 and "Al" in r.json()["detail"]
    ins = [{"action": "insert", "to": "Al", "coords": [0.25, 0.25, 0.25]}]
    r = client.post("/api/apply-atomic-operations", json={"session_id": sid, "operations": ins})
    assert r.status_code == 400
    # Cu is supported by fake-narrow
    ok = [{"action": "insert", "to": "Cu", "coords": [0.25, 0.25, 0.25]}]
    assert client.post("/api/apply-atomic-operations",
                       json={"session_id": sid, "operations": ok}).status_code == 200
    # evaluate endpoints validate the target element against the session model
    r = client.post("/api/evaluate-candidate-energies", json={
        "session_id": sid, "candidates": [{"id": 1, "action": "substitute", "index": 0, "to": "Al"}]})
    assert r.status_code == 400
    r = client.post("/api/evaluate-insertion-energies", json={
        "session_id": sid, "element": "Al",
        "sites": [{"id": 0, "frac_coords": [0.25, 0.25, 0.25]}]})
    assert r.status_code == 400


def test_edit_validation_default_model_session_allows_substitution(client, fake_backends):
    sid = _loaded_sid(client)  # fake-emt supports Al
    ops = [{"action": "substitute", "index": 0, "to": "Al"}]
    r = client.post("/api/apply-atomic-operations", json={"session_id": sid, "operations": ops})
    assert r.status_code == 200


def test_reset_session_structure_keeps_model(client, fake_backends):
    sid = _loaded_sid(client, "fake-narrow")
    r = client.post("/api/reset-session-structure", json={"session_id": sid})
    assert r.status_code == 200
    assert session_manager.sessions[sid]["model"] == "fake-narrow"
    assert _relax(client, sid).json()["model_info"]["id"] == "fake-narrow"


# -- MACE load error mapping (mocked, no mace import) -------------------------

@pytest.mark.parametrize("exc", [AttributeError("bad torch"), ValueError("bad config"),
                                 ImportError("e3nn"), OSError("no net")])
def test_mace_construct_errors_map_to_model_unavailable(monkeypatch, exc):
    import sys
    import types
    from mlip.base import ModelUnavailableError

    def boom(**kwargs):
        raise exc

    fake = types.ModuleType("mace.calculators")
    fake.mace_mp = boom
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", fake)
    with pytest.raises(ModelUnavailableError) as excinfo:
        MACEBackend()._load_sync()
    assert excinfo.value.__cause__ is exc  # traceback preserved


def test_mace_import_attribute_error_maps_to_model_unavailable(monkeypatch):
    import sys
    import types
    from mlip.base import ModelUnavailableError

    fake = types.ModuleType("mace.calculators")  # has no mace_mp -> ImportError
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", fake)
    with pytest.raises(ModelUnavailableError):
        MACEBackend()._load_sync()


def test_mace_download_failure_still_maps_to_unavailable(monkeypatch):
    import sys
    import types
    from mlip.base import ModelUnavailableError

    def boom(**kwargs):
        raise RuntimeError("Model download failed")

    fake = types.ModuleType("mace.calculators")
    fake.mace_mp = boom
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", fake)
    with pytest.raises(ModelUnavailableError):
        MACEBackend()._load_sync()
