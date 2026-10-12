"""Tests for the ML electronic DOS endpoint (/api/predict-dos, PET-MAD-DOS).

Fast tests replace the model manager with a fake so no `upet` model is
loaded; the @pytest.mark.slow test at the end runs the real model.
"""
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from pymatgen.core import Structure

import main
from main import session_manager


def _structure(name="Cu.cif", folder="Metals"):
    return Structure.from_file(Path("sample_cif") / folder / name)


def _create_session(structure, filename="Cu.cif"):
    return session_manager.create_session(str(uuid.uuid4()), filename, structure)


def _relaxed_session(structure=None, filename="Cu.cif"):
    structure = structure if structure is not None else _structure()
    session_id = _create_session(structure.copy(), filename)
    session_manager.get_session_info(session_id)["relaxed_structure"] = structure.copy()
    return session_id


class FakeDOSManager:
    """Stands in for main.dos_manager: a smooth fake DOS on a 0.05 eV grid."""
    model_name = "PET-MAD-DOS"

    def __init__(self, gap=1.0, delay=0.0, on_predict=None):
        self.gap = gap
        self.delay = delay
        self.on_predict = on_predict
        self.calls = 0

    async def get_calculator(self):
        return object()

    def predict_sync(self, atoms):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.on_predict:
            self.on_predict()
        step = 0.05
        energy = np.arange(4806) * step
        return {
            "energy": energy,
            "dos": np.abs(np.sin(energy)) * len(atoms),
            "fermi_level": 100.0,
            "band_gap": self.gap,
            "sigma": 0.3,
            "energy_step": step,
        }


@pytest.fixture
def fake_dos(monkeypatch):
    fake = FakeDOSManager()
    monkeypatch.setattr(main, "dos_manager", fake)
    return fake


def test_dos_requires_session_id(client):
    assert client.post("/api/predict-dos", json={}).status_code == 400


def test_dos_unknown_session_404(client, fake_dos):
    response = client.post("/api/predict-dos", json={"session_id": "nope"})
    assert response.status_code == 404


def test_dos_requires_relaxation(client, fake_dos):
    session_id = _create_session(_structure())
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 400
    assert "relax" in response.json()["detail"].lower()
    assert fake_dos.calls == 0


def test_dos_success_schema_and_cache(client, fake_dos):
    session_id = _relaxed_session()
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 200, response.text
    data = response.json()

    n = len(data["energies_rel_ef"])
    assert n > 0
    for key in ("dos", "dos_per_atom", "integrated_dos", "energy_grid"):
        assert len(data[key]) == n
    lo, hi = main.DOS_WINDOW_EV
    assert min(data["energies_rel_ef"]) >= lo and max(data["energies_rel_ef"]) <= hi
    assert data["fermi_level"] == pytest.approx(100.0)
    assert data["band_gap"] == pytest.approx(1.0)
    assert data["n_atoms"] == len(_structure())
    assert data["formula"] == "Cu"
    assert data["window"] == [lo, hi]
    assert data["model_info"]["name"] == "PET-MAD-DOS"
    assert data["model_info"]["sigma"] == pytest.approx(0.3)
    assert data["elapsed_seconds"] >= 0
    # Cumulative integral is monotonic for a non-negative DOS.
    assert all(b >= a - 1e-9 for a, b in zip(data["integrated_dos"], data["integrated_dos"][1:]))

    cached = session_manager.get_session_info(session_id)["dos_result"]
    assert cached["structure_version"] == session_manager.get_session_info(session_id)["structure_version"]
    assert cached["result"]["formula"] == "Cu"


def test_dos_strips_magmom_site_property(client, fake_dos):
    structure = _structure()
    structure.add_site_property("magmom", [0.5] * len(structure))
    session_id = _relaxed_session(structure)
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 200, response.text


def test_dos_cache_cleared_after_structure_edit(client, fake_dos):
    session_id = _relaxed_session()
    assert client.post("/api/predict-dos", json={"session_id": session_id}).status_code == 200
    assert "dos_result" in session_manager.get_session_info(session_id)

    session_manager.update_structure(session_id, _structure())
    info = session_manager.get_session_info(session_id)
    assert "dos_result" not in info
    assert "relaxed_structure" not in info


def test_dos_discarded_when_structure_changes_during_inference(client, monkeypatch):
    session_id = _relaxed_session()

    def edit():
        session_manager.update_structure(session_id, _structure())

    monkeypatch.setattr(main, "dos_manager", FakeDOSManager(on_predict=edit))
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 409
    assert "dos_result" not in session_manager.get_session_info(session_id)


def test_dos_atom_limit(client, fake_dos, monkeypatch):
    monkeypatch.setattr(main, "MAX_DOS_ATOMS", 2)
    structure = _structure()
    structure.make_supercell((2, 2, 2))
    session_id = _relaxed_session(structure)
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 413
    assert fake_dos.calls == 0


def test_dos_timeout_returns_504(client, monkeypatch):
    monkeypatch.setattr(main, "DOS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main, "dos_manager", FakeDOSManager(delay=0.5))
    session_id = _relaxed_session()
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 504
    time.sleep(0.6)  # let the worker thread finish before the next test


def test_dos_rejects_concurrent_request(client, fake_dos):
    session_id = _relaxed_session()
    assert main._dos_semaphore.locked() is False
    main._dos_semaphore._value = 0  # simulate a running prediction
    try:
        response = client.post("/api/predict-dos", json={"session_id": session_id})
        assert response.status_code == 429
    finally:
        main._dos_semaphore._value = 1


def test_dos_missing_upet_returns_503(client, monkeypatch):
    class Missing:
        model_name = "PET-MAD-DOS"

        async def get_calculator(self):
            raise main.DOSModelUnavailable("DOS prediction requires the 'upet' package")

    monkeypatch.setattr(main, "dos_manager", Missing())
    session_id = _relaxed_session()
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 503
    assert "upet" in response.json()["detail"]


def test_dos_loader_reports_missing_upet(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("upet"):
            raise ImportError("No module named 'upet'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(main.DOSModelUnavailable, match="upet"):
        main.DOSModelManager._load_sync()


def _load_download_script():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "download_dos_model.py"
    spec = importlib.util.spec_from_file_location("download_dos_model_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_manifest(tmp_path, tools):
    """Tiny fake files plus a matching manifest (no network)."""
    import hashlib
    manifest = {}
    for name, payload in (("a.ckpt", b"alpha"), ("b.pt", b"bravo!")):
        (tmp_path / name).write_bytes(payload)
        manifest[name] = ("models/" + name, len(payload), hashlib.sha256(payload).hexdigest())
    return manifest


def test_download_script_check_logic(tmp_path):
    tools = _load_download_script()
    manifest = _fake_manifest(tmp_path, tools)
    assert tools.check_model_dir(tmp_path, full=True, manifest=manifest) == []

    # Same size, different content: only the SHA256 check notices.
    (tmp_path / "a.ckpt").write_bytes(b"ALPHA")
    assert tools.check_model_dir(tmp_path, full=False, manifest=manifest) == []
    problems = tools.check_model_dir(tmp_path, full=True, manifest=manifest)
    assert len(problems) == 1 and "SHA256" in problems[0]

    (tmp_path / "b.pt").write_bytes(b"x")
    assert any("size mismatch" in p for p in tools.check_model_dir(tmp_path, full=False, manifest=manifest))
    (tmp_path / "b.pt").unlink()
    assert any("missing" in p for p in tools.check_model_dir(tmp_path, full=False, manifest=manifest))


def test_download_script_manifest_is_pinned():
    tools = _load_download_script()
    assert len(tools.REVISION) == 40 and all(c in "0123456789abcdef" for c in tools.REVISION)
    for _, size, sha in tools.MANIFEST.values():
        assert size > 0 and len(sha) == 64


def test_download_script_check_cli(tmp_path, monkeypatch, capsys):
    tools = _load_download_script()
    monkeypatch.setattr(tools, "MANIFEST", _fake_manifest(tmp_path, tools))
    assert tools.main(["--check", "--dir", str(tmp_path)]) == 0
    (tmp_path / "a.ckpt").unlink()
    assert tools.main(["--check", "--dir", str(tmp_path)]) == 1
    assert "download_dos_model.py" in capsys.readouterr().err


def test_download_script_env_override(tmp_path, monkeypatch):
    tools = _load_download_script()
    monkeypatch.setenv("CRYSTALNEXUS_DOS_MODEL_DIR", str(tmp_path))
    assert tools.default_model_dir() == tmp_path


def test_dos_missing_model_files_returns_503_with_hint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("CRYSTALNEXUS_DOS_MODEL_DIR", str(tmp_path))  # empty dir
    monkeypatch.setattr(main, "dos_manager", main.DOSModelManager())
    monkeypatch.setattr(main.DOSModelManager, "_calculator", None)
    session_id = _relaxed_session()
    response = client.post("/api/predict-dos", json={"session_id": session_id})
    assert response.status_code == 503
    assert "scripts/download_dos_model.py" in response.json()["detail"]


def test_dos_status_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("CRYSTALNEXUS_DOS_MODEL_DIR", str(tmp_path))
    data = client.get("/api/dos-status").json()
    assert data["available"] is False and data["problems"]
    assert "download_dos_model.py" in data["hint"]


def test_dos_loader_uses_explicit_local_paths_and_offline(monkeypatch, tmp_path):
    import sys
    import types

    captured = {}

    class FakeCalc:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Stand-in for upet.ase.dos so no real model is touched.
    pkg_upet = types.ModuleType("upet")
    pkg_ase = types.ModuleType("upet.ase")
    pkg_dos = types.ModuleType("upet.ase.dos")
    pkg_dos.PETMADDOSCalculator = FakeCalc
    monkeypatch.setitem(sys.modules, "upet", pkg_upet)
    monkeypatch.setitem(sys.modules, "upet.ase", pkg_ase)
    monkeypatch.setitem(sys.modules, "upet.ase.dos", pkg_dos)

    monkeypatch.setenv("CRYSTALNEXUS_DOS_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(main, "dos_model_problems", lambda full=False: [])

    main.DOSModelManager._load_sync()

    assert Path(captured["model_path"]) == tmp_path / "pet-mad-dos-v1.0.ckpt"
    assert Path(captured["bandgap_model_path"]) == tmp_path / "bandgap-model.pt"
    assert Path(captured["fermi_model_path"]) == tmp_path / "fermi-model.pt"
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.slow
def test_dos_real_model_si_gap_and_cu_metal(client):
    pytest.importorskip("upet")
    if main.dos_model_problems(full=False):
        pytest.skip("PET-MAD-DOS model files not installed; run python scripts/download_dos_model.py")
    si = _structure("Si.cif", "Metals")
    si_session = _relaxed_session(si, "Si.cif")
    cu_session = _relaxed_session(_structure(), "Cu.cif")

    t0 = time.time()
    si_resp = client.post("/api/predict-dos", json={"session_id": si_session})
    cu_resp = client.post("/api/predict-dos", json={"session_id": cu_session})
    assert time.time() - t0 < 60

    assert si_resp.status_code == 200, si_resp.text
    assert cu_resp.status_code == 200, cu_resp.text
    assert si_resp.json()["band_gap"] > 0
    assert cu_resp.json()["band_gap"] == pytest.approx(0.0, abs=1e-6)


# --- Static asset cache busting -------------------------------------------

def test_static_url_hash_and_change(tmp_path, monkeypatch):
    import re
    (tmp_path / "css").mkdir()
    f = tmp_path / "css" / "a.css"
    f.write_text("body{}")
    monkeypatch.setattr(main, "STATIC_DIR", str(tmp_path))
    main._static_hash_cache.clear()
    url1 = main.static_url("css/a.css")
    assert re.fullmatch(r"/static/css/a\.css\?v=[0-9a-f]{10}", url1)
    assert main.static_url("css/a.css") == url1
    f.write_text("body{color:red}")
    url2 = main.static_url("css/a.css")
    assert url2 != url1 and url2.startswith("/static/css/a.css?v=")
    assert main.static_url("css/missing.css") == "/static/css/missing.css"


def test_index_html_versioned_assets_and_no_gap_shading(client):
    html = client.get("/").text
    assert "/static/css/styles.css?v=" in html
    assert "/static/js/dos_chart.js?v=" in html
    assert "shaded" not in html.lower()
    js = client.get("/static/js/dos_chart.js").text
    assert "_gapInterval" not in js and "bandGap" not in js


def test_static_cache_control_headers(client):
    r = client.get("/static/css/styles.css?v=x")
    assert r.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    r = client.get("/static/css/styles.css")
    assert r.headers["Cache-Control"] == "no-cache"


def test_new_cif_reset_path_closes_dos_modal(client):
    """Reset paths must call resetDOSState so no stale DOS result stays on screen."""
    html = client.get("/").text
    start = html.index("function resetAndLoadNewCIF()")
    end = html.index("function resetApplicationState()")
    assert "resetDOSState();" in html[start:end]
    # resetApplicationState also resets via hideRelaxCharts -> resetDOSState
    hide = html.index("function hideRelaxCharts()")
    assert "resetDOSState();" in html[hide:hide + 600]
    app_end = html.index("function ", end + 40)
    assert "hideRelaxCharts();" in html[end:app_end]
