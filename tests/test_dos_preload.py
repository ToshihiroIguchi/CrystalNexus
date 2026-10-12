"""Tests for the startup PRELOAD of PET-MAD-DOS (no real model is loaded)."""
import asyncio
import logging
import time

import pytest

import main


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " Off "])
def test_preload_disabled_values(monkeypatch, value):
    monkeypatch.setenv("CRYSTALNEXUS_DOS_PRELOAD", value)
    assert main.dos_preload_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_preload_enabled_values(monkeypatch, value):
    monkeypatch.setenv("CRYSTALNEXUS_DOS_PRELOAD", value)
    assert main.dos_preload_enabled() is True


def test_preload_enabled_by_default(monkeypatch):
    monkeypatch.delenv("CRYSTALNEXUS_DOS_PRELOAD", raising=False)
    assert main.dos_preload_enabled() is True


def test_conftest_disables_preload_for_tests():
    assert main.dos_preload_enabled() is False


@pytest.fixture
def fresh_manager(monkeypatch):
    """Real DOSModelManager logic with a counting fake loader / predictor."""
    state = {"loads": 0, "predicts": 0}

    def fake_load():
        state["loads"] += 1
        time.sleep(0.2)
        return object()

    # get_calculator() stores the calculator on the instance, so reset that
    monkeypatch.setattr(main.dos_manager, "_calculator", None, raising=False)
    monkeypatch.setattr(main.DOSModelManager, "_lock", asyncio.Lock())
    monkeypatch.setattr(main.DOSModelManager, "_load_sync", staticmethod(fake_load))

    def fake_predict(atoms):
        state["predicts"] += 1
        return {}

    monkeypatch.setattr(main.dos_manager, "predict_sync", fake_predict, raising=False)
    monkeypatch.setattr(main, "dos_model_problems", lambda full=False: [])
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: object())
    return state


def test_preload_success(fresh_manager, caplog):
    caplog.set_level(logging.INFO, logger=main.logger.name)
    assert asyncio.run(main.preload_dos_model()) is True
    assert fresh_manager == {"loads": 1, "predicts": 1}
    assert main.dos_manager._calculator is not None
    assert any("PET-MAD-DOS preloaded in" in r.getMessage() for r in caplog.records)
    assert main._dos_preloading is False


def test_preload_failure_keeps_lazy_fallback(fresh_manager, monkeypatch, caplog):
    def boom():
        raise main.DOSModelUnavailable("broken weights")

    monkeypatch.setattr(main.DOSModelManager, "_load_sync", staticmethod(boom))
    caplog.set_level(logging.WARNING, logger=main.logger.name)
    assert asyncio.run(main.preload_dos_model()) is False
    assert main.dos_manager._calculator is None
    assert any("preload failed" in r.getMessage() and "broken weights" in r.getMessage()
               for r in caplog.records)
    assert main._dos_preloading is False


def test_preload_skipped_when_files_missing(fresh_manager, monkeypatch, caplog):
    monkeypatch.setattr(main, "dos_model_problems", lambda full=False: ["missing x.ckpt"])
    caplog.set_level(logging.WARNING, logger=main.logger.name)
    assert asyncio.run(main.preload_dos_model()) is False
    assert fresh_manager["loads"] == 0
    assert any("skipped" in r.getMessage() for r in caplog.records)


def test_preload_skipped_when_upet_missing(fresh_manager, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **k: None)
    assert asyncio.run(main.preload_dos_model()) is False
    assert fresh_manager["loads"] == 0


def test_concurrent_user_request_loads_only_once(fresh_manager):
    async def scenario():
        preload = asyncio.create_task(main.preload_dos_model())
        await asyncio.sleep(0.05)  # preload is now inside the loader
        user = await main.dos_manager.get_calculator()
        await preload
        return user

    calc = asyncio.run(scenario())
    assert calc is main.dos_manager._calculator
    assert fresh_manager["loads"] == 1


def test_status_reports_loaded_and_preloading(client, monkeypatch):
    monkeypatch.setattr(main.dos_manager, "_calculator", None, raising=False)
    data = client.get("/api/dos-status").json()
    assert data["loaded"] is False and data["preloading"] is False
    assert {"available", "problems", "hint"} <= set(data)
    monkeypatch.setattr(main.dos_manager, "_calculator", object(), raising=False)
    assert client.get("/api/dos-status").json()["loaded"] is True
