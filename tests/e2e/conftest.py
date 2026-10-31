"""Fixtures for the real-browser (Playwright) E2E suite.

A real uvicorn server runs ``main:app`` in a background thread of the test
process, with the EMT-based fake backend from ``tests/conftest.py`` as the only
registered model (no model download, relaxations take milliseconds). Production
code is untouched: the registry is monkeypatched for the session.

Run with ``python -m pytest -m e2e`` (needs ``python -m playwright install
chromium``; 3Dmol.js and JSZip are fetched from their CDN by the page).
"""
import socket
import threading
import time

import pytest
import uvicorn

import main
from analytics_db import AnalyticsDatabase
from mlip import registry as mlip_registry
from tests.conftest import FakeBackend, FakeMagmomBackend

# Messages that headless Chromium without a GPU produces for 3Dmol/WebGL and
# that say nothing about the app under test.
IGNORED_CONSOLE_PATTERNS = (
    "webgl",
    "gpu stall",
    "swiftshader",
    "GL Driver Message",
    "GroupMarker",
    "favicon",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """Start uvicorn serving main:app with the fake EMT model as the default."""
    mp = pytest.MonkeyPatch()
    # Only the fake backends are registered, so the model dialog offers
    # exactly one usable model and /api/models never touches CHGNet/MACE.
    # (main.py imported the dict by name, so mutate it in place.)
    for model_id in list(mlip_registry.MODEL_REGISTRY):
        mp.delitem(mlip_registry.MODEL_REGISTRY, model_id)
    mp.setitem(mlip_registry.MODEL_REGISTRY, "fake-emt", FakeBackend)
    mp.setitem(mlip_registry.MODEL_REGISTRY, "fake-emt-magmom", FakeMagmomBackend)
    mp.setenv("CRYSTALNEXUS_DEFAULT_MODEL", "fake-emt")
    # The per-test rate-limit fixture of tests/conftest.py is function scoped;
    # raise the limit for the whole server lifetime too.
    mp.setattr(main, "RATE_LIMIT_BURST", 100_000)
    mp.setattr(main, "RATE_LIMIT_REQUESTS_PER_MINUTE", 100_000 * 60)
    mp.setattr(main, "analytics_db",
               AnalyticsDatabase(db_path=str(tmp_path_factory.mktemp("e2e") / "analytics.db")))

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("E2E uvicorn server failed to start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
    mp.undo()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {**browser_context_args, "viewport": {"width": 1500, "height": 950},
            "accept_downloads": True}


class PageLog:
    """Console errors, failed requests and dialogs seen by a page."""

    def __init__(self):
        self.console_errors = []
        self.page_errors = []
        self.failed_requests = []   # (status or failure text, method, url)
        self.dialogs = []           # (type, message)
        self.allow_errors = False   # tests that provoke errors on purpose

    def problems(self):
        out = [f"console.error: {m}" for m in self.console_errors]
        out += [f"pageerror: {m}" for m in self.page_errors]
        out += [f"request failed: {r}" for r in self.failed_requests]
        return out


@pytest.fixture
def page(page, live_server):
    """Playwright page that records errors; checked at the end of each test."""
    log = PageLog()
    page.log = log

    def on_console(msg):
        if msg.type != "error":
            return
        text = msg.text
        if any(p.lower() in text.lower() for p in IGNORED_CONSOLE_PATTERNS):
            return
        # The browser logs every non-2xx fetch as a generic console error;
        # the failed request itself is tracked via the response handler.
        if text.startswith("Failed to load resource"):
            return
        log.console_errors.append(text)

    def on_response(resp):
        if resp.status >= 400 and resp.url.startswith(live_server) \
                and not resp.url.endswith("/favicon.ico"):
            log.failed_requests.append((resp.status, resp.request.method, resp.url))

    def on_request_failed(req):
        # Aborted fetches/SSE streams are normal when the app cancels work.
        if req.url.startswith(live_server) and "aborted" not in (req.failure or "").lower() \
                and "ERR_ABORTED" not in (req.failure or ""):
            log.failed_requests.append((req.failure, req.method, req.url))

    def on_dialog(dialog):
        log.dialogs.append((dialog.type, dialog.message))
        dialog.accept()

    page.on("console", on_console)
    page.on("pageerror", lambda e: log.page_errors.append(str(e)))
    page.on("response", on_response)
    page.on("requestfailed", on_request_failed)
    page.on("dialog", on_dialog)
    page.set_default_timeout(20_000)
    yield page


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """After a passing test body, fail it if the page logged unexpected errors.

    Done in the call phase (not fixture teardown) so that ``xfail`` applies.
    """
    result = yield
    page = item.funcargs.get("page")
    log = getattr(page, "log", None)
    if log is not None and not log.allow_errors and log.problems():
        pytest.fail("unexpected browser errors:\n  " + "\n  ".join(log.problems()),
                    pytrace=False)
    return result
