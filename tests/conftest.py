"""Shared pytest fixtures for the CrystalNexus test suite."""
import sys
from pathlib import Path

import pytest

# Make the repository root importable so tests can `from main import ...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from main import app  # noqa: E402
from analytics_db import AnalyticsDatabase  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """Shared TestClient for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolated_analytics_db(tmp_path):
    """
    Redirect every analytics write main.py makes during a test to a
    throwaway sqlite file instead of the real analytics.db in the repo
    root.

    main.py does `from analytics_db import analytics_db` (a module-level
    singleton instance), so every route handler looks up the name
    `analytics_db` in main's own module namespace at call time -- patching
    that attribute here (rather than the one on the analytics_db module)
    is what actually redirects main.py's calls, since it already copied
    the reference in at import time.
    """
    test_db = AnalyticsDatabase(db_path=str(tmp_path / "test_analytics.db"))
    original = main.analytics_db
    main.analytics_db = test_db
    yield
    main.analytics_db = original


@pytest.fixture(scope="session")
def sample_cif_dir():
    """Path to the bundled sample CIF directory."""
    return REPO_ROOT / "sample_cif"


@pytest.fixture(autouse=True)
def _generous_rate_limit_for_tests():
    """
    rate_limit_middleware (main.py) applies a per-client-IP token bucket to
    every POST /api/* request. TestClient's synthetic client address is
    always "testclient", so the whole suite -- hundreds of POSTs across
    many test functions -- shares a single bucket; at the production
    default (RATE_LIMIT_BURST=20) it would 429 partway through the run.

    Raise the limit for the duration of each test instead of disabling
    the middleware, so the middleware itself still runs (exercising the
    same code path as production) and a specific test can still opt back
    into a low limit via monkeypatch to exercise the 429 path.
    """
    old_burst = main.RATE_LIMIT_BURST
    old_rate = main.RATE_LIMIT_REQUESTS_PER_MINUTE
    main.RATE_LIMIT_BURST = 100_000
    main.RATE_LIMIT_REQUESTS_PER_MINUTE = 100_000 * 60
    yield
    main.RATE_LIMIT_BURST = old_burst
    main.RATE_LIMIT_REQUESTS_PER_MINUTE = old_rate
