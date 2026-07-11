"""Shared pytest fixtures for the CrystalNexus test suite."""
import sys
from pathlib import Path

import pytest

# Make the repository root importable so tests can `from main import ...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """Shared TestClient for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(scope="session")
def sample_cif_dir():
    """Path to the bundled sample CIF directory."""
    return REPO_ROOT / "sample_cif"
