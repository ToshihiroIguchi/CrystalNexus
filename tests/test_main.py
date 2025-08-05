import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "CrystalNexus"}

def test_root_endpoint():
    """Test the root endpoint returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_sample_cif_files():
    """Test getting sample CIF files"""
    response = client.get("/api/sample-cif-files")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert isinstance(data["files"], list)

def test_analyze_cif_sample_missing_filename():
    """Test analyze sample CIF with missing filename"""
    response = client.post("/api/analyze-cif-sample", json={})
    assert response.status_code == 400

def test_analyze_cif_sample_nonexistent_file():
    """Test analyze sample CIF with nonexistent file"""
    response = client.post("/api/analyze-cif-sample", json={"filename": "nonexistent.cif"})
    assert response.status_code == 404