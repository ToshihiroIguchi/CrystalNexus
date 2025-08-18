import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import platform

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import app, CHGNET_AVAILABLE, ALLOWED_ELEMENTS

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "CrystalNexus"}

def test_root_endpoint(client):
    """Test the root endpoint returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Check for Reset & Load New CIF button
    assert "🔄 Reset & Load New CIF" in response.text

def test_sample_cif_files(client):
    """Test getting sample CIF files"""
    response = client.get("/api/sample-cif-files")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert isinstance(data["files"], list)
    assert len(data["files"]) == 5  # Should have 5 sample files

def test_chgnet_elements(client):
    """Test CHGNet elements endpoint"""
    response = client.get("/api/chgnet-elements")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "elements" in data
    assert "total_elements" in data
    assert isinstance(data["elements"], list)
    assert data["total_elements"] > 50  # Should have many elements

def test_windows_compatibility():
    """Test Windows compatibility detection"""
    from main import WINDOWS_PLATFORM
    current_platform = platform.system() == "Windows"
    assert WINDOWS_PLATFORM == current_platform

def test_analyze_cif_sample_missing_filename(client):
    """Test analyze sample CIF with missing filename"""
    response = client.post("/api/analyze-cif-sample", json={})
    assert response.status_code == 400

def test_analyze_cif_sample_nonexistent_file(client):
    """Test analyze sample CIF with nonexistent file"""
    response = client.post("/api/analyze-cif-sample", json={"filename": "nonexistent.cif"})
    assert response.status_code == 404

def test_analyze_cif_sample_valid(client):
    """Test analyze sample CIF with valid file"""
    response = client.post("/api/analyze-cif-sample", json={"filename": "C.cif"})
    assert response.status_code == 200
    data = response.json()
    assert "formula" in data
    assert "num_atoms" in data
    assert "density" in data
    assert "lattice_parameters" in data
    assert "filename" in data

def test_create_supercell(client):
    """Test supercell creation"""
    crystal_data = {
        "formula": "C4",
        "num_atoms": 4,
        "density": 1.94,
        "volume": 41.14,
        "num_sites": 4,
        "filename": "C.cif"
    }
    
    response = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [2, 2, 2],
        "session_id": "test-session-123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "supercell_created"
    assert "supercell_info" in data
    assert data["supercell_info"]["scaling_factor"] == 8

def test_security_validation():
    """Test security validation functions"""
    from main import validate_element, safe_filename, validate_supercell_size
    
    # Test element validation
    assert validate_element("Fe") == "Fe"
    with pytest.raises(ValueError):
        validate_element("InvalidElement")
    
    # Test filename validation
    assert safe_filename("test.cif") == "test.cif"
    with pytest.raises(ValueError):
        safe_filename("test.py")  # Non-CIF file
    with pytest.raises(ValueError):
        safe_filename("")  # Empty filename
    
    # Test supercell validation
    assert validate_supercell_size([2, 2, 2]) == [2, 2, 2]
    with pytest.raises(ValueError):
        validate_supercell_size([0, 2, 2])  # Zero dimension
    with pytest.raises(ValueError):
        validate_supercell_size([15, 2, 2])  # Too large dimension

def test_element_count():
    """Test that we have expected number of elements"""
    assert len(ALLOWED_ELEMENTS) >= 70  # Should have at least 70 elements
    
    # Test common elements are present
    common_elements = ["H", "C", "N", "O", "Fe", "Cu", "Zn", "Al", "Si"]
    for element in common_elements:
        assert element in ALLOWED_ELEMENTS, f"Element {element} should be supported"

@pytest.mark.skipif(not CHGNET_AVAILABLE, reason="CHGNet not available")
def test_chgnet_predict(client):
    """Test CHGNet prediction endpoint (only if CHGNet is available)"""
    response = client.post("/api/chgnet-predict", json={
        "filename": "C.cif",
        "operations": [],
        "supercell_size": [1, 1, 1]
    })
    # Should return either success or 503 (service unavailable)
    assert response.status_code in [200, 503]