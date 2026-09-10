import platform

import pytest

from main import ALLOWED_ELEMENTS, CHGNET_AVAILABLE


def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "CrystalNexus"}


def test_root_endpoint(client):
    """Test the root endpoint returns the application HTML page"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>CrystalNexus</title>" in response.text


def test_sample_cif_files(client):
    """Test getting sample CIF files"""
    response = client.get("/api/sample-cif-files")
    assert response.status_code == 200
    data = response.json()
    assert "structure" in data
    structure = data["structure"]
    assert "Metals" in structure["subdirs"]

    def count_cif_files(node):
        total = len(node["files"])
        for subdir in node["subdirs"].values():
            total += count_cif_files(subdir)
        return total

    assert count_cif_files(structure) > 0


def test_chgnet_elements(client):
    """Test CHGNet elements endpoint"""
    response = client.get("/api/chgnet-elements")
    assert response.status_code == 200
    data = response.json()
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
    response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    assert response.status_code == 200
    data = response.json()
    assert "formula" in data
    assert "num_atoms" in data
    assert "density" in data
    assert "lattice_parameters" in data
    assert "filename" in data


def test_create_supercell(client):
    """Test supercell creation"""
    # Analyze a real sample file to get consistent crystal data
    analyze_response = client.post("/api/analyze-cif-sample", json={"filename": "Metals/Cu.cif"})
    assert analyze_response.status_code == 200
    crystal_data = analyze_response.json()

    response = client.post("/api/create-supercell", json={
        "crystal_data": crystal_data,
        "supercell_size": [2, 2, 2],
        "session_id": "test-session-123"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "supercell_created"
    assert "supercell_info" in data
    supercell_info = data["supercell_info"]
    assert supercell_info["scaling_factor"] == 8
    assert supercell_info["size"] == [2, 2, 2]
    assert supercell_info["num_sites"] == crystal_data["num_sites"] * 8
    assert supercell_info["volume"] == pytest.approx(crystal_data["volume"] * 8)


def test_element_count():
    """Test that we have expected number of elements"""
    assert len(ALLOWED_ELEMENTS) >= 70  # Should have at least 70 elements

    # Test common elements are present
    common_elements = ["H", "C", "N", "O", "Fe", "Cu", "Zn", "Al", "Si"]
    for element in common_elements:
        assert element in ALLOWED_ELEMENTS, f"Element {element} should be supported"


@pytest.mark.slow
@pytest.mark.skipif(not CHGNET_AVAILABLE, reason="CHGNet not available")
def test_chgnet_predict(client):
    """Test CHGNet prediction endpoint returns a physical result"""
    response = client.post("/api/chgnet-predict", json={
        "filename": "Metals/Cu.cif",
        "operations": [],
        "supercell_size": [1, 1, 1]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    prediction = data["prediction"]
    energy = prediction["energy_eV_per_atom"]
    assert isinstance(energy, float)
    # Cohesive/formation energy per atom for bulk Cu must be negative
    assert energy < 0


def test_chgnet_predict_validates_supercell_size(client):
    """supercell_size must be validated before the (expensive, model-
    loading) CHGNet path runs, so this stays fast even without CHGNet."""
    response = client.post("/api/chgnet-predict", json={
        "filename": "Metals/Cu.cif",
        "operations": [],
        "supercell_size": [99, 1, 1],
    })
    assert response.status_code == 400


@pytest.mark.slow
@pytest.mark.skipif(not CHGNET_AVAILABLE, reason="CHGNet not available")
def test_chgnet_predict_with_substitution(client):
    """A substitution passed through /api/chgnet-predict must actually
    apply and be reflected in the returned structure info."""
    response = client.post("/api/chgnet-predict", json={
        "filename": "Metals/Cu.cif",
        "operations": [{"action": "substitute", "index": 0, "to": "Ni"}],
        "supercell_size": [1, 1, 1],
    })
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert "Ni" in prediction["formula"]
    assert prediction["operations_applied"] == 1
    assert prediction["operations_skipped"] == []


@pytest.mark.slow
@pytest.mark.skipif(not CHGNET_AVAILABLE, reason="CHGNet not available")
def test_chgnet_predict_reports_skipped_operations(client):
    """An out-of-range operation must be reported in operations_skipped
    rather than silently dropped (the field the frontend now reads)."""
    response = client.post("/api/chgnet-predict", json={
        "filename": "Metals/Cu.cif",
        "operations": [{"action": "substitute", "index": 9999, "to": "Ni"}],
        "supercell_size": [1, 1, 1],
    })
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert len(prediction["operations_skipped"]) == 1
