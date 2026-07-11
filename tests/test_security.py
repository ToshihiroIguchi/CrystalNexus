"""Tests for the security/validation functions defined in main.py.

These tests import the real implementations from main (no local copies),
so any regression in the production validation logic is caught here.
"""
import pytest

from main import (
    ALLOWED_ELEMENTS,
    safe_filename,
    safe_path,
    validate_element,
    validate_supercell_size,
)


# ---------------------------------------------------------------------------
# validate_element
# ---------------------------------------------------------------------------

def test_validate_element_accepts_valid_element():
    assert validate_element("Fe") == "Fe"


def test_validate_element_strips_whitespace():
    assert validate_element(" Cu ") == "Cu"


def test_validate_element_rejects_unknown_element():
    with pytest.raises(ValueError):
        validate_element("InvalidElement")


def test_validate_element_rejects_empty_string():
    with pytest.raises(ValueError):
        validate_element("")


def test_validate_element_rejects_non_string():
    with pytest.raises(ValueError):
        validate_element(42)


def test_allowed_elements_contains_common_elements():
    for element in ["H", "C", "N", "O", "Fe", "Cu", "Zn", "Al", "Si"]:
        assert element in ALLOWED_ELEMENTS, f"Element {element} should be supported"


# ---------------------------------------------------------------------------
# safe_filename
# ---------------------------------------------------------------------------

def test_safe_filename_accepts_valid_cif():
    assert safe_filename("test.cif") == "test.cif"


def test_safe_filename_strips_directory_components():
    # basename is applied, so a path collapses to its final component
    assert safe_filename("some/dir/structure.cif") == "structure.cif"


def test_safe_filename_rejects_empty():
    with pytest.raises(ValueError):
        safe_filename("")


def test_safe_filename_rejects_non_cif_extension():
    with pytest.raises(ValueError):
        safe_filename("malicious.exe")


def test_safe_filename_rejects_traversal_to_non_cif():
    with pytest.raises(ValueError):
        safe_filename("../../../etc/passwd")


# ---------------------------------------------------------------------------
# safe_path (subdirectory-aware path validation)
# ---------------------------------------------------------------------------

def test_safe_path_accepts_valid_subdirectory_path():
    assert safe_path("Metals/Cu.cif") == "Metals/Cu.cif"


def test_safe_path_normalizes_backslashes():
    assert safe_path("Metals\\Cu.cif") == "Metals/Cu.cif"


def test_safe_path_rejects_empty():
    with pytest.raises(ValueError):
        safe_path("")


def test_safe_path_rejects_simple_traversal():
    with pytest.raises(ValueError):
        safe_path("../x.cif")


def test_safe_path_rejects_nested_traversal():
    with pytest.raises(ValueError):
        safe_path("Metals/../../x.cif")


def test_safe_path_rejects_posix_absolute_path():
    with pytest.raises(ValueError):
        safe_path("/etc/x.cif")


def test_safe_path_rejects_backslash_absolute_path():
    with pytest.raises(ValueError):
        safe_path("\\Windows\\x.cif")


def test_safe_path_rejects_windows_drive_absolute_path():
    with pytest.raises(ValueError):
        safe_path("C:/Windows/x.cif")


def test_safe_path_rejects_non_cif_extension():
    with pytest.raises(ValueError):
        safe_path("Metals/notes.txt")


# ---------------------------------------------------------------------------
# validate_supercell_size
# ---------------------------------------------------------------------------

def test_validate_supercell_size_accepts_valid_size():
    assert validate_supercell_size([2, 2, 2]) == [2, 2, 2]


def test_validate_supercell_size_rejects_zero_dimension():
    with pytest.raises(ValueError):
        validate_supercell_size([0, 2, 2])


def test_validate_supercell_size_rejects_oversized_dimension():
    with pytest.raises(ValueError):
        validate_supercell_size([100, 100, 100])


def test_validate_supercell_size_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_supercell_size([1, 2])


def test_validate_supercell_size_rejects_non_integer():
    with pytest.raises(ValueError):
        validate_supercell_size([1.5, 2, 2])
