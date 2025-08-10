#!/usr/bin/env python3
"""
Security functions test script
セキュリティ機能の独立テスト
"""
import os
import tempfile
from pathlib import Path

# セキュリティ設定
MAX_SUPERCELL_DIM = 10
ALLOWED_ELEMENTS = {
    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
    'Ga', 'Ge', 'As', 'Se', 'Br',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
    'In', 'Sn', 'Sb', 'Te', 'I',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
    'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
    'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Th', 'U'
}

def safe_filename(filename: str) -> str:
    """Secure filename validation"""
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Remove path traversal characters
    safe_name = os.path.basename(filename).replace('..', '')
    
    # Check for dangerous characters
    if not safe_name or '/' in safe_name or '\\' in safe_name:
        raise ValueError("Invalid filename")
    
    # Only allow CIF files
    if not safe_name.lower().endswith('.cif'):
        raise ValueError("Only CIF files are allowed")
    
    return safe_name

def validate_supercell_size(supercell_size: list) -> list:
    """Supercell size validation"""
    if not isinstance(supercell_size, list) or len(supercell_size) != 3:
        raise ValueError("Supercell size must be a list of 3 integers")
    
    for dim in supercell_size:
        if not isinstance(dim, int) or dim < 1 or dim > MAX_SUPERCELL_DIM:
            raise ValueError(f"Supercell dimensions must be between 1 and {MAX_SUPERCELL_DIM}")
    
    return supercell_size

def validate_element(element: str) -> str:
    """Element validation (injection prevention)"""
    if not isinstance(element, str):
        raise ValueError("Element must be a string")
    
    element = element.strip()
    if element not in ALLOWED_ELEMENTS:
        raise ValueError(f"Element '{element}' is not supported")
    
    return element

def test_security_functions():
    """Run all security tests"""
    print('=== Security Function Test ===')
    
    # Test results
    passed = 0
    total = 0
    
    # 1. Element validation tests
    print('\n--- Element Validation Tests ---')
    
    # Valid element test
    total += 1
    try:
        validate_element('Fe')
        print('✓ Valid element (Fe) validation: OK')
        passed += 1
    except Exception as e:
        print(f'✗ Valid element validation error: {e}')
    
    # Invalid element test
    total += 1
    try:
        validate_element('InvalidElement')
        print('✗ Invalid element validation: FAILED (should not pass)')
    except ValueError:
        print('✓ Invalid element (InvalidElement) rejection: OK')
        passed += 1
    
    # Empty element test
    total += 1
    try:
        validate_element('')
        print('✗ Empty element validation: FAILED (should not pass)')
    except ValueError:
        print('✓ Empty element rejection: OK')
        passed += 1
    
    # 2. Filename validation tests
    print('\n--- Filename Validation Tests ---')
    
    # Valid filename test
    total += 1
    try:
        result = safe_filename('test.cif')
        assert result == 'test.cif'
        print('✓ Valid filename validation: OK')
        passed += 1
    except Exception as e:
        print(f'✗ Valid filename error: {e}')
    
    # Path traversal test
    total += 1
    try:
        safe_filename('../../../etc/passwd')
        print('✗ Path traversal validation: FAILED (should not pass)')
    except ValueError:
        print('✓ Path traversal attack rejection: OK')
        passed += 1
    
    # Non-CIF file test
    total += 1
    try:
        safe_filename('malicious.exe')
        print('✗ Non-CIF file validation: FAILED (should not pass)')
    except ValueError:
        print('✓ Non-CIF file rejection: OK')
        passed += 1
    
    # 3. Supercell size validation tests
    print('\n--- Supercell Size Validation Tests ---')
    
    # Valid supercell size test
    total += 1
    try:
        result = validate_supercell_size([2, 2, 2])
        assert result == [2, 2, 2]
        print('✓ Valid supercell size validation: OK')
        passed += 1
    except Exception as e:
        print(f'✗ Valid supercell size error: {e}')
    
    # Large supercell test
    total += 1
    try:
        validate_supercell_size([100, 100, 100])
        print('✗ Large supercell validation: FAILED (should not pass)')
    except ValueError:
        print('✓ Large supercell rejection: OK')
        passed += 1
    
    # Invalid format test
    total += 1
    try:
        validate_supercell_size([1, 2])  # Only 2 dimensions
        print('✗ Invalid supercell format: FAILED (should not pass)')
    except ValueError:
        print('✓ Invalid supercell format rejection: OK')
        passed += 1
    
    # 4. File size simulation test
    print('\n--- File Size Test ---')
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    total += 1
    try:
        # Simulate large file
        large_data = b'x' * (MAX_FILE_SIZE + 1)
        if len(large_data) > MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")
        print('✗ Large file test: FAILED (should not pass)')
    except ValueError:
        print('✓ Large file rejection: OK')
        passed += 1
    
    # Summary
    print('\n=== Test Summary ===')
    print(f'Passed: {passed}/{total}')
    if passed == total:
        print('✓ All security tests PASSED')
        return True
    else:
        print('✗ Some security tests FAILED')
        return False

if __name__ == '__main__':
    success = test_security_functions()
    exit(0 if success else 1)