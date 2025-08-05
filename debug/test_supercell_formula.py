"""
Test script for supercell formula calculation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import calculate_supercell_formula

def test_supercell_formula():
    """Test supercell formula calculation with various inputs"""
    
    test_cases = [
        # (original_formula, scaling_factor, expected_result)
        ("Zr4 O8", 8, "Zr32 O64"),  # 2x2x2 supercell
        ("Ba Ti O3", 1, "Ba1 Ti1 O3"),  # 1x1x1 (no change)
        ("Ba Ti O3", 8, "Ba8 Ti8 O24"),  # 2x2x2 supercell
        ("Fe2 O3", 4, "Fe8 O12"),  # 2x2x1 supercell
        ("Ca", 27, "Ca27"),  # Simple element, 3x3x3
        ("Mg2 Si O4", 2, "Mg4 Si2 O8"),  # 2x1x1 supercell
    ]
    
    print("Testing supercell formula calculation:")
    print("=" * 60)
    
    for original, factor, expected in test_cases:
        result = calculate_supercell_formula(original, factor)
        status = "✓" if result == expected else "✗"
        
        print(f"{status} {original} × {factor} = {result}")
        if result != expected:
            print(f"   Expected: {expected}")
        print()
    
    # Test with ZrO2 specifically
    print("ZrO2 test case:")
    print("-" * 30)
    zro2_formula = "Zr4 O8"  # Example from actual data
    supercell_2x2x2 = calculate_supercell_formula(zro2_formula, 8)
    print(f"Original: {zro2_formula}")
    print(f"2×2×2 Supercell: {supercell_2x2x2}")
    print(f"Expected atoms: Zr=32, O=64, Total=96")
    
    # Calculate total atoms
    import re
    pattern = r'([A-Z][a-z]?)(\d+)'
    matches = re.findall(pattern, supercell_2x2x2)
    total_atoms = sum(int(count) for _, count in matches)
    print(f"Calculated total atoms: {total_atoms}")

if __name__ == "__main__":
    test_supercell_formula()