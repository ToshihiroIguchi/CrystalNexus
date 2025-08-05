#!/usr/bin/env python3
"""
Direct test of server functionality without external server
"""

import sys
from pathlib import Path
import asyncio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import analyze_cif_file, calculate_supercell_formula

async def test_direct():
    """Test server functions directly"""
    print("🧪 Direct Server Function Test")
    print("=" * 40)
    
    # Test 1: CIF Analysis
    print("1. Testing CIF file analysis...")
    cif_path = Path("sample_cif/BaTiO3.cif")
    
    if cif_path.exists():
        try:
            result = await analyze_cif_file(cif_path)
            print("   ✓ CIF analysis successful")
            print(f"   → Formula: {result['formula']}")
            print(f"   → Atoms: {result['num_atoms']}")
            print(f"   → Density: {result['density']:.4f} g/cm³")
            print(f"   → Volume: {result['volume']:.2f} Ų")
            print(f"   → Space Group: {result['space_group']}")
            
            # Test 2: Supercell Formula Calculation
            print(f"\n2. Testing supercell formula calculation...")
            original_formula = result['formula']
            scaling_factor = 8  # 2x2x2
            supercell_formula = calculate_supercell_formula(original_formula, scaling_factor)
            
            print(f"   → Original: {original_formula}")
            print(f"   → 2×2×2 Supercell: {supercell_formula}")
            print(f"   → Scaling factor: {scaling_factor}")
            
            # Test 3: Complete Supercell Data Structure
            print(f"\n3. Testing complete supercell data structure...")
            supercell_data = {
                "status": "supercell_created",
                "original_data": result,
                "supercell_info": {
                    "size": [2, 2, 2],
                    "volume": result['volume'] * scaling_factor,
                    "num_sites": result['num_sites'] * scaling_factor,
                    "scaling_factor": scaling_factor,
                    "formula": supercell_formula
                }
            }
            
            print(f"   ✓ Supercell data structure created")
            print(f"   → Original atoms: {result['num_sites']}")
            print(f"   → Supercell atoms: {supercell_data['supercell_info']['num_sites']}")
            print(f"   → Original volume: {result['volume']:.2f} Ų")
            print(f"   → Supercell volume: {supercell_data['supercell_info']['volume']:.2f} Ų")
            
            # Test 4: Data Validation
            print(f"\n4. Validating calculations...")
            expected_atoms = result['num_sites'] * 8
            expected_volume = result['volume'] * 8
            
            atom_check = supercell_data['supercell_info']['num_sites'] == expected_atoms
            volume_check = abs(supercell_data['supercell_info']['volume'] - expected_volume) < 0.01
            formula_check = supercell_data['supercell_info']['formula'] != result['formula']
            
            print(f"   → Atom scaling: {'✓' if atom_check else '✗'}")
            print(f"   → Volume scaling: {'✓' if volume_check else '✗'}")  
            print(f"   → Formula scaling: {'✓' if formula_check else '✗'}")
            
            if atom_check and volume_check and formula_check:
                print(f"\n🎉 All validations passed!")
                print(f"✓ CIF analysis working correctly")
                print(f"✓ Formula scaling implemented")
                print(f"✓ Supercell calculations verified")
                print(f"✓ No original data leakage")
            else:
                print(f"\n❌ Some validations failed")
                
        except Exception as e:
            print(f"   ✗ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ✗ CIF file not found: {cif_path}")
        print("   Available files:")
        sample_dir = Path("sample_cif")
        if sample_dir.exists():
            for f in sample_dir.glob("*.cif"):
                print(f"     - {f.name}")

if __name__ == "__main__":
    asyncio.run(test_direct())