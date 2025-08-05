#!/usr/bin/env python3
"""
Test element label generation functionality
"""

import sys
from pathlib import Path
import asyncio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import analyze_cif_file, assign_unique_labels
from pymatgen.core.structure import Structure

async def test_element_labels():
    """Test element label generation"""
    print("🧪 Element Label Generation Test")
    print("=" * 50)
    
    # Test with BaTiO3
    print("1. Testing with BaTiO3...")
    cif_path = Path("sample_cif/BaTiO3.cif")
    
    if cif_path.exists():
        try:
            # Analyze original structure
            result = await analyze_cif_file(cif_path)
            print(f"   ✓ Original structure analyzed")
            print(f"   → Formula: {result['formula']}")
            print(f"   → Atoms: {result['num_atoms']}")
            
            # Create actual structure for label testing
            structure = Structure.from_file(str(cif_path))
            print(f"   → Unit cell sites: {len(structure.sites)}")
            
            # Test label generation on unit cell
            unit_labels = assign_unique_labels(structure)
            print(f"   → Unit cell labels: {unit_labels}")
            
            # Create supercell (2x2x2)
            supercell = structure.copy()
            supercell.make_supercell([2, 2, 2])
            print(f"   → Supercell sites: {len(supercell.sites)}")
            
            # Test label generation on supercell
            supercell_labels = assign_unique_labels(supercell)
            print(f"   → Supercell labels (first 10): {supercell_labels[:10]}")
            print(f"   → Total supercell labels: {len(supercell_labels)}")
            
            # Verify label counts by element
            from collections import Counter
            label_counts = Counter()
            for label in supercell_labels:
                element = ''.join(filter(str.isalpha, label))
                label_counts[element] += 1
            
            print(f"   → Label counts by element: {dict(label_counts)}")
            
            # Test API simulation
            print(f"\n2. Simulating API response...")
            import re
            formula = result['formula']
            pattern = r'([A-Z][a-z]?)(\d*)'
            matches = re.findall(pattern, formula)
            scaling_factor = 8  # 2x2x2
            
            api_labels = []
            for element, count_str in matches:
                count = int(count_str) if count_str else 1
                supercell_count = count * scaling_factor
                
                for i in range(supercell_count):
                    api_labels.append(f"{element}{i}")
            
            print(f"   → API simulated labels (first 10): {api_labels[:10]}")
            print(f"   → Total API labels: {len(api_labels)}")
            
            # Verify consistency
            actual_total = len(supercell_labels)
            api_total = len(api_labels)
            consistency_check = actual_total == api_total
            
            print(f"\n3. Consistency check: {'✓' if consistency_check else '✗'}")
            print(f"   → Actual structure labels: {actual_total}")
            print(f"   → API simulated labels: {api_total}")
            
            if consistency_check:
                print(f"   ✓ Label generation is consistent!")
            else:
                print(f"   ✗ Mismatch detected!")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ✗ CIF file not found: {cif_path}")

if __name__ == "__main__":
    asyncio.run(test_element_labels())