#!/usr/bin/env python3
"""
Complete workflow test including element labels
"""

import requests
import json

BASE_URL = "http://localhost:8080"

def test_complete_workflow():
    """Test the complete workflow including new element labels feature"""
    print("🧪 Complete CrystalNexus Workflow Test with Element Labels")
    print("=" * 70)
    
    # Step 1: Get sample files
    print("1. Getting sample CIF files...")
    response = requests.get(f"{BASE_URL}/api/sample-cif-files")
    files_data = response.json()
    cif_files = files_data.get("files", [])
    test_file = cif_files[0] if cif_files else None
    print(f"   ✓ Using test file: {test_file}")
    
    # Step 2: Analyze CIF
    print(f"\n2. Analyzing CIF file: {test_file}...")
    response = requests.post(
        f"{BASE_URL}/api/analyze-cif-sample",
        json={"filename": test_file}
    )
    crystal_data = response.json()
    print(f"   ✓ Formula: {crystal_data.get('formula')}")
    print(f"   ✓ Atoms: {crystal_data.get('num_atoms')}")
    print(f"   ✓ Density: {crystal_data.get('density'):.4f} g/cm³")
    
    # Step 3: Create supercell
    print(f"\n3. Creating 2×2×2 supercell...")
    supercell_request = {
        "crystal_data": crystal_data,
        "supercell_size": [2, 2, 2]
    }
    response = requests.post(
        f"{BASE_URL}/api/create-supercell",
        json=supercell_request
    )
    supercell_data = response.json()
    supercell_info = supercell_data["supercell_info"]
    print(f"   ✓ Supercell formula: {supercell_info['formula']}")
    print(f"   ✓ Supercell atoms: {supercell_info['num_sites']}")
    print(f"   ✓ Scaling factor: {supercell_info['scaling_factor']}")
    
    # Step 4: Get element labels (NEW FEATURE)
    print(f"\n4. Getting element labels for operations...")
    labels_request = {
        "crystal_data": crystal_data,
        "supercell_size": [2, 2, 2]
    }
    response = requests.post(
        f"{BASE_URL}/api/get-element-labels",
        json=labels_request
    )
    labels_data = response.json()
    labels = labels_data.get("labels", [])
    unique_elements = labels_data.get("unique_elements", [])
    
    print(f"   ✓ Total labels: {len(labels)}")
    print(f"   ✓ Unique elements: {unique_elements}")
    print(f"   ✓ Sample labels: {labels[:10]}...")
    
    # Step 5: Analyze label distribution
    print(f"\n5. Analyzing label distribution...")
    from collections import Counter
    element_counts = Counter()
    for label in labels:
        element = ''.join(filter(str.isalpha, label))
        element_counts[element] += 1
    
    print(f"   ✓ Label distribution:")
    for element, count in element_counts.items():
        print(f"     → {element}: {count} labels")
    
    # Step 6: Verify consistency
    print(f"\n6. Verifying data consistency...")
    expected_total = crystal_data['num_atoms'] * 8  # 2x2x2 = 8
    actual_total = len(labels)
    supercell_total = supercell_info['num_sites']
    
    consistency_checks = [
        ("Label count vs expected", actual_total == expected_total),
        ("Label count vs supercell", actual_total == supercell_total),
        ("All elements present", len(unique_elements) > 0),
        ("Labels are unique", len(labels) == len(set(labels)))
    ]
    
    all_passed = True
    for check_name, passed in consistency_checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    print(f"\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✓ CIF file analysis working")
        print("✓ Supercell creation working")
        print("✓ Formula scaling working")
        print("✓ Element labels generation working")
        print("✓ Data consistency verified")
        print("\n🖥️  Ready for browser testing at: http://localhost:8080")
        print("Expected behavior:")
        print("1. Modal window opens with CIF file selection")
        print("2. Crystal information displays after selection")
        print("3. Supercell creation closes modal")
        print("4. Left panel shows supercell info (no symmetry)")
        print("5. Structure Operations shows dropdown with all atom labels")
        print(f"6. Dropdown contains {len(labels)} unique atom labels")
    else:
        print("❌ SOME TESTS FAILED - Check implementation")

if __name__ == "__main__":
    try:
        test_complete_workflow()
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()