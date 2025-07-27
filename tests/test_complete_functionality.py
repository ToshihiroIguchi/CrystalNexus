#!/usr/bin/env python3
"""
Complete functionality test - Real workflow
"""

import requests
import json
import time

def test_complete_functionality():
    SERVER_URL = 'http://localhost:8080'
    
    print("=== Complete Functionality Test ===\n")
    
    # 1. Server health status check
    print("🔬 Step 1: Server health status check")
    try:
        health_response = requests.get(f'{SERVER_URL}/health')
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"  ✅ Server healthy: {health_data.get('status')}")
            print(f"  Pymatgen: {health_data.get('pymatgen_version')}")
        else:
            print(f"  ❌ Server abnormal: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
        return False
    
    # 2. Load original CIF
    with open('BaTiO3.cif', 'r') as f:
        original_cif = f.read()
    
    # 3. Standard CIF parsing test
    print("\n🔬 Step 2: Standard CIF parsing")
    parse_response = requests.post(f'{SERVER_URL}/parse_cif', json={
        'cif_content': original_cif
    })
    
    if parse_response.status_code == 200:
        parse_data = parse_response.json()
        print(f"  ✅ Parsing successful")
        print(f"  Space group: {parse_data.get('space_group')}")
        print(f"  Crystal system: {parse_data.get('crystal_system')}")
        print(f"  Formula: {parse_data.get('formula')}")
        print(f"  Atom count: {parse_data.get('atom_count')}")
        
        original_space_group = parse_data.get('space_group')
    else:
        print(f"  ❌ Parsing failed: {parse_response.status_code}")
        return False
    
    # 4. Supercell creation test (main functionality)
    print("\n🔬 Step 3: Supercell creation (P1 forcing verification)")
    supercell_response = requests.post(f'{SERVER_URL}/create_supercell', json={
        'cif_content': original_cif,
        'a_multiplier': 2,
        'b_multiplier': 2,
        'c_multiplier': 2
    })
    
    if supercell_response.status_code == 200:
        supercell_data = supercell_response.json()
        supercell_info = supercell_data.get('supercell_info', {})
        
        print(f"  ✅ Supercell creation successful")
        print(f"  Space group: {supercell_info.get('space_group')}")
        print(f"  Crystal system: {supercell_info.get('crystal_system')}")
        print(f"  Atom count: {supercell_info.get('atom_count')}")
        print(f"  Lattice parameter a: {supercell_info.get('lattice_parameters', {}).get('a')}Å")
        
        supercell_cif = supercell_data['supercell_cif']
        atom_info = supercell_info.get('atom_info', [])
        
        # P1 forcing verification
        p1_forced = "P1" in str(supercell_info.get('space_group'))
        triclinic_forced = "triclinic" in str(supercell_info.get('crystal_system'))
        
        print(f"  P1 forcing: {'✅' if p1_forced else '❌'}")
        print(f"  Triclinic forcing: {'✅' if triclinic_forced else '❌'}")
        
    else:
        print(f"  ❌ Supercell creation failed: {supercell_response.status_code}")
        return False
    
    # 5. Atom information verification (unique labels)
    print("\n🔬 Step 4: Atom information verification (unique labels)")
    
    print(f"  Total atom count: {len(atom_info)}")
    
    if len(atom_info) > 0:
        labels = [atom.get('label') for atom in atom_info]
        unique_labels = set(labels)
        
        print(f"  Unique label count: {len(unique_labels)}")
        print(f"  Label uniqueness: {'✅' if len(labels) == len(unique_labels) else '❌'}")
        print(f"  First 5 labels: {labels[:5]}")
        print(f"  Last 5 labels: {labels[-5:]}")
    
    # 6. Atom replacement test (supercell preservation)
    print("\n🔬 Step 5: Atom replacement test (supercell preservation)")
    
    supercell_metadata = {
        'multipliers': {'a': 2, 'b': 2, 'c': 2},
        'original_atoms': 5,
        'is_supercell': True
    }
    
    replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [0],
        'new_element': 'Sr',
        'supercell_metadata': supercell_metadata
    })
    
    if replace_response.status_code == 200:
        replace_data = replace_response.json()
        replace_info = replace_data.get('modified_structure_info', {})
        
        print(f"  ✅ Atom replacement successful")
        print(f"  Atom count preserved: {replace_data.get('modified_atom_count')}/40")
        print(f"  Lattice constant preserved: {replace_info.get('lattice_parameters', {}).get('a')}Å")
        print(f"  Space group: {replace_info.get('space_group')}")
        print(f"  Supercell maintained: {replace_info.get('is_supercell')}")
        
        # Supercell preservation verification
        atoms_preserved = replace_data.get('modified_atom_count') == 40
        lattice_preserved = abs(replace_info.get('lattice_parameters', {}).get('a', 0) - 7.98) < 0.1
        
        print(f"  Supercell preservation: {'✅' if atoms_preserved and lattice_preserved else '❌'}")
        
    else:
        print(f"  ❌ Atom replacement failed: {replace_response.status_code}")
        return False
    
    # 7. Atom deletion test
    print("\n🔬 Step 6: Atom deletion test")
    
    delete_response = requests.post(f'{SERVER_URL}/delete_atoms', json={
        'cif_content': supercell_cif,
        'atom_indices': [39],  # Delete last atom
        'supercell_metadata': supercell_metadata
    })
    
    if delete_response.status_code == 200:
        delete_data = delete_response.json()
        delete_info = delete_data.get('modified_structure_info', {})
        
        print(f"  ✅ Atom deletion successful")
        print(f"  Atom count after deletion: {delete_data.get('modified_atom_count')}")
        print(f"  Space group: {delete_info.get('space_group')}")
        
        # Deletion operation verification
        atoms_reduced = delete_data.get('modified_atom_count') == 39
        print(f"  Deletion operation: {'✅' if atoms_reduced else '❌'}")
        
    else:
        print(f"  ❌ Atom deletion failed: {delete_response.status_code}")
        return False
    
    # 8. Atom editing in normal structure (symmetry preservation verification)
    print("\n🔬 Step 7: Atom editing in normal structure (symmetry preservation)")
    
    normal_replace_response = requests.post(f'{SERVER_URL}/replace_atoms', json={
        'cif_content': original_cif,
        'atom_indices': [0],
        'new_element': 'Sr'
        # No supercell_metadata
    })
    
    if normal_replace_response.status_code == 200:
        normal_replace_data = normal_replace_response.json()
        normal_replace_info = normal_replace_data.get('modified_structure_info', {})
        
        print(f"  ✅ Normal structure editing successful")
        print(f"  Space group: {normal_replace_info.get('space_group')}")
        print(f"  Crystal system: {normal_replace_info.get('crystal_system')}")
        
        # Symmetry preservation verification
        preserved_tetragonal = "tetragonal" in str(normal_replace_info.get('crystal_system'))
        print(f"  Original symmetry consideration: {'✅' if preserved_tetragonal else '❌'}")
        
    else:
        print(f"  ❌ Normal structure editing failed: {normal_replace_response.status_code}")
        return False
    
    # 9. Overall evaluation
    print("\n🔍 Overall evaluation:")
    
    key_features = {
        "Server healthy operation": health_response.status_code == 200,
        "CIF parsing": parse_response.status_code == 200,
        "Supercell creation": supercell_response.status_code == 200,
        "P1 forcing": p1_forced and triclinic_forced,
        "Unique label generation": len(labels) == len(unique_labels) if atom_info else False,
        "Supercell preservation": atoms_preserved and lattice_preserved,
        "Atom editing functionality": replace_response.status_code == 200,
        "Atom deletion functionality": delete_response.status_code == 200,
        "Symmetry branch processing": normal_replace_response.status_code == 200
    }
    
    print(f"  Feature-by-feature evaluation:")
    for feature, status in key_features.items():
        print(f"    {feature}: {'✅' if status else '❌'}")
    
    all_working = all(key_features.values())
    
    print(f"\n📊 Final evaluation: {'✅ All functions normal' if all_working else '❌ Some functions have issues'}")
    
    if all_working:
        print("🎉 CIF viewer is working perfectly!")
        print("💡 Main features:")
        print("   ✅ BaTiO3 structure display and analysis")
        print("   ✅ Supercell creation (P1 forcing)")
        print("   ✅ Individual atom selection and editing")
        print("   ✅ Atom replacement/deletion (supercell preservation)")
        print("   ✅ Symmetry analysis (branch processing)")
        print("   ✅ Unique atom label generation")
        
        print(f"\n🌐 Access: http://127.0.0.1:5000")
        print("📋 Operation procedure:")
        print("   1. Access with browser")
        print("   2. Load BaTiO3.cif")
        print("   3. Create supercell (2×2×2 recommended)")
        print("   4. Select individual atoms with atom selection")
        print("   5. Execute replacement/deletion operations")
        
    else:
        print("💡 Some functions have issues. Please check the logs.")
    
    return all_working

if __name__ == "__main__":
    success = test_complete_functionality()
    exit(0 if success else 1)