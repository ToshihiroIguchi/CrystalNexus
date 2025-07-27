#!/usr/bin/env python3
"""
Atom Selection Debug Test
Tests the atom selection dropdown functionality by simulating the frontend workflow
"""

import requests
import json
import time
import os

def test_supercell_atom_info_flow():
    """Test the complete flow of supercell creation and atom info extraction"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    
    print("=== Supercell Atom Info Flow Test ===")
    print(f"Testing server: {base_url}")
    
    # Step 1: Get BaTiO3 sample file (simulating frontend)
    try:
        print("\n1. Getting BaTiO3 sample file...")
        response = requests.get(f"{base_url}/sample_cif/BaTiO3.cif", timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to get sample file: {response.status_code}")
            return False
        
        cif_content = response.text
        print(f"✅ Sample file loaded: {len(cif_content)} characters")
        
    except Exception as e:
        print(f"❌ Failed to load sample file: {e}")
        return False
    
    # Step 2: Parse original CIF to get baseline atom info
    try:
        print("\n2. Parsing original CIF...")
        response = requests.post(
            f"{base_url}/parse_cif",
            headers={'Content-Type': 'application/json'},
            json={'cif_content': cif_content},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to parse CIF: {response.status_code}")
            return False
        
        original_data = response.json()
        print(f"✅ Original CIF parsed successfully")
        print(f"   Formula: {original_data.get('formula', 'Unknown')}")
        print(f"   Atom count: {original_data.get('atom_count', 'Unknown')}")
        print(f"   Has atom_info: {'atom_info' in original_data}")
        
        if 'atom_info' in original_data:
            print(f"   Original atom_info entries: {len(original_data['atom_info'])}")
            if original_data['atom_info']:
                sample_atom = original_data['atom_info'][0]
                print(f"   Sample atom: {sample_atom}")
        
    except Exception as e:
        print(f"❌ Failed to parse original CIF: {e}")
        return False
    
    # Step 3: Create supercell and get atom info
    try:
        print("\n3. Creating 2x2x2 supercell...")
        response = requests.post(
            f"{base_url}/create_supercell",
            headers={'Content-Type': 'application/json'},
            json={
                'cif_content': cif_content,
                'a_multiplier': 2,
                'b_multiplier': 2,
                'c_multiplier': 2
            },
            timeout=15
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to create supercell: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
        
        supercell_data = response.json()
        print(f"✅ Supercell created successfully")
        
        # Analyze supercell response structure
        print(f"\n--- Supercell Response Analysis ---")
        print(f"Success: {supercell_data.get('success', False)}")
        print(f"Has supercell_info: {'supercell_info' in supercell_data}")
        
        if 'supercell_info' in supercell_data:
            sc_info = supercell_data['supercell_info']
            print(f"Supercell formula: {sc_info.get('formula', 'Unknown')}")
            print(f"Supercell atom count: {sc_info.get('atom_count', 'Unknown')}")
            print(f"Has atom_info in supercell_info: {'atom_info' in sc_info}")
            
            if 'atom_info' in sc_info:
                atom_info = sc_info['atom_info']
                print(f"✅ Supercell atom_info found: {len(atom_info)} entries")
                
                # Analyze atom info structure
                if atom_info:
                    sample_atom = atom_info[0]
                    print(f"Sample supercell atom: {sample_atom}")
                    
                    # Check for expected labels
                    labels = [atom.get('label', 'No_Label') for atom in atom_info[:10]]
                    print(f"First 10 atom labels: {labels}")
                    
                    # Count unique elements
                    elements = {}
                    for atom in atom_info:
                        elem = atom.get('type_symbol', 'Unknown')
                        elements[elem] = elements.get(elem, 0) + 1
                    print(f"Element distribution: {elements}")
                    
                    return True
                else:
                    print("❌ Supercell atom_info is empty")
                    return False
            else:
                print("❌ No atom_info in supercell_info")
                return False
        else:
            print("❌ No supercell_info in response")
            return False
        
    except Exception as e:
        print(f"❌ Failed to create supercell: {e}")
        return False

def test_manual_cif_parsing():
    """Test manual CIF parsing with BaTiO3 to ensure atom_info extraction works"""
    
    server_host = os.getenv('SERVER_HOST', '127.0.0.1')
    server_port = int(os.getenv('SERVER_PORT', 8080))
    base_url = f"http://{server_host}:{server_port}"
    
    print("\n=== Manual CIF Parsing Test ===")
    
    try:
        # Get BaTiO3 sample
        response = requests.get(f"{base_url}/sample_cif/BaTiO3.cif", timeout=10)
        cif_content = response.text
        
        # Parse with server
        response = requests.post(
            f"{base_url}/parse_cif",
            headers={'Content-Type': 'application/json'},
            json={'cif_content': cif_content},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Parse result success: {data.get('success', False)}")
            print(f"Has atom_info: {'atom_info' in data}")
            
            if 'atom_info' in data and data['atom_info']:
                atom_info = data['atom_info']
                print(f"✅ Atom info extracted: {len(atom_info)} atoms")
                
                for i, atom in enumerate(atom_info):
                    print(f"  Atom {i}: {atom}")
                
                return True
            else:
                print("❌ No atom_info in parse result")
                return False
        else:
            print(f"❌ Parse failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Manual CIF parsing test failed: {e}")
        return False

def test_atom_selection_workflow():
    """Test the complete atom selection workflow that frontend follows"""
    
    print("\n=== Complete Atom Selection Workflow Test ===")
    
    # Test Step 1: Manual CIF parsing
    manual_success = test_manual_cif_parsing()
    
    # Test Step 2: Supercell creation and atom info
    supercell_success = test_supercell_atom_info_flow()
    
    # Summary
    print(f"\n=== Workflow Test Summary ===")
    print(f"Manual CIF parsing: {'✅ PASS' if manual_success else '❌ FAIL'}")
    print(f"Supercell atom info: {'✅ PASS' if supercell_success else '❌ FAIL'}")
    
    overall_success = manual_success and supercell_success
    print(f"Overall workflow: {'✅ WORKING' if overall_success else '❌ BROKEN'}")
    
    if not overall_success:
        print("\n🔧 Debugging Recommendations:")
        if not manual_success:
            print("  1. Check server /parse_cif endpoint atom_info extraction")
        if not supercell_success:
            print("  2. Check supercell creation atom_info generation")
            print("  3. Verify supercell_info.atom_info is properly returned")
        print("  4. Check frontend currentAtomInfo assignment logic")
        print("  5. Verify initializeAtomLabelSelector() is called after CIF loading")
    
    return overall_success

def main():
    """Run atom selection debug tests"""
    print("Atom Selection Debug Test Suite")
    print("=" * 60)
    
    try:
        success = test_atom_selection_workflow()
        return success
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)