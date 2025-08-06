#!/usr/bin/env python3
"""
Test actual density behavior in the complete workflow
"""

import requests
import json
import time

BASE_URL = "http://localhost:8080"

def test_real_density_workflow():
    """Test the complete workflow and check if density actually changes"""
    print("🔍 Real Density Behavior Test")
    print("=" * 60)
    
    # Step 1: Complete initial workflow
    print("1. Setting up initial workflow...")
    try:
        # Get sample files
        response = requests.get(f"{BASE_URL}/api/sample-cif-files")
        files = response.json().get("files", [])
        test_file = files[0]
        print(f"   → Using: {test_file}")
        
        # Analyze CIF
        response = requests.post(f"{BASE_URL}/api/analyze-cif-sample", 
                               json={"filename": test_file})
        crystal_data = response.json()
        original_density = crystal_data.get('density')
        original_formula = crystal_data.get('formula')
        print(f"   → Original: {original_formula} - {original_density:.6f} g/cm³")
        
        # Create supercell
        response = requests.post(f"{BASE_URL}/api/create-supercell", 
                               json={"crystal_data": crystal_data, "supercell_size": [2, 2, 2]})
        supercell_data = response.json()
        supercell_formula = supercell_data['supercell_info']['formula']
        supercell_volume = supercell_data['supercell_info']['volume']
        print(f"   → Supercell: {supercell_formula} - Volume: {supercell_volume:.6f} Ų")
        
        # Get element labels
        response = requests.post(f"{BASE_URL}/api/get-element-labels", 
                               json={"crystal_data": crystal_data, "supercell_size": [2, 2, 2]})
        labels_data = response.json()
        labels = labels_data.get("labels", [])[:10]  # Just first 10 for testing
        print(f"   → Available atoms: {labels}")
        
    except Exception as e:
        print(f"   ✗ Setup error: {e}")
        return False
    
    # Step 2: Test density recalculation API directly
    print(f"\n2. Testing density API directly...")
    
    # Test original formula
    density_request = {
        "formula": supercell_formula,
        "volume": supercell_volume,
        "lattice_parameters": crystal_data.get('lattice_parameters', {})
    }
    
    response = requests.post(f"{BASE_URL}/api/recalculate-density", json=density_request)
    if response.status_code == 200:
        result = response.json()
        api_density = result['density']
        print(f"   ✓ API density for {supercell_formula}: {api_density:.6f} g/cm³")
        print(f"   ✓ Expected vs API: {original_density:.6f} vs {api_density:.6f}")
        print(f"   ✓ Ratio: {api_density/original_density:.6f} (should be ~1.0)")
    else:
        print(f"   ✗ API error: {response.status_code} - {response.text}")
        return False
    
    # Step 3: Test modified formulas
    print(f"\n3. Testing formula modifications...")
    
    test_modifications = [
        ("Zr31 Na1 O64", "Zr→Na substitution"),
        ("Zr32 Al1 O63", "O→Al substitution"), 
        ("Zr31 O64", "Zr deletion")
    ]
    
    for modified_formula, description in test_modifications:
        density_request["formula"] = modified_formula
        response = requests.post(f"{BASE_URL}/api/recalculate-density", json=density_request)
        
        if response.status_code == 200:
            result = response.json()
            modified_density = result['density']
            change = modified_density - api_density
            change_percent = (change / api_density) * 100
            
            print(f"   → {description}:")
            print(f"     Formula: {modified_formula}")
            print(f"     Density: {modified_density:.6f} g/cm³")
            print(f"     Change: {change:+.6f} g/cm³ ({change_percent:+.2f}%)")
        else:
            print(f"   ✗ Failed to test {description}: {response.text}")
    
    # Step 4: Check what's actually being called in the frontend
    print(f"\n4. Checking frontend integration...")
    
    # Check if the recalculateDensityWithPymatgen function exists
    response = requests.get(f"{BASE_URL}/")
    html_content = response.text
    
    if "recalculateDensityWithPymatgen" in html_content:
        print("   ✓ Frontend function recalculateDensityWithPymatgen exists")
    else:
        print("   ✗ Frontend function recalculateDensityWithPymatgen missing")
    
    if "await recalculateDensityWithPymatgen" in html_content:
        print("   ✓ Frontend calls API asynchronously")
    else:
        print("   ✗ Frontend may not be calling API properly")
    
    if "/api/recalculate-density" in html_content:
        print("   ✓ Frontend has correct API endpoint")
    else:
        print("   ✗ Frontend missing API endpoint")
    
    # Check for hardcoded values
    if "6.2060" in html_content or "original.density" in html_content:
        print("   ⚠ Potential hardcoded density values found")
    else:
        print("   ✓ No obvious hardcoded density values")
    
    # Step 5: Trace the actual data flow
    print(f"\n5. Tracing data flow...")
    
    # Check displaySupercellInfo function
    if "supercell.density ||" in html_content:
        print("   ✓ displaySupercellInfo uses current density")
    else:
        print("   ✗ displaySupercellInfo may be using fixed density")
    
    # Check update functions
    if "currentStructureData.supercell_info.density = newDensity" in html_content:
        print("   ✓ Update functions set new density")
    else:
        print("   ✗ Update functions may not be setting density")
    
    print(f"\n" + "=" * 60)
    print("🔬 REALITY CHECK COMPLETE")
    
    # Final assessment
    if (api_density > 0 and 
        "recalculateDensityWithPymatgen" in html_content and
        "currentStructureData.supercell_info.density = newDensity" in html_content):
        print("✓ Implementation appears genuine - API works, frontend integrated")
    else:
        print("❌ Implementation may have issues - check hardcoding")
    
    print(f"\n💡 To verify manually:")
    print("1. Open http://localhost:8080")
    print("2. Create ZrO2 supercell")
    print("3. Substitute Zr0 → Na")
    print("4. Check if density changes from ~6.2060 to ~6.0986")
    
    return True

if __name__ == "__main__":
    try:
        test_real_density_workflow()
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()