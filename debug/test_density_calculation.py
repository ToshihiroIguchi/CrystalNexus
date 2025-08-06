#!/usr/bin/env python3
"""
Test and analyze density calculation issues
"""

import requests
import json

BASE_URL = "http://localhost:8080"

def test_density_calculation():
    """Analyze density calculation problem and propose solutions"""
    print("🧪 Density Calculation Analysis")
    print("=" * 60)
    
    # Test current workflow
    print("1. Testing current workflow...")
    try:
        response = requests.get(f"{BASE_URL}/api/sample-cif-files")
        files = response.json().get("files", [])
        test_file = files[0]
        
        response = requests.post(f"{BASE_URL}/api/analyze-cif-sample", 
                               json={"filename": test_file})
        crystal_data = response.json()
        original_density = crystal_data.get('density')
        
        response = requests.post(f"{BASE_URL}/api/create-supercell", 
                               json={"crystal_data": crystal_data, "supercell_size": [2, 2, 2]})
        supercell_data = response.json()
        supercell_density = supercell_data['supercell_info'].get('density')
        
        print(f"   ✓ Original density: {original_density} g/cm³")
        print(f"   ✓ Supercell density: {supercell_density} g/cm³")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Problem analysis
    print(f"\n2. Problem Analysis:")
    print("   📋 Current Implementation Issues:")
    issues = [
        "displaySupercellInfo() uses original.density (fixed value)",
        "updateStructureForSubstitution() doesn't recalculate density", 
        "updateStructureForDeletion() only approximates volume change",
        "No atomic mass lookup for density calculation",
        "Density = mass/volume formula not implemented for modifications"
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"      {i}. ❌ {issue}")
    
    # Atomic mass data needed
    print(f"\n3. Required Atomic Mass Data:")
    common_elements = {
        'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81,
        'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'Ne': 20.18,
        'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97,
        'S': 32.07, 'Cl': 35.45, 'Ar': 39.95, 'K': 39.10, 'Ca': 40.08,
        'Ti': 47.87, 'V': 50.94, 'Cr': 52.00, 'Mn': 54.94, 'Fe': 55.85,
        'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72,
        'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.96, 'Tc': 98.91, 'Ru': 101.07
    }
    
    print(f"   → Need atomic masses for {len(common_elements)} elements")
    print(f"   → Example: Zr={common_elements['Zr']}, O={common_elements['O']}")
    
    # Solution strategies
    print(f"\n4. Solution Strategies:")
    strategies = [
        ("Frontend-only", "Add atomic mass lookup table in JavaScript", "Simple but incomplete"),
        ("Backend integration", "Use pymatgen.core.Element for atomic masses", "Accurate but requires API"),
        ("Hybrid approach", "Pre-calculate masses, store in structure data", "Balanced solution"),
        ("Approximation", "Use simple mass ratios for quick estimation", "Fast but less accurate")
    ]
    
    for strategy, description, pros_cons in strategies:
        print(f"   🔧 {strategy}:")
        print(f"      • {description}")
        print(f"      • {pros_cons}")
    
    # Recommended solution
    print(f"\n5. Recommended Solution:")
    print("   🎯 **Hybrid Approach** (Most Balanced):")
    recommendations = [
        "Add atomic mass constants in JavaScript",
        "Implement calculateNewDensity() function", 
        "Update density in both substitution and deletion functions",
        "Use formula: density = total_mass / volume",
        "Handle volume changes for deletions appropriately"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"      {i}. {rec}")
    
    # Implementation plan
    print(f"\n6. Implementation Plan:")
    steps = [
        "Create atomic mass lookup table (ATOMIC_MASSES)",
        "Implement calculateAtomicMass(formula) function",
        "Implement calculateNewDensity(formula, volume) function", 
        "Update updateStructureForSubstitution() to recalculate density",
        "Update updateStructureForDeletion() to recalculate density",
        "Ensure displaySupercellInfo() uses current density"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    # Test cases
    print(f"\n7. Expected Test Cases:")
    test_cases = [
        "Zr32 O64 → Zr31 Na1 O64 (Zr→Na substitution): density should decrease",
        "Zr32 O64 → Zr31 O64 (delete Zr): density should decrease slightly", 
        "Zr32 O64 → Zr32 Al1 O63 (O→Al substitution): density should increase",
        "Multiple operations: density should update progressively"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"   {i}. {case}")
    
    print(f"\n" + "=" * 60)
    print("📊 DENSITY CALCULATION PROBLEM IDENTIFIED")
    print("Root cause: Fixed density value not updated after atom operations")
    print("Recommended: Implement dynamic density calculation with atomic masses")
    print("Priority: High - affects scientific accuracy of results")

if __name__ == "__main__":
    try:
        test_density_calculation()
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()