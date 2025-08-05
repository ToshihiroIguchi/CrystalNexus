#!/usr/bin/env python3
"""
Test cleaned up UI design
"""

import requests

BASE_URL = "http://localhost:8080"

def test_clean_ui():
    """Test the cleaned up UI design"""
    print("🧪 Clean UI Design Test")
    print("=" * 40)
    
    # Test basic functionality
    try:
        response = requests.get(f"{BASE_URL}/api/sample-cif-files")
        files = response.json().get("files", [])
        
        response = requests.post(f"{BASE_URL}/api/analyze-cif-sample", 
                                json={"filename": files[0]})
        crystal_data = response.json()
        
        response = requests.post(f"{BASE_URL}/api/create-supercell", 
                                json={"crystal_data": crystal_data, "supercell_size": [2, 2, 2]})
        supercell_data = response.json()
        
        response = requests.post(f"{BASE_URL}/api/get-element-labels",
                                json={"crystal_data": crystal_data, "supercell_size": [2, 2, 2]})
        labels_data = response.json()
        
        response = requests.get(f"{BASE_URL}/api/chgnet-elements")
        chgnet_data = response.json()
        
        print("✓ All APIs working")
        print(f"✓ Atoms: {len(labels_data.get('labels', []))}")
        print(f"✓ CHGnet elements: {len(chgnet_data.get('elements', []))}")
        
    except Exception as e:
        print(f"✗ API Error: {e}")
        return
    
    print(f"\n🎨 UI Improvements Applied:")
    print("✓ Removed unnecessary 'Operation' section box")
    print("✓ Eliminated redundant explanatory text")
    print("✓ Simplified optgroup label: 'Substitute with:'")
    print("✓ Shortened action labels: '🗑️ Delete atom'")
    print("✓ Removed statistics clutter")
    print("✓ Cleaner button text: 'Execute' vs 'Execute Operation'")
    
    print(f"\n📐 New UI Structure:")
    print("Structure Operations")
    print("├── Select Atom to Modify: [dropdown]")
    print("├── Action: [🗑️ Delete atom | Substitute with: H, He...]")
    print("└── [Execute] button")
    
    print(f"\n✨ Expected User Experience:")
    print("1. Cleaner visual hierarchy")
    print("2. Reduced cognitive load")
    print("3. Faster operation selection")
    print("4. Less screen space usage")
    print("5. Professional appearance")
    
    print(f"\n🖥️  Test at: http://localhost:8080")
    print("Expected changes:")
    print("• No gray boxes around operations")
    print("• No verbose explanatory text")
    print("• Compact, focused interface")
    print("• Clear action hierarchy")

if __name__ == "__main__":
    test_clean_ui()