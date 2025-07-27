#!/usr/bin/env python3
"""
Test pymatgen CifParser API to identify the correct usage
"""

from io import StringIO
import tempfile
import os

try:
    from pymatgen.io.cif import CifParser
    print("✅ Successfully imported CifParser")
    
    # Read BaTiO3.cif content
    with open('BaTiO3.cif', 'r') as f:
        cif_content = f.read()
    
    print(f"📄 CIF content length: {len(cif_content)} characters")
    print(f"📄 First 200 chars: {cif_content[:200]}...")
    
    # Test Method 1: StringIO
    print("\n🔬 Testing Method 1: StringIO")
    try:
        parser1 = CifParser(StringIO(cif_content))
        structures1 = parser1.get_structures()
        print(f"✅ StringIO method: Found {len(structures1)} structures")
        if structures1:
            structure = structures1[0]
            print(f"✅ Structure sites: {len(structure.sites)}")
            print(f"✅ Structure lattice: {structure.lattice}")
        else:
            print("❌ No structures found with StringIO method")
    except Exception as e:
        print(f"❌ StringIO method failed: {e}")
    
    # Test Method 2: Temporary file
    print("\n🔬 Testing Method 2: Temporary file")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp_file:
            tmp_file.write(cif_content)
            tmp_file.flush()
            
            parser2 = CifParser(tmp_file.name)
            structures2 = parser2.get_structures()
            print(f"✅ Temp file method: Found {len(structures2)} structures")
            if structures2:
                structure = structures2[0]
                print(f"✅ Structure sites: {len(structure.sites)}")
                print(f"✅ Structure lattice: {structure.lattice}")
            else:
                print("❌ No structures found with temp file method")
        
        os.unlink(tmp_file.name)
    except Exception as e:
        print(f"❌ Temp file method failed: {e}")
    
    # Test Method 3: Direct file path
    print("\n🔬 Testing Method 3: Direct file path")
    try:
        parser3 = CifParser('BaTiO3.cif')
        structures3 = parser3.get_structures()
        print(f"✅ Direct file method: Found {len(structures3)} structures")
        if structures3:
            structure = structures3[0]
            print(f"✅ Structure sites: {len(structure.sites)}")
            print(f"✅ Structure lattice: {structure.lattice}")
            
            # Test supercell creation
            print("\n🔬 Testing supercell creation...")
            supercell = structure.make_supercell([2, 2, 2])
            print(f"✅ Supercell created with {len(supercell.sites)} atoms")
            
            # Test CIF writing
            from pymatgen.io.cif import CifWriter
            writer = CifWriter(supercell)
            supercell_cif = str(writer)
            print(f"✅ Supercell CIF generated ({len(supercell_cif)} chars)")
            print("✅ All tests passed!")
            
        else:
            print("❌ No structures found with direct file method")
    except Exception as e:
        print(f"❌ Direct file method failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Failed to import pymatgen: {e}")