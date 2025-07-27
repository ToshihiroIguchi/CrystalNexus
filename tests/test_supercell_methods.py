#!/usr/bin/env python3
"""
Test different supercell creation methods in pymatgen
"""

from pymatgen.io.cif import CifParser
from pymatgen.core import Structure

try:
    # Parse CIF
    parser = CifParser('BaTiO3.cif')
    structures = parser.get_structures()
    structure = structures[0]
    
    print(f"Original structure: {len(structure.sites)} atoms")
    print(f"Structure type: {type(structure)}")
    print(f"Structure lattice: {structure.lattice}")
    
    # Method 1: make_supercell (in-place modification)
    print("\n🔬 Testing Method 1: make_supercell")
    try:
        result = structure.make_supercell([2, 2, 2])
        print(f"make_supercell result: {result}")
        print(f"Structure after make_supercell: {len(structure.sites)} atoms")
    except Exception as e:
        print(f"❌ make_supercell failed: {e}")
    
    # Method 2: Create new structure with repeated lattice
    print("\n🔬 Testing Method 2: Manual supercell creation")
    try:
        # Re-parse to get fresh structure
        parser2 = CifParser('BaTiO3.cif')
        original_structure = parser2.get_structures()[0]
        
        # Get supercell lattice
        supercell_lattice = original_structure.lattice.matrix * [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        
        # Create supercell manually
        from pymatgen.core.lattice import Lattice
        supercell_lattice_obj = Lattice(supercell_lattice)
        
        # Generate supercell sites
        supercell_sites = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for site in original_structure.sites:
                        new_coords = site.frac_coords + [i/2, j/2, k/2]
                        supercell_sites.append((site.specie, new_coords))
        
        # Create supercell structure
        supercell = Structure(supercell_lattice_obj, 
                            [site[0] for site in supercell_sites],
                            [site[1] for site in supercell_sites])
        
        print(f"✅ Manual supercell: {len(supercell.sites)} atoms")
        
        # Test CIF writing
        from pymatgen.io.cif import CifWriter
        writer = CifWriter(supercell)
        supercell_cif = str(writer)
        print(f"✅ CIF written: {len(supercell_cif)} characters")
        
    except Exception as e:
        print(f"❌ Manual method failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: Using transformation
    print("\n🔬 Testing Method 3: SupercellTransformation")
    try:
        from pymatgen.transformations.standard_transformations import SupercellTransformation
        
        # Re-parse to get fresh structure
        parser3 = CifParser('BaTiO3.cif')
        original_structure = parser3.get_structures()[0]
        
        # Create transformation
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        transformation = SupercellTransformation(supercell_matrix)
        
        # Apply transformation
        supercell = transformation.apply_transformation(original_structure)
        print(f"✅ Transformation supercell: {len(supercell.sites)} atoms")
        
        # Test CIF writing
        writer = CifWriter(supercell)
        supercell_cif = str(writer)
        print(f"✅ CIF written: {len(supercell_cif)} characters")
        print("✅ SupercellTransformation method works!")
        
    except Exception as e:
        print(f"❌ SupercellTransformation failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ General error: {e}")
    import traceback
    traceback.print_exc()