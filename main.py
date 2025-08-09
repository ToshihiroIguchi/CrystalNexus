import os
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# CHGnet import
try:
    from chgnet.model import CHGNet
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False
    print("Warning: CHGnet not available. Install with: pip install chgnet")

app = FastAPI(title="CrystalNexus")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SAMPLE_CIF_DIR = Path("sample_cif")
PORT = 8080

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CrystalNexus"}

@app.get("/api/sample-cif-files")
async def get_sample_cif_files():
    try:
        cif_files = [f.name for f in SAMPLE_CIF_DIR.glob("*.cif")]
        return {"files": cif_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sample CIF files: {str(e)}")

@app.post("/api/analyze-cif-sample")
async def analyze_sample_cif(data: dict):
    try:
        filename = data.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_path = SAMPLE_CIF_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="CIF file not found")
        
        result = await analyze_cif_file(file_path)
        result["filename"] = filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing CIF file: {str(e)}")

@app.post("/api/analyze-cif-upload")
async def analyze_uploaded_cif(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.cif'):
            raise HTTPException(status_code=400, detail="File must be a CIF file")
        
        contents = await file.read()
        temp_path = Path(f"temp_{file.filename}")
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        try:
            result = await analyze_cif_file(temp_path)
            result["filename"] = file.filename
            return result
        finally:
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing uploaded CIF file: {str(e)}")

async def analyze_cif_file(file_path: Path) -> Dict:
    try:
        # CIF file parsing - same as ZrO2test.py
        structure = Structure.from_file(str(file_path))
        
        # Basic structure information
        lattice = structure.lattice
        
        # Symmetry analysis - same as ZrO2test.py
        analyzer = SpacegroupAnalyzer(structure)
        
        return {
            # Basic structure information (same as ZrO2test.py)
            "formula": str(structure.formula),
            "num_atoms": len(structure),
            "density": float(structure.density),
            
            # Lattice parameters (same as ZrO2test.py)
            "lattice_parameters": {
                "a": float(lattice.a),
                "b": float(lattice.b),
                "c": float(lattice.c),
                "alpha": float(lattice.alpha),
                "beta": float(lattice.beta),
                "gamma": float(lattice.gamma)
            },
            
            # Volume (same as ZrO2test.py)
            "volume": float(lattice.volume),
            
            # Symmetry information (same as ZrO2test.py)
            "space_group": analyzer.get_space_group_symbol(),
            "space_group_number": analyzer.get_space_group_number(),
            "point_group": analyzer.get_point_group_symbol(),
            "crystal_system": analyzer.get_crystal_system(),
            
            # Additional information for compatibility
            "num_sites": len(structure.sites)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in pymatgen analysis: {str(e)}")

@app.post("/api/create-supercell")
async def create_supercell(data: dict):
    try:
        crystal_data = data.get("crystal_data")
        supercell_size = data.get("supercell_size", [1, 1, 1])
        filename = data.get("filename")
        
        if not crystal_data:
            raise HTTPException(status_code=400, detail="Crystal data is required")
        
        # Calculate supercell information
        a_mult, b_mult, c_mult = supercell_size
        scaling_factor = a_mult * b_mult * c_mult
        
        original_volume = crystal_data["volume"]
        supercell_volume = original_volume * scaling_factor
        supercell_sites = crystal_data["num_sites"] * scaling_factor
        
        # Calculate supercell formula by scaling the original formula
        original_formula = crystal_data["formula"]
        supercell_formula = calculate_supercell_formula(original_formula, scaling_factor)
        
        # Generate actual pymatgen Structure for 3D visualization
        try:
            from pymatgen.io.cif import CifParser
            from pymatgen.core import Structure
            import os
            
            # Try to load original structure from CIF file if available
            filename = crystal_data.get("filename")
            if filename:
                cif_path = os.path.join("sample_cif", filename)
                if os.path.exists(cif_path):
                    # Parse original CIF file
                    parser = CifParser(cif_path)
                    original_structure = parser.get_structures()[0]
                    
                    # Create supercell
                    supercell_structure = original_structure.copy()
                    supercell_structure.make_supercell([a_mult, b_mult, c_mult])
                    
                    # Get structure dictionary for CIF generation
                    structure_dict = supercell_structure.as_dict()
                    print(f"Successfully created supercell structure from {filename}")
                else:
                    print(f"CIF file not found: {cif_path}")
                    structure_dict = None
            else:
                print("No filename provided for structure generation")
                structure_dict = None
                
        except Exception as e:
            print(f"Warning: Could not create structure object: {e}")
            structure_dict = None
        
        return {
            "status": "supercell_created",
            "original_data": crystal_data,
            "supercell_info": {
                "size": supercell_size,
                "volume": supercell_volume,
                "num_sites": supercell_sites,
                "scaling_factor": scaling_factor,
                "formula": supercell_formula
            },
            "structure_dict": structure_dict,  # Add structure_dict for 3D visualization
            "message": f"Supercell {a_mult}x{b_mult}x{c_mult} created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating supercell: {str(e)}")

def assign_unique_labels(structure):
    """
    Assign unique labels to each site based on element symbol + index
    Similar to elementedit.py assign_unique_labels_bare function
    """
    from collections import Counter
    
    counter = Counter()
    labels = []
    
    for site in structure:
        element = site.specie
        element_symbol = element.symbol if hasattr(element, "symbol") else element.name
        index = counter[element_symbol]
        label = f"{element_symbol}{index}"
        labels.append(label)
        counter[element_symbol] += 1
    
    return labels

@app.post("/api/get-element-labels")
async def get_element_labels(data: dict):
    """
    Get unique element labels for the current supercell structure
    Based on elementedit.py approach
    """
    try:
        crystal_data = data.get("crystal_data")
        supercell_size = data.get("supercell_size", [1, 1, 1])
        
        if not crystal_data:
            raise HTTPException(status_code=400, detail="Crystal data is required")
        
        # Recreate structure from crystal data
        # For now, we'll simulate the structure based on the supercell info
        # In a complete implementation, we would reconstruct the actual Structure object
        
        # Get element information from formula
        import re
        formula = crystal_data.get("formula", "")
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)
        
        # Calculate supercell scaling
        scaling_factor = supercell_size[0] * supercell_size[1] * supercell_size[2]
        
        # Generate labels for supercell
        labels = []
        for element, count_str in matches:
            count = int(count_str) if count_str else 1
            supercell_count = count * scaling_factor
            
            # Generate individual labels for each atom
            for i in range(supercell_count):
                labels.append(f"{element}{i}")
        
        return {
            "status": "success",
            "labels": labels,
            "unique_elements": [match[0] for match in matches],
            "total_labels": len(labels)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting element labels: {str(e)}")

def get_chgnet_supported_elements():
    """
    Get list of elements supported by CHGnet
    """
    if not CHGNET_AVAILABLE:
        return []
    
    try:
        # CHGnet supported elements (based on CHGnet documentation and model)
        # These are the elements that CHGnet was trained on
        chgnet_elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        ]
        return chgnet_elements
    except Exception as e:
        print(f"Error getting CHGnet elements: {e}")
        return []

@app.get("/api/chgnet-elements")
async def get_chgnet_elements():
    """
    Get elements supported by CHGnet for substitution
    """
    try:
        elements = get_chgnet_supported_elements()
        return {
            "status": "success",
            "chgnet_available": CHGNET_AVAILABLE,
            "elements": elements,
            "total_elements": len(elements)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CHGnet elements: {str(e)}")

@app.post("/api/recalculate-density")
async def recalculate_density(request: dict):
    """
    Recalculate density using pymatgen after atom operations
    """
    try:
        formula = request.get("formula")
        volume = request.get("volume")  # in Ų
        lattice_parameters = request.get("lattice_parameters", {})
        
        if not all([formula, volume]):
            raise HTTPException(status_code=400, detail="Missing required parameters: formula, volume")
        
        # Parse the formula to create a dummy structure for density calculation
        import re
        from pymatgen.core import Structure, Lattice, Element
        from pymatgen.core.composition import Composition
        
        # Create composition from formula
        comp = Composition(formula.replace(" ", ""))
        
        # Create dummy lattice with correct volume
        # Use cubic lattice for simplicity (actual lattice shape doesn't affect density)
        a = (volume) ** (1/3)  # Convert volume to lattice parameter
        lattice = Lattice.cubic(a)
        
        # Get atomic masses and calculate density
        total_mass = 0
        for element, amount in comp.items():
            atomic_mass = Element(element).atomic_mass
            total_mass += atomic_mass * amount
        
        # Convert to g/cm³
        # Density = mass (g/mol) / (volume (Ų) * N_A * 1e-24 (cm³/Ų))
        avogadro = 6.02214076e23
        volume_cm3 = volume * 1e-24  # Convert Ų to cm³
        density = total_mass / (avogadro * volume_cm3)
        
        return {
            "status": "success",
            "density": density,
            "formula": formula,
            "volume": volume,
            "total_mass": total_mass,
            "calculation_method": "pymatgen_composition"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recalculating density: {str(e)}")

@app.post("/api/generate-modified-structure-cif")
async def generate_modified_structure_cif(request: dict):
    """
    Generate CIF from structure with applied atomic operations
    Supports complete atomic operation history replay
    """
    try:
        filename = request.get("filename")
        supercell_size = request.get("supercell_size", [1, 1, 1])
        operations = request.get("operations", [])
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        print(f"Generating modified structure CIF for {filename}")
        print(f"Supercell size: {supercell_size}")
        print(f"Operations to apply: {len(operations)}")
        
        # Construct CIF file path
        import os
        cif_path = os.path.join("sample_cif", filename)
        
        if not os.path.exists(cif_path):
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        # Parse original CIF file
        from pymatgen.io.cif import CifParser, CifWriter
        from pymatgen.core import Element
        parser = CifParser(cif_path)
        structure = parser.get_structures(primitive=False)[0]  # Keep original lattice
        
        print(f"Original structure: {structure.formula}")
        
        # Create supercell
        structure.make_supercell(supercell_size)
        print(f"Supercell created: {structure.formula} ({len(structure.sites)} sites)")
        
        # Apply atomic operations in sequence
        operations_applied = 0
        for i, operation in enumerate(operations):
            try:
                if operation["action"] == "substitute":
                    site_index = operation["index"]
                    new_element = operation["to"]
                    
                    if site_index < len(structure.sites):
                        old_coords = structure[site_index].frac_coords
                        old_element = str(structure[site_index].specie)
                        
                        # Replace the site with new element
                        structure[site_index] = Element(new_element), old_coords
                        
                        print(f"  Operation {i+1}: Substituted {old_element} → {new_element} at site {site_index}")
                        operations_applied += 1
                    else:
                        print(f"  Operation {i+1}: SKIPPED - Invalid site index {site_index}")
                        
                elif operation["action"] == "delete":
                    site_index = operation["index"]
                    
                    if site_index < len(structure.sites):
                        deleted_element = str(structure[site_index].specie)
                        structure.remove_sites([site_index])
                        
                        print(f"  Operation {i+1}: Deleted {deleted_element} at site {site_index}")
                        operations_applied += 1
                        
                        # Adjust subsequent indices for deletions
                        for j in range(i + 1, len(operations)):
                            if operations[j].get("index", 0) > site_index:
                                operations[j]["index"] -= 1
                    else:
                        print(f"  Operation {i+1}: SKIPPED - Invalid site index {site_index}")
                
            except Exception as op_error:
                print(f"  Operation {i+1}: ERROR - {str(op_error)}")
                continue
        
        print(f"Applied {operations_applied}/{len(operations)} operations successfully")
        print(f"Final structure: {structure.formula} ({len(structure.sites)} sites)")
        
        # Generate CIF using pymatgen CifWriter
        cif_writer = CifWriter(
            structure,
            write_magmoms=False,
            significant_figures=6
        )
        
        cif_content = str(cif_writer)
        
        # Add metadata header
        operations_summary = f"{len(operations)} operations applied"
        metadata_lines = [
            f"# Modified structure CIF generated by CrystalNexus",
            f"# Original file: {filename}",
            f"# Supercell size: {'x'.join(map(str, supercell_size))}",
            f"# Operations: {operations_summary}",
            f"# Final formula: {structure.formula}",
            f"# Number of atoms: {len(structure.sites)}",
            f"# Volume: {structure.volume:.2f} Ų",
            ""
        ]
        
        final_cif = "\n".join(metadata_lines) + cif_content
        
        print(f"Modified CIF generated successfully, length: {len(final_cif)} characters")
        
        return Response(
            content=final_cif,
            media_type="chemical/x-cif",
            headers={
                "Content-Disposition": f"inline; filename={filename.replace('.cif', '')}_modified.cif",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        error_msg = f"Failed to generate modified structure CIF: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/generate-supercell-cif-direct")
async def generate_supercell_cif_direct(request: dict):
    """
    Generate supercell CIF directly from original CIF file using pymatgen
    This is the most reliable approach - no intermediate data conversion
    """
    try:
        # Extract request parameters
        filename = request.get("filename")
        supercell_size = request.get("supercell_size", [1, 1, 1])
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        print(f"Generating supercell CIF for {filename} with size {supercell_size}")
        
        # Construct CIF file path
        import os
        cif_path = os.path.join("sample_cif", filename)
        
        if not os.path.exists(cif_path):
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        print(f"Reading CIF file: {cif_path}")
        
        # Parse original CIF file
        from pymatgen.io.cif import CifParser, CifWriter
        parser = CifParser(cif_path)
        structures = parser.get_structures(primitive=False)  # Keep original lattice - don't convert to primitive
        
        if not structures:
            raise HTTPException(status_code=400, detail="No structures found in CIF file")
        
        original_structure = structures[0]
        print(f"Original structure loaded: {original_structure.formula}")
        
        # Create supercell
        supercell_structure = original_structure.copy()
        supercell_structure.make_supercell(supercell_size)
        
        print(f"Supercell created: {supercell_structure.formula}")
        print(f"Number of sites: {len(supercell_structure.sites)}")
        
        # Generate CIF using pymatgen CifWriter
        cif_writer = CifWriter(
            supercell_structure,
            write_magmoms=False,
            significant_figures=6
        )
        
        cif_content = str(cif_writer)
        
        # Add metadata header
        size_str = "x".join(map(str, supercell_size))
        metadata_lines = [
            f"# Supercell CIF generated by CrystalNexus",
            f"# Original file: {filename}",
            f"# Supercell size: {size_str}",
            f"# Formula: {supercell_structure.formula}",
            f"# Number of atoms: {len(supercell_structure.sites)}",
            f"# Volume: {supercell_structure.volume:.2f} Ų",
            ""
        ]
        
        final_cif = "\n".join(metadata_lines) + cif_content
        
        print(f"CIF generated successfully, length: {len(final_cif)} characters")
        
        return Response(
            content=final_cif,
            media_type="chemical/x-cif",
            headers={
                "Content-Disposition": f"inline; filename={filename.replace('.cif', '')}_supercell_{size_str}.cif",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        error_msg = f"Failed to generate supercell CIF: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def calculate_supercell_formula(original_formula: str, scaling_factor: int) -> str:
    """
    Calculate supercell formula by scaling the original formula
    Example: "Zr4 O8" with scaling_factor=8 becomes "Zr32 O64"
    """
    import re
    
    # Parse the formula using regex to find elements and their counts
    # Pattern matches: Element (uppercase + optional lowercase) + optional number
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, original_formula)
    
    supercell_parts = []
    for element, count_str in matches:
        # If no number specified, count is 1
        count = int(count_str) if count_str else 1
        # Scale by supercell factor
        supercell_count = count * scaling_factor
        supercell_parts.append(f"{element}{supercell_count}")
    
    return " ".join(supercell_parts)

def check_backend_status():
    try:
        import requests
        response = requests.get(f"http://localhost:{PORT}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_backend():
    try:
        return subprocess.Popen([
            "uvicorn", "main:app", 
            "--host", "0.0.0.0", 
            "--port", str(PORT),
            "--reload"
        ])
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

if __name__ == "__main__":
    if not check_backend_status():
        print("Starting CrystalNexus backend...")
        process = start_backend()
        if process:
            print(f"Backend started on port {PORT}")
        else:
            print("Failed to start backend")
    else:
        print("Backend is already running")
        uvicorn.run(app, host="0.0.0.0", port=PORT)