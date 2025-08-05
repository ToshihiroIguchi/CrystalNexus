import os
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
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