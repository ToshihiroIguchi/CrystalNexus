import os
import json
import asyncio
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ログ設定（早期初期化）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CHGnet import
try:
    from chgnet.model import CHGNet
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False
    logger.warning("CHGnet not available. Install with: pip install chgnet")

def get_chgnet_supported_elements() -> Set[str]:
    """
    CHGnetからサポートされている元素を動的に取得
    フォールバックメカニズム付き
    """
    if not CHGNET_AVAILABLE:
        logger.warning("CHGNet not available, using fallback element list")
        return _get_fallback_elements()
    
    try:
        # 方法1: CHGnetモデルから直接取得を試行
        supported_elements = _get_elements_from_chgnet()
        if supported_elements:
            logger.info(f"Retrieved {len(supported_elements)} elements from CHGNet")
            return supported_elements
            
    except Exception as e:
        logger.warning(f"Failed to get elements from CHGNet: {e}")
    
    try:
        # 方法2: pymatgenの周期表から安全な範囲を取得
        supported_elements = _get_elements_from_periodic_table()
        if supported_elements:
            logger.info(f"Using periodic table range: {len(supported_elements)} elements")
            return supported_elements
            
    except Exception as e:
        logger.error(f"Failed to get elements from periodic table: {e}")
    
    # 方法3: 最終フォールバック
    logger.warning("Using hardcoded fallback element list")
    return _get_fallback_elements()

def _get_elements_from_chgnet() -> Optional[Set[str]]:
    """CHGnetモデルから元素リストを取得"""
    try:
        from chgnet.model import CHGNet
        
        # CHGnetモデルをロード
        model = CHGNet.load()
        
        # モデルの設定から元素情報を取得
        if hasattr(model, 'atom_embedding'):
            # 原子番号の範囲を推定（通常1-94程度）
            max_atomic_num = 94
            
            # 実際に使用されている原子番号を確認
            supported_elements = set()
            for z in range(1, min(max_atomic_num + 1, 95)):  # 1から94まで
                try:
                    element = Element.from_Z(z)
                    # ノーブルガス・アクチノイド除外（CHGnetの特性に基づく）
                    if z not in [2, 10, 18, 36, 54, 86] and z <= 92:  # He, Ne, Ar, Kr, Xe, Rn, U以降除外
                        supported_elements.add(element.symbol)
                except:
                    continue
                    
            return supported_elements if supported_elements else None
            
    except Exception as e:
        logger.debug(f"Direct CHGNet element extraction failed: {e}")
        return None

def _get_elements_from_periodic_table() -> Optional[Set[str]]:
    """pymatgenの周期表から妥当な元素範囲を取得"""
    try:
        supported_elements = set()
        
        # Materials Project/CHGnetで一般的にサポートされる範囲
        # 原子番号1-92（Uまで、ノーブルガス・放射性元素の一部除外）
        excluded_elements = {
            # ノーブルガス
            'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn',
            # アクチノイドの一部（安定性が低い）
            'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
        }
        
        for z in range(1, 93):  # 1-92 (H to U)
            try:
                element = Element.from_Z(z)
                if element.symbol not in excluded_elements:
                    supported_elements.add(element.symbol)
            except:
                continue
                
        return supported_elements if len(supported_elements) > 50 else None
        
    except Exception:
        return None

def _get_fallback_elements() -> Set[str]:
    """
    最終フォールバック: 確実にサポートされている元素のみ
    Materials Projectで高頻度使用される安全な元素リスト
    """
    return {
        # 1-2周期
        'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
        # 3周期  
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
        # 4周期
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
        'Ga', 'Ge', 'As', 'Se', 'Br',
        # 5周期
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
        'In', 'Sn', 'Sb', 'Te', 'I',
        # 6周期（主要）
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
        'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
        'Au', 'Hg', 'Tl', 'Pb', 'Bi',
        # 7周期（安定）
        'Th', 'U'
    }

# グローバル変数として一度だけ初期化
ALLOWED_ELEMENTS: Set[str] = get_chgnet_supported_elements()

# セキュリティ関数群
def safe_filename(filename: str) -> str:
    """ファイル名の安全化（パストラバーサル対策）"""
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # パストラバーサル文字を除去
    safe_name = os.path.basename(filename).replace('..', '')
    
    # 危険な文字をチェック
    if not safe_name or '/' in safe_name or '\\' in safe_name:
        raise ValueError("Invalid filename")
    
    # CIFファイル以外を拒否
    if not safe_name.lower().endswith('.cif'):
        raise ValueError("Only CIF files are allowed")
    
    return safe_name

def validate_supercell_size(supercell_size: List[int]) -> List[int]:
    """スーパーセルサイズの検証"""
    if not isinstance(supercell_size, list) or len(supercell_size) != 3:
        raise ValueError("Supercell size must be a list of 3 integers")
    
    for dim in supercell_size:
        if not isinstance(dim, int) or dim < 1 or dim > MAX_SUPERCELL_DIM:
            raise ValueError(f"Supercell dimensions must be between 1 and {MAX_SUPERCELL_DIM}")
    
    return supercell_size

def validate_element(element: str) -> str:
    """元素の検証（インジェクション対策）"""
    if not isinstance(element, str):
        raise ValueError("Element must be a string")
    
    element = element.strip()
    if element not in ALLOWED_ELEMENTS:
        raise ValueError(f"Element '{element}' is not supported by CHGNet")
    
    return element

def refresh_supported_elements():
    """元素リストを再取得（設定変更時などに使用）"""
    global ALLOWED_ELEMENTS
    ALLOWED_ELEMENTS = get_chgnet_supported_elements()
    logger.info(f"Refreshed element list: {len(ALLOWED_ELEMENTS)} elements available")

app = FastAPI(title="CrystalNexus")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SAMPLE_CIF_DIR = Path("sample_cif")

# セキュアなデフォルト設定
HOST = os.getenv('CRYSTALNEXUS_HOST', '127.0.0.1')
PORT = int(os.getenv('CRYSTALNEXUS_PORT', '8080'))
DEBUG = os.getenv('CRYSTALNEXUS_DEBUG', 'False').lower() == 'true'
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(50 * 1024 * 1024)))  # 50MB
MAX_SUPERCELL_DIM = int(os.getenv('MAX_SUPERCELL_DIM', '10'))

# DEBUG設定を適用
if DEBUG:
    logger.setLevel(logging.DEBUG)

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
        
        # セキュアなファイル名検証
        safe_name = safe_filename(filename)
        file_path = SAMPLE_CIF_DIR / safe_name
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="CIF file not found")
        
        result = await analyze_cif_file(file_path)
        result["filename"] = safe_name
        return result
    except ValueError as e:
        logger.warning(f"Invalid input in analyze_sample_cif: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_sample_cif: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/analyze-cif-upload")
async def analyze_uploaded_cif(file: UploadFile = File(...)):
    try:
        # ファイル名とサイズの検証
        if not file.filename or not file.filename.lower().endswith('.cif'):
            raise HTTPException(status_code=400, detail="File must be a CIF file")
        
        # ファイルサイズ制限
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")
        
        # セキュアな一時ファイル作成
        with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as tmp_file:
            tmp_file.write(contents)
            temp_path = Path(tmp_file.name)
        
        try:
            result = await analyze_cif_file(temp_path)
            safe_name = safe_filename(file.filename)
            result["filename"] = safe_name
            return result
        finally:
            if temp_path.exists():
                temp_path.unlink()
                
    except ValueError as e:
        logger.warning(f"Invalid input in analyze_uploaded_cif: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in analyze_uploaded_cif: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
        supercell_size_raw = data.get("supercell_size", [1, 1, 1])
        filename = data.get("filename")
        
        if not crystal_data:
            raise HTTPException(status_code=400, detail="Crystal data is required")
        
        # スーパーセルサイズの検証
        supercell_size = validate_supercell_size(supercell_size_raw)
        
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
                    logger.info(f"Successfully created supercell structure from {filename}")
                else:
                    logger.warning(f"CIF file not found: {cif_path}")
                    structure_dict = None
            else:
                logger.warning("No filename provided for structure generation")
                structure_dict = None
                
        except Exception as e:
            logger.warning(f"Could not create structure object: {e}")
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

def get_legacy_chgnet_elements():
    """
    旧ハードコーディング関数（互換性のため保持）
    動的取得した元素リストを返す
    """
    return list(ALLOWED_ELEMENTS)

@app.get("/api/chgnet-elements")
async def get_chgnet_elements():
    """
    CHGnet supported elements API (dynamic detection version)
    Returns elements sorted by atomic number
    """
    try:
        elements = list(ALLOWED_ELEMENTS)
        # Sort by atomic number for consistent UI ordering
        elements.sort(key=lambda el: Element(el).Z)
        return {
            "status": "success",
            "chgnet_available": CHGNET_AVAILABLE,
            "elements": elements,
            "total_elements": len(elements),
            "source": "dynamic_detection" if CHGNET_AVAILABLE else "fallback"
        }
    except Exception as e:
        logger.error(f"Error getting CHGnet elements: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported elements")

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
                    new_element_raw = operation["to"]
                    
                    # 元素の検証（セキュリティ対策）
                    new_element = validate_element(new_element_raw)
                    
                    if site_index < len(structure.sites):
                        old_coords = structure[site_index].frac_coords
                        old_element = str(structure[site_index].specie)
                        
                        # Replace the site with new element
                        structure[site_index] = Element(new_element), old_coords
                        
                        logger.info(f"Operation {i+1}: Substituted {old_element} → {new_element} at site {site_index}")
                        operations_applied += 1
                    else:
                        logger.warning(f"Operation {i+1}: SKIPPED - Invalid site index {site_index}")
                        
                elif operation["action"] == "delete":
                    site_index = operation["index"]
                    
                    if site_index < len(structure.sites):
                        deleted_element = str(structure[site_index].specie)
                        structure.remove_sites([site_index])
                        
                        logger.info(f"Operation {i+1}: Deleted {deleted_element} at site {site_index}")
                        operations_applied += 1
                        
                        # Adjust subsequent indices for deletions
                        for j in range(i + 1, len(operations)):
                            if operations[j].get("index", 0) > site_index:
                                operations[j]["index"] -= 1
                    else:
                        logger.warning(f"Operation {i+1}: SKIPPED - Invalid site index {site_index}")
                
            except ValueError as ve:
                logger.error(f"Operation {i+1}: Invalid element - {str(ve)}")
                continue
            except Exception as op_error:
                logger.error(f"Operation {i+1}: ERROR - {str(op_error)}")
                continue
        
        logger.info(f"Applied {operations_applied}/{len(operations)} operations successfully")
        logger.info(f"Final structure: {structure.formula} ({len(structure.sites)} sites)")
        
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
        
        logger.info(f"Modified CIF generated successfully, length: {len(final_cif)} characters")
        
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
        logger.error(error_msg)
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
        
        logger.info(f"Generating supercell CIF for {filename} with size {supercell_size}")
        
        # Construct CIF file path
        import os
        cif_path = os.path.join("sample_cif", filename)
        
        if not os.path.exists(cif_path):
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        logger.debug(f"Reading CIF file: {cif_path}")
        
        # Parse original CIF file
        from pymatgen.io.cif import CifParser, CifWriter
        parser = CifParser(cif_path)
        structures = parser.get_structures(primitive=False)  # Keep original lattice - don't convert to primitive
        
        if not structures:
            raise HTTPException(status_code=400, detail="No structures found in CIF file")
        
        original_structure = structures[0]
        logger.info(f"Original structure loaded: {original_structure.formula}")
        
        # Create supercell
        supercell_structure = original_structure.copy()
        supercell_structure.make_supercell(supercell_size)
        
        logger.info(f"Supercell created: {supercell_structure.formula}")
        logger.info(f"Number of sites: {len(supercell_structure.sites)}")
        
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
        
        logger.info(f"CIF generated successfully, length: {len(final_cif)} characters")
        
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
        logger.error(error_msg)
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

def safe_get_prediction(pred):
    """Extract prediction results safely from CHGNet output"""
    out = {}
    for k in ("energy", "e", "energy_per_atom", "e_per_atom"):
        if k in pred:
            out["energy_eV_per_atom"] = float(getattr(pred[k], "item", lambda: pred[k])())
            break
    for k in ("forces", "f", "force"):
        if k in pred:
            arr = pred[k]
            out["forces_eV_per_A"] = [list(a) for a in arr.tolist()]
            break
    for k in ("stress", "s", "virial"):
        if k in pred:
            arr = pred[k]
            out["stress_GPa"] = [list(a) for a in arr.tolist()]
            break
    for k in ("magmom", "m", "magmoms"):
        if k in pred:
            arr = pred[k]
            out["magmoms_muB"] = [float(x) for x in arr.tolist()]
            break
    # Other fields
    for k, v in pred.items():
        if k in ("energy", "e", "forces", "f", "stress", "s", "magmom", "m"):
            continue
        try:
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        except:
            continue
    return out

@app.post("/api/chgnet-predict")
async def chgnet_predict_structure(request: dict):
    """
    Predict structure properties using CHGNet
    """
    try:
        filename = request.get("filename")
        operations = request.get("operations", [])
        supercell_size = request.get("supercell_size", [1, 1, 1])
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        logger.info(f"CHGNet prediction for {filename} with {len(operations)} operations")
        
        # Get the modified structure
        from pymatgen.io.cif import CifParser
        cif_path = SAMPLE_CIF_DIR / safe_filename(filename)
        
        if not cif_path.exists():
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        # Parse and create supercell
        parser = CifParser(str(cif_path))
        structure = parser.get_structures(primitive=False)[0]
        structure.make_supercell(supercell_size)
        
        # Apply operations
        for operation in operations:
            if operation["action"] == "substitute":
                site_index = operation["index"]
                new_element = validate_element(operation["to"])
                if site_index < len(structure.sites):
                    old_coords = structure[site_index].frac_coords
                    structure[site_index] = Element(new_element), old_coords
            elif operation["action"] == "delete":
                site_index = operation["index"]
                if site_index < len(structure.sites):
                    structure.remove_sites([site_index])
                    # Adjust subsequent indices
                    for j, op in enumerate(operations[operations.index(operation)+1:], operations.index(operation)+1):
                        if op.get("index", 0) > site_index:
                            op["index"] -= 1
        
        # Load CHGNet model
        if not CHGNET_AVAILABLE:
            raise HTTPException(status_code=503, detail="CHGNet not available. Please install with: pip install chgnet")
        
        from chgnet.model.model import CHGNet
        chgnet = CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=False)
        
        # Predict structure properties
        try:
            pred = chgnet.predict_structure(structure,
                                          return_site_energies=True,
                                          return_atom_feas=False,
                                          return_crystal_feas=False)
        except TypeError:
            pred = chgnet.predict_structure(structure)
        
        # Extract results
        results = safe_get_prediction(pred)
        
        # Add structure information
        results.update({
            "formula": str(structure.formula),
            "num_sites": len(structure.sites),
            "volume": float(structure.volume),
            "density": float(structure.density),
            "operations_applied": len(operations),
            "supercell_size": supercell_size
        })
        
        logger.info(f"CHGNet prediction completed: {results.get('energy_eV_per_atom', 'N/A')} eV/atom")
        
        return {
            "status": "success",
            "prediction": results,
            "model_info": {
                "version": chgnet.version if hasattr(chgnet, 'version') else "0.3.0",
                "device": "cpu",
                "parameters": chgnet.n_params if hasattr(chgnet, 'n_params') else None
            }
        }
        
    except ValueError as e:
        logger.error(f"CHGNet prediction validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CHGNet prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CHGNet prediction failed: {str(e)}")

@app.post("/api/chgnet-relax")
async def chgnet_relax_structure(request: dict):
    """
    Relax structure using CHGNet
    """
    try:
        filename = request.get("filename")
        operations = request.get("operations", [])
        supercell_size = request.get("supercell_size", [1, 1, 1])
        fmax = float(request.get("fmax", 0.1))
        max_steps = int(request.get("max_steps", 100))
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        logger.info(f"CHGNet relaxation for {filename} with fmax={fmax}, max_steps={max_steps}")
        
        # Get the modified structure (same as predict)
        from pymatgen.io.cif import CifParser
        cif_path = SAMPLE_CIF_DIR / safe_filename(filename)
        
        if not cif_path.exists():
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        parser = CifParser(str(cif_path))
        structure = parser.get_structures(primitive=False)[0]
        structure.make_supercell(supercell_size)
        
        # Apply operations
        for operation in operations:
            if operation["action"] == "substitute":
                site_index = operation["index"]
                new_element = validate_element(operation["to"])
                if site_index < len(structure.sites):
                    old_coords = structure[site_index].frac_coords
                    structure[site_index] = Element(new_element), old_coords
            elif operation["action"] == "delete":
                site_index = operation["index"]
                if site_index < len(structure.sites):
                    structure.remove_sites([site_index])
        
        # Load CHGNet and relaxer
        if not CHGNET_AVAILABLE:
            raise HTTPException(status_code=503, detail="CHGNet not available")
        
        from chgnet.model.model import CHGNet
        from chgnet.model import StructOptimizer
        
        chgnet = CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=False)
        relaxer = StructOptimizer(model=chgnet, use_device="cpu", optimizer_class="FIRE")
        
        # Predict initial structure
        try:
            pred_initial = chgnet.predict_structure(structure)
            initial_results = safe_get_prediction(pred_initial)
        except Exception as e:
            logger.warning(f"Initial prediction failed: {e}")
            initial_results = {"energy_eV_per_atom": None}
        
        # Relax structure
        result = relaxer.relax(structure, fmax=fmax, steps=max_steps, verbose=False, relax_cell=True)
        
        final_structure = result.get("final_structure")
        if final_structure is None:
            raise RuntimeError("Relaxation failed: no final structure returned")
        
        # Predict relaxed structure
        try:
            pred_final = chgnet.predict_structure(final_structure)
            final_results = safe_get_prediction(pred_final)
        except Exception as e:
            logger.warning(f"Final prediction failed: {e}")
            final_results = {"energy_eV_per_atom": None}
        
        # Calculate energy difference
        energy_diff = None
        if (initial_results.get("energy_eV_per_atom") is not None and 
            final_results.get("energy_eV_per_atom") is not None):
            energy_diff = final_results["energy_eV_per_atom"] - initial_results["energy_eV_per_atom"]
        
        relaxation_info = {
            "converged": result.get("converged", False),
            "steps": len(result.get("trajectory", [])) if result.get("trajectory") else 0,
            "fmax": fmax,
            "max_steps": max_steps,
            "energy_change_eV_per_atom": energy_diff
        }
        
        # Add structure information
        final_results.update({
            "formula": str(final_structure.formula),
            "num_sites": len(final_structure.sites),
            "volume": float(final_structure.volume),
            "density": float(final_structure.density)
        })
        
        logger.info(f"CHGNet relaxation completed: {relaxation_info['steps']} steps, converged: {relaxation_info['converged']}")
        
        # Generate relaxed structure info
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(final_structure)
        
        relaxed_structure_info = {
            "formula": str(final_structure.formula),
            "num_atoms": len(final_structure),
            "density": float(final_structure.density),
            "lattice_parameters": {
                "a": float(final_structure.lattice.a),
                "b": float(final_structure.lattice.b),
                "c": float(final_structure.lattice.c),
                "alpha": float(final_structure.lattice.alpha),
                "beta": float(final_structure.lattice.beta),
                "gamma": float(final_structure.lattice.gamma)
            },
            "volume": float(final_structure.lattice.volume),
            "space_group": analyzer.get_space_group_symbol(),
            "space_group_number": analyzer.get_space_group_number(),
            "point_group": analyzer.get_point_group_symbol(),
            "crystal_system": analyzer.get_crystal_system(),
            "num_sites": len(final_structure.sites)
        }
        
        return {
            "status": "success",
            "initial_prediction": initial_results,
            "final_prediction": final_results,
            "relaxation_info": relaxation_info,
            "relaxed_structure_info": relaxed_structure_info,
            "model_info": {
                "version": chgnet.version if hasattr(chgnet, 'version') else "0.3.0",
                "device": "cpu"
            }
        }
        
    except ValueError as e:
        logger.error(f"CHGNet relaxation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CHGNet relaxation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"CHGNet relaxation failed: {str(e)}")

@app.post("/api/generate-relaxed-structure-cif")
async def generate_relaxed_structure_cif(request: dict):
    """
    Generate CIF from relaxed structure with applied atomic operations
    """
    try:
        filename = request.get("filename")
        operations = request.get("operations", [])
        supercell_size = request.get("supercell_size", [1, 1, 1])
        fmax = float(request.get("fmax", 0.1))
        max_steps = int(request.get("max_steps", 100))
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        logger.info(f"Generating relaxed structure CIF for {filename}")
        
        # Get the modified structure (same process as relax)
        from pymatgen.io.cif import CifParser, CifWriter
        cif_path = SAMPLE_CIF_DIR / safe_filename(filename)
        
        if not cif_path.exists():
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        parser = CifParser(str(cif_path))
        structure = parser.get_structures(primitive=False)[0]
        structure.make_supercell(supercell_size)
        
        # Apply operations
        for operation in operations:
            if operation["action"] == "substitute":
                site_index = operation["index"]
                new_element = validate_element(operation["to"])
                if site_index < len(structure.sites):
                    old_coords = structure[site_index].frac_coords
                    structure[site_index] = Element(new_element), old_coords
            elif operation["action"] == "delete":
                site_index = operation["index"]
                if site_index < len(structure.sites):
                    structure.remove_sites([site_index])
        
        # Load CHGNet and relax
        if not CHGNET_AVAILABLE:
            raise HTTPException(status_code=503, detail="CHGNet not available")
        
        from chgnet.model.model import CHGNet
        from chgnet.model import StructOptimizer
        
        chgnet = CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=False)
        relaxer = StructOptimizer(model=chgnet, use_device="cpu", optimizer_class="FIRE")
        
        # Relax structure
        result = relaxer.relax(structure, fmax=fmax, steps=max_steps, verbose=False, relax_cell=True)
        final_structure = result.get("final_structure")
        
        if final_structure is None:
            raise RuntimeError("Relaxation failed: no final structure returned")
        
        # Generate CIF using pymatgen CifWriter
        cif_writer = CifWriter(
            final_structure,
            write_magmoms=False,
            significant_figures=6
        )
        
        cif_content = str(cif_writer)
        
        # Add metadata header
        operations_summary = f"{len(operations)} operations applied"
        size_str = "x".join(map(str, supercell_size))
        metadata_lines = [
            f"# Relaxed structure CIF generated by CrystalNexus",
            f"# Original file: {filename}",
            f"# Supercell size: {size_str}",
            f"# Operations: {operations_summary}",
            f"# CHGNet relaxation: fmax={fmax}, steps={result.get('converged', 'N/A')}",
            f"# Final formula: {final_structure.formula}",
            f"# Number of atoms: {len(final_structure.sites)}",
            f"# Volume: {final_structure.volume:.2f} Ų",
            ""
        ]
        
        final_cif = "\n".join(metadata_lines) + cif_content
        
        logger.info(f"Relaxed CIF generated successfully, length: {len(final_cif)} characters")
        
        return Response(
            content=final_cif,
            media_type="chemical/x-cif",
            headers={
                "Content-Disposition": f"inline; filename={filename.replace('.cif', '')}_relaxed.cif",
                "Cache-Control": "no-cache"
            }
        )
        
    except Exception as e:
        error_msg = f"Failed to generate relaxed structure CIF: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def check_backend_status():
    try:
        import requests
        response = requests.get(f"http://localhost:{PORT}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Backend status check failed: {e}")
        return False

def start_backend():
    try:
        cmd = [
            "uvicorn", "main:app", 
            "--host", HOST,
            "--port", str(PORT)
        ]
        
        if DEBUG:
            cmd.append("--reload")
        
        return subprocess.Popen(cmd)
    except Exception as e:
        logger.error(f"Error starting backend: {e}")
        return None

if __name__ == "__main__":
    if not check_backend_status():
        logger.info("Starting CrystalNexus backend...")
        process = start_backend()
        if process:
            logger.info(f"Backend started on port {PORT}")
        else:
            logger.error("Failed to start backend")
    else:
        logger.info("Backend is already running")
        uvicorn.run(app, host=HOST, port=PORT, reload=DEBUG)