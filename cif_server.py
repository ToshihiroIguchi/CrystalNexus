#!/usr/bin/env python3
"""
Improved CIF Parser Server with Security, Caching, and Error Handling
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
import logging
import hashlib
import time
import tempfile
from functools import lru_cache
from typing import Dict, Any, Optional
import warnings
from io import StringIO

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*from_string.*")

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

# Configuration
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', 5000))
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB limit
CACHE_SIZE = int(os.getenv('CACHE_SIZE', 100))

# Logging configuration
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Secure CORS configuration
allowed_origins = [
    f'http://127.0.0.1:{SERVER_PORT}',
    f'http://localhost:{SERVER_PORT}',
    'http://127.0.0.1:5000',
    'http://localhost:5000',
    'http://0.0.0.0:5000'
]

CORS(app, origins=allowed_origins, methods=['GET', 'POST'], 
     allow_headers=['Content-Type'], supports_credentials=False)

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://3dmol.csb.pitt.edu; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response

# Cache for CIF parsing results
cif_cache: Dict[str, Dict[str, Any]] = {}
cache_timestamps: Dict[str, float] = {}
CACHE_EXPIRY = 3600  # 1 hour

def get_cache_key(cif_content: str) -> str:
    """Generate cache key from CIF content"""
    return hashlib.md5(cif_content.encode('utf-8')).hexdigest()

def is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry is still valid"""
    if cache_key not in cache_timestamps:
        return False
    return (time.time() - cache_timestamps[cache_key]) < CACHE_EXPIRY

def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached parsing result if valid"""
    if cache_key in cif_cache and is_cache_valid(cache_key):
        logger.info(f"Cache hit for key: {cache_key[:8]}...")
        return cif_cache[cache_key]
    return None

def cache_result(cache_key: str, result: Dict[str, Any]) -> None:
    """Cache parsing result with size limit"""
    if len(cif_cache) >= CACHE_SIZE:
        # Remove oldest entry
        oldest_key = min(cache_timestamps.keys(), key=lambda k: cache_timestamps[k])
        del cif_cache[oldest_key]
        del cache_timestamps[oldest_key]
        logger.info(f"Cache evicted oldest entry: {oldest_key[:8]}...")
    
    cif_cache[cache_key] = result
    cache_timestamps[cache_key] = time.time()
    logger.info(f"Cached result for key: {cache_key[:8]}...")

def generate_custom_supercell_cif(structure, supercell_metadata):
    """Generate custom CIF for supercell that preserves all atoms"""
    try:
        # Get lattice parameters
        lattice = structure.lattice
        
        # Generate formula
        formula = str(structure.composition.reduced_formula)
        element_formula = str(structure.composition)
        
        # Generate CIF header
        cif_lines = [
            "# generated using pymatgen (supercell preserved)",
            f"data_{formula}",
            "_symmetry_space_group_name_H-M   'P 1'",
            f"_cell_length_a   {lattice.a:.8f}",
            f"_cell_length_b   {lattice.b:.8f}", 
            f"_cell_length_c   {lattice.c:.8f}",
            f"_cell_angle_alpha   {lattice.alpha:.8f}",
            f"_cell_angle_beta   {lattice.beta:.8f}",
            f"_cell_angle_gamma   {lattice.gamma:.8f}",
            "_symmetry_Int_Tables_number   1",
            f"_chemical_formula_structural   {formula}",
            f"_chemical_formula_sum   '{element_formula}'",
            f"_cell_volume   {lattice.volume:.8f}",
            "_cell_formula_units_Z   1",
            "loop_",
            " _symmetry_equiv_pos_site_id",
            " _symmetry_equiv_pos_as_xyz",
            "  1  'x, y, z'",
            "loop_",
            " _atom_site_type_symbol",
            " _atom_site_label", 
            " _atom_site_symmetry_multiplicity",
            " _atom_site_fract_x",
            " _atom_site_fract_y",
            " _atom_site_fract_z",
            " _atom_site_occupancy"
        ]
        
        # Add all atomic sites with proper element type and unique labels
        element_counts = {}
        for i, site in enumerate(structure.sites):
            element = str(site.specie)
            
            # 元素ごとのカウンターを管理
            if element not in element_counts:
                element_counts[element] = 0
            element_counts[element] += 1
            
            label = f"{element}{element_counts[element]}"
            frac_coords = site.frac_coords
            
            cif_lines.append(
                f"  {element}  {label}  1  {frac_coords[0]:.8f}  "
                f"{frac_coords[1]:.8f}  {frac_coords[2]:.8f}  1.0"
            )
        
        return "\n".join(cif_lines) + "\n"
        
    except Exception as e:
        logger.error(f"Failed to generate custom supercell CIF: {e}")
        # Fallback to standard CifWriter
        from pymatgen.io.cif import CifWriter
        writer = CifWriter(structure)
        return str(writer)

def validate_cif_content(cif_content: str) -> Dict[str, Any]:
    """Validate CIF content and return validation result"""
    errors = []
    warnings_list = []
    
    if not cif_content or not cif_content.strip():
        errors.append("Empty CIF content")
        return {"valid": False, "errors": errors, "warnings": warnings_list}
    
    if len(cif_content) > MAX_CONTENT_LENGTH:
        errors.append(f"CIF content too large (max: {MAX_CONTENT_LENGTH} bytes)")
        return {"valid": False, "errors": errors, "warnings": warnings_list}
    
    # Basic CIF format checks
    if 'data_' not in cif_content:
        errors.append("No data block found in CIF")
    
    if not any(keyword in cif_content.lower() for keyword in ['_cell_length', '_atom_site']):
        warnings_list.append("CIF may not contain crystal structure data")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list
    }

def safe_parse_cif(cif_content: str) -> Dict[str, Any]:
    """Safely parse CIF with comprehensive error handling"""
    try:
        if not PYMATGEN_AVAILABLE:
            return {
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }
        
        # Use the correct, non-deprecated API
        parser = CifParser(StringIO(cif_content))
        structures = parser.get_structures()
        
        if not structures:
            return {
                "success": False,
                "error": "No valid crystal structures found in CIF file",
                "error_type": "parsing_error"
            }
        
        structure = structures[0]
        logger.info(f"Structure parsed successfully: {structure.formula}")
        
        result = {"success": True}
        
        # Safe extraction of each property with individual error handling
        try:
            result["formula"] = str(structure.composition.reduced_formula)
        except Exception as e:
            logger.warning(f"Could not extract formula: {e}")
            result["formula"] = "Unknown"
        
        try:
            analyzer = SpacegroupAnalyzer(structure)
            result["crystal_system"] = analyzer.get_crystal_system()
        except Exception as e:
            logger.warning(f"Could not determine crystal system: {e}")
            result["crystal_system"] = "Unknown"
        
        try:
            spacegroup_info = structure.get_space_group_info()
            if len(spacegroup_info) > 1:
                # Get both symbol and number for comprehensive display
                space_group_symbol = spacegroup_info[1]  # Symbol like 'P4mm'
                space_group_number = spacegroup_info[0]  # Number like 99
                result["space_group"] = f"{space_group_symbol} (#{space_group_number})"
            else:
                result["space_group"] = "Unknown"
        except Exception as e:
            logger.warning(f"Could not get space group: {e}")
            result["space_group"] = "Unknown"
        
        try:
            lattice = structure.lattice
            result["lattice_parameters"] = {
                "a": round(lattice.a, 6),
                "b": round(lattice.b, 6),
                "c": round(lattice.c, 6),
                "alpha": round(lattice.alpha, 3),
                "beta": round(lattice.beta, 3),
                "gamma": round(lattice.gamma, 3)
            }
            result["volume"] = round(lattice.volume, 6)
        except Exception as e:
            logger.warning(f"Could not extract lattice parameters: {e}")
            result["lattice_parameters"] = None
            result["volume"] = None
        
        try:
            result["atom_count"] = len(structure.sites)
            element_counts = {}
            atom_info = []
            
            for i, site in enumerate(structure.sites):
                element = str(site.specie)
                element_counts[element] = element_counts.get(element, 0) + 1
                
                # 原子の詳細情報を追加（CIFラベル含む）
                atom_data = {
                    "index": i,
                    "element": element,
                    "label": getattr(site, 'label', f'{element}{i}'),
                    "coords": [round(coord, 6) for coord in site.coords],
                    "frac_coords": [round(coord, 6) for coord in site.frac_coords]
                }
                atom_info.append(atom_data)
            
            result["element_counts"] = element_counts
            result["atom_info"] = atom_info
        except Exception as e:
            logger.warning(f"Could not count atoms: {e}")
            result["atom_count"] = 0
            result["element_counts"] = {}
            result["atom_info"] = []
        
        try:
            result["density"] = round(structure.density, 3)
        except Exception as e:
            logger.warning(f"Could not calculate density: {e}")
            result["density"] = None
        
        return result
        
    except Exception as e:
        logger.error(f"Critical error in CIF parsing: {e}")
        if DEBUG_MODE:
            logger.error(traceback.format_exc())
        
        error_type = "unknown_error"
        if "Invalid CIF file" in str(e):
            error_type = "invalid_cif"
        elif "structures" in str(e).lower():
            error_type = "structure_error"
        
        return {
            "success": False,
            "error": f"Failed to parse CIF file: {str(e)}",
            "error_type": error_type,
            "traceback": traceback.format_exc() if DEBUG_MODE else None
        }

# Static file serving
@app.route('/')
def serve_index():
    """Serve main application"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files with security checks"""
    # Prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "Invalid file path"}), 400
    
    # Only allow specific file extensions
    allowed_extensions = {'.html', '.css', '.js', '.cif', '.json'}
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        return send_from_directory('.', filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route('/parse_cif', methods=['POST'])
def parse_cif():
    """Enhanced CIF parsing API with caching and validation"""
    try:
        # Input validation
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False,
                "error_type": "invalid_content_type"
            }), 400
        
        data = request.get_json()
        if not data or 'cif_content' not in data:
            return jsonify({
                "error": "Missing 'cif_content' in request body",
                "success": False,
                "error_type": "missing_parameter"
            }), 400
        
        cif_content = data['cif_content']
        
        # Validate CIF content
        validation_result = validate_cif_content(cif_content)
        if not validation_result["valid"]:
            return jsonify({
                "error": "Invalid CIF content",
                "errors": validation_result["errors"],
                "success": False,
                "error_type": "validation_error"
            }), 400
        
        # Check cache first
        cache_key = get_cache_key(cif_content)
        cached_result = get_cached_result(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        logger.info(f"Processing CIF content (length: {len(cif_content)})")
        
        # Parse CIF
        result = safe_parse_cif(cif_content)
        
        # Add validation warnings to successful results
        if result.get("success") and validation_result["warnings"]:
            result["warnings"] = validation_result["warnings"]
        
        # Cache successful results
        if result.get("success"):
            cache_result(cache_key, result)
        
        logger.info(f"CIF parsing completed: {'success' if result.get('success') else 'failed'}")
        
        status_code = 200 if result.get("success") else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Unexpected error in parse_cif endpoint: {e}")
        if DEBUG_MODE:
            logger.error(traceback.format_exc())
        
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False,
            "error_type": "server_error",
            "traceback": traceback.format_exc() if DEBUG_MODE else None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with dependency verification"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "pymatgen_available": PYMATGEN_AVAILABLE,
            "cache_size": len(cif_cache),
            "debug_mode": DEBUG_MODE
        }
        
        if not PYMATGEN_AVAILABLE:
            health_status["status"] = "degraded"
            health_status["warning"] = "Pymatgen library not available"
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/test_cif', methods=['GET'])
def test_cif():
    """Test endpoint with enhanced error handling"""
    try:
        if not os.path.exists('BaTiO3.cif'):
            return jsonify({
                "error": "Test file BaTiO3.cif not found",
                "success": False,
                "error_type": "file_not_found"
            }), 404
        
        with open('BaTiO3.cif', 'r', encoding='utf-8') as f:
            cif_content = f.read()
        
        # Validate test file
        validation_result = validate_cif_content(cif_content)
        if not validation_result["valid"]:
            return jsonify({
                "error": "Test CIF file is invalid",
                "errors": validation_result["errors"],
                "success": False,
                "error_type": "invalid_test_file"
            }), 500
        
        # Parse test file
        result = safe_parse_cif(cif_content)
        
        status_code = 200 if result.get("success") else 500
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({
            "error": f"Test failed: {str(e)}",
            "success": False,
            "error_type": "test_error"
        }), 500

@app.route('/sample_files', methods=['GET'])
def get_sample_files():
    """Get list of available sample CIF files with basic info"""
    try:
        sample_dir = 'sample_cif'
        
        if not os.path.exists(sample_dir):
            return jsonify({
                "success": False,
                "error": "Sample directory not found",
                "error_type": "directory_not_found"
            }), 404
        
        sample_files = []
        
        # Scan for .cif files in sample_cif directory
        for filename in os.listdir(sample_dir):
            if filename.lower().endswith('.cif') and not filename.endswith('.cif:Zone.Identifier'):
                file_path = os.path.join(sample_dir, filename)
                
                try:
                    # Get file info
                    file_stat = os.stat(file_path)
                    file_size = file_stat.st_size
                    
                    # Quick validation: try to read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Basic CIF validation
                    validation_result = validate_cif_content(content)
                    
                    sample_info = {
                        "filename": filename,
                        "display_name": filename.replace('.cif', ''),
                        "file_size": file_size,
                        "valid": validation_result["valid"],
                        "path": f"sample_cif/{filename}"
                    }
                    
                    # If valid, try to get basic structure info
                    if validation_result["valid"]:
                        try:
                            parse_result = safe_parse_cif(content)
                            if parse_result.get("success"):
                                sample_info.update({
                                    "formula": parse_result.get("formula", "Unknown"),
                                    "space_group": parse_result.get("space_group", "Unknown"),
                                    "crystal_system": parse_result.get("crystal_system", "Unknown"),
                                    "atom_count": parse_result.get("atom_count", 0)
                                })
                        except Exception as e:
                            logger.warning(f"Could not parse sample file {filename}: {e}")
                            sample_info.update({
                                "formula": "Parse Error",
                                "space_group": "Unknown",
                                "crystal_system": "Unknown",
                                "atom_count": 0
                            })
                    
                    sample_files.append(sample_info)
                    
                except Exception as e:
                    logger.error(f"Error processing sample file {filename}: {e}")
                    # Add as invalid file
                    sample_files.append({
                        "filename": filename,
                        "display_name": filename.replace('.cif', ''),
                        "valid": False,
                        "error": str(e),
                        "path": f"sample_cif/{filename}"
                    })
        
        # Sort by filename
        sample_files.sort(key=lambda x: x["filename"])
        
        return jsonify({
            "success": True,
            "sample_files": sample_files,
            "count": len(sample_files)
        })
        
    except Exception as e:
        logger.error(f"Sample files endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to get sample files: {str(e)}",
            "error_type": "sample_files_error"
        }), 500

@app.route('/sample_cif/<filename>', methods=['GET'])
def get_sample_file(filename):
    """Get specific sample CIF file content"""
    try:
        # Security: validate filename
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({
                "success": False,
                "error": "Invalid filename",
                "error_type": "security_error"
            }), 400
        
        file_path = os.path.join('sample_cif', filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": f"Sample file {filename} not found",
                "error_type": "file_not_found"
            }), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            cif_content = f.read()
        
        # Validate and parse
        validation_result = validate_cif_content(cif_content)
        if not validation_result["valid"]:
            return jsonify({
                "success": False,
                "error": "Invalid CIF file",
                "errors": validation_result["errors"],
                "error_type": "invalid_cif"
            }), 400
        
        # Parse CIF for detailed info
        parse_result = safe_parse_cif(cif_content)
        
        response_data = {
            "success": True,
            "filename": filename,
            "cif_content": cif_content,
            "file_info": parse_result
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Sample file endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to get sample file: {str(e)}",
            "error_type": "sample_file_error"
        }), 500

@app.route('/create_supercell', methods=['POST'])
def create_supercell():
    """Create supercell from CIF content"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False,
                "error_type": "invalid_content_type"
            }), 400
        
        data = request.get_json()
        if not data or 'cif_content' not in data:
            return jsonify({
                "error": "Missing 'cif_content' in request body",
                "success": False,
                "error_type": "missing_parameter"
            }), 400
        
        cif_content = data['cif_content']
        a_mult = data.get('a_multiplier', 1)
        b_mult = data.get('b_multiplier', 1)
        c_mult = data.get('c_multiplier', 1)
        
        # Validate multipliers
        if not all(isinstance(x, int) and 1 <= x <= 10 for x in [a_mult, b_mult, c_mult]):
            return jsonify({
                "error": "Multipliers must be integers between 1 and 10",
                "success": False,
                "error_type": "invalid_multipliers"
            }), 400
        
        if not PYMATGEN_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }), 500
        
        # Parse original structure with enhanced debugging
        try:
            logger.info(f"Attempting to parse CIF content (length: {len(cif_content)} chars)")
            
            # Try different parsing methods
            try:
                # Method 1: StringIO approach
                parser = CifParser(StringIO(cif_content))
                structures = parser.get_structures()
                logger.info(f"StringIO method: Found {len(structures) if structures else 0} structures")
            except Exception as e1:
                logger.warning(f"StringIO method failed: {e1}")
                try:
                    # Method 2: Write to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp_file:
                        tmp_file.write(cif_content)
                        tmp_file.flush()
                        parser = CifParser(tmp_file.name)
                        structures = parser.get_structures()
                        logger.info(f"Temp file method: Found {len(structures) if structures else 0} structures")
                    os.unlink(tmp_file.name)
                except Exception as e2:
                    logger.error(f"Both parsing methods failed: StringIO={e1}, TempFile={e2}")
                    raise e1
            
            if not structures:
                logger.error("No structures found after parsing")
                return jsonify({
                    "success": False,
                    "error": "No valid crystal structures found in CIF file",
                    "error_type": "parsing_error"
                }), 400
            
            structure = structures[0]
            logger.info(f"Successfully parsed structure with {len(structure.sites)} atoms")
            
            if structure is None:
                logger.error("Parsed structure is None")
                return jsonify({
                    "success": False,
                    "error": "Parsed structure is None",
                    "error_type": "parsing_error"
                }), 400
                
        except Exception as parse_error:
            logger.error(f"Structure parsing failed: {parse_error}")
            return jsonify({
                "success": False,
                "error": f"Failed to parse CIF structure: {str(parse_error)}",
                "error_type": "parsing_error"
            }), 400
        
        # Create supercell using SupercellTransformation (more reliable)
        try:
            from pymatgen.transformations.standard_transformations import SupercellTransformation
            
            supercell_matrix = [[a_mult, 0, 0], [0, b_mult, 0], [0, 0, c_mult]]
            transformation = SupercellTransformation(supercell_matrix)
            supercell = transformation.apply_transformation(structure)
            
            logger.info(f"Supercell created: {len(structure.sites)} -> {len(supercell.sites)} atoms")
            
        except Exception as e:
            logger.error(f"SupercellTransformation failed: {e}")
            # Fallback to make_supercell (in-place modification)
            try:
                structure.make_supercell([a_mult, b_mult, c_mult])
                supercell = structure  # make_supercell modifies in-place
                logger.info(f"Fallback supercell created: {len(supercell.sites)} atoms")
            except Exception as e2:
                logger.error(f"Both supercell methods failed: {e2}")
                raise e2
        
        # Generate new CIF content using custom generation to preserve supercell
        supercell_metadata_for_generation = {
            'multipliers': {'a': a_mult, 'b': b_mult, 'c': c_mult},
            'original_atoms': len(structure.sites),
            'is_supercell': True
        }
        supercell_cif = generate_custom_supercell_cif(supercell, supercell_metadata_for_generation)
        
        logger.info(f"Created supercell {a_mult}x{b_mult}x{c_mult} with {len(supercell.sites)} atoms")
        
        # Extract supercell properties directly from the supercell object using pymatgen
        # This is more reliable than re-parsing the CIF because CifWriter has known issues
        supercell_info = {"success": True}
        
        # Safe extraction using pymatgen methods (same as safe_parse_cif)
        try:
            supercell_info["formula"] = str(supercell.composition.reduced_formula)
        except Exception as e:
            logger.warning(f"Could not extract supercell formula: {e}")
            supercell_info["formula"] = "Unknown"
        
        # スーパーセルは構造の複雑性により強制的にP1 (triclinic)に設定
        supercell_info["crystal_system"] = "triclinic"
        supercell_info["space_group"] = "P1 (#1)"
        logger.info("Supercell symmetry forced to P1 (triclinic) due to structural complexity")
        
        try:
            supercell_info["lattice_parameters"] = {
                "a": round(supercell.lattice.a, 6),
                "b": round(supercell.lattice.b, 6),
                "c": round(supercell.lattice.c, 6),
                "alpha": round(supercell.lattice.alpha, 3),
                "beta": round(supercell.lattice.beta, 3),
                "gamma": round(supercell.lattice.gamma, 3)
            }
            supercell_info["volume"] = round(supercell.lattice.volume, 6)
        except Exception as e:
            logger.warning(f"Could not extract supercell lattice parameters: {e}")
            supercell_info["lattice_parameters"] = None
            supercell_info["volume"] = None
        
        try:
            supercell_info["atom_count"] = len(supercell.sites)
            
            # スーパーセルの原子情報も追加（一意のラベル生成）
            atom_info = []
            for i, site in enumerate(supercell.sites):
                element = str(site.specie)
                # スーパーセルでは各原子に一意のラベルを生成
                unique_label = f'{element}{i}'
                atom_data = {
                    "index": i,
                    "element": element,
                    "label": unique_label,
                    "coords": [round(coord, 6) for coord in site.coords],
                    "frac_coords": [round(coord, 6) for coord in site.frac_coords]
                }
                atom_info.append(atom_data)
            
            supercell_info["atom_info"] = atom_info
        except Exception as e:
            logger.warning(f"Could not count supercell atoms: {e}")
            supercell_info["atom_count"] = 0
            supercell_info["atom_info"] = []
        
        try:
            supercell_info["density"] = round(supercell.density, 3)
        except Exception as e:
            logger.warning(f"Could not calculate supercell density: {e}")
            supercell_info["density"] = None
        
        supercell_info["direct_from_pymatgen"] = True  # Flag to indicate this is directly from pymatgen
        
        # Compare original and supercell properties
        original_info = safe_parse_cif(cif_content)
        
        result = {
            "success": True,
            "supercell_cif": supercell_cif,
            "original_atoms": len(structure.sites),
            "supercell_atoms": len(supercell.sites),
            "multipliers": {"a": a_mult, "b": b_mult, "c": c_mult},
            "is_supercell": True,
            "supercell_info": supercell_info if supercell_info.get("success") else None,
            "original_info": original_info if original_info.get("success") else None
        }
        
        # Add comparison data if both analyses succeeded
        if (supercell_info.get("success") and original_info.get("success") and 
            supercell_info.get("lattice_parameters") and original_info.get("lattice_parameters")):
            
            orig_lp = original_info["lattice_parameters"]
            super_lp = supercell_info["lattice_parameters"]
            
            result["lattice_comparison"] = {
                "original": {
                    "a": orig_lp["a"], "b": orig_lp["b"], "c": orig_lp["c"]
                },
                "supercell": {
                    "a": super_lp["a"], "b": super_lp["b"], "c": super_lp["c"]
                },
                "scaling_factors": {
                    "a": round(super_lp["a"] / orig_lp["a"], 2),
                    "b": round(super_lp["b"] / orig_lp["b"], 2),
                    "c": round(super_lp["c"] / orig_lp["c"], 2)
                }
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Supercell creation error: {e}")
        return jsonify({
            "error": f"Failed to create supercell: {str(e)}",
            "success": False,
            "error_type": "supercell_error"
        }), 500

@app.route('/delete_atoms', methods=['POST'])
def delete_atoms():
    """Delete specified atoms from structure"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False,
                "error_type": "invalid_content_type"
            }), 400
        
        data = request.get_json()
        if not data or 'cif_content' not in data or 'atom_indices' not in data:
            return jsonify({
                "error": "Missing 'cif_content' or 'atom_indices' in request body",
                "success": False,
                "error_type": "missing_parameter"
            }), 400
        
        cif_content = data['cif_content']
        atom_indices = data['atom_indices']  # リスト形式で原子のインデックス
        
        # スーパーセル情報を保持するためのメタデータ（オプション）
        supercell_metadata = data.get('supercell_metadata', None)
        
        if not PYMATGEN_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }), 500
        
        # 元の構造を解析
        parser = CifParser(StringIO(cif_content))
        structures = parser.get_structures()
        
        if not structures:
            return jsonify({
                "success": False,
                "error": "No valid crystal structures found in CIF file",
                "error_type": "parsing_error"
            }), 400
        
        structure = structures[0]
        
        # スーパーセルメタデータがある場合、構造を再構築
        if supercell_metadata and supercell_metadata.get('is_supercell'):
            # 期待されるスーパーセル原子数を計算
            original_atoms = supercell_metadata.get('original_atoms', 0)
            multipliers = supercell_metadata.get('multipliers', {})
            a_mult = multipliers.get('a', 1)
            b_mult = multipliers.get('b', 1) 
            c_mult = multipliers.get('c', 1)
            expected_supercell_atoms = original_atoms * a_mult * b_mult * c_mult
            
            current_atoms = len(structure.sites)
            
            # 既にスーパーセル状態かどうかを判定
            if current_atoms == expected_supercell_atoms:
                logger.info(f"Structure already is supercell for deletion: {current_atoms} atoms (expected: {expected_supercell_atoms})")
                # 既にスーパーセル状態なので再構築不要
            else:
                logger.info(f"Reconstructing supercell structure from metadata for deletion: {current_atoms} -> {expected_supercell_atoms} atoms")
                try:
                    from pymatgen.transformations.standard_transformations import SupercellTransformation
                    
                    # 元の単位格子構造を取得（CifParserが自動的に縮約した結果）
                    unit_cell = structure
                    
                    # スーパーセル変換を再適用
                    supercell_matrix = [[a_mult, 0, 0], [0, b_mult, 0], [0, 0, c_mult]]
                    transformation = SupercellTransformation(supercell_matrix)
                    structure = transformation.apply_transformation(unit_cell)
                    
                    logger.info(f"Reconstructed supercell for deletion: {len(structure.sites)} atoms, a={structure.lattice.a:.6f}")
                    
                except Exception as e:
                    logger.error(f"Failed to reconstruct supercell for deletion: {e}")
                    # フォールバック：元の構造を使用
        
        # 削除する原子のインデックスを検証
        if not all(isinstance(i, int) and 0 <= i < len(structure.sites) for i in atom_indices):
            return jsonify({
                "success": False,
                "error": f"Invalid atom indices. Must be integers between 0 and {len(structure.sites)-1}",
                "error_type": "invalid_indices"
            }), 400
        
        # 新しい構造を作成（削除対象以外の原子のみ）
        remaining_sites = []
        for i, site in enumerate(structure.sites):
            if i not in atom_indices:
                remaining_sites.append(site)
        
        if len(remaining_sites) == 0:
            return jsonify({
                "success": False,
                "error": "Cannot delete all atoms",
                "error_type": "invalid_operation"
            }), 400
        
        # 新しい構造を作成
        from pymatgen.core.structure import Structure
        modified_structure = Structure(
            lattice=structure.lattice,
            species=[site.specie for site in remaining_sites],
            coords=[site.coords for site in remaining_sites],
            coords_are_cartesian=True
        )
        
        # 新しいCIF文字列を生成（スーパーセル保持対応）
        from pymatgen.io.cif import CifWriter
        
        if supercell_metadata and supercell_metadata.get('is_supercell'):
            logger.info("Generating supercell-preserving CIF for deletion")
            modified_cif = generate_custom_supercell_cif(modified_structure, supercell_metadata)
        else:
            writer = CifWriter(modified_structure)
            modified_cif = str(writer)
        
        logger.info(f"Deleted {len(atom_indices)} atoms, {len(remaining_sites)} atoms remaining")
        
        # 編集後の構造情報を解析
        modified_structure_info = {"success": True}
        
        # スーパーセル情報が提供されている場合は、それを参考に解析
        if supercell_metadata:
            logger.info("Using supercell metadata for analysis")
            # スーパーセル情報を基に修正された構造の解析を行う
            try:
                modified_structure_info.update({
                    "is_supercell": True,
                    "supercell_multipliers": supercell_metadata.get('multipliers', {}),
                    "original_unit_cell_atoms": supercell_metadata.get('original_atoms', 0)
                })
            except Exception as e:
                logger.warning(f"Could not apply supercell metadata: {e}")
        
        # 修正された構造の詳細情報を取得
        try:
            logger.info(f"Analyzing modified structure after deletion with {len(modified_structure.sites)} sites")
            
            # 基本的な構造情報
            modified_structure_info.update({
                "formula": str(modified_structure.composition.reduced_formula),
                "atom_count": len(modified_structure.sites)
            })
            
            # 格子定数の取得
            try:
                lattice = modified_structure.lattice
                modified_structure_info["lattice_parameters"] = {
                    "a": round(lattice.a, 6),
                    "b": round(lattice.b, 6),
                    "c": round(lattice.c, 6),
                    "alpha": round(lattice.alpha, 3),
                    "beta": round(lattice.beta, 3),
                    "gamma": round(lattice.gamma, 3)
                }
                modified_structure_info["volume"] = round(lattice.volume, 6)
                logger.info(f"Lattice parameters extracted: a={lattice.a:.4f}")
            except Exception as e:
                logger.error(f"Failed to extract lattice parameters: {e}")
                modified_structure_info["lattice_parameters"] = None
                modified_structure_info["volume"] = None
            
            # 原子情報の生成（修正後の構造）
            try:
                atom_info = []
                element_counts = {}
                
                logger.info(f"Starting atom info generation for modified_structure with {len(modified_structure.sites)} sites")
                
                for i, site in enumerate(modified_structure.sites):
                    element = str(site.specie)
                    if element not in element_counts:
                        element_counts[element] = 0
                    
                    # 一意な原子ラベル生成（スーパーセル対応）
                    label = f"{site.specie}{element_counts[element]}"
                    element_counts[element] += 1
                    
                    atom_data = {
                        "index": i,
                        "element": element,
                        "label": label,
                        "coords": [round(coord, 6) for coord in site.coords],
                        "frac_coords": [round(coord, 6) for coord in site.frac_coords]
                    }
                    atom_info.append(atom_data)
                    
                    if i < 3:  # 最初の3個の原子をログに出力
                        logger.info(f"  Atom {i}: {label} at {atom_data['coords']}")
                
                modified_structure_info["atom_info"] = atom_info
                logger.info(f"Successfully generated atom info for {len(atom_info)} atoms after deletion")
                
            except Exception as e:
                logger.error(f"Failed to generate atom info: {e}")
                logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                modified_structure_info["atom_info"] = []
            
            # 対称性解析（構造タイプに応じた分岐処理 - delete_atoms）
            try:
                if supercell_metadata and supercell_metadata.get('is_supercell'):
                    # スーパーセルの場合：厳密な対称性解析
                    analyzer = SpacegroupAnalyzer(modified_structure, symprec=1e-6, angle_tolerance=1)
                    analysis_type = "厳密解析（スーパーセル）"
                    logger.info("Using strict symmetry analysis for supercell structure")
                    
                    # スーパーセル関連情報を保持
                    modified_structure_info["is_supercell"] = True
                    modified_structure_info["supercell_multipliers"] = supercell_metadata.get('multipliers', {})
                    modified_structure_info["original_unit_cell_atoms"] = supercell_metadata.get('original_atoms', 0)
                else:
                    # 通常構造の場合：標準パラメータで元の対称性に応じた解析
                    analyzer = SpacegroupAnalyzer(modified_structure)
                    analysis_type = "標準解析（通常構造）"
                    logger.info("Using standard symmetry analysis for normal structure")
                    
                    modified_structure_info["is_supercell"] = False
                
                modified_structure_info["crystal_system"] = analyzer.get_crystal_system()
                
                # SpacegroupAnalyzerから直接取得
                space_group_symbol = analyzer.get_space_group_symbol()
                space_group_number = analyzer.get_space_group_number()
                modified_structure_info["space_group"] = f"{space_group_symbol} (#{space_group_number})"
                
                logger.info(f"Symmetry analysis (deletion) completed: {modified_structure_info['space_group']} [{analysis_type}]")
            except Exception as e:
                logger.error(f"Failed to analyze symmetry: {e}")
                modified_structure_info["crystal_system"] = "Unknown"
                modified_structure_info["space_group"] = "Unknown"
                
        except Exception as e:
            logger.error(f"Critical error in deletion structure analysis: {e}")
            modified_structure_info.update({
                "formula": "Unknown",
                "atom_count": 0,
                "lattice_parameters": None,
                "volume": None,
                "crystal_system": "Unknown", 
                "space_group": "Unknown"
            })
        
        return jsonify({
            "success": True,
            "modified_cif": modified_cif,
            "original_atom_count": len(structure.sites),
            "modified_atom_count": len(remaining_sites),
            "deleted_indices": atom_indices,
            "modified_structure_info": modified_structure_info
        })
        
    except Exception as e:
        logger.error(f"Atom deletion error: {e}")
        return jsonify({
            "error": f"Failed to delete atoms: {str(e)}",
            "success": False,
            "error_type": "deletion_error"
        }), 500

@app.route('/replace_atoms', methods=['POST'])
def replace_atoms():
    """Replace specified atoms with new element"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False,
                "error_type": "invalid_content_type"
            }), 400
        
        data = request.get_json()
        if not data or 'cif_content' not in data or 'atom_indices' not in data or 'new_element' not in data:
            return jsonify({
                "error": "Missing 'cif_content', 'atom_indices', or 'new_element' in request body",
                "success": False,
                "error_type": "missing_parameter"
            }), 400
        
        cif_content = data['cif_content']
        atom_indices = data['atom_indices']
        new_element = data['new_element']
        
        # スーパーセル情報を保持するためのメタデータ（オプション）
        supercell_metadata = data.get('supercell_metadata', None)
        
        if not PYMATGEN_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }), 500
        
        # 元の構造を解析
        parser = CifParser(StringIO(cif_content))
        structures = parser.get_structures()
        
        if not structures:
            return jsonify({
                "success": False,
                "error": "No valid crystal structures found in CIF file",
                "error_type": "parsing_error"
            }), 400
        
        structure = structures[0]
        
        # スーパーセルメタデータがある場合、構造を再構築
        if supercell_metadata and supercell_metadata.get('is_supercell'):
            # 期待されるスーパーセル原子数を計算
            original_atoms = supercell_metadata.get('original_atoms', 0)
            multipliers = supercell_metadata.get('multipliers', {})
            a_mult = multipliers.get('a', 1)
            b_mult = multipliers.get('b', 1) 
            c_mult = multipliers.get('c', 1)
            expected_supercell_atoms = original_atoms * a_mult * b_mult * c_mult
            
            current_atoms = len(structure.sites)
            
            # 既にスーパーセル状態かどうかを判定
            if current_atoms == expected_supercell_atoms:
                logger.info(f"Structure already is supercell: {current_atoms} atoms (expected: {expected_supercell_atoms})")
                # 既にスーパーセル状態なので再構築不要
            else:
                logger.info(f"Reconstructing supercell structure from metadata: {current_atoms} -> {expected_supercell_atoms} atoms")
                try:
                    from pymatgen.transformations.standard_transformations import SupercellTransformation
                    
                    # 元の単位格子構造を取得（CifParserが自動的に縮約した結果）
                    unit_cell = structure
                    
                    # スーパーセル変換を再適用
                    supercell_matrix = [[a_mult, 0, 0], [0, b_mult, 0], [0, 0, c_mult]]
                    transformation = SupercellTransformation(supercell_matrix)
                    structure = transformation.apply_transformation(unit_cell)
                    
                    logger.info(f"Reconstructed supercell: {len(structure.sites)} atoms, a={structure.lattice.a:.6f}")
                    
                except Exception as e:
                    logger.error(f"Failed to reconstruct supercell: {e}")
                    # フォールバック：元の構造を使用
        
        # 置換する原子のインデックスを検証
        if not all(isinstance(i, int) and 0 <= i < len(structure.sites) for i in atom_indices):
            return jsonify({
                "success": False,
                "error": f"Invalid atom indices. Must be integers between 0 and {len(structure.sites)-1}",
                "error_type": "invalid_indices"
            }), 400
        
        # 新しい元素を検証
        try:
            from pymatgen.core.periodic_table import Element
            new_element_obj = Element(new_element)
        except Exception:
            return jsonify({
                "success": False,
                "error": f"Invalid element symbol: {new_element}",
                "error_type": "invalid_element"
            }), 400
        
        # 構造をコピーして置換
        modified_structure = structure.copy()
        
        for index in atom_indices:
            original_site = structure.sites[index]
            # 同じ座標で元素のみ置換
            modified_structure.replace(index, new_element_obj, coords=original_site.coords, coords_are_cartesian=True)
        
        # 新しいCIF文字列を生成（スーパーセル保持対応）
        from pymatgen.io.cif import CifWriter
        
        if supercell_metadata and supercell_metadata.get('is_supercell'):
            logger.info("Generating supercell-preserving CIF for replacement")
            modified_cif = generate_custom_supercell_cif(modified_structure, supercell_metadata)
        else:
            writer = CifWriter(modified_structure)
            modified_cif = str(writer)
        
        logger.info(f"Replaced {len(atom_indices)} atoms with {new_element}")
        
        # 編集後の構造情報を解析
        modified_structure_info = {"success": True}
        
        # スーパーセル情報が提供されている場合は、それを参考に解析
        if supercell_metadata:
            logger.info("Using supercell metadata for replacement analysis")
            try:
                modified_structure_info.update({
                    "is_supercell": True,
                    "supercell_multipliers": supercell_metadata.get('multipliers', {}),
                    "original_unit_cell_atoms": supercell_metadata.get('original_atoms', 0)
                })
            except Exception as e:
                logger.warning(f"Could not apply supercell metadata: {e}")
        
        # 修正された構造の詳細情報を取得
        try:
            logger.info(f"Analyzing modified structure after replacement with {len(modified_structure.sites)} sites")
            
            # 基本的な構造情報
            modified_structure_info.update({
                "formula": str(modified_structure.composition.reduced_formula),
                "atom_count": len(modified_structure.sites)
            })
            
            # 格子定数の取得
            try:
                lattice = modified_structure.lattice
                modified_structure_info["lattice_parameters"] = {
                    "a": round(lattice.a, 6),
                    "b": round(lattice.b, 6),
                    "c": round(lattice.c, 6),
                    "alpha": round(lattice.alpha, 3),
                    "beta": round(lattice.beta, 3),
                    "gamma": round(lattice.gamma, 3)
                }
                modified_structure_info["volume"] = round(lattice.volume, 6)
                logger.info(f"Lattice parameters extracted: a={lattice.a:.4f}")
            except Exception as e:
                logger.error(f"Failed to extract lattice parameters: {e}")
                modified_structure_info["lattice_parameters"] = None
                modified_structure_info["volume"] = None
            
            # 原子情報の生成（修正後の構造）
            try:
                atom_info = []
                element_counts = {}
                
                logger.info(f"Starting atom info generation for modified_structure with {len(modified_structure.sites)} sites")
                
                for i, site in enumerate(modified_structure.sites):
                    element = str(site.specie)
                    if element not in element_counts:
                        element_counts[element] = 0
                    
                    # 一意な原子ラベル生成（スーパーセル対応）
                    label = f"{site.specie}{element_counts[element]}"
                    element_counts[element] += 1
                    
                    atom_data = {
                        "index": i,
                        "element": element,
                        "label": label,
                        "coords": [round(coord, 6) for coord in site.coords],
                        "frac_coords": [round(coord, 6) for coord in site.frac_coords]
                    }
                    atom_info.append(atom_data)
                    
                    if i < 3:  # 最初の3個の原子をログに出力
                        logger.info(f"  Atom {i}: {label} at {atom_data['coords']}")
                
                modified_structure_info["atom_info"] = atom_info
                logger.info(f"Successfully generated atom info for {len(atom_info)} atoms after replacement")
                
            except Exception as e:
                logger.error(f"Failed to generate atom info: {e}")
                logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                modified_structure_info["atom_info"] = []
            
            # 対称性解析（構造タイプに応じた分岐処理 - replace_atoms）
            try:
                if supercell_metadata and supercell_metadata.get('is_supercell'):
                    # スーパーセルの場合：厳密な対称性解析
                    analyzer = SpacegroupAnalyzer(modified_structure, symprec=1e-6, angle_tolerance=1)
                    analysis_type = "厳密解析（スーパーセル）"
                    logger.info("Using strict symmetry analysis for supercell structure")
                    
                    # スーパーセル関連情報を保持
                    modified_structure_info["is_supercell"] = True
                    modified_structure_info["supercell_multipliers"] = supercell_metadata.get('multipliers', {})
                    modified_structure_info["original_unit_cell_atoms"] = supercell_metadata.get('original_atoms', 0)
                else:
                    # 通常構造の場合：標準パラメータで元の対称性に応じた解析
                    analyzer = SpacegroupAnalyzer(modified_structure)
                    analysis_type = "標準解析（通常構造）"
                    logger.info("Using standard symmetry analysis for normal structure")
                    
                    modified_structure_info["is_supercell"] = False
                
                modified_structure_info["crystal_system"] = analyzer.get_crystal_system()
                
                # SpacegroupAnalyzerから直接取得
                space_group_symbol = analyzer.get_space_group_symbol()
                space_group_number = analyzer.get_space_group_number()
                modified_structure_info["space_group"] = f"{space_group_symbol} (#{space_group_number})"
                
                logger.info(f"Symmetry analysis (replacement) completed: {modified_structure_info['space_group']} [{analysis_type}]")
            except Exception as e:
                logger.error(f"Failed to analyze symmetry: {e}")
                modified_structure_info["crystal_system"] = "Unknown"
                modified_structure_info["space_group"] = "Unknown"
                
        except Exception as e:
            logger.error(f"Critical error in replacement structure analysis: {e}")
            modified_structure_info.update({
                "formula": "Unknown",
                "atom_count": 0,
                "lattice_parameters": None,
                "volume": None,
                "crystal_system": "Unknown", 
                "space_group": "Unknown"
            })
        
        return jsonify({
            "success": True,
            "modified_cif": modified_cif,
            "original_atom_count": len(structure.sites),
            "modified_atom_count": len(modified_structure.sites),
            "replaced_indices": atom_indices,
            "new_element": new_element,
            "modified_structure_info": modified_structure_info
        })
        
    except Exception as e:
        logger.error(f"Atom replacement error: {e}")
        return jsonify({
            "error": f"Failed to replace atoms: {str(e)}",
            "success": False,
            "error_type": "replacement_error"
        }), 500

@app.route('/cache_info', methods=['GET'])
def cache_info():
    """Cache statistics endpoint"""
    return jsonify({
        "cache_size": len(cif_cache),
        "cache_limit": CACHE_SIZE,
        "cache_expiry_seconds": CACHE_EXPIRY,
        "cached_items": [
            {
                "key": key[:8] + "...",
                "age_seconds": int(time.time() - cache_timestamps[key])
            }
            for key in cache_timestamps.keys()
        ]
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "error": f"File too large. Maximum size: {MAX_CONTENT_LENGTH} bytes",
        "success": False,
        "error_type": "file_too_large"
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "success": False,
        "error_type": "not_found"
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "error": "Internal server error",
        "success": False,
        "error_type": "internal_error"
    }), 500

if __name__ == '__main__':
    if not PYMATGEN_AVAILABLE:
        logger.warning("Pymatgen not available. CIF parsing will be limited.")
    
@app.route('/parse_cif_enhanced', methods=['POST'])
def parse_cif_enhanced():
    """Enhanced CIF parsing with multiple Pymatgen methods and fallbacks"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False,
                "error_type": "invalid_content_type"
            }), 400
        
        data = request.get_json()
        if not data or 'cif_content' not in data:
            return jsonify({
                "error": "Missing 'cif_content' in request body",
                "success": False,
                "error_type": "missing_parameter"
            }), 400
        
        cif_content = data['cif_content']
        logger.info(f"Enhanced CIF parsing requested (length: {len(cif_content)})")
        
        if not PYMATGEN_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }), 500
        
        # Method 1: Standard CifParser
        try:
            parser = CifParser(StringIO(cif_content))
            structures = parser.get_structures()
            if structures:
                structure = structures[0]
                result = extract_structure_info_enhanced(structure, "CifParser")
                if result.get("success"):
                    logger.info("Enhanced parsing successful with CifParser")
                    return jsonify(result)
        except Exception as e:
            logger.warning(f"CifParser method failed: {e}")
        
        # Method 2: Structure.from_str()
        try:
            from pymatgen.core.structure import Structure
            structure = Structure.from_str(cif_content, fmt="cif")
            result = extract_structure_info_enhanced(structure, "Structure.from_str")
            if result.get("success"):
                logger.info("Enhanced parsing successful with Structure.from_str")
                return jsonify(result)
        except Exception as e:
            logger.warning(f"Structure.from_str method failed: {e}")
        
        # Method 3: Manual CifParser with different settings
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp_file:
                tmp_file.write(cif_content)
                tmp_file.flush()
                parser = CifParser(tmp_file.name, occupancy_tolerance=1.0)
                structures = parser.get_structures(primitive=False)
                if structures:
                    structure = structures[0]
                    result = extract_structure_info_enhanced(structure, "CifParser_relaxed")
                    if result.get("success"):
                        logger.info("Enhanced parsing successful with relaxed CifParser")
                        os.unlink(tmp_file.name)
                        return jsonify(result)
                os.unlink(tmp_file.name)
        except Exception as e:
            logger.warning(f"Relaxed CifParser method failed: {e}")
        
        # All Pymatgen methods failed
        return jsonify({
            "success": False,
            "error": "All Pymatgen parsing methods failed",
            "error_type": "parsing_error"
        }), 500
        
    except Exception as e:
        logger.error(f"Enhanced CIF parsing failed: {e}")
        return jsonify({
            "error": f"Enhanced parsing failed: {str(e)}",
            "success": False,
            "error_type": "server_error"
        }), 500

def extract_structure_info_enhanced(structure, method):
    """Extract structure information with enhanced error handling"""
    result = {"success": True, "method": method}
    
    try:
        # Basic information
        result["formula"] = str(structure.composition.reduced_formula)
        result["atom_count"] = len(structure.sites)
        
        # Lattice parameters
        lattice = structure.lattice
        result["lattice_parameters"] = {
            "a": round(lattice.a, 6),
            "b": round(lattice.b, 6), 
            "c": round(lattice.c, 6),
            "alpha": round(lattice.alpha, 3),
            "beta": round(lattice.beta, 3),
            "gamma": round(lattice.gamma, 3)
        }
        result["volume"] = round(lattice.volume, 6)
        
        # Space group with multiple attempts
        try:
            spacegroup_info = structure.get_space_group_info()
            if len(spacegroup_info) > 1:
                result["space_group"] = f"{spacegroup_info[1]} (#{spacegroup_info[0]})"
            else:
                result["space_group"] = "Unknown"
        except:
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
                sg_number = analyzer.get_space_group_number()
                sg_symbol = analyzer.get_space_group_symbol()
                result["space_group"] = f"{sg_symbol} (#{sg_number})"
            except:
                result["space_group"] = "Unknown"
        
        # Crystal system
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            result["crystal_system"] = analyzer.get_crystal_system()
        except:
            result["crystal_system"] = "Unknown"
        
        # Atom information
        atom_info = []
        element_counts = {}
        for i, site in enumerate(structure.sites):
            element = str(site.specie)
            element_counts[element] = element_counts.get(element, 0) + 1
            
            atom_data = {
                "index": i,
                "element": element,
                "label": f"{element}{element_counts[element]}",
                "coords": [round(coord, 6) for coord in site.coords],
                "frac_coords": [round(coord, 6) for coord in site.frac_coords]
            }
            atom_info.append(atom_data)
        
        result["atom_info"] = atom_info
        
        logger.info(f"Enhanced extraction successful: {result['formula']}, {result['atom_count']} atoms")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced structure info extraction failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Improved CIF Parser Server Starting...")
    print("Available endpoints:")
    print("  GET  /health         - Health check with dependency info")
    print("  GET  /test_cif       - Test with BaTiO3.cif")
    print("  GET  /cache_info     - Cache statistics")
    print("  POST /parse_cif      - Parse CIF content")
    print("  POST /parse_cif_enhanced - Enhanced CIF parsing with multiple methods")
    print("  POST /create_supercell - Create supercell structure")
    print("  POST /delete_atoms   - Delete specified atoms")
    print("  POST /replace_atoms  - Replace atoms with new element")
    print("  GET  /               - Serve index.html")
    print("  GET  /<filename>     - Serve static files")
    print(f"Security: CORS limited to {allowed_origins}")
    print(f"Cache: {CACHE_SIZE} items, {CACHE_EXPIRY}s expiry")
    print(f"Max file size: {MAX_CONTENT_LENGTH} bytes")
    
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE)