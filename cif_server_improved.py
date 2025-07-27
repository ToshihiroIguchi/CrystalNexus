#!/usr/bin/env python3
"""
CrystalNexus Server - Crystal Structure Analysis Platform
Advanced CIF parsing with security, caching, and comprehensive error handling
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback
import logging
import hashlib
import time
from functools import lru_cache
from typing import Dict, Any, Optional
import warnings

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
SERVER_HOST = os.getenv('SERVER_HOST', '127.0.0.1')
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

# Secure CORS configuration - Dynamic port support
allowed_origins = [
    f'http://{SERVER_HOST}:{SERVER_PORT}',
    f'https://{SERVER_HOST}:{SERVER_PORT}',
    f'http://localhost:{SERVER_PORT}',
    f'http://127.0.0.1:{SERVER_PORT}',
    'http://localhost:5000',  # Keep legacy support
    'http://127.0.0.1:5000'   # Keep legacy support
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
        parser = CifParser.from_str(cif_content)
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
            result["space_group"] = spacegroup_info[1] if len(spacegroup_info) > 1 else "Unknown"
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
            for site in structure.sites:
                element = str(site.specie)
                element_counts[element] = element_counts.get(element, 0) + 1
            result["element_counts"] = element_counts
        except Exception as e:
            logger.warning(f"Could not count atoms: {e}")
            result["atom_count"] = 0
            result["element_counts"] = {}
        
        try:
            result["density"] = round(structure.density, 3)
        except Exception as e:
            logger.warning(f"Could not calculate density: {e}")
            result["density"] = None
        
        # Extract atom information for atom selection (same as supercell creation)
        try:
            atom_info = []
            element_counts = {}
            
            for i, site in enumerate(structure.sites):
                element = str(site.specie.symbol)
                
                # Count occurrences to generate unique labels like Ba0, Ba1, Ti0, etc.
                if element not in element_counts:
                    element_counts[element] = 0
                
                label = f"{element}{element_counts[element]}"
                element_counts[element] += 1
                
                atom_info.append({
                    "index": i,
                    "type_symbol": element,
                    "label": label,
                    "fract_x": round(site.frac_coords[0], 6),
                    "fract_y": round(site.frac_coords[1], 6),
                    "fract_z": round(site.frac_coords[2], 6),
                    "occupancy": 1.0,
                    "multiplicity": 1
                })
            
            result["atom_info"] = atom_info
            logger.info(f"Generated {len(atom_info)} atom entries for atom selection")
            
        except Exception as e:
            logger.warning(f"Could not extract atom info: {e}")
            result["atom_info"] = []
        
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

@app.route('/sample_files', methods=['GET'])
def get_sample_files():
    """Get list of available sample CIF files"""
    try:
        sample_dir = 'sample_cif'
        if not os.path.exists(sample_dir):
            return jsonify({
                "error": "Sample directory not found",
                "success": False,
                "error_type": "directory_not_found"
            }), 404
        
        sample_files = []
        for filename in os.listdir(sample_dir):
            if filename.lower().endswith('.cif') and not filename.endswith('.cif:Zone.Identifier'):
                file_path = os.path.join(sample_dir, filename)
                try:
                    # Get file info
                    stat_info = os.stat(file_path)
                    
                    # Basic file information
                    file_info = {
                        "filename": filename,
                        "display_name": filename.replace('.cif', ''),
                        "path": f"sample_cif/{filename}",
                        "size": stat_info.st_size,
                        "modified": stat_info.st_mtime
                    }
                    
                    # Try to get basic crystal info
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract basic info from CIF
                        if '_chemical_formula_sum' in content:
                            import re
                            formula_match = re.search(r'_chemical_formula_sum\s+[\'"]?([^\n\'\"]+)', content)
                            if formula_match:
                                file_info["formula"] = formula_match.group(1).strip()
                        
                        if '_space_group_name_H-M_alt' in content:
                            import re
                            sg_match = re.search(r'_space_group_name_H-M_alt\s+[\'"]?([^\n\'\"]+)', content)
                            if sg_match:
                                file_info["space_group"] = sg_match.group(1).strip()
                                
                        file_info["description"] = f"Crystal structure: {file_info.get('formula', 'Unknown formula')}"
                        
                    except Exception as e:
                        logger.warning(f"Could not extract info from {filename}: {e}")
                        file_info["description"] = "Crystal structure file"
                    
                    sample_files.append(file_info)
                    
                except Exception as e:
                    logger.warning(f"Could not process file {filename}: {e}")
                    continue
        
        # Sort by filename
        sample_files.sort(key=lambda x: x['filename'])
        
        return jsonify({
            "success": True,
            "files": sample_files,
            "count": len(sample_files)
        })
        
    except Exception as e:
        logger.error(f"Error getting sample files: {e}")
        return jsonify({
            "error": f"Failed to get sample files: {str(e)}",
            "success": False,
            "error_type": "server_error"
        }), 500

@app.route('/sample_cif/<filename>', methods=['GET'])
def serve_sample_cif(filename):
    """Serve sample CIF files with security checks"""
    try:
        # Security: prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        # Only allow .cif files
        if not filename.lower().endswith('.cif'):
            return jsonify({"error": "Only CIF files are allowed"}), 400
        
        file_path = os.path.join('sample_cif', filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "Sample file not found"}), 404
        
        # Read and return file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
    except Exception as e:
        logger.error(f"Error serving sample CIF {filename}: {e}")
        return jsonify({
            "error": f"Failed to serve sample file: {str(e)}",
            "success": False,
            "error_type": "server_error"
        }), 500

@app.route('/create_supercell', methods=['POST'])
def create_supercell():
    """Create supercell from CIF content with enhanced error handling"""
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
        
        logger.info(f"Creating supercell with multipliers: {a_mult}x{b_mult}x{c_mult}")
        
        # Parse original structure
        try:
            parser = CifParser.from_str(cif_content)
            structures = parser.get_structures()
            
            if not structures:
                return jsonify({
                    "success": False,
                    "error": "No valid crystal structures found in CIF",
                    "error_type": "parsing_error"
                }), 400
                
            structure = structures[0]
            logger.info(f"Original structure: {structure.formula}")
            
        except Exception as e:
            logger.error(f"CIF parsing failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Failed to parse CIF: {str(e)}",
                "error_type": "parsing_error"
            }), 400
        
        # Create supercell
        try:
            supercell = structure.make_supercell([a_mult, b_mult, c_mult])
            logger.info(f"Supercell created: {supercell.formula}")
            
            # Convert back to CIF
            from pymatgen.io.cif import CifWriter
            cif_writer = CifWriter(supercell)
            supercell_cif = str(cif_writer)
            
            # Calculate supercell properties
            result = {
                "success": True,
                "supercell_cif": supercell_cif,
                "multipliers": {
                    "a": a_mult,
                    "b": b_mult, 
                    "c": c_mult
                },
                "original_info": {
                    "formula": str(structure.composition.reduced_formula),
                    "atom_count": len(structure.sites),
                    "volume": round(structure.lattice.volume, 6)
                },
                "supercell_info": {
                    "formula": str(supercell.composition.reduced_formula),
                    "atom_count": len(supercell.sites),
                    "volume": round(supercell.lattice.volume, 6),
                    "volume_ratio": round(supercell.lattice.volume / structure.lattice.volume, 3)
                }
            }
            
            # Add lattice parameters if available
            try:
                result["supercell_info"]["lattice_parameters"] = {
                    "a": round(supercell.lattice.a, 6),
                    "b": round(supercell.lattice.b, 6),
                    "c": round(supercell.lattice.c, 6),
                    "alpha": round(supercell.lattice.alpha, 3),
                    "beta": round(supercell.lattice.beta, 3),
                    "gamma": round(supercell.lattice.gamma, 3)
                }
            except Exception as e:
                logger.warning(f"Could not extract supercell lattice parameters: {e}")
            
            # Extract atom information for supercell for atom selection
            try:
                atom_info = []
                element_counts = {}
                
                for i, site in enumerate(supercell.sites):
                    element = str(site.specie.symbol)
                    
                    # Count occurrences to generate unique labels like Ba0, Ba1, Ti0, etc.
                    if element not in element_counts:
                        element_counts[element] = 0
                    
                    label = f"{element}{element_counts[element]}"
                    element_counts[element] += 1
                    
                    atom_info.append({
                        "index": i,
                        "type_symbol": element,
                        "label": label,
                        "fract_x": round(site.frac_coords[0], 6),
                        "fract_y": round(site.frac_coords[1], 6),
                        "fract_z": round(site.frac_coords[2], 6),
                        "occupancy": 1.0,
                        "multiplicity": 1
                    })
                
                result["supercell_info"]["atom_info"] = atom_info
                logger.info(f"Generated {len(atom_info)} atom entries for supercell atom selection")
                
            except Exception as e:
                logger.warning(f"Could not extract supercell atom info: {e}")
                result["supercell_info"]["atom_info"] = []
            
            logger.info(f"Supercell creation completed successfully")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Supercell creation failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Failed to create supercell: {str(e)}",
                "error_type": "supercell_creation_error"
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in create_supercell: {e}")
        if DEBUG_MODE:
            logger.error(traceback.format_exc())
        
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "success": False,
            "error_type": "server_error",
            "traceback": traceback.format_exc() if DEBUG_MODE else None
        }), 500

@app.route('/delete_atoms', methods=['POST'])
def delete_atoms():
    """Delete specific atoms from structure"""
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
        atom_indices = data['atom_indices']
        
        if not PYMATGEN_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }), 500
        
        # Parse structure
        parser = CifParser.from_str(cif_content)
        structures = parser.get_structures()
        
        if not structures:
            return jsonify({
                "success": False,
                "error": "No valid crystal structures found in CIF",
                "error_type": "parsing_error"
            }), 400
        
        structure = structures[0]
        original_atom_count = len(structure)
        
        # Create new structure without specified atoms
        sites_to_keep = []
        for i, site in enumerate(structure):
            if i not in atom_indices:
                sites_to_keep.append(site)
        
        if not sites_to_keep:
            return jsonify({
                "success": False,
                "error": "Cannot delete all atoms",
                "error_type": "invalid_operation"
            }), 400
        
        # Create new structure
        from pymatgen.core.structure import Structure
        new_structure = Structure(
            lattice=structure.lattice,
            species=[site.specie for site in sites_to_keep],
            coords=[site.frac_coords for site in sites_to_keep]
        )
        
        # Convert back to CIF with proper formatting
        from pymatgen.io.cif import CifWriter
        cif_writer = CifWriter(new_structure, symprec=None, write_magmoms=False)
        modified_cif = str(cif_writer)
        
        # Ensure unit cell information is preserved
        if "loop_" not in modified_cif:
            # Add proper CIF headers if missing
            lattice = new_structure.lattice
            cif_header = f"""# Modified structure - atoms deleted
data_modified_structure
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   {lattice.a:.6f}
_cell_length_b   {lattice.b:.6f}
_cell_length_c   {lattice.c:.6f}
_cell_angle_alpha   {lattice.alpha:.6f}
_cell_angle_beta   {lattice.beta:.6f}
_cell_angle_gamma   {lattice.gamma:.6f}
_symmetry_Int_Tables_number   1
_chemical_formula_structural   {new_structure.composition.reduced_formula}
_cell_volume   {lattice.volume:.6f}
_cell_formula_units_Z   1

"""
            modified_cif = cif_header + modified_cif.split('\n', 2)[-1] if '\n' in modified_cif else cif_header + modified_cif
        
        return jsonify({
            "success": True,
            "modified_cif": modified_cif,
            "original_atom_count": original_atom_count,
            "modified_atom_count": len(new_structure),
            "deleted_atoms": len(atom_indices)
        })
        
    except Exception as e:
        logger.error(f"Error deleting atoms: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Failed to delete atoms: {str(e)}",
            "error_type": "processing_error"
        }), 500

@app.route('/replace_atoms', methods=['POST'])
def replace_atoms():
    """Replace specific atoms with new element"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "success": False,
                "error_type": "invalid_content_type"
            }), 400
        
        data = request.get_json()
        required_fields = ['cif_content', 'atom_indices', 'new_element']
        if not data or not all(field in data for field in required_fields):
            return jsonify({
                "error": f"Missing required fields: {required_fields}",
                "success": False,
                "error_type": "missing_parameter"
            }), 400
        
        cif_content = data['cif_content']
        atom_indices = data['atom_indices']
        new_element = data['new_element']
        supercell_metadata = data.get('supercell_metadata')  # スーパーセル情報を取得
        
        logger.info(f"Replacing atoms with element: {new_element}")
        logger.info(f"Target atom indices: {atom_indices}")
        if supercell_metadata:
            logger.info(f"Supercell metadata received: is_supercell={supercell_metadata.get('is_supercell')}")
            logger.info(f"Supercell multipliers: {supercell_metadata.get('multipliers')}")
        
        if not PYMATGEN_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Pymatgen library not available",
                "error_type": "dependency_error"
            }), 500
        
        # Parse structure
        parser = CifParser.from_str(cif_content)
        structures = parser.get_structures()
        
        if not structures:
            return jsonify({
                "success": False,
                "error": "No valid crystal structures found in CIF",
                "error_type": "parsing_error"
            }), 400
        
        structure = structures[0]
        original_atom_count = len(structure)
        
        logger.info(f"Parsed structure: {structure.composition.reduced_formula}")
        logger.info(f"Structure has {original_atom_count} atoms")
        
        # スーパーセルの場合の特別処理
        if supercell_metadata and supercell_metadata.get('is_supercell'):
            logger.warning("CRITICAL: CifParser縮約問題 - スーパーセルが単位格子に縮約されました")
            logger.warning(f"期待されるスーパーセル原子数: {supercell_metadata.get('supercell_info', {}).get('atom_count', 'unknown')}")
            logger.warning(f"実際に解析された原子数: {original_atom_count}")
        
        # スーパーセルの場合は縮約を無効化してパースし直し
        if supercell_metadata and supercell_metadata.get('is_supercell') and original_atom_count < 20:
            logger.info("Re-parsing with primitive=False to preserve supercell structure")
            parser = CifParser.from_str(cif_content)
            structures = parser.get_structures(primitive=False)
            if structures:
                structure = structures[0]
                original_atom_count = len(structure)
                logger.info(f"After disabling primitive reduction: {original_atom_count} atoms")
        
        # Replace atoms
        from pymatgen.core.periodic_table import Element
        try:
            new_elem = Element(new_element)
        except:
            return jsonify({
                "success": False,
                "error": f"Invalid element symbol: {new_element}",
                "error_type": "invalid_element"
            }), 400
        
        # Create new structure with replaced atoms
        new_species = []
        for i, site in enumerate(structure):
            if i in atom_indices:
                new_species.append(new_elem)
            else:
                new_species.append(site.specie)
        
        new_structure = Structure(
            lattice=structure.lattice,
            species=new_species,
            coords=[site.frac_coords for site in structure]
        )
        
        # Convert back to CIF with proper formatting
        from pymatgen.io.cif import CifWriter
        cif_writer = CifWriter(new_structure, symprec=None, write_magmoms=False)
        modified_cif = str(cif_writer)
        
        # Ensure unit cell information is preserved
        if "loop_" not in modified_cif:
            # Add proper CIF headers if missing
            lattice = new_structure.lattice
            cif_header = f"""# Modified structure - atoms replaced
data_modified_structure
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   {lattice.a:.6f}
_cell_length_b   {lattice.b:.6f}
_cell_length_c   {lattice.c:.6f}
_cell_angle_alpha   {lattice.alpha:.6f}
_cell_angle_beta   {lattice.beta:.6f}
_cell_angle_gamma   {lattice.gamma:.6f}
_symmetry_Int_Tables_number   1
_chemical_formula_structural   {new_structure.composition.reduced_formula}
_cell_volume   {lattice.volume:.6f}
_cell_formula_units_Z   1

"""
            modified_cif = cif_header + modified_cif.split('\n', 2)[-1] if '\n' in modified_cif else cif_header + modified_cif
        
        # レスポンス構築
        response_data = {
            "success": True,
            "modified_cif": modified_cif,
            "original_atom_count": original_atom_count,
            "modified_atom_count": len(new_structure),
            "replaced_atoms": len(atom_indices),
            "new_element": new_element
        }
        
        # スーパーセル情報がある場合は保持
        if supercell_metadata and supercell_metadata.get('is_supercell'):
            logger.info("Preserving supercell metadata in response")
            
            # 置換後の原子情報を生成（スーパーセル用）
            atom_info = []
            element_counts = {}
            
            for i, site in enumerate(new_structure.sites):
                element = str(site.specie.symbol)
                
                if element not in element_counts:
                    element_counts[element] = 0
                
                label = f"{element}{element_counts[element]}"
                element_counts[element] += 1
                
                atom_info.append({
                    "index": i,
                    "type_symbol": element,
                    "label": label,
                    "fract_x": round(site.frac_coords[0], 6),
                    "fract_y": round(site.frac_coords[1], 6),
                    "fract_z": round(site.frac_coords[2], 6),
                    "occupancy": 1.0,
                    "multiplicity": 1
                })
            
            # スーパーセル構造情報を保持
            modified_structure_info = {
                "is_supercell": True,
                "supercell_multipliers": supercell_metadata.get('multipliers'),
                "original_unit_cell_atoms": supercell_metadata.get('original_atoms'),
                "atom_count": len(new_structure),
                "atom_info": atom_info,
                "formula": new_structure.composition.reduced_formula,
                "volume": new_structure.lattice.volume,
                "lattice_parameters": {
                    "a": new_structure.lattice.a,
                    "b": new_structure.lattice.b,
                    "c": new_structure.lattice.c,
                    "alpha": new_structure.lattice.alpha,
                    "beta": new_structure.lattice.beta,
                    "gamma": new_structure.lattice.gamma
                }
            }
            
            response_data["modified_structure_info"] = modified_structure_info
            logger.info(f"Added supercell info to response: {len(atom_info)} atoms, multipliers={supercell_metadata.get('multipliers')}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error replacing atoms: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": f"Failed to replace atoms: {str(e)}",
            "error_type": "processing_error"
        }), 500

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
    
    print("CrystalNexus Server Starting...")
    print("Available endpoints:")
    print("  GET  /health         - Health check with dependency info")
    print("  GET  /test_cif       - Test with BaTiO3.cif")
    print("  GET  /cache_info     - Cache statistics")
    print("  POST /parse_cif      - Parse CIF content")
    print("  POST /create_supercell - Create supercell structure")
    print("  POST /delete_atoms   - Delete specific atoms")
    print("  POST /replace_atoms  - Replace atoms with new element")
    print("  GET  /sample_files   - List available sample CIF files")
    print("  GET  /sample_cif/<f> - Serve sample CIF files")
    print("  GET  /               - Serve index.html")
    print("  GET  /<filename>     - Serve static files")
    print(f"Security: CORS limited to {allowed_origins}")
    print(f"Cache: {CACHE_SIZE} items, {CACHE_EXPIRY}s expiry")
    print(f"Max file size: {MAX_CONTENT_LENGTH} bytes")
    
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE)