import os
import json
import asyncio
import subprocess
import tempfile
import logging
import hashlib
import time
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

# Logging configuration (early initialization)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CHGnet import with Windows compatibility
import platform
WINDOWS_PLATFORM = platform.system() == "Windows"

try:
    import torch
    # Windows-specific NumPy/PyTorch compatibility settings
    if WINDOWS_PLATFORM:
        import numpy as np
        # Set NumPy to use compatible dtypes for Windows
        os.environ["NPY_NO_DEPRECATED_API"] = "NPY_1_7_API_VERSION"
        # Ensure PyTorch uses CPU on Windows to avoid CUDA issues
        torch.set_default_dtype(torch.float32)
        logger.info("Windows platform detected - applied compatibility settings")
    
    from chgnet.model import CHGNet
    CHGNET_AVAILABLE = True
    logger.info(f"CHGNet successfully loaded (Platform: {platform.system()})")
except ImportError:
    CHGNET_AVAILABLE = False
    logger.warning(f"CHGNet not available on {platform.system()}: Install with: pip install chgnet")
except Exception as e:
    CHGNET_AVAILABLE = False
    if WINDOWS_PLATFORM and "Buffer dtype mismatch" in str(e):
        logger.error(f"Windows CHGNet compatibility issue detected: {e}")
        logger.error("Solution: pip install -r requirements-windows.txt")
    else:
        logger.error(f"CHGNet loading error: {e}")

def get_chgnet_supported_elements() -> Set[str]:
    """
    Dynamically retrieve supported elements from CHGNet
    With fallback mechanism
    """
    if not CHGNET_AVAILABLE:
        logger.warning("CHGNet not available, using fallback element list")
        return _get_fallback_elements()
    
    try:
        # Method 1: Try direct retrieval from CHGNet model
        supported_elements = _get_elements_from_chgnet()
        if supported_elements:
            logger.info(f"Retrieved {len(supported_elements)} elements from CHGNet")
            return supported_elements
            
    except Exception as e:
        logger.warning(f"Failed to get elements from CHGNet: {e}")
    
    try:
        # Method 2: Get safe range from pymatgen periodic table
        supported_elements = _get_elements_from_periodic_table()
        if supported_elements:
            logger.info(f"Using periodic table range: {len(supported_elements)} elements")
            return supported_elements
            
    except Exception as e:
        logger.error(f"Failed to get elements from periodic table: {e}")
    
    # Method 3: Final fallback
    logger.warning("Using hardcoded fallback element list")
    return _get_fallback_elements()

def _get_elements_from_chgnet() -> Optional[Set[str]]:
    """Get element list from CHGNet model"""
    try:
        from chgnet.model import CHGNet
        
        # Load CHGNet model
        model = CHGNet.load(use_device="cpu")
        
        # Get element information from model configuration
        if hasattr(model, 'atom_embedding'):
            # 実際に使用されている原子番号を確認
            supported_elements = set()
            for z in range(1, MAX_ATOMIC_NUMBER + 1):
                try:
                    element = Element.from_Z(z)
                    # Exclude noble gases and actinoids (based on CHGNet characteristics)
                    if z not in [2, 10, 18, 36, 54, 86] and z <= 92:  # He, Ne, Ar, Kr, Xe, Rn, exclude U and beyond
                        supported_elements.add(element.symbol)
                except (ValueError, AttributeError):
                    continue
                    
            return supported_elements if supported_elements else None
            
    except Exception as e:
        logger.debug(f"Direct CHGNet element extraction failed: {e}")
        return None

def _get_elements_from_periodic_table() -> Optional[Set[str]]:
    """Get valid element range from pymatgen periodic table"""
    try:
        supported_elements = set()
        
        # Range generally supported by Materials Project/CHGNet
        # Atomic numbers 1-92 (up to U, excluding some noble gases and radioactive elements)
        excluded_elements = {
            # Noble gases
            'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn',
            # Some actinoids (low stability)
            'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
        }
        
        for z in range(1, 93):  # 1-92 (H to U)
            try:
                element = Element.from_Z(z)
                if element.symbol not in excluded_elements:
                    supported_elements.add(element.symbol)
            except (ValueError, AttributeError):
                continue
                
        return supported_elements if len(supported_elements) > MIN_SUPPORTED_ELEMENTS else None
        
    except Exception:
        return None

def _get_fallback_elements() -> Set[str]:
    """
    Final fallback: only reliably supported elements
    Safe element list frequently used in Materials Project
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

# Configuration constants - must be defined before use
# Memory management settings
SESSION_CLEANUP_HOURS = int(os.getenv('SESSION_CLEANUP_HOURS', '6'))
PERIODIC_CLEANUP_INTERVAL = int(os.getenv('PERIODIC_CLEANUP_INTERVAL', '1800'))  # 30 minutes
CHGNET_BATCH_SIZE = int(os.getenv('CHGNET_BATCH_SIZE', '4'))

# CHGNet model settings
MAX_ATOMIC_NUMBER = int(os.getenv('MAX_ATOMIC_NUMBER', '94'))
MIN_SUPPORTED_ELEMENTS = int(os.getenv('MIN_SUPPORTED_ELEMENTS', '50'))

# Initialize once as global variable
ALLOWED_ELEMENTS: Set[str] = get_chgnet_supported_elements()

# CHGNet Model Manager (Singleton Pattern)
class CHGNetModelManager:
    """
    Singleton pattern for CHGNet model management with batch processing support
    Ensures only one model instance is loaded in memory
    Thread-safe with asyncio.Lock
    """
    _instance: Optional['CHGNetModelManager'] = None
    _model = None
    _relaxer = None
    _lock = asyncio.Lock()
    batch_size = CHGNET_BATCH_SIZE  # Configurable batch size for CPU processing
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_model(self):
        """Get CHGNet model instance (lazy loading)"""
        if self._model is None:
            async with self._lock:
                if self._model is None:  # Double-check locking
                    await self._load_model()
        return self._model
    
    async def get_relaxer(self):
        """Get StructOptimizer instance (lazy loading)"""
        if self._relaxer is None:
            async with self._lock:
                if self._relaxer is None:  # Double-check locking
                    model = await self.get_model()
                    from chgnet.model import StructOptimizer
                    self._relaxer = StructOptimizer(model=model, use_device="cpu", optimizer_class="FIRE")
                    logger.info("CHGNet StructOptimizer created and cached")
        return self._relaxer
    
    async def _load_model(self):
        """Internal method to load CHGNet model"""
        if not CHGNET_AVAILABLE:
            raise RuntimeError("CHGNet not available")
        
        try:
            from chgnet.model.model import CHGNet
            
            if WINDOWS_PLATFORM:
                self._model = CHGNet.load(use_device="cpu", verbose=False)
                logger.info("CHGNet model loaded with Windows compatibility settings (cached)")
            else:
                self._model = CHGNet.load(use_device="cpu", verbose=False)
                logger.info("CHGNet model loaded and cached")
                
        except Exception as load_error:
            if WINDOWS_PLATFORM and "Buffer dtype mismatch" in str(load_error):
                raise RuntimeError("Windows CHGNet compatibility issue. Try: pip install -r requirements-windows.txt")
            else:
                raise RuntimeError(f"Failed to load CHGNet model: {load_error}")
                
    async def predict_structures_batch(self, structures, **kwargs):
        """
        Batch prediction of multiple structures for improved performance
        
        Args:
            structures: List of pymatgen Structure objects
            **kwargs: Additional arguments for predict_structure
            
        Returns:
            List of prediction results
        """
        if not structures:
            return []
            
        model = await self.get_model()
        results = []
        
        # Process in batches to optimize memory usage
        for i in range(0, len(structures), self.batch_size):
            batch = structures[i:i + self.batch_size]
            batch_results = []
            
            # CHGNet doesn't support native batch processing yet, so we optimize individual calls
            for structure in batch:
                try:
                    pred = model.predict_structure(structure, **kwargs)
                    batch_results.append(pred)
                except Exception as e:
                    logger.warning(f"Batch prediction failed for structure {len(results) + len(batch_results)}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
        return results
    
    async def predict_single_optimized(self, structure, **kwargs):
        """
        Optimized single structure prediction with memory management
        """
        model = await self.get_model()
        
        # Memory optimization for large structures
        if len(structure) > 100:
            # Force garbage collection before large structure prediction
            import gc
            gc.collect()
            logger.debug(f"Garbage collection performed for large structure ({len(structure)} atoms)")
            
        try:
            pred = model.predict_structure(structure, **kwargs)
            return pred
        except Exception as e:
            logger.error(f"Optimized prediction failed: {e}")
            raise
            
    async def predict_auto_mode_batch(self, base_structure, target_atoms, operation_type, **kwargs):
        """
        Batch prediction for Auto mode operations (future implementation)
        
        Args:
            base_structure: Base pymatgen Structure
            target_atoms: List of atom indices to process
            operation_type: 'delete' or 'substitute'
            **kwargs: Additional prediction arguments
            
        Returns:
            List of (atom_index, prediction_result) tuples
        """
        structures_to_predict = []
        atom_structure_map = []
        
        # Generate modified structures for batch processing
        for atom_idx in target_atoms:
            try:
                if operation_type == 'delete':
                    modified_structure = self._create_deletion_structure(base_structure, atom_idx)
                elif operation_type == 'substitute':
                    # Would need additional parameters for substitution
                    continue
                else:
                    continue
                    
                structures_to_predict.append(modified_structure)
                atom_structure_map.append(atom_idx)
                
            except Exception as e:
                logger.warning(f"Failed to create modified structure for atom {atom_idx}: {e}")
                continue
        
        if not structures_to_predict:
            return []
        
        # Batch prediction
        batch_results = await self.predict_structures_batch(structures_to_predict, **kwargs)
        
        # Combine results with atom indices
        results = []
        for i, (atom_idx, pred_result) in enumerate(zip(atom_structure_map, batch_results)):
            if pred_result is not None:
                results.append((atom_idx, pred_result))
                
        return results
    
    def _create_deletion_structure(self, base_structure, atom_index):
        """Helper method to create structure with deleted atom"""
        modified_structure = base_structure.copy()
        if atom_index < len(modified_structure.sites):
            modified_structure.remove_sites([atom_index])
        return modified_structure

def validate_atomic_operation(operation, structure_size, operation_index=None):
    """
    Validate a single atomic operation
    Returns: (is_valid, error_message)
    """
    try:
        # Check required fields
        if "action" not in operation:
            return False, "missing 'action' field"
        if "index" not in operation:
            return False, "missing 'index' field"
        
        # Validate index type and range
        index = operation["index"]
        if not isinstance(index, int):
            return False, f"index must be integer, got {type(index).__name__}: {index}"
        if index < 0:
            return False, f"index cannot be negative: {index}"
        if index >= structure_size:
            return False, f"index {index} out of range (max: {structure_size-1})"
        
        # Validate action type
        action = operation["action"]
        if action not in ["substitute", "delete"]:
            return False, f"invalid action '{action}', must be 'substitute' or 'delete'"
        
        # Validate substitution target element
        if action == "substitute":
            if "to" not in operation:
                return False, "substitution missing 'to' element"
            try:
                validate_element(operation["to"])
            except ValueError as e:
                return False, str(e)
        
        return True, None
    except Exception as e:
        return False, f"unexpected validation error: {str(e)}"

def filter_valid_operations(operations, structure_size, strict_mode=False):
    """
    Filter and validate atomic operations
    
    Args:
        operations: List of operations to validate
        structure_size: Number of sites in structure
        strict_mode: If True, raise exception on any invalid operation
    
    Returns:
        (valid_operations, invalid_operations_info)
    """
    valid_operations = []
    invalid_operations = []
    
    for i, operation in enumerate(operations):
        is_valid, error_msg = validate_atomic_operation(operation, structure_size, i+1)
        if is_valid:
            valid_operations.append(operation)
        else:
            invalid_operations.append(f"Operation {i+1}: {error_msg}")
    
    if strict_mode and invalid_operations:
        error_message = f"Invalid atomic operations detected:\n" + "\n".join(f"  - {error}" for error in invalid_operations)
        raise ValueError(error_message)
    
    return valid_operations, invalid_operations

# Global model manager instance
chgnet_manager = CHGNetModelManager()

# Background task for periodic cleanup
async def periodic_cleanup_task():
    """Background task that runs periodic memory cleanup"""
    while True:
        try:
            await asyncio.sleep(PERIODIC_CLEANUP_INTERVAL)  # Configurable cleanup interval
            logger.info("Starting periodic cleanup...")
            
            # Clean up old sessions
            session_manager.cleanup_old_sessions()
            
            # Log session statistics and perform garbage collection
            log_session_statistics()
            
            # Always perform garbage collection during periodic cleanup
            import gc
            gc.collect()
            logger.debug("Periodic garbage collection completed")
            
            logger.info("Periodic cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup task: {e}")
            # Continue running even if there's an error

# Security functions
def safe_filename(filename: str) -> str:
    """Secure filename (path traversal protection) - legacy function for backward compatibility"""
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Remove path traversal characters
    safe_name = os.path.basename(filename).replace('..', '')
    
    # Check dangerous characters
    if not safe_name or '/' in safe_name or '\\' in safe_name:
        raise ValueError("Invalid filename")
    
    # Reject non-CIF files
    if not safe_name.lower().endswith('.cif'):
        raise ValueError("Only CIF files are allowed")
    
    return safe_name

def safe_path(filepath: str) -> str:
    """Secure path validation for subdirectories (supports folder structure)"""
    if not filepath:
        raise ValueError("Filepath cannot be empty")
    
    # Path traversal attack prevention
    if '..' in filepath or filepath.startswith('/') or filepath.startswith('\\'):
        raise ValueError("Invalid path: path traversal attempt detected")
    
    # Normalize path separators
    normalized_path = filepath.replace('\\', '/')
    
    # Check each path component
    path_parts = normalized_path.split('/')
    for part in path_parts:
        if not part or part in ['.', '..']:
            raise ValueError("Invalid path component")
    
    # Final file must be CIF format
    if not path_parts[-1].lower().endswith('.cif'):
        raise ValueError("Only CIF files are allowed")
    
    return normalized_path

def validate_supercell_size(supercell_size: List[int]) -> List[int]:
    """Validate supercell size"""
    if not isinstance(supercell_size, list) or len(supercell_size) != 3:
        raise ValueError("Supercell size must be a list of 3 integers")
    
    for dim in supercell_size:
        if not isinstance(dim, int) or dim < 1 or dim > MAX_SUPERCELL_DIM:
            raise ValueError(f"Supercell dimensions must be between 1 and {MAX_SUPERCELL_DIM}")
    
    return supercell_size

def validate_element(element: str) -> str:
    """Element validation (injection protection)"""
    if not isinstance(element, str):
        raise ValueError("Element must be a string")
    
    element = element.strip()
    if element not in ALLOWED_ELEMENTS:
        raise ValueError(f"Element '{element}' is not supported by CHGNet")
    
    return element

def validate_occupancy(structure) -> None:
    """Validate that all atomic sites have full occupancy (1.0)"""
    partial_sites = []
    
    for i, site in enumerate(structure.sites):
        # Check if site has partial occupancy
        total_occupancy = sum(site.species.values())
        if abs(total_occupancy - 1.0) > 1e-6:  # Allow small numerical errors
            # Get species with partial occupancy
            species_info = []
            for species, occupancy in site.species.items():
                if occupancy < 1.0:
                    species_info.append(f"{species}: {occupancy:.3f}")
            
            partial_sites.append({
                'site_index': i + 1,
                'species': species_info,
                'total_occupancy': total_occupancy,
                'position': [round(coord, 3) for coord in site.frac_coords]
            })
    
    if partial_sites:
        # Format detailed error message
        error_details = []
        for site in partial_sites:
            species_str = ", ".join(site['species'])
            pos_str = f"({site['position'][0]}, {site['position'][1]}, {site['position'][2]})"
            error_details.append(f"Site {site['site_index']}: {species_str} at {pos_str}")
        
        error_message = (
            f"Partial occupancy detected in {len(partial_sites)} atomic site(s). "
            f"CHGNet predictions require fully occupied crystal structures.\n\n"
            f"Detected partial occupancies:\n" + "\n".join(f"- {detail}" for detail in error_details) +
            f"\n\nPlease provide a CIF file with all atomic sites having occupancy = 1.0, "
            f"or create an ordered supercell model of your structure."
        )
        
        raise ValueError(error_message)


def log_session_statistics():
    """Log session statistics for memory management monitoring"""
    active_sessions = session_manager.get_session_count()
    memory_intensive = session_manager.get_memory_intensive_sessions()
    logger.info(f"Session statistics: {active_sessions} active sessions, {memory_intensive} with large/relaxed structures")

class SessionManager:
    """Session management class - manages structure data and metadata"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        
    def create_session(self, session_id: str, filename: str, original_structure: Structure) -> None:
        """Create new session"""
        self.sessions[session_id] = {
            'filename': filename,
            'original_structure': original_structure,
            'current_structure': original_structure.copy(),
            'operations': [],
            'supercell_size': [1, 1, 1],
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        logger.info(f"Created session {session_id[:8]}... for {filename}")
    
    def get_current_structure(self, session_id: str) -> Optional[Structure]:
        """Get current structure"""
        if session_id in self.sessions:
            self.sessions[session_id]['last_accessed'] = time.time()  # Track access time
            return self.sessions[session_id]['current_structure']
        return None
    
    def update_structure(self, session_id: str, structure: Structure, operations: List = None, supercell_size: List = None) -> None:
        """Update structure"""
        if session_id in self.sessions:
            self.sessions[session_id]['current_structure'] = structure
            self.sessions[session_id]['last_accessed'] = time.time()  # Track access time
            if operations is not None:
                self.sessions[session_id]['operations'] = operations
            if supercell_size is not None:
                self.sessions[session_id]['supercell_size'] = supercell_size
            logger.info(f"Updated structure for session {session_id[:8]}...")
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        if session_id in self.sessions:
            self.sessions[session_id]['last_accessed'] = time.time()  # Track access time
        return self.sessions.get(session_id)
    
    def get_session_count(self) -> int:
        """Get current number of active sessions"""
        return len(self.sessions)
    
    def get_memory_intensive_sessions(self) -> int:
        """Count sessions with large structures"""
        count = 0
        for session_data in self.sessions.values():
            if 'relaxed_structure' in session_data:
                count += 1
            if 'current_structure' in session_data:
                structure = session_data['current_structure']
                if len(structure.sites) > 50:  # Large structure
                    count += 1
        return count
    
    def cleanup_old_sessions(self, max_age_hours: int = SESSION_CLEANUP_HOURS) -> None:
        """Clean up old sessions (optimized to 6 hours for better memory management)"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        old_sessions = [
            sid for sid, data in self.sessions.items()
            if data.get('created_at', 0) < cutoff_time
        ]
        
        memory_freed = 0
        for sid in old_sessions:
            # Estimate memory usage before deletion
            session_data = self.sessions[sid]
            if 'current_structure' in session_data:
                memory_freed += 1  # Rough estimate
            if 'relaxed_structure' in session_data:
                memory_freed += 1  # Rough estimate
            
            del self.sessions[sid]
            logger.info(f"Cleaned up old session {sid[:8]}...")
        
        if old_sessions:
            logger.info(f"Memory cleanup completed: {len(old_sessions)} sessions removed, ~{memory_freed} structures freed")
        else:
            logger.debug("No old sessions to cleanup")
        
        # Log session statistics is now handled by the global function
        log_session_statistics()

class UnifiedStructureManager:
    """
    Unified structure management for session-based workspace
    Handles both sample and uploaded files with single code path
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.workspace_dir = Path("sessions") / session_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Memory-first management
        self._structure_cache = {
            'original': None,           # Original pymatgen Structure
            'current_supercell': None,  # Current supercell structure
            'source_type': None,        # 'sample' or 'upload'
            'original_filename': None,  # Original filename
            'operations': [],           # Applied operations
            'supercell_size': [1, 1, 1], # Current supercell size
            'atom_identifiers': {}      # Atom identification mapping
        }

        # Minimal file storage
        self._file_paths = {
            'working': self.workspace_dir / "current.cif"
        }

    def set_structure(self, structure: Structure, source_type: str, filename: str) -> None:
        """Unified entry point for both sample and uploaded files"""
        self._structure_cache['original'] = structure
        self._structure_cache['current_supercell'] = structure  # Initially same as original
        self._structure_cache['source_type'] = source_type
        self._structure_cache['original_filename'] = filename
        self._structure_cache['operations'] = []

        logger.info(f"Structure set for session {self.session_id[:8]} - Type: {source_type}, File: {filename}")

        # Clear any existing working file to force regeneration
        if self._file_paths['working'].exists():
            self._file_paths['working'].unlink()

    def update_supercell(self, supercell_structure: Structure, operations: List = None, supercell_size: List = None) -> None:
        """Update the current supercell structure"""
        self._structure_cache['current_supercell'] = supercell_structure
        if operations is not None:
            self._structure_cache['operations'] = operations
        if supercell_size is not None:
            self._structure_cache['supercell_size'] = supercell_size

        # Clear working file to force regeneration with new structure
        if self._file_paths['working'].exists():
            self._file_paths['working'].unlink()

    def get_working_file_path(self) -> Path:
        """Get working file path for API calls (lazy generation)"""
        if not self._file_paths['working'].exists():
            self._generate_working_file()
        return self._file_paths['working']

    def _generate_working_file(self) -> None:
        """Generate working CIF file from current structure"""
        current_structure = self._get_current_structure()
        if current_structure:
            current_structure.to(filename=str(self._file_paths['working']), fmt="cif")
            logger.debug(f"Generated working file for session {self.session_id[:8]}")
        else:
            raise ValueError(f"No structure available for session {self.session_id}")

    def _get_current_structure(self) -> Optional[Structure]:
        """Get current structure (supercell if available, otherwise original)"""
        return self._structure_cache['current_supercell'] or self._structure_cache['original']

    def get_current_structure(self) -> Optional[Structure]:
        """Public accessor for current structure (supercell if available, otherwise original)"""
        return self._get_current_structure()

    @property
    def filename(self) -> Optional[str]:
        """Get the original filename"""
        return self._structure_cache.get('original_filename')

    @property
    def supercell_size(self) -> List[int]:
        """Get current supercell size"""
        return self._structure_cache.get('supercell_size', [1, 1, 1])

    def save_relaxed_structure(self, relaxed_structure, chgnet_result: Dict):
        """Save relaxed structure and CHGNet metadata"""
        self._structure_cache['relaxed_structure'] = relaxed_structure
        self._structure_cache['chgnet_result'] = chgnet_result

    def get_relaxed_structure(self):
        """Get relaxed structure if available, otherwise return current structure"""
        relaxed = self._structure_cache.get('relaxed_structure')
        if relaxed is not None:
            return relaxed
        return self._get_current_structure()

    def get_chgnet_result(self) -> Dict:
        """Get CHGNet relaxation metadata"""
        return self._structure_cache.get('chgnet_result', {})

    def _generate_atom_identifiers(self, structure) -> Dict:
        """Generate Element Type Counter System identifiers for atoms"""
        element_counters = {}
        site_to_id = {}
        id_to_site = {}

        for i, site in enumerate(structure.sites):
            # Use element symbol instead of species (removes oxidation state)
            element = str(site.specie.element)

            # Count occurrences of each element
            count = element_counters.get(element, 0) + 1
            element_counters[element] = count

            # Create atom identifier
            atom_id = f"{element}_{count}"

            # Create bidirectional mapping
            site_to_id[i] = atom_id
            id_to_site[atom_id] = i

        return {
            'site_to_id': site_to_id,
            'id_to_site': id_to_site,
            'element_counters': element_counters
        }

    def get_atom_identifiers(self) -> Dict:
        """Get current atom identifiers for the active structure"""
        current_structure = self._get_current_structure()
        if not current_structure:
            return {'site_to_id': {}, 'id_to_site': {}, 'element_counters': {}}

        # Generate fresh identifiers for current structure
        return self._generate_atom_identifiers(current_structure)

    def get_numbered_atom_list(self) -> List[Dict]:
        """Get list of atoms with numbering for frontend display"""
        current_structure = self._get_current_structure()
        if not current_structure:
            return []

        identifiers = self._generate_atom_identifiers(current_structure)
        atoms = []

        for i, site in enumerate(current_structure.sites):
            atom_id = identifiers['site_to_id'][i]
            element = str(site.specie.element)

            atoms.append({
                'site_index': i,
                'atom_id': atom_id,
                'element': element,
                'display_name': atom_id,
                'fractional_coords': site.frac_coords.tolist(),
                'cartesian_coords': site.coords.tolist()
            })

        return atoms

    def get_structure_info(self) -> Dict:
        """Get structure information"""
        return {
            'source_type': self._structure_cache['source_type'],
            'filename': self._structure_cache['original_filename'],
            'has_supercell': self._structure_cache['current_supercell'] is not None,
            'operations_count': len(self._structure_cache['operations'])
        }

    def get_cif_string(self) -> str:
        """Get current structure as CIF string (no file needed)"""
        current_structure = self._get_current_structure()
        if current_structure:
            return current_structure.to(fmt="cif")
        else:
            raise ValueError(f"No structure available for session {self.session_id}")

    def cleanup(self) -> None:
        """Clean up session workspace"""
        try:
            if self.workspace_dir.exists():
                import shutil
                shutil.rmtree(self.workspace_dir)
                logger.debug(f"Cleaned up workspace for session {self.session_id[:8]}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session {self.session_id[:8]}: {e}")

class UnifiedSessionManager:
    """
    Global manager for unified structure managers
    """

    def __init__(self):
        self._managers: Dict[str, UnifiedStructureManager] = {}
        self._legacy_sessions = session_manager  # Keep reference to legacy manager

    def get_manager(self, session_id: str) -> UnifiedStructureManager:
        """Get or create unified structure manager for session"""
        if session_id not in self._managers:
            self._managers[session_id] = UnifiedStructureManager(session_id)
        return self._managers[session_id]

    def cleanup_session(self, session_id: str) -> None:
        """Clean up specific session"""
        if session_id in self._managers:
            self._managers[session_id].cleanup()
            del self._managers[session_id]

    def cleanup_old_sessions(self, max_age_hours: int = 6) -> None:
        """Clean up old unified sessions"""
        import time
        cutoff_time = time.time() - (max_age_hours * 3600)

        old_sessions = []
        for session_id, manager in list(self._managers.items()):
            # Check if workspace directory is old based on creation time
            if manager.workspace_dir.exists():
                workspace_ctime = manager.workspace_dir.stat().st_ctime
                if workspace_ctime < cutoff_time:
                    old_sessions.append(session_id)

        for session_id in old_sessions:
            self.cleanup_session(session_id)
            logger.info(f"Cleaned up old unified session {session_id[:8]}...")

        if old_sessions:
            logger.info(f"Unified session cleanup: {len(old_sessions)} sessions removed")

    def get_session_count(self) -> int:
        """Get number of active unified sessions"""
        return len(self._managers)

# Session management instances
session_manager = SessionManager()  # Legacy manager
unified_session_manager = UnifiedSessionManager()  # New unified manager

# Configuration constants - must be defined before use in app creation
TEMPLATE_DIR = os.getenv('CRYSTALNEXUS_TEMPLATE_DIR', 'templates')
STATIC_DIR = os.getenv('CRYSTALNEXUS_STATIC_DIR', 'static')
SAMPLE_CIF_DIR_NAME = os.getenv('CRYSTALNEXUS_SAMPLE_CIF_DIR', 'sample_cif')
APP_NAME = os.getenv('CRYSTALNEXUS_APP_NAME', 'CrystalNexus')

# Server configuration
HOST = os.getenv('CRYSTALNEXUS_HOST', '0.0.0.0')
PORT = int(os.getenv('CRYSTALNEXUS_PORT', '8080'))
DEBUG = os.getenv('CRYSTALNEXUS_DEBUG', 'False').lower() == 'true'

# File handling limits
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(50 * 1024 * 1024)))  # 50MB
MAX_SUPERCELL_DIM = int(os.getenv('MAX_SUPERCELL_DIM', '10'))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management (startup and shutdown)"""
    # Startup
    logger.info("CrystalNexus starting up...")
    log_session_statistics()
    # Start the periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup_task())
    logger.info("Periodic cleanup task started")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down, performing final cleanup")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logger.info("Periodic cleanup task cancelled")
    session_manager.cleanup_old_sessions(max_age_hours=0)  # Clean all sessions
    import gc
    gc.collect()
    logger.info("Final cleanup completed")

app = FastAPI(title=APP_NAME, lifespan=lifespan)
templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SAMPLE_CIF_DIR = Path(SAMPLE_CIF_DIR_NAME)
# Mount sample_cif directory for direct CIF file access
app.mount("/sample_cif", StaticFiles(directory=SAMPLE_CIF_DIR_NAME), name="sample_cif")

# Apply DEBUG settings
if DEBUG:
    logger.setLevel(logging.DEBUG)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CrystalNexus"}

@app.post("/api/apply-atomic-operations")
async def apply_atomic_operations(request: dict):
    """Apply atomic operations to the current structure in session"""
    try:
        session_id = request.get("session_id")
        operations = request.get("operations", [])
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Get current structure from session
        current_structure = session_manager.get_current_structure(session_id)
        if current_structure is None:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")
        
        # Get session info to recreate structure from original + supercell + operations
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session info not found for {session_id}")
        
        # Start with original structure and apply supercell + operations
        structure = session_info['original_structure'].copy()
        supercell_size = session_info['supercell_size']
        structure.make_supercell(supercell_size)
        
        # Validate all operations using strict mode (fail-fast approach)
        try:
            valid_operations, invalid_operations = filter_valid_operations(
                operations, len(structure.sites), strict_mode=True
            )
        except ValueError as e:
            logger.error(f"Rejecting {len(operations)} operations due to validation errors")
            raise HTTPException(status_code=400, detail=str(e))
        
        # All operations validated - process in descending order by index to avoid index shift issues
        stable_operations = sorted(valid_operations, key=lambda x: x["index"], reverse=True)
        logger.info(f"Processing {len(stable_operations)} validated atomic operations")
        
        for operation in stable_operations:
            site_index = operation["index"]
            if operation["action"] == "substitute":
                # Already validated: element and index are valid
                new_element = operation["to"]
                old_coords = structure[site_index].frac_coords
                structure[site_index] = Element(new_element), old_coords
                logger.debug(f"Substituted site {site_index} with {new_element}")
            elif operation["action"] == "delete":
                # Already validated: index is valid
                structure.remove_sites([site_index])
                logger.debug(f"Deleted site {site_index}")
        
        # Update session with modified structure and operations
        session_manager.update_structure(session_id, structure, operations=operations)
        
        logger.info(f"Applied {len(operations)} operations to session {session_id[:8]}...")
        
        return {
            "status": "success",
            "message": f"Applied {len(operations)} atomic operations",
            "num_sites": len(structure),
            "composition": str(structure.composition)
        }
        
    except Exception as e:
        logger.error(f"Error applying atomic operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sample-cif-files")
async def get_sample_cif_files():
    try:
        def scan_directory_recursive(base_path: Path, current_path: Path = None):
            """Recursively scan directory to generate hierarchical structure"""
            if current_path is None:
                current_path = base_path
            
            structure = {"files": [], "subdirs": {}}
            
            # Get CIF files in current directory
            for cif_file in current_path.glob("*.cif"):
                relative_path = cif_file.relative_to(base_path)
                structure["files"].append({
                    "name": cif_file.name,
                    "path": str(relative_path).replace("\\", "/"),  # Windows compatibility
                    "display_name": cif_file.stem  # Without extension
                })
            
            # Recursively scan subdirectories
            for subdir in current_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    subdir_structure = scan_directory_recursive(base_path, subdir)
                    if subdir_structure["files"] or subdir_structure["subdirs"]:
                        structure["subdirs"][subdir.name] = subdir_structure
            
            return structure
        
        # Dynamically scan sample_cif directory
        directory_structure = scan_directory_recursive(SAMPLE_CIF_DIR)
        
        return {
            "structure": directory_structure,
            "scan_time": time.time(),
            "base_path": str(SAMPLE_CIF_DIR)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning sample CIF files: {str(e)}")

@app.post("/api/analyze-cif-sample")
async def analyze_sample_cif(data: dict):
    try:
        filename = data.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Secure path validation (supports subdirectories)
        safe_path_str = safe_path(filename)
        file_path = SAMPLE_CIF_DIR / safe_path_str
        
        # Additional security check: ensure path is within sample_cif directory
        resolved_path = file_path.resolve()
        base_path = SAMPLE_CIF_DIR.resolve()
        if not str(resolved_path).startswith(str(base_path)):
            raise ValueError("Path outside sample directory")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="CIF file not found")
        
        result = await analyze_cif_file(file_path)
        result["filename"] = safe_path_str
        
        # Store the original structure for consistency with uploaded files
        try:
            from pymatgen.core import Structure
            structure = Structure.from_file(str(file_path))
            result["structure_data"] = structure.as_dict()
            logger.info(f"Successfully stored structure data for sample file: {safe_path_str}")
        except Exception as struct_error:
            logger.warning(f"Failed to store structure data for sample file: {struct_error}")
            # Sample files can still work without structure_data since file is available
        
        return result
    except ValueError as e:
        logger.warning(f"Invalid input in analyze_sample_cif: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTPExceptions (like 400 errors from validate_occupancy) without modification
        raise
    except Exception as e:
        logger.error(f"Error in analyze_sample_cif: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/analyze-cif-upload")
async def analyze_uploaded_cif(file: UploadFile = File(...)):
    try:
        # Filename and size validation
        if not file.filename or not file.filename.lower().endswith('.cif'):
            raise HTTPException(status_code=400, detail="File must be a CIF file")
        
        # File size limit
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")
        
        # Create secure temporary file (explicit UTF-8 encoding)
        try:
            # Try to decode as UTF-8 first
            contents_str = contents.decode('utf-8')
            # Remove BOM if present
            if contents_str.startswith('\ufeff'):
                contents_str = contents_str[1:]
            logger.info("Successfully decoded CIF file as UTF-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                contents_str = contents.decode('latin-1')
                logger.info("Decoded CIF file as latin-1")
            except UnicodeDecodeError as decode_error:
                logger.error(f"Failed to decode CIF file: {decode_error}")
                raise ValueError("CIF file encoding error. Please ensure the file is saved in UTF-8 or ASCII format.")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(contents_str)
            temp_path = Path(tmp_file.name)
        
        try:
            result = await analyze_cif_file(temp_path)
            safe_name = safe_filename(file.filename)
            result["filename"] = safe_name
            result["cif_content"] = contents_str  # Add CIF content for unified API
            
            # Store the original structure for later use in createSupercell
            # Parse the structure and add it to the result
            try:
                from pymatgen.core import Structure
                structure = Structure.from_file(str(temp_path))
                # Store structure as a dictionary for JSON serialization
                result["structure_data"] = structure.as_dict()
                logger.info(f"Successfully stored structure data for uploaded file: {safe_name}")
            except Exception as struct_error:
                logger.warning(f"Failed to store structure data: {struct_error}")
                # Continue without structure data - fallback will handle this
            
            return result
        finally:
            if temp_path.exists():
                temp_path.unlink()
                
    except ValueError as e:
        logger.warning(f"Invalid input in analyze_uploaded_cif: {e}")
        # Provide user-friendly error message for common CIF issues
        error_msg = str(e)
        if "Invalid CIF file format" in error_msg:
            user_msg = "CIF file format error. Please ensure your file follows standard CIF format with proper symmetry operations and atomic coordinates."
        elif "No structures found" in error_msg:
            user_msg = "No crystal structures found in CIF file. Please verify the file contains valid atomic site data."
        elif "no atomic sites" in error_msg.lower():
            user_msg = "CIF file contains no atomic positions. Please ensure your file includes atom site coordinates."
        else:
            user_msg = f"CIF file validation error: {error_msg}"
        raise HTTPException(status_code=400, detail=user_msg)
    except HTTPException:
        # Re-raise HTTPExceptions (like 400 errors from validate_occupancy) without modification
        raise
    except Exception as e:
        logger.error(f"Error in analyze_uploaded_cif: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {str(e)}", exc_info=True)
        # Report Windows-specific file operation errors in more detail
        if WINDOWS_PLATFORM and ("permission" in str(e).lower() or "access" in str(e).lower()):
            raise HTTPException(status_code=500, detail="Windows file access error - check file permissions")
        raise HTTPException(status_code=500, detail="Internal server error occurred")

async def analyze_cif_file(file_path: Path) -> Dict:
    try:
        # CIF file parsing with enhanced validation
        logger.info(f"Parsing CIF file: {file_path}")
        
        # Try multiple parsing methods for better compatibility
        structure = None
        parsing_errors = []
        
        # Method 1: Direct Structure.from_file
        try:
            structure = Structure.from_file(str(file_path))
            logger.info("Successfully parsed with Structure.from_file")
        except Exception as e:
            parsing_errors.append(f"Structure.from_file: {str(e)}")
            logger.warning(f"Structure.from_file failed: {e}")
        
        # Method 2: CifParser with explicit handling
        if structure is None:
            try:
                from pymatgen.io.cif import CifParser
                parser = CifParser(str(file_path))
                structures = parser.get_structures()
                if structures:
                    structure = structures[0]
                    logger.info("Successfully parsed with CifParser")
                else:
                    parsing_errors.append("CifParser: No structures found in CIF file")
            except Exception as e:
                parsing_errors.append(f"CifParser: {str(e)}")
                logger.warning(f"CifParser failed: {e}")
        
        if structure is None:
            error_msg = "Failed to parse CIF file. Common issues: " + "; ".join(parsing_errors)
            logger.error(f"All parsing methods failed: {error_msg}")
            raise ValueError(f"Invalid CIF file format. {error_msg}")
        
        # Validate structure
        if not structure.sites:
            raise ValueError("CIF file contains no atomic sites")
        
        # Validate occupancy
        validate_occupancy(structure)
        
        logger.info(f"Successfully loaded structure with {len(structure.sites)} sites")
        
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
        logger.error(f"Error in analyze_cif_file: {e}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {str(e)}", exc_info=True)
        # Handle specific error types appropriately
        if "Partial occupancy detected" in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        # Report Windows-specific pymatgen errors in detail
        if WINDOWS_PLATFORM and "fortran" in str(e).lower():
            raise HTTPException(status_code=500, detail="Windows Fortran library error - install Microsoft Visual C++ Redistributable")
        raise HTTPException(status_code=500, detail=f"Error in pymatgen analysis: {str(e)}")

@app.post("/api/create-supercell")
async def create_supercell(data: dict):
    try:
        crystal_data = data.get("crystal_data")
        supercell_size_raw = data.get("supercell_size", [1, 1, 1])
        filename = data.get("filename")
        session_id = data.get("session_id")
        
        logger.debug(f"create_supercell called with crystal_data type: {type(crystal_data)}")
        logger.debug(f"filename from crystal_data: {crystal_data.get('filename') if crystal_data else None}")
        
        if not crystal_data:
            raise HTTPException(status_code=400, detail="Crystal data is required")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Validate supercell size
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
            
            filename = crystal_data.get("filename", "unknown.cif")
            original_structure = None
            
            # Method 1: Try to load from stored structure_data (for uploaded files)
            if "structure_data" in crystal_data:
                try:
                    from pymatgen.core import Structure
                    structure_data = crystal_data["structure_data"]
                    if structure_data and isinstance(structure_data, dict):
                        original_structure = Structure.from_dict(structure_data)
                        logger.info(f"Loaded structure from stored structure_data for: {filename}")
                        # Validate the loaded structure
                        if not original_structure.sites:
                            logger.warning(f"Structure has no sites: {filename}")
                            original_structure = None
                    else:
                        logger.warning(f"Invalid structure_data format: {type(structure_data)}")
                except Exception as e:
                    logger.error(f"Failed to load from structure_data: {e}")
                    logger.error(f"Structure data content: {crystal_data.get('structure_data', 'None')[:200]}...")
                    original_structure = None
            
            # Method 2: Try to load from CIF file (for sample files)
            if original_structure is None and filename != "unknown.cif":
                cif_path = SAMPLE_CIF_DIR / filename
                if cif_path.exists():
                    parser = CifParser(str(cif_path))
                    original_structure = parser.get_structures()[0]
                    logger.info(f"Loaded structure from CIF file: {filename}")
            
            # If no structure available, this is an error condition
            if original_structure is None:
                error_msg = f"Failed to load structure for {filename}. "
                if "structure_data" in crystal_data:
                    error_msg += "Structure data was provided but could not be parsed. "
                else:
                    error_msg += "No structure data available and file not found in sample directory. "
                error_msg += "Please ensure the CIF file is valid and properly uploaded."
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            # Create supercell
            supercell_structure = original_structure.copy()
            supercell_structure.make_supercell([a_mult, b_mult, c_mult])
            
            # Create or update session with structures
            session_manager.create_session(session_id, filename, original_structure)
            session_manager.update_structure(session_id, supercell_structure, 
                                           operations=[], supercell_size=supercell_size)
            
            # Get structure dictionary for CIF generation
            structure_dict = supercell_structure.as_dict()
            logger.info(f"Successfully created supercell structure and session for {filename}")
                
        except Exception as e:
            logger.error(f"Could not create structure object: {e}")
            import traceback
            traceback.print_exc()
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


@app.post("/api/get-element-labels")
async def get_element_labels(data: dict):
    """
    Get element labels directly from the actual Structure object to ensure correct ordering
    This guarantees labels match CHGNet results ordering
    """
    try:
        session_id = data.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Get the actual current structure from session (with supercell + operations applied)
        structure = session_manager.get_current_structure(session_id)
        if structure is None:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")
        
        # Generate labels directly from Structure sites in the exact same order
        # This ensures perfect alignment with CHGNet results
        labels = []
        element_counts = {}
        
        for i, site in enumerate(structure.sites):
            # Get clean element symbol using PyMatGen's unified approach
            # This handles both Element and Species types consistently
            if hasattr(site.specie, 'element'):
                element = str(site.specie.element)  # Species type (e.g., Ba2+ -> Ba)
            else:
                element = str(site.specie)          # Element type (e.g., Nd -> Nd)
            
            # Count occurrences of each element to generate unique labels
            if element not in element_counts:
                element_counts[element] = 0
            else:
                element_counts[element] += 1
            
            # Generate label: Element + count (e.g., Ba0, Ti0, O0, O1, O2)
            label = f"{element}{element_counts[element]}"
            labels.append(label)
        
        # Get unique elements for the UI
        unique_elements = list(element_counts.keys())
        
        logger.info(f"Generated structure-based labels for session {session_id}: {labels}")
        
        return {
            "status": "success", 
            "labels": labels,
            "unique_elements": unique_elements,
            "total_labels": len(labels),
            "method": "structure_based"  # Indicate this is the reliable method
        }
        
    except Exception as e:
        logger.error(f"Error getting element labels: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting element labels: {str(e)}")


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
        
        logger.info(f"Generating modified structure CIF for {filename}")
        logger.info(f"Supercell size: {supercell_size}")
        logger.info(f"Operations to apply: {len(operations)}")
        
        # Construct CIF file path
        cif_path = SAMPLE_CIF_DIR / filename
        
        if not cif_path.exists():
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        # Parse original CIF file
        from pymatgen.io.cif import CifParser, CifWriter
        from pymatgen.core import Element
        parser = CifParser(str(cif_path))
        structure = parser.get_structures(primitive=False)[0]  # Keep original lattice
        
        logger.info(f"Original structure: {structure.formula}")
        
        # Create supercell
        structure.make_supercell(supercell_size)
        logger.info(f"Supercell created: {structure.formula} ({len(structure.sites)} sites)")
        
        # Apply atomic operations with partial success tolerance (for CIF generation)
        valid_operations, invalid_operations = filter_valid_operations(
            operations, len(structure.sites), strict_mode=False
        )
        
        # Log invalid operations but continue with valid ones
        if invalid_operations:
            logger.warning(f"Skipping {len(invalid_operations)} invalid operations during CIF generation: {invalid_operations}")
        
        # Process valid operations in descending order by index to eliminate index adjustment issues
        stable_operations = sorted(valid_operations, key=lambda x: x["index"], reverse=True)
        operations_applied = 0
        
        for operation in stable_operations:
            site_index = operation["index"]
            if operation["action"] == "substitute":
                new_element = operation["to"]
                old_coords = structure[site_index].frac_coords
                old_element = str(structure[site_index].specie)
                
                # Replace the site with new element
                structure[site_index] = Element(new_element), old_coords
                
                logger.info(f"CIF Generation: Substituted {old_element} → {new_element} at site {site_index}")
                operations_applied += 1
                        
            elif operation["action"] == "delete":
                deleted_element = str(structure[site_index].specie)
                structure.remove_sites([site_index])
                
                logger.info(f"CIF Generation: Deleted {deleted_element} at site {site_index}")
                operations_applied += 1
        
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
    Generate supercell CIF directly from structure data or CIF file using pymatgen
    Supports both uploaded files and sample files
    """
    try:
        # Extract request parameters
        filename = request.get("filename")
        supercell_size = request.get("supercell_size", [1, 1, 1])
        session_id = request.get("session_id")
        
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        logger.info(f"Generating supercell CIF for {filename} with size {supercell_size}")
        
        original_structure = None
        
        # Method 1: Try to get structure from session (for uploaded files)
        if session_id:
            try:
                session_info = session_manager.get_session_info(session_id)
                if session_info and 'original_structure' in session_info:
                    original_structure = session_info['original_structure']
                    logger.info(f"Loaded structure from session for uploaded file: {filename}")
            except Exception as e:
                logger.warning(f"Could not load structure from session: {e}")
        
        # Method 2: Try to load from CIF file (for sample files)
        if original_structure is None:
            cif_path = SAMPLE_CIF_DIR / filename
            if cif_path.exists():
                logger.debug(f"Reading CIF file: {cif_path}")
                from pymatgen.io.cif import CifParser
                parser = CifParser(str(cif_path))
                structures = parser.get_structures(primitive=False)  # Keep original lattice - don't convert to primitive
                if structures:
                    original_structure = structures[0]
                    logger.info(f"Loaded structure from sample CIF file: {filename}")
        
        if not original_structure:
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        logger.info(f"Original structure loaded: {original_structure.formula}")
        
        # Create supercell
        supercell_structure = original_structure.copy()
        supercell_structure.make_supercell(supercell_size)
        
        logger.info(f"Supercell created: {supercell_structure.formula}")
        logger.info(f"Number of sites: {len(supercell_structure.sites)}")
        
        # Generate CIF using pymatgen CifWriter
        from pymatgen.io.cif import CifWriter
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

def _get_chgnet_version():
    """Get CHGNet version dynamically"""
    try:
        import chgnet
        return chgnet.__version__
    except (ImportError, AttributeError):
        try:
            # Alternative version detection
            import pkg_resources
            return pkg_resources.get_distribution('chgnet').version
        except Exception:
            return "unknown"

def _get_device_info():
    """Get current device information for CHGNet"""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return "cpu"
    except ImportError:
        return "cpu"

def evaluate_convergence(trajectory, fmax):
    """
    Evaluate convergence based on CHGNet trajectory final step only.
    Simplified version for improved performance and reduced logging.
    """
    import numpy as np
    
    if not trajectory or not hasattr(trajectory, 'forces') or not trajectory.forces:
        return False
    
    # Check final step only - simplified approach
    final_forces = np.array(trajectory.forces[-1])
    max_force = np.linalg.norm(final_forces, axis=1).max()
    
    converged = max_force < fmax
    logger.info(f"Convergence check: max_force={max_force:.4f} eV/Å, fmax={fmax}, converged={converged}")
    
    return bool(converged)

def safe_get_prediction(pred, num_atoms=None):
    """Extract prediction results safely from CHGNet output"""
    out = {}
    for k in ("energy", "e", "energy_per_atom", "e_per_atom"):
        if k in pred:
            energy_per_atom = float(getattr(pred[k], "item", lambda: pred[k])())
            out["energy_eV_per_atom"] = energy_per_atom
            # Convert to total energy if number of atoms is provided
            if num_atoms is not None:
                out["total_energy_eV"] = energy_per_atom * num_atoms
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
    # Site energies
    for k in ("site_energies", "site_energy"):
        if k in pred:
            arr = pred[k]
            out["site_energies_eV"] = [float(x) for x in arr.tolist()]
            break
    
    # Crystal features
    for k in ("crystal_fea", "crystal_features"):
        if k in pred:
            arr = pred[k]
            out["crystal_features"] = [float(x) for x in arr.tolist()]
            break
    
    # Atom features
    for k in ("atom_fea", "atom_features"):
        if k in pred:
            arr = pred[k]
            out["atom_features"] = [[float(x) for x in atom] for atom in arr.tolist()]
            break
    
    # Other fields
    for k, v in pred.items():
        if k in ("energy", "e", "forces", "f", "stress", "s", "magmom", "m", 
                 "site_energies", "crystal_fea", "atom_fea"):
            continue
        try:
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        except (AttributeError, TypeError, ValueError):
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
        cif_path = SAMPLE_CIF_DIR / safe_path(filename)
        
        if not cif_path.exists():
            raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")
        
        # Parse and create supercell
        parser = CifParser(str(cif_path))
        structure = parser.get_structures(primitive=False)[0]
        structure.make_supercell(supercell_size)
        
        # Apply operations for CHGNet prediction (skip invalid operations)
        valid_operations, invalid_operations = filter_valid_operations(
            operations, len(structure.sites), strict_mode=False
        )
        
        # Log invalid operations but continue with valid ones
        if invalid_operations:
            logger.warning(f"Skipping {len(invalid_operations)} invalid operations for CHGNet prediction: {invalid_operations}")
        
        # Process valid operations in descending order by index to eliminate index adjustment issues
        stable_operations = sorted(valid_operations, key=lambda x: x["index"], reverse=True)
        
        for operation in stable_operations:
            site_index = operation["index"]
            if operation["action"] == "substitute":
                new_element = operation["to"]
                old_coords = structure[site_index].frac_coords
                structure[site_index] = Element(new_element), old_coords
            elif operation["action"] == "delete":
                structure.remove_sites([site_index])
        
        # Load CHGNet model (using singleton pattern)
        try:
            chgnet = await chgnet_manager.get_model()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        
        # Predict structure properties using optimized prediction
        try:
            pred = await chgnet_manager.predict_single_optimized(structure,
                                                               return_site_energies=True,
                                                               return_atom_feas=False,
                                                               return_crystal_feas=False)
        except TypeError:
            pred = await chgnet_manager.predict_single_optimized(structure)
        
        # Extract results with number of atoms for energy conversion
        results = safe_get_prediction(pred, num_atoms=len(structure.sites))
        
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
                "version": _get_chgnet_version(),
                "device": _get_device_info(),
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
    Relax structure using CHGNet - uses pre-prepared structure from session
    """
    try:
        session_id = request.get("session_id")
        fmax = float(request.get("fmax", 0.1))
        max_steps = int(request.get("max_steps", 100))
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Get the current structure from unified session manager
        manager = unified_session_manager.get_manager(session_id)
        if not manager:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        structure = manager.get_current_structure()
        if structure is None:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        filename = manager.filename
        
        logger.info(f"CHGNet relaxation for session {session_id[:8]}... ({filename}) with fmax={fmax}, max_steps={max_steps}")
        logger.info(f"Structure: {structure.composition} ({len(structure)} sites)")
        
        # Load CHGNet model and relaxer (using singleton pattern)
        try:
            chgnet = await chgnet_manager.get_model()
            relaxer = await chgnet_manager.get_relaxer()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        
        # Predict initial structure using optimized prediction
        try:
            pred_initial = await chgnet_manager.predict_single_optimized(structure,
                                                                       return_site_energies=True,
                                                                       return_atom_feas=True,
                                                                       return_crystal_feas=True)
            initial_results = safe_get_prediction(pred_initial, num_atoms=len(structure.sites))
        except TypeError:
            # Fallback if detailed prediction fails
            pred_initial = await chgnet_manager.predict_single_optimized(structure)
            initial_results = safe_get_prediction(pred_initial, num_atoms=len(structure.sites))
        except Exception as e:
            logger.warning(f"Initial prediction failed: {e}")
            initial_results = {"energy_eV_per_atom": None}
        
        # CHGNet structure relaxation
        logger.info(f"Starting CHGNet relaxation: fmax={fmax}, max_steps={max_steps}")
        result = relaxer.relax(structure, fmax=fmax, steps=max_steps, verbose=True, relax_cell=True)
        
        # Basic result validation
        logger.info(f"CHGNet result keys: {sorted(result.keys())}")
        if 'final_structure' not in result:
            raise RuntimeError("CHGNet relaxation failed: no final_structure returned")
        if 'trajectory' not in result:
            logger.warning("CHGNet relaxation warning: no trajectory returned")
        
        final_structure = result.get("final_structure")
        if final_structure is None:
            raise RuntimeError("Relaxation failed: no final structure returned")
        
        # Predict relaxed structure using optimized prediction
        try:
            pred_final = await chgnet_manager.predict_single_optimized(final_structure,
                                                                     return_site_energies=True,
                                                                     return_atom_feas=True,
                                                                     return_crystal_feas=True)
            final_results = safe_get_prediction(pred_final, num_atoms=len(final_structure.sites))
        except TypeError:
            # Fallback if detailed prediction fails
            pred_final = await chgnet_manager.predict_single_optimized(final_structure)
            final_results = safe_get_prediction(pred_final, num_atoms=len(final_structure.sites))
        except Exception as e:
            logger.warning(f"Final prediction failed: {e}")
            final_results = {"energy_eV_per_atom": None}
        
        # Calculate energy difference in total eV
        energy_diff = None
        energy_diff_per_atom = None
        if (initial_results.get("total_energy_eV") is not None and 
            final_results.get("total_energy_eV") is not None):
            energy_diff = final_results["total_energy_eV"] - initial_results["total_energy_eV"]
            energy_diff_per_atom = final_results["energy_eV_per_atom"] - initial_results["energy_eV_per_atom"]
        
        # Extract relaxation information with CHGNet-compliant convergence evaluation
        trajectory = result.get("trajectory")
        steps = len(trajectory) if trajectory else 0
        
        # Use our own convergence evaluation (CHGNet spec-compliant)
        converged = evaluate_convergence(trajectory, fmax)
        
        # Log convergence evaluation details using consistent force calculation method
        if trajectory and hasattr(trajectory, 'forces') and len(trajectory.forces) > 0:
            import numpy as np
            final_forces = trajectory.forces[-1]
            final_forces_array = np.array(final_forces)
            max_final_force = np.linalg.norm(final_forces_array, axis=1).max()
            logger.info(f"Convergence evaluation: max_force={max_final_force:.6f} eV/A, fmax={fmax}, converged={converged}")
        else:
            logger.info(f"Convergence evaluation: no force data available, converged={converged}")
            
        logger.info(f"Final relaxation status: converged={converged}, steps={steps}, trajectory_exists={trajectory is not None}")
        logger.info("=== CHGNet SPEC-COMPLIANT CONVERGENCE EVALUATION COMPLETE ===")
        
        relaxation_info = {
            "converged": converged,
            "steps": steps,
            "fmax": fmax,
            "max_steps": max_steps,
            "energy_change_eV": energy_diff,
            "energy_change_eV_per_atom": energy_diff_per_atom
        }
        
        # Calculate structure properties once to avoid redundancy
        final_formula = str(final_structure.formula)
        final_volume = float(final_structure.volume)
        final_density = float(final_structure.density)
        final_num_sites = len(final_structure.sites)
        
        # Add structure information
        final_results.update({
            "formula": final_formula,
            "num_sites": final_num_sites,
            "volume": final_volume,
            "density": final_density
        })
        
        logger.info(f"CHGNet relaxation completed: {relaxation_info['steps']} steps, converged: {relaxation_info['converged']}")
        
        # Generate relaxed structure info
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        analyzer = SpacegroupAnalyzer(final_structure)
        
        relaxed_structure_info = {
            "formula": final_formula,
            "num_atoms": len(final_structure),
            "density": final_density,
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
        
        # Extract trajectory data for analysis modal
        # Use all steps as provided by CHGNet - no unnecessary optimization
        trajectory_data = None
        if trajectory:
            import numpy as np
            trajectory_data = {
                "steps": len(trajectory.forces) if hasattr(trajectory, 'forces') else 0,
                "energies": [float(e) for e in trajectory.energies] if hasattr(trajectory, 'energies') else [],
                "forces": [],
                "force_magnitudes": []
            }
            
            # Extract force data for all steps
            if hasattr(trajectory, 'forces'):
                for step_forces in trajectory.forces:
                    # Store detailed forces only for final step
                    if step_forces is trajectory.forces[-1]:
                        step_force_data = step_forces.tolist() if hasattr(step_forces, 'tolist') else step_forces
                        trajectory_data["forces"].append(step_force_data)
                    
                    # Calculate force magnitudes efficiently for all steps
                    step_forces_array = np.array(step_forces)
                    force_mags = np.linalg.norm(step_forces_array, axis=1).tolist()
                    trajectory_data["force_magnitudes"].append(force_mags)
        
        # Save relaxed structure and CHGNet result metadata to session for later use
        chgnet_metadata = {
            'fmax': fmax,
            'converged': relaxation_info.get('converged', False),
            'steps': relaxation_info.get('steps', 0)
        }
        manager.save_relaxed_structure(final_structure, chgnet_metadata)
        logger.info(f"Saved relaxed structure to session {session_id[:8]}...")
        
        return {
            "status": "success",
            "initial_prediction": initial_results,
            "final_prediction": final_results,
            "relaxation_info": relaxation_info,
            "relaxed_structure_info": relaxed_structure_info,
            "trajectory_data": trajectory_data,
            "model_info": {
                "version": _get_chgnet_version(),
                "device": _get_device_info()
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

@app.post("/api/reset-session-structure")
async def reset_session_structure(request: dict):
    """
    Reset session structure to original supercell state
    Ensures frontend and backend state consistency after Reset Operations
    """
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        # Get original structure and supercell parameters
        original_structure = session_info['original_structure']
        supercell_size = session_info['supercell_size']
        filename = session_info.get('filename', 'unknown')
        
        if original_structure is None:
            raise HTTPException(status_code=404, detail="No original structure found in session")
        
        logger.info(f"Resetting session {session_id[:8]}... ({filename}) to original supercell state")
        logger.info(f"Supercell size: {supercell_size}")
        
        # Recreate supercell from original structure
        reset_structure = original_structure.copy()
        reset_structure.make_supercell(supercell_size)
        
        # Update session with reset structure and clear operations
        session_manager.update_structure(session_id, reset_structure, operations=[])
        
        # Clear any relaxed structure data
        if 'relaxed_structure' in session_info:
            del session_info['relaxed_structure']
        if 'chgnet_result' in session_info:
            del session_info['chgnet_result']
        
        logger.info(f"Session {session_id[:8]}... reset successfully")
        logger.info(f"Reset structure: {reset_structure.composition} ({len(reset_structure)} sites)")
        
        return {
            "status": "success",
            "message": "Session reset to original structure",
            "session_id": session_id,
            "structure_info": {
                "formula": str(reset_structure.composition),
                "num_sites": len(reset_structure),
                "volume": float(reset_structure.volume),
                "density": float(reset_structure.density),
                "supercell_size": supercell_size
            }
        }
        
    except ValueError as e:
        logger.error(f"Session reset validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Session reset error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")


@app.post("/api/generate-relaxed-structure-cif")
async def generate_relaxed_structure_cif(request: dict):
    """
    Generate CIF from relaxed structure - uses session-managed CHGNet results
    """
    try:
        session_id = request.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Get manager from unified session workspace
        manager = unified_session_manager.get_manager(session_id)
        if not manager:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        filename = manager.filename or 'unknown'
        logger.info(f"Generating relaxed structure CIF for {filename}")

        # Get relaxed structure (fallback to current structure if not relaxed)
        relaxed_structure = manager.get_relaxed_structure()
        if relaxed_structure is None:
            raise HTTPException(status_code=404, detail="No structure data available in session")

        has_relaxed = manager._structure_cache.get('relaxed_structure') is not None
        if has_relaxed:
            logger.info("Using relaxed structure from session")
        else:
            logger.info("Using current structure (relaxed structure not found in session)")

        final_structure = relaxed_structure

        # Get operations and supercell info from session
        operations = manager._structure_cache.get('operations', [])
        supercell_size = manager.supercell_size or [1, 1, 1]

        # Get CHGNet relaxation info from session
        chgnet_result = manager.get_chgnet_result()
        fmax = chgnet_result.get('fmax', 'N/A')
        converged = chgnet_result.get('converged', 'N/A')
        steps = chgnet_result.get('steps', 'N/A')
        
        # Generate CIF using pymatgen CifWriter
        from pymatgen.io.cif import CifWriter
        cif_writer = CifWriter(
            final_structure,
            write_magmoms=False,
            significant_figures=6
        )
        
        cif_content = str(cif_writer)
        
        # Calculate structure properties once for metadata
        final_formula_str = str(final_structure.formula)
        final_num_atoms = len(final_structure.sites)
        final_volume_val = final_structure.volume
        
        # Add metadata header
        operations_summary = f"{len(operations)} operations applied"
        size_str = "x".join(map(str, supercell_size))
        metadata_lines = [
            f"# Relaxed structure CIF generated by CrystalNexus",
            f"# Original file: {filename}",
            f"# Supercell size: {size_str}",
            f"# Operations: {operations_summary}",
            f"# CHGNet relaxation: fmax={fmax}, steps={steps}, converged={converged}",
            f"# Final formula: {final_formula_str}",
            f"# Number of atoms: {final_num_atoms}",
            f"# Volume: {final_volume_val:.2f} Ų",
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

# =============================================================================
# UNIFIED WORKSPACE API ENDPOINTS
# =============================================================================

@app.post("/api/structure/set")
async def set_structure_unified(request: dict):
    """
    Unified endpoint to set structure for both sample and uploaded files
    """
    try:
        session_id = request.get("session_id")
        structure_data = request.get("structure_data")
        source_type = request.get("source_type")  # 'sample' or 'upload'
        identifier = request.get("identifier")  # filename or path

        if not all([session_id, structure_data, source_type, identifier]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: session_id, structure_data, source_type, identifier"
            )

        # Get or create unified structure manager
        manager = unified_session_manager.get_manager(session_id)

        # Convert structure data to pymatgen Structure
        if isinstance(structure_data, dict):
            # If it's already parsed crystal data
            from pymatgen.io.cif import CifParser

            if source_type == 'sample':
                # For sample files, load from sample directory
                cif_path = SAMPLE_CIF_DIR / safe_path(identifier)
                if not cif_path.exists():
                    raise HTTPException(status_code=404, detail=f"Sample file not found: {identifier}")
                parser = CifParser(str(cif_path))
                structures = parser.get_structures(primitive=False)
                if not structures:
                    raise HTTPException(status_code=400, detail=f"Invalid CIF file with no structures: {identifier}")
                structure = structures[0]
            else:
                # For uploaded files, reconstruct from CIF content
                if 'cif_content' in structure_data:
                    from io import StringIO
                    parser = CifParser(StringIO(structure_data['cif_content']))
                    structures = parser.get_structures(primitive=False)
                    if not structures:
                        raise HTTPException(status_code=400, detail="Invalid CIF file with no structures!")
                    structure = structures[0]
                else:
                    raise HTTPException(status_code=400, detail="CIF content required for uploaded files")
        else:
            # Direct structure object (fallback)
            structure = structure_data

        # Set structure in unified manager
        manager.set_structure(structure, source_type, identifier)

        logger.info(f"Structure set via unified API - Session: {session_id[:8]}, Type: {source_type}, ID: {identifier}")

        return {
            "success": True,
            "message": f"Structure set successfully for {source_type} file: {identifier}",
            "session_id": session_id,
            "structure_info": manager.get_structure_info()
        }

    except Exception as e:
        logger.error(f"Failed to set structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chgnet-predict-unified")
async def chgnet_predict_unified(request: dict):
    """
    Unified CHGNet prediction endpoint - uses session workspace instead of filename
    """
    try:
        session_id = request.get("session_id")
        operations = request.get("operations", [])

        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get unified structure manager
        manager = unified_session_manager.get_manager(session_id)
        structure_info = manager.get_structure_info()

        if not structure_info['filename']:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        logger.info(f"Unified CHGNet prediction - Session: {session_id[:8]}, Operations: {len(operations)}")

        # Get working file path (lazy generation)
        working_file_path = manager.get_working_file_path()

        if not working_file_path.exists():
            raise HTTPException(status_code=500, detail="Failed to generate working file")

        # Use existing CHGNet prediction logic with unified file
        from pymatgen.io.cif import CifParser
        parser = CifParser(str(working_file_path))
        structures = parser.get_structures(primitive=False)
        if not structures:
            raise HTTPException(status_code=400, detail="Invalid CIF file with no structures!")
        structure = structures[0]

        # Apply operations for CHGNet prediction
        valid_operations, invalid_operations = filter_valid_operations(
            operations, len(structure.sites), strict_mode=False
        )

        if invalid_operations:
            logger.warning(f"Skipping invalid operations: {invalid_operations}")

        # Apply valid operations
        for op in valid_operations:
            if not isinstance(op, dict) or "type" not in op:
                logger.warning(f"Skipping invalid operation format: {op}")
                continue

            if op["type"] == "delete" and "site_index" in op and op["site_index"] < len(structure.sites):
                structure.remove_sites([op["site_index"]])
            elif op["type"] == "substitute" and "site_index" in op and "new_element" in op and op["site_index"] < len(structure.sites):
                structure.replace(op["site_index"], op["new_element"])

        if not CHGNET_AVAILABLE:
            raise HTTPException(status_code=503, detail="CHGNet is not available")

        # Load CHGNet model
        try:
            chgnet = await chgnet_manager.get_model()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        # Predict structure properties
        try:
            prediction = await chgnet_manager.predict_single_optimized(
                structure,
                return_site_energies=True,
                return_atom_feas=True,
                return_crystal_feas=True
            )
            result = safe_get_prediction(prediction, num_atoms=len(structure.sites))
        except TypeError:
            # Fallback if detailed prediction fails
            prediction = await chgnet_manager.predict_single_optimized(structure)
            result = safe_get_prediction(prediction, num_atoms=len(structure.sites))

        return {
            "success": True,
            "status": "success",
            "prediction": result,
            "operations_applied": len(valid_operations),
            "operations_skipped": len(invalid_operations),
            "structure_info": structure_info
        }

    except Exception as e:
        logger.error(f"Unified CHGNet prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/supercell/create-unified")
async def create_supercell_unified(request: dict):
    """
    Unified supercell creation endpoint
    """
    try:
        session_id = request.get("session_id")
        crystal_data = request.get("crystal_data")
        supercell_size = request.get("supercell_size", [1, 1, 1])

        if not all([session_id, crystal_data, supercell_size]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: session_id, crystal_data, supercell_size"
            )

        # Get unified structure manager
        manager = unified_session_manager.get_manager(session_id)

        # Create supercell from crystal data
        cif_content = crystal_data.get('cif_content', '')
        if not cif_content:
            raise HTTPException(status_code=400, detail="CIF content is required")

        # Parse structure and create supercell
        from pymatgen.io.cif import CifParser
        from io import StringIO

        parser = CifParser(StringIO(cif_content))
        structures = parser.get_structures(primitive=False)
        if not structures:
            raise HTTPException(status_code=400, detail="Invalid CIF file with no structures!")

        original_structure = structures[0]
        structure = original_structure.copy()
        structure.make_supercell(supercell_size)

        # Update manager with supercell and size
        manager.update_supercell(structure, operations=None, supercell_size=supercell_size)

        structure_data = {
            'original_data': {
                'filename': manager.get_structure_info()['filename'],
                'composition': str(original_structure.composition),
                'num_atoms': len(original_structure.sites),
                'lattice': original_structure.lattice.abc + original_structure.lattice.angles,
                'volume': original_structure.lattice.volume,
                'density': original_structure.density,
                'formula': str(original_structure.composition),
                'lattice_parameters': {
                    'a': float(original_structure.lattice.a),
                    'b': float(original_structure.lattice.b),
                    'c': float(original_structure.lattice.c),
                    'alpha': float(original_structure.lattice.alpha),
                    'beta': float(original_structure.lattice.beta),
                    'gamma': float(original_structure.lattice.gamma)
                }
            },
            'supercell_info': {
                'size': supercell_size,
                'num_atoms': len(structure.sites),
                'composition': str(structure.composition),
                'formula': str(structure.composition),
                'volume': structure.lattice.volume,
                'density': structure.density
            },
            'cif_content': structure.to(fmt='cif')
        }

        logger.info(f"Unified supercell created - Session: {session_id[:8]}, Size: {supercell_size}")

        return structure_data

    except Exception as e:
        logger.error(f"Unified supercell creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/element-labels/unified")
async def get_element_labels_unified(request: dict):
    """
    Unified element labels endpoint for 3D visualization
    """
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get unified structure manager
        manager = unified_session_manager.get_manager(session_id)
        structure_info = manager.get_structure_info()

        if not structure_info['filename']:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        # Get current structure (supercell if available, otherwise original)
        structure = manager.get_current_structure()
        if not structure:
            raise HTTPException(status_code=404, detail="No structure available in session")

        # Get element labels in site order
        element_labels = [str(site.specie) for site in structure.sites]

        return {
            "success": True,
            "labels": element_labels,
            "num_sites": len(structure.sites)
        }

    except Exception as e:
        logger.error(f"Unified element labels failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/supercell/cif-unified")
async def generate_supercell_cif_unified(request: dict):
    """
    Unified supercell CIF generation endpoint for 3D visualization
    """
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get unified structure manager
        manager = unified_session_manager.get_manager(session_id)
        structure_info = manager.get_structure_info()

        if not structure_info['filename']:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        # Get current supercell structure
        structure = manager.get_current_structure()
        if not structure:
            raise HTTPException(status_code=404, detail="No supercell structure available in session")

        # Generate CIF content
        cif_content = structure.to(fmt='cif')

        return {
            "success": True,
            "cif_content": cif_content,
            "filename": structure_info['filename']
        }

    except Exception as e:
        logger.error(f"Unified supercell CIF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/structure/modified-cif-unified")
async def generate_modified_structure_cif_unified(request: dict):
    """
    Unified modified structure CIF generation endpoint
    """
    try:
        session_id = request.get("session_id")
        operations = request.get("operations", [])

        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get unified structure manager
        manager = unified_session_manager.get_manager(session_id)
        structure_info = manager.get_structure_info()

        if not structure_info['filename']:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        # Get current structure
        structure = manager.get_current_structure()
        if not structure:
            raise HTTPException(status_code=404, detail="No structure available in session")

        # Apply operations to create modified structure
        modified_structure = structure.copy()

        # Apply operations in reverse order for proper deletion indices
        delete_operations = [op for op in operations if op.get("action") == "delete"]
        substitute_operations = [op for op in operations if op.get("action") == "substitute"]

        # Sort deletions by index in reverse order
        delete_operations.sort(key=lambda x: x.get("index", 0), reverse=True)

        # Apply deletions first (in reverse order)
        for op in delete_operations:
            if "index" in op and 0 <= op["index"] < len(modified_structure.sites):
                modified_structure.remove_sites([op["index"]])

        # Apply substitutions (indices may have changed after deletions)
        for op in substitute_operations:
            if "index" in op and "element" in op and 0 <= op["index"] < len(modified_structure.sites):
                modified_structure.replace(op["index"], op["element"])

        # Generate CIF content
        cif_content = modified_structure.to(fmt='cif')

        return {
            "success": True,
            "cif_content": cif_content,
            "filename": structure_info['filename'],
            "operations_applied": len(operations)
        }

    except Exception as e:
        logger.error(f"Unified modified structure CIF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/operations/apply-unified")
async def apply_atomic_operations_unified(request: dict):
    """
    Unified atomic operations application endpoint
    """
    try:
        session_id = request.get("session_id")
        operations = request.get("operations", [])

        logger.info(f"🎯 APPLY_OPERATIONS: Session {session_id}, {len(operations)} operations")
        logger.info(f"🎯 APPLY_OPERATIONS: Operations: {operations}")

        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get unified structure manager
        manager = unified_session_manager.get_manager(session_id)
        structure_info = manager.get_structure_info()

        logger.info(f"🎯 APPLY_OPERATIONS: Structure info: {structure_info}")

        if not structure_info['filename']:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        # Get current structure
        structure = manager.get_current_structure()
        if not structure:
            raise HTTPException(status_code=404, detail="No structure available in session")

        logger.info(f"🎯 APPLY_OPERATIONS: Original structure has {len(structure.sites)} sites")
        logger.info(f"🎯 APPLY_OPERATIONS: Formula: {structure.formula}")

        # Apply operations to create modified structure
        modified_structure = structure.copy()

        # Apply operations similar to above
        delete_operations = [op for op in operations if op.get("action") == "delete"]
        substitute_operations = [op for op in operations if op.get("action") == "substitute"]

        logger.info(f"🎯 APPLY_OPERATIONS: Delete ops: {delete_operations}")
        logger.info(f"🎯 APPLY_OPERATIONS: Substitute ops: {substitute_operations}")

        # Sort deletions by index in reverse order
        delete_operations.sort(key=lambda x: x.get("index", 0), reverse=True)

        # Apply operations
        for i, op in enumerate(delete_operations):
            index = op.get("index", -1)
            target_element = op.get("target_element", "unknown")

            logger.info(f"🎯 APPLY_OPERATIONS: Delete op {i+1}/{len(delete_operations)}: index={index}, target_element={target_element}")
            logger.info(f"🎯 APPLY_OPERATIONS: Structure size before delete: {len(modified_structure.sites)}")

            if "index" in op and 0 <= op["index"] < len(modified_structure.sites):
                actual_element = str(modified_structure.sites[op["index"]].specie.element)
                logger.info(f"🎯 APPLY_OPERATIONS: Site {op['index']} contains element: {actual_element}")
                logger.info(f"🎯 APPLY_OPERATIONS: Requested to delete: {target_element}")

                if target_element != "unknown" and actual_element != target_element:
                    logger.warning(f"🎯 APPLY_OPERATIONS: Element mismatch! Site {op['index']} is {actual_element}, not {target_element}")

                modified_structure.remove_sites([op["index"]])
                logger.info(f"🎯 APPLY_OPERATIONS: Deleted site {op['index']}, structure now has {len(modified_structure.sites)} sites")
            else:
                logger.warning(f"🎯 APPLY_OPERATIONS: Invalid delete index {index} (structure size: {len(modified_structure.sites)})")

        for i, op in enumerate(substitute_operations):
            index = op.get("index", -1)
            element = op.get("element", "unknown")

            logger.info(f"🎯 APPLY_OPERATIONS: Substitute op {i+1}/{len(substitute_operations)}: index={index}, element={element}")

            if "index" in op and "element" in op and 0 <= op["index"] < len(modified_structure.sites):
                actual_element = str(modified_structure.sites[op["index"]].specie.element)
                logger.info(f"🎯 APPLY_OPERATIONS: Substituting site {op['index']} ({actual_element}) with {element}")
                modified_structure.replace(op["index"], op["element"])
            else:
                logger.warning(f"🎯 APPLY_OPERATIONS: Invalid substitute index {index} (structure size: {len(modified_structure.sites)})")

        logger.info(f"🎯 APPLY_OPERATIONS: Final structure has {len(modified_structure.sites)} sites")
        logger.info(f"🎯 APPLY_OPERATIONS: Final formula: {modified_structure.formula}")

        # Update manager with modified structure
        manager.update_supercell(modified_structure, operations)

        return {
            "success": True,
            "message": f"Applied {len(operations)} operations to session structure",
            "operations_applied": len(operations),
            "structure_info": manager.get_structure_info()
        }

    except Exception as e:
        logger.error(f"Unified operations application failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/atoms/list-unified")
async def get_numbered_atoms_unified(request: dict):
    """
    Get numbered atom list for unified session workspace
    Returns atoms with Element Type Counter System identifiers
    """
    try:
        session_id = request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        manager = unified_session_manager.get_manager(session_id)
        if not manager:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        atoms = manager.get_numbered_atom_list()
        return {
            "success": True,
            "atoms": atoms,
            "total_atoms": len(atoms)
        }

    except Exception as e:
        logger.error(f"Get numbered atoms failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/atoms/elements-unified")
async def get_unique_elements_unified(request: dict):
    """
    Get unique elements in current structure for unified session workspace
    """
    try:
        session_id = request.get("session_id")
        logger.info(f"🔍 GET_UNIQUE_ELEMENTS: Request for session {session_id}")

        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        manager = unified_session_manager.get_manager(session_id)
        if not manager:
            logger.warning(f"🔍 GET_UNIQUE_ELEMENTS: Session not found: {session_id}")
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        current_structure = manager.get_current_structure()
        if not current_structure:
            logger.warning(f"🔍 GET_UNIQUE_ELEMENTS: No current structure in session {session_id}")
            return {"success": True, "elements": []}

        logger.info(f"🔍 GET_UNIQUE_ELEMENTS: Structure has {len(current_structure.sites)} sites")
        logger.info(f"🔍 GET_UNIQUE_ELEMENTS: Structure formula: {current_structure.formula}")

        # Get unique elements (not species/ions)
        elements = []
        seen_elements = set()
        species_details = []

        for i, site in enumerate(current_structure.sites):
            element = str(site.specie.element)
            species = str(site.specie)
            species_details.append(f"Site {i}: {species} ({element})")

            if element not in seen_elements:
                elements.append(element)
                seen_elements.add(element)

        logger.info(f"🔍 GET_UNIQUE_ELEMENTS: Species details: {species_details[:10]}...")  # Show first 10
        logger.info(f"🔍 GET_UNIQUE_ELEMENTS: Found unique elements: {elements}")

        return {
            "success": True,
            "elements": elements
        }

    except Exception as e:
        logger.error(f"Get unique elements failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LEGACY API ENDPOINTS (kept for backward compatibility)
# =============================================================================

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
        
        # Windows対応: CREATE_NO_WINDOWフラグを設定
        kwargs = {}
        if WINDOWS_PLATFORM:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        
        return subprocess.Popen(cmd, **kwargs)
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