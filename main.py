import os
import asyncio
import functools
import itertools
import json
import math
import logging
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from scipy.spatial import Voronoi

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from pymatgen.core import Structure, Element, PeriodicSite
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.defects.utils import remove_collisions, cluster_nodes
from analytics_db import analytics_db

# Logging configuration (early initialization)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import platform
WINDOWS_PLATFORM = platform.system() == "Windows"

# MLIP backends. Importing the registry also performs the model imports and
# the Windows torch compatibility setup.
from mlip import (MODEL_REGISTRY, MLIPError, ModelUnavailableError,
                  UnknownModelError, get_backend)
from mlip.chgnet_backend import CHGNetBackend
from mlip.base import CAP_STRESS
from mlip.registry import (is_available as _model_is_available, default_model_id,
                           list_models)
from mlip.executor import inference_executor as _inference_executor

# Configuration constants - must be defined before use
# Memory management settings
SESSION_CLEANUP_HOURS = int(os.getenv('SESSION_CLEANUP_HOURS', '6'))
PERIODIC_CLEANUP_INTERVAL = int(os.getenv('PERIODIC_CLEANUP_INTERVAL', '1800'))  # 30 minutes
# How long to keep analytics_db rows (visitor IP/User-Agent history,
# uploaded filenames/formulas). Nothing previously bounded this at all.
ANALYTICS_RETENTION_DAYS = int(os.getenv('ANALYTICS_RETENTION_DAYS', '90'))

# Atom insertion settings
MAX_VOID_CANDIDATES = int(os.getenv('MAX_VOID_CANDIDATES', '50'))
MAX_INSERTION_BATCH = int(os.getenv('MAX_INSERTION_BATCH', '8'))

# Atom substitution/deletion Auto-mode settings
MAX_CANDIDATE_BATCH = int(os.getenv('MAX_CANDIDATE_BATCH', '8'))

# Structure relaxation settings
# LBFGS reaches the same or a lower energy minimum than FIRE on every
# structure we benchmarked, and needs fewer force evaluations in most
# cases (BaTiO3 2x2x2 + 1 substitution: 35 frames/24.6s with FIRE vs
# 19 frames/10.1s with LBFGS, final energy -337.8715 vs -337.8779 eV).
# Set RELAX_OPTIMIZER=FIRE to restore the previous default.
RELAX_OPTIMIZER = os.getenv('RELAX_OPTIMIZER', 'LBFGS')
VALID_RELAX_OPTIMIZERS = ('LBFGS', 'FIRE', 'BFGS', 'LBFGSLineSearch')
MAX_RELAX_STEPS = int(os.getenv('MAX_RELAX_STEPS', '500'))
MIN_RELAX_FMAX = float(os.getenv('MIN_RELAX_FMAX', '0.01'))
MAX_RELAX_FMAX = float(os.getenv('MAX_RELAX_FMAX', '1.0'))
MAX_RELAX_ATOMS = int(os.getenv('MAX_RELAX_ATOMS', '400'))
# Prediction (/api/predict, the Auto-mode candidate/insertion-energy
# screening endpoints, and the insertion-void search) is comparably
# expensive to relaxation per structure, so it shares the same cap.
MAX_PREDICT_ATOMS = MAX_RELAX_ATOMS
# Minimum distance (A, periodic) between an inserted atom and any existing
# atom. Below the candidate generator's min_dist=0.9 so UI candidates pass.
MIN_INSERT_DISTANCE = 0.5
RELAX_TIMEOUT_SECONDS = float(os.getenv('RELAX_TIMEOUT_SECONDS', '600'))
# Wall-clock limit for the optional elastic-constants step that follows a relaxation.
ELASTIC_TIMEOUT_SECONDS = float(os.getenv('ELASTIC_TIMEOUT_SECONDS', '600'))
ELASTIC_STRAIN_MIN_PCT = 0.1
ELASTIC_STRAIN_MAX_PCT = 2.0
ELASTIC_FMAX_WARN = 0.01
# Initialize once as global variable
def _initial_allowed_elements() -> Set[str]:
    """Structure-edit element whitelist: the union of every AVAILABLE model's
    supported elements. Model-specific limits are enforced at prediction/
    relaxation time (422). No model is loaded here: CHGNet's list is derived
    from a fixed atomic-number range and MACE's is a static set until loaded.
    Falls back to CHGNet's own list if no model is available."""
    elements: Set[str] = set()
    for model_id, cls in MODEL_REGISTRY.items():
        if cls.is_available():
            elements.update(cls().supported_elements())
    if not elements:
        elements = set(CHGNetBackend().supported_elements())
    return elements


ALLOWED_ELEMENTS: Set[str] = _initial_allowed_elements()

# The single-worker inference executor lives in mlip.executor (imported above
# as _inference_executor) so every backend shares ONE executor.


# Relaxation can run for minutes; block a second concurrent relaxation
# immediately with a clear error instead of silently queuing behind the
# only MLIP worker thread (which would also starve Auto-mode screening).
_relax_semaphore = asyncio.Semaphore(1)

# session_id -> {"state": {...}} published by RelaxationWatcher from the
# executor thread and polled/streamed by GET /api/relax-progress/{id}.
_relax_progress: Dict[str, dict] = {}

def validate_atomic_operation(operation, structure_size, operation_index=None, allowed=None):
    """
    Validate a single atomic operation
    Returns: (is_valid, error_message)
    """
    try:
        # Check required fields
        if "action" not in operation:
            return False, "missing 'action' field"
        
        action = operation["action"]
        if action not in ["substitute", "delete", "insert"]:
            return False, f"invalid action '{action}', must be 'substitute', 'delete' or 'insert'"
            
        if action in ["substitute", "delete"]:
            if "index" not in operation:
                return False, "missing 'index' field"
            
            # Validate index type and range
            index = operation["index"]
            if isinstance(index, bool) or not isinstance(index, int):
                return False, f"index must be integer, got {type(index).__name__}: {index}"
            if index < 0:
                return False, f"index cannot be negative: {index}"
            if index >= structure_size:
                return False, f"index {index} out of range (max: {structure_size-1})"
                
        if action == "insert":
            if "coords" not in operation:
                return False, "missing 'coords' field"
            try:
                validate_frac_coords(operation["coords"], "coords")
            except ValueError as e:
                return False, str(e)
            
        # Validate substitution target element
        if action in ["substitute", "insert"]:
            if "to" not in operation:
                return False, "missing 'to' element"
            try:
                validate_element(operation["to"], allowed)
            except ValueError as e:
                return False, str(e)
        
        return True, None
    except Exception as e:
        return False, f"unexpected validation error: {str(e)}"

def _site_element_symbol(specie) -> str:
    """
    Clean element symbol for a site's species using PyMatGen's unified
    approach (e.g. Species Ba2+ -> "Ba", plain Element Nd -> "Nd"). Species
    (an Element with an oxidation state, e.g. from a CIF with oxidation-state
    labels) expose the underlying Element via `.element`; plain Element
    instances don't.

    Shared by build_element_labels, apply_operations_to_structure's
    same-element substitution check, /api/evaluate-candidate-energies, and
    /api/equivalent-sites, so all of them agree on what "the element at a
    site" means.
    """
    if hasattr(specie, 'element'):
        return str(specie.element)  # Species type (e.g., Ba2+ -> Ba)
    return str(specie)              # Element type (e.g., Nd -> Nd)


def build_element_labels(structure: Structure) -> "tuple[List[str], List[str]]":
    """
    Generate per-element sequential labels (Ba0, Ti0, O0, O1, O2, ...) directly
    from a Structure's sites, in site order.

    Site order here is the same order MLIP's site-energy/force results
    come back in and the same order the client's currentLabels array is
    built in, so these labels stay aligned with both. Extracted from
    /api/get-element-labels so /api/apply-atomic-operations can return the
    same labels the client would otherwise have to re-derive itself.

    Returns:
        (labels, unique_elements) -- unique_elements is in first-appearance
        order, not alphabetical; sort it yourself if you need the latter
        (the client's own extractUniqueElements() already does).
    """
    labels: List[str] = []
    element_counts: Dict[str, int] = {}

    for site in structure.sites:
        element = _site_element_symbol(site.specie)

        if element not in element_counts:
            element_counts[element] = 0
        else:
            element_counts[element] += 1

        labels.append(f"{element}{element_counts[element]}")

    return labels, list(element_counts.keys())


def apply_element_labels_to_structure(structure: Structure) -> Structure:
    """
    Set each site's CIF label to match the same per-element sequential
    numbering build_element_labels()/the client's currentLabels use (Ba0,
    Ba1, Ti0, O0, O1, ...), so CifWriter emits unique, UI-consistent
    _atom_site_label values instead of pymatgen's default (which repeats
    the bare element symbol for every site of that element and produces
    invalid duplicate labels in the CIF). Mutates and returns `structure`.
    """
    labels, _ = build_element_labels(structure)
    for site, label in zip(structure.sites, labels):
        site.label = label
    return structure


def substitute_site(structure: Structure, site_index: int, new_element: str) -> bool:
    """
    Replace `structure[site_index]` with a neutral `new_element` at the same
    fractional coordinates. Shared by apply_operations_to_structure (the
    authoritative replay path) and /api/evaluate-candidate-energies (the Auto
    mode screening path) so both potential loss sites stay in sync.

    Returns True if the replaced site carried information this discards: an
    oxidation state (pymatgen Species, e.g. Ba2+) or non-empty site
    properties (e.g. magmom). The new site is always a plain Element with no
    properties -- pymatgen does not support preserving these across a
    species substitution.
    """
    site = structure[site_index]
    old_specie = site.specie
    lossy = bool(getattr(old_specie, "oxi_state", 0)) or bool(site.properties)
    structure[site_index] = Element(new_element), site.frac_coords
    return lossy


def apply_operations_to_structure(structure: Structure, operations: List[dict],
                                   strict_mode: bool = False,
                                   allowed: Optional[Set[str]] = None) -> "tuple[int, List[str], List[str]]":
    """
    Apply atomic operations to `structure` in place, in the exact order given.

    Each operation's 'index' refers to the structure state produced by every
    operation that precedes it in this list -- the same "live array" frame
    the client uses when it records operations, since a delete/insert
    immediately shifts what index everything after it refers to on the
    client. This deliberately replaces the previous approach of reordering
    the batch by descending index before a single-pass apply: that
    reordering only kept a batch's *own* indices consistent with each
    other, while the client had already recorded those indices against a
    structure it was progressively mutating -- so replaying out of order
    silently applied an operation to the wrong site whenever a delete or
    insert was followed by another operation in the same batch.

    Args:
        structure: Structure to mutate in place.
        operations: Atomic operations in client-recorded order.
        strict_mode: If True, raise ValueError on the first invalid
            operation (fail-fast; used for the authoritative session
            endpoint). If False, skip invalid operations and keep going,
            returning a description of each one skipped.
        allowed: Element whitelist for substitute/insert targets (the
            session model's supported elements). None -> ALLOWED_ELEMENTS.

    Returns:
        (operations_applied, skipped_messages, property_warnings)
    """
    operations_applied = 0
    skipped: List[str] = []
    property_warnings: List[str] = []

    for i, operation in enumerate(operations):
        is_valid, error_msg = validate_atomic_operation(operation, len(structure), allowed=allowed)
        if not is_valid:
            message = f"Operation {i + 1}: {error_msg}"
            if strict_mode:
                raise ValueError(message)
            skipped.append(message)
            continue

        action = operation["action"]
        site_index = operation.get("index")

        if action == "substitute":
            # validate_atomic_operation above already confirmed operation["to"]
            # passes validate_element(); write the *stripped* value back so a
            # symbol with leading/trailing whitespace (e.g. " Fe") doesn't
            # reach pymatgen's Element() unstripped and raise an unhandled
            # ValueError (500).
            new_element = validate_element(operation["to"], allowed)
            old_element = structure[site_index].specie
            old_symbol = _site_element_symbol(old_element)
            # validate_atomic_operation has no access to `structure`, so it
            # can't know the *current* element at this index -- this check
            # has to live here instead (see plan doc / commit message).
            if new_element == old_symbol:
                message = (f"Operation {i + 1}: cannot substitute site {site_index} "
                           f"({old_symbol}) with the same element")
                if strict_mode:
                    raise ValueError(message)
                skipped.append(message)
                continue
            if substitute_site(structure, site_index, new_element):
                property_warnings.append(
                    f"Operation {i + 1}: site {site_index} ({old_element}) had an "
                    f"oxidation state and/or site properties (e.g. magmom) that "
                    f"substitution to {new_element} does not preserve"
                )
            logger.info(f"Substituted site {site_index}: {old_element} -> {new_element}")
        elif action == "delete":
            deleted_element = structure[site_index].specie
            structure.remove_sites([site_index])
            logger.info(f"Deleted site {site_index} ({deleted_element})")
        elif action == "insert":
            new_element = validate_element(operation["to"], allowed)
            frac_coords = [float(c) % 1.0 for c in operation["coords"]]
            # Defense in depth: the UI only sends vetted candidates, but a
            # direct API call could drop an atom on top of an existing one.
            # get_sites_in_sphere accounts for periodic images.
            cart_coords = structure.lattice.get_cartesian_coords(frac_coords)
            if structure.get_sites_in_sphere(cart_coords, MIN_INSERT_DISTANCE):
                message = (f"Operation {i + 1}: cannot insert {new_element} at "
                           f"{frac_coords}; closer than {MIN_INSERT_DISTANCE} A "
                           f"to an existing atom")
                if strict_mode:
                    raise ValueError(message)
                skipped.append(message)
                continue
            structure.append(Element(new_element), frac_coords)
            logger.info(f"Inserted {new_element} at {frac_coords}")

        operations_applied += 1

    if strict_mode and len(structure.sites) == 0:
        raise ValueError("Operation would delete all atoms; at least one atom must remain.")

    return operations_applied, skipped, property_warnings


def _compute_equivalent_indices(structure: Structure) -> "Optional[List[List[int]]]":
    """
    Group site indices into symmetry-equivalent classes via SpacegroupAnalyzer.

    Used by /api/equivalent-sites to let the Auto-mode substitution/deletion
    sweep (runOptimalCandidateSweep in templates/index.html) skip redundant
    MLIP evaluation of candidate sites that are symmetry-equivalent:
    substituting/deleting any one site in a class produces a structure that
    is symmetrically identical to substituting/deleting any other site in
    that same class, so scoring more than one representative per class is
    wasted MLIP work (see plan doc / commit message for measurements).

    A per-call symprec of 0.01 is used -- this matches SpacegroupAnalyzer's
    own default, and is a practical tolerance for as-loaded/supercell
    structures (CIF-derived coordinates are rarely exact to machine
    precision).

    Returns None if symmetry analysis fails (e.g. an unusual structure
    SpacegroupAnalyzer can't handle); callers should fall back to treating
    every site as its own singleton equivalence class in that case, which
    matches the old "evaluate everything" behavior.
    """
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
        sym = analyzer.get_symmetrized_structure()
        return list(sym.equivalent_indices)
    except Exception:
        return None


# Background task for periodic cleanup
async def periodic_cleanup_task():
    """Background task that runs periodic memory cleanup"""
    while True:
        try:
            await asyncio.sleep(PERIODIC_CLEANUP_INTERVAL)  # Configurable cleanup interval
            logger.info("Starting periodic cleanup...")
            
            # Clean up old sessions
            session_manager.cleanup_old_sessions()

            # Purge analytics rows past the retention window (sqlite write, offload)
            await asyncio.to_thread(analytics_db.purge_old_data, ANALYTICS_RETENTION_DAYS)

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

    # Check dangerous characters. Control characters (CR/LF in particular)
    # are rejected here too: this name is later echoed back into a
    # Content-Disposition header and a CIF comment line (see
    # sanitize_display_filename), and neither the '/'/'\\' checks below
    # nor endswith('.cif') would otherwise catch a name like
    # "a\r\nX-Injected: 1.cif".
    if not safe_name or '/' in safe_name or '\\' in safe_name or not safe_name.isprintable():
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

    # Reject absolute paths (including Windows drive-absolute forms like 'C:/...')
    if ':' in filepath or Path(filepath).is_absolute():
        raise ValueError("Invalid path: absolute paths are not allowed")
    
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

_MSONABLE_MARKER_KEYS = ("@module", "@class", "@callable", "@bound")

def _reject_msonable_payload(obj, *, depth: int = 0, max_depth: int = 25) -> None:
    """
    Recursively reject any nested dict carrying MSONable/MontyDecoder
    marker keys ("@module"/"@class"/"@callable"/"@bound"), anywhere below
    the top level of obj.

    Why: pymatgen's Structure.from_dict() round-trips each site's
    `properties` dict through monty.json.MontyDecoder (see
    PeriodicSite.from_dict in pymatgen/core/sites.py), which performs a
    dynamic `__import__(modname)` + `getattr` + `cls.from_dict(...)` on ANY
    dict shaped like {"@module": ..., "@class": ...} it finds -- with
    modname/classname taken verbatim from the payload. A shallow check of
    only the outer Structure dict (as done by the caller before this) does
    not see markers nested inside `sites[*].properties`, so this walks the
    whole payload to close that gap.

    The caller is expected to have already validated/consumed the
    top-level dict's own @module/@class (that's the legitimate
    pymatgen.core.structure.Structure marker), so depth=0's own keys are
    intentionally not checked here -- only what's nested inside its values.

    Raises:
        HTTPException(400) if a marker key is found below the top level,
        or the payload nests deeper than max_depth (a cheap guard against
        a pathological/recursion-bomb payload).
    """
    if depth > max_depth:
        raise HTTPException(status_code=400, detail="Invalid structure_data: nesting too deep")
    if isinstance(obj, dict):
        if depth > 0 and any(key in obj for key in _MSONABLE_MARKER_KEYS):
            raise HTTPException(
                status_code=400,
                detail="Invalid structure_data: nested MSONable markers are not allowed"
            )
        for value in obj.values():
            _reject_msonable_payload(value, depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, list):
        for item in obj:
            _reject_msonable_payload(item, depth=depth + 1, max_depth=max_depth)

def sanitize_display_filename(filename: str) -> str:
    """
    Strip characters that could break out of a single Content-Disposition
    header value or a single '# ...' CIF comment line (CR/LF and other
    control characters).

    This is NOT a path-safety check -- safe_path()/safe_filename() above
    cover that wherever a filename touches the filesystem. This exists
    because several CIF-generation endpoints echo the client-supplied
    filename into both a response header and the generated CIF body's
    metadata comment even when the actual structure came from the
    session (skipping safe_path() entirely, since no filesystem lookup
    happens on that path) -- an unsanitized value there let a newline
    inject an extra line into the downloaded CIF file.
    """
    if not isinstance(filename, str):
        return str(filename)
    return "".join(ch for ch in filename if ch.isprintable())

def build_content_disposition(display_name: str) -> str:
    """
    Build a Content-Disposition header value that is always safe to encode
    (headers are Latin-1 in this stack) and safe to parse as a single
    unquoted-or-quoted value, even when display_name contains non-ASCII
    characters, spaces, quotes, or path separators.

    Provides both a quoted ASCII-safe fallback filename and an RFC 5987
    filename* for clients that support UTF-8 names.
    """
    import re
    from urllib.parse import quote

    # Never let a path separator escape into the filename.
    safe_display = display_name.replace('\\', '_').replace('/', '_')
    ascii_fallback = safe_display.encode('ascii', errors='ignore').decode('ascii')
    ascii_fallback = re.sub(r'[\"\\\\]', '_', ascii_fallback).strip() or 'structure.cif'
    encoded_utf8 = quote(safe_display, safe='')
    return f'inline; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded_utf8}'

def validate_frac_coords(coords, label: str = "coords") -> List[float]:
    """
    Validate a fractional-coordinate triple before it reaches
    Structure.append()/pymatgen, which otherwise raises an unhandled
    exception (500) for a malformed shape, a non-numeric entry, or a
    non-finite value (e.g. float('inf') % 1.0 -> NaN, silently accepted
    as a coordinate today).
    """
    if not isinstance(coords, (list, tuple)) or len(coords) != 3:
        raise ValueError(f"{label} must be a list of 3 numbers")
    result = []
    for c in coords:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise ValueError(f"{label} values must be numbers, got {type(c).__name__}")
        c = float(c)
        if not np.isfinite(c):
            raise ValueError(f"{label} values must be finite numbers")
        result.append(c)
    return result

def validate_supercell_size(supercell_size: List[int]) -> List[int]:
    """Validate supercell size"""
    if not isinstance(supercell_size, list) or len(supercell_size) != 3:
        raise ValueError("Supercell size must be a list of 3 integers")

    for dim in supercell_size:
        # bool is a subclass of int in Python (isinstance(True, int) is
        # True), so without this explicit check `True` would silently pass
        # as dimension 1 -- mirror the same guard validate_atomic_operation
        # uses for the 'index' field.
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 or dim > MAX_SUPERCELL_DIM:
            raise ValueError(f"Supercell dimensions must be between 1 and {MAX_SUPERCELL_DIM}")

    return supercell_size

def check_supercell_site_limit(base_num_sites: int, supercell_size: List[int]) -> None:
    """
    Reject a supercell expansion whose total site count would exceed
    MAX_TOTAL_SITES.

    validate_supercell_size() bounds each dimension independently
    (<= MAX_SUPERCELL_DIM), but never their product: a 20-atom cell at the
    per-dimension limit (10x10x10) still expands to 20,000 atoms. This
    catches that multiplicative blow-up before make_supercell() allocates
    it, and must be called on every path that runs make_supercell() with a
    size that ultimately came from the request body.
    """
    scaling_factor = 1
    for dim in supercell_size:
        scaling_factor *= dim
    total_sites = base_num_sites * scaling_factor
    if total_sites > MAX_TOTAL_SITES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Requested supercell would contain {total_sites} atoms, "
                f"exceeding the limit of {MAX_TOTAL_SITES}"
            )
        )

def validate_element(element: str, allowed: Optional[Set[str]] = None) -> str:
    """Element validation (injection protection).

    ``allowed`` is the session model's supported-element set; when omitted the
    union whitelist ALLOWED_ELEMENTS is used."""
    if not isinstance(element, str):
        raise ValueError("Element must be a string")
    
    element = element.strip()
    if element not in (ALLOWED_ELEMENTS if allowed is None else allowed):
        raise ValueError(f"Element '{element}' is not supported by the ML potential")
    
    return element

def validate_occupancy(structure) -> None:
    """Validate that all atomic sites have full occupancy (1.0)"""
    partial_sites = []
    
    for i, site in enumerate(structure.sites):
        # Check if site has partial occupancy
        total_occupancy = sum(site.species.values())
        if not site.is_ordered or abs(total_occupancy - 1.0) > 1e-6:  # Allow small numerical errors
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
            f"ML potential predictions require fully occupied crystal structures.\n\n"
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
        
    def create_session(self, session_id: Optional[str], filename: str, original_structure: Structure,
                        owner_ip: Optional[str] = None) -> str:
        """
        Create (or replace) a session and return the id it was stored
        under.

        session_id is the sole authorization token every session-scoped
        endpoint checks (see get_session_info/get_current_structure), so it
        must never be a value the client gets to choose: a client-supplied
        id is honored only when it already names a session this server
        previously issued (continuing it -- e.g. the browser tab loading a
        different sample file into the same session). Any other value
        (missing, or unknown to this server) is discarded in favor of a
        freshly minted, cryptographically random id, which is what the
        caller must return to the client so it can adopt it.
        """
        is_new_session = not session_id or session_id not in self.sessions
        if is_new_session:
            session_id = secrets.token_urlsafe(32)
            self._evict_lru_for_ip_if_at_capacity(owner_ip)
            self._evict_lru_if_at_capacity()
        self.sessions[session_id] = {
            'filename': filename,
            'original_structure': original_structure,
            'current_structure': original_structure.copy(),
            'structure_version': 0,
            'operations': [],
            'supercell_size': [1, 1, 1],
            'created_at': time.time(),
            'last_accessed': time.time(),
            'owner_ip': owner_ip,
        }
        logger.info(f"Created session {session_id[:8]}... for {filename}")
        return session_id

    def _evict_lru_for_ip_if_at_capacity(self, owner_ip: Optional[str]) -> None:
        """
        Evict this IP's own least-recently-accessed session if it already
        holds MAX_SESSIONS_PER_IP sessions, before minting a new one.

        Runs before the global _evict_lru_if_at_capacity check: caps how
        much of the shared MAX_SESSIONS pool a single IP can claim, so a
        caller minting sessions as fast as rate_limit_middleware allows can
        only ever evict its own sessions, never another client's. An
        unknown/missing owner_ip (e.g. request.client is None) is not
        tracked against any cap.
        """
        if owner_ip is None:
            return
        own_ids = [sid for sid, s in self.sessions.items() if s.get('owner_ip') == owner_ip]
        while len(own_ids) >= MAX_SESSIONS_PER_IP:
            oldest_id = min(own_ids, key=lambda sid: self.sessions[sid].get('last_accessed', 0))
            del self.sessions[oldest_id]
            own_ids.remove(oldest_id)
            logger.warning(
                f"Per-IP session limit ({MAX_SESSIONS_PER_IP}) reached; evicted "
                f"that client's least-recently-accessed session {oldest_id[:8]}...")

    def _evict_lru_if_at_capacity(self) -> None:
        """
        Evict the least-recently-accessed session(s) so a new one can be
        added without exceeding MAX_SESSIONS. Called only when actually
        minting a new id -- updating an existing session never grows the
        dict, so it never needs to evict.
        """
        while len(self.sessions) >= MAX_SESSIONS:
            oldest_id = min(self.sessions, key=lambda sid: self.sessions[sid].get('last_accessed', 0))
            del self.sessions[oldest_id]
            logger.warning(
                f"Session limit ({MAX_SESSIONS}) reached; evicted least-recently-accessed "
                f"session {oldest_id[:8]}..."
            )

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
            self.sessions[session_id]['structure_version'] = self.sessions[session_id].get('structure_version', 0) + 1
            self.sessions[session_id]['last_accessed'] = time.time()  # Track access time
            if operations is not None:
                self.sessions[session_id]['operations'] = operations
            if supercell_size is not None:
                self.sessions[session_id]['supercell_size'] = supercell_size
            # Any previously relaxed structure and its MLIP metadata are
            # invalidated by this update: /api/generate-relaxed-structure-cif
            # prefers session_info['relaxed_structure'] over the current
            # structure (main.py ~2226), so leaving a stale one in place made
            # Analyze -> substitute -> download silently return the
            # pre-substitution relaxed cell.
            self.sessions[session_id].pop('relaxed_structure', None)
            self.sessions[session_id].pop('relax_result', None)
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
            if data.get('last_accessed', data.get('created_at', 0)) < cutoff_time
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

# Session management instance
session_manager = SessionManager()

# Configuration constants - must be defined before use in app creation
TEMPLATE_DIR = os.getenv('CRYSTALNEXUS_TEMPLATE_DIR', 'templates')
STATIC_DIR = os.getenv('CRYSTALNEXUS_STATIC_DIR', 'static')
SAMPLE_CIF_DIR_NAME = os.getenv('CRYSTALNEXUS_SAMPLE_CIF_DIR', 'sample_cif')
APP_NAME = os.getenv('CRYSTALNEXUS_APP_NAME', 'CrystalNexus')

# Server configuration (default to loopback; set CRYSTALNEXUS_HOST=0.0.0.0 to expose on the network)
HOST = os.getenv('CRYSTALNEXUS_HOST', '127.0.0.1')
PORT = int(os.getenv('CRYSTALNEXUS_PORT', '8090'))
DEBUG = os.getenv('CRYSTALNEXUS_DEBUG', 'False').lower() == 'true'

# Gates the /analytics dashboard and /api/analytics/* endpoints, which
# otherwise expose every visitor's IP/User-Agent history plus uploaded
# filenames and formulas to anyone who can reach this server. Set this to
# require a shared token (via the X-Analytics-Token header or a ?token=
# query parameter) for access from beyond localhost -- see
# require_analytics_access().
CRYSTALNEXUS_ANALYTICS_TOKEN = os.getenv('CRYSTALNEXUS_ANALYTICS_TOKEN')

# File handling limits
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', str(50 * 1024 * 1024)))  # 50MB
MAX_SUPERCELL_DIM = int(os.getenv('MAX_SUPERCELL_DIM', '10'))
# Caps the *product* of the supercell dimensions against the base
# structure's site count; MAX_SUPERCELL_DIM only bounds each dimension
# independently (see check_supercell_site_limit).
MAX_TOTAL_SITES = int(os.getenv('MAX_TOTAL_SITES', '20000'))
# Upper bound on concurrently-held sessions (each holds at least two full
# pymatgen Structures). Without this, an anonymous caller could grow
# session_manager.sessions without limit -- cleanup_old_sessions() only
# evicts entries older than SESSION_CLEANUP_HOURS, on a
# PERIODIC_CLEANUP_INTERVAL timer, so a fast loop of create-supercell
# calls could exhaust memory long before either ever runs. See
# SessionManager.create_session's least-recently-accessed eviction.
MAX_SESSIONS = int(os.getenv('MAX_SESSIONS', '100'))

# Upper bound on concurrently-held sessions from a single client IP.
# MAX_SESSIONS alone does not stop one IP from claiming most of that
# shared pool by calling /api/create-supercell repeatedly (the existing
# rate_limit_middleware only limits request *frequency*, not how many
# sessions a caller can hold at once) -- see SessionManager.create_session.
MAX_SESSIONS_PER_IP = int(os.getenv('MAX_SESSIONS_PER_IP', '10'))

# Upper bound on the number of atomic operations accepted in a single
# /api/apply-atomic-operations request, so a pathologically long
# operations list (e.g. thousands of 'insert' ops) cannot be used to blow
# past MAX_TOTAL_SITES worth of work before that check even runs, or to
# simply pin the event loop applying an enormous list.
MAX_OPERATIONS_PER_REQUEST = int(os.getenv('MAX_OPERATIONS_PER_REQUEST', '1000'))

# Per-client-IP rate limit for POST /api/* (see rate_limit_middleware):
# there is otherwise no request-volume protection anywhere in this app,
# so a single caller could hammer the CIF-parsing/MLIP endpoints as
# fast as the network allows.
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv('RATE_LIMIT_REQUESTS_PER_MINUTE', '60'))
RATE_LIMIT_BURST = int(os.getenv('RATE_LIMIT_BURST', '20'))

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

# path -> ((mtime_ns, size), hash). Re-hashed only when the file changes on disk.
_static_hash_cache: Dict[str, Tuple[Tuple[int, int], str]] = {}


def static_url(path: str) -> str:
    """Return '/static/<path>?v=<first 10 hex of the content SHA256>'.

    Used for cache busting: the URL changes whenever the file content changes,
    so /static/ responses with a `v` query can be cached as immutable. The hash
    is cached per file keyed by (mtime, size), so edits during development are
    picked up on the next render. A missing file yields the URL without `v`.
    """
    import hashlib
    rel = path.lstrip('/')
    full = Path(STATIC_DIR) / rel
    base = f"/static/{rel}"
    try:
        st = full.stat()
        sig = (st.st_mtime_ns, st.st_size)
        cached = _static_hash_cache.get(rel)
        if cached and cached[0] == sig:
            return f"{base}?v={cached[1]}"
        digest = hashlib.sha256(full.read_bytes()).hexdigest()[:10]
        _static_hash_cache[rel] = (sig, digest)
        return f"{base}?v={digest}"
    except OSError:
        return base


templates.env.globals['static_url'] = static_url

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """
    Minimal, response-shape-preserving security headers (S-8 in the
    security review): this app set none at all. `frame-ancestors 'none'`
    (plus the older X-Frame-Options for browsers that don't honor CSP)
    stops /  and /analytics from being framed -- meaningful here because
    this app has no CSRF token and several destructive same-origin POSTs
    (Reset, Execute) that a framed clickjack could trigger.

    Deliberately NOT a full Content-Security-Policy: the pages rely on
    inline <script> blocks and dozens of inline onclick= handlers, so a
    script-src policy would need that inline code extracted first (a
    separate, larger change) to avoid breaking the UI.
    """
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/static/"):
        # Versioned URLs (?v=<hash>) never change content -> cache for a year;
        # unversioned ones must be revalidated.
        if "v" in request.query_params:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
    return response

class _TokenBucket:
    """Mutable per-client token-bucket state. A plain object (not a
    tuple/dataclass) so rate_limit_middleware can update tokens/last_refill
    in place via a single dict lookup."""
    __slots__ = ('tokens', 'last_refill')

    def __init__(self, tokens: float, last_refill: float):
        self.tokens = tokens
        self.last_refill = last_refill

# In-process only, like session_manager.sessions and _relax_progress below
# -- fine for this app's single-worker deployment model, and only ever
# touched from the event loop thread, so (also like those) no lock is
# needed. Not evicted on a timer: at realistic client counts this stays
# far smaller than session_manager's own cap, and a stale entry is only a
# few floats.
_rate_limit_buckets: Dict[str, "_TokenBucket"] = {}

def _rate_limit_check(client_host: str) -> bool:
    """
    Token-bucket rate limit: refills at RATE_LIMIT_REQUESTS_PER_MINUTE/60
    tokens per second, capped at RATE_LIMIT_BURST. Returns True if the
    request may proceed (and consumes one token), False if it should be
    rejected with 429.
    """
    now = time.time()
    refill_per_second = RATE_LIMIT_REQUESTS_PER_MINUTE / 60.0
    bucket = _rate_limit_buckets.get(client_host)
    if bucket is None:
        # Spend the first token immediately so a burst still starts at
        # BURST - 1, not BURST + 1 on the very next refill calculation.
        _rate_limit_buckets[client_host] = _TokenBucket(tokens=RATE_LIMIT_BURST - 1, last_refill=now)
        return True

    elapsed = max(0.0, now - bucket.last_refill)
    bucket.tokens = min(RATE_LIMIT_BURST, bucket.tokens + elapsed * refill_per_second)
    bucket.last_refill = now

    if bucket.tokens >= 1:
        bucket.tokens -= 1
        return True
    return False

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Per-client-IP rate limit on POST /api/* (see RATE_LIMIT_* config):
    there is no other protection anywhere in this app against a single
    caller hammering the CIF-parsing/MLIP endpoints as fast as the
    network allows.
    """
    if request.method == "POST" and request.url.path.startswith("/api/"):
        client_host = request.client.host if request.client else "unknown"
        if not _rate_limit_check(client_host):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down and try again shortly."}
            )
    return await call_next(request)

@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Log to analytics db (skip static files, health checks, and the
    # analytics dashboard/API itself to reduce noise -- /health polling and
    # viewing the analytics dashboard would otherwise inflate the very
    # access-count analytics that dashboard reports on).
    _ANALYTICS_EXCLUDED_PREFIXES = ("/static", "/health", "/analytics", "/api/analytics")
    if not request.url.path.startswith(_ANALYTICS_EXCLUDED_PREFIXES):
        try:
            client_host = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            # Run sqlite write in a thread so it doesn't block the event loop
            await asyncio.to_thread(
                analytics_db.log_access,
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                client_host=client_host,
                user_agent=user_agent
            )
        except Exception as e:
            logger.error(f"Analytics logging failed: {e}")
            
    return response

SAMPLE_CIF_DIR = Path(SAMPLE_CIF_DIR_NAME)

# Apply DEBUG settings
if DEBUG:
    logger.setLevel(logging.DEBUG)

def load_base_supercell(session_id: Optional[str], filename: Optional[str],
                         supercell_size: List[int]) -> Structure:
    """
    Resolve the pre-operation supercell that atomic operations should be
    replayed against.

    Prefers the session's own original structure + supercell size: this
    works uniformly for both sample and uploaded files, since the parsed
    Structure already lives in memory and needs no filesystem lookup.
    Falls back to the sample CIF directory by filename only when no
    session is available.

    Uploaded files are deliberately NOT looked up by filename on disk here:
    the saved filename is a random UUID (see /api/analyze-cif-upload) with
    no reliable mapping back to the original name, and matching by the
    original name risked resolving to a different session's upload.

    Raises:
        HTTPException(404) if no structure can be resolved, or (400) if
        filename fails path-safety validation.
    """
    if session_id:
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            # An explicit but expired/unknown session_id must fail loudly
            # instead of silently falling through to the sample-file-by-
            # filename fallback below: that fallback would happily load an
            # unrelated structure and apply this request's operations to
            # it, which is even more confusing than a plain 404.
            raise HTTPException(status_code=404, detail="Session not found or expired")
        structure = session_info['original_structure'].copy()
        effective_size = supercell_size or session_info.get('supercell_size', [1, 1, 1])
        check_supercell_site_limit(len(structure.sites), effective_size)
        structure.make_supercell(effective_size)
        return structure

    if filename:
        try:
            sample_path = SAMPLE_CIF_DIR / safe_path(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid filename: {e}")
        if sample_path.exists():
            parser = CifParser(str(sample_path))
            structure = parser.parse_structures(primitive=False)[0]
            check_supercell_site_limit(len(structure.sites), supercell_size)
            structure.make_supercell(supercell_size)
            return structure

    raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "CrystalNexus"}

def _get_session_lock(session_info: Dict) -> asyncio.Lock:
    """
    Return this session's asyncio.Lock, creating it on first use.

    Stored inside the session's own dict (session_manager.sessions[session_id])
    rather than in a separate global registry, so it is per-session, not
    global, and requires no separate cleanup: when a session is evicted or
    replaced, its lock goes with it.

    Serializes the read-modify-write sequence used by
    /api/apply-atomic-operations and /api/reset-session-structure, so two
    concurrent requests against the same session cannot interleave their
    (thread-pool-scheduled, possibly slow) pymatgen work and have the
    slower one silently clobber the faster one's session update.

    Note: a session can also be replaced wholesale (a brand new dict for
    the same session_id, e.g. via /api/create-supercell) while this lock
    is held on the OLD dict -- that does not block a request operating on
    the new dict, since it acquires the new dict's own lock. Callers must
    therefore still re-check `session_manager.sessions.get(session_id) is
    session_info` after re-acquiring, before writing anything back.
    """
    return session_info.setdefault('lock', asyncio.Lock())


def _apply_atomic_operations_sync(original_structure: Structure, supercell_size: List[int],
                                   operations: List[dict],
                                   allowed: Optional[Set[str]] = None) -> "Tuple[Structure, int]":
    """
    CPU-bound: rebuild the supercell from `original_structure` and replay
    `operations` against it in strict mode (fail-fast on the first invalid
    operation). Call via asyncio.to_thread.

    Raises ValueError if an operation is invalid, or HTTPException(400) if
    the resulting structure exceeds MAX_TOTAL_SITES -- both propagate
    through asyncio.to_thread into the caller unchanged.
    """
    structure = original_structure.copy()
    logger.info(f" OPERATIONS: Applying supercell {supercell_size}")
    structure.make_supercell(supercell_size)
    logger.info(f"✅ OPERATIONS: Supercell applied - Formula: {structure.formula}, Sites: {len(structure.sites)}")

    # Apply operations in the exact order the client recorded them (strict
    # mode: fail-fast on the first invalid operation). Each operation's
    # index refers to the structure state produced by the operations
    # before it -- the same frame the client's own atomicOperations
    # history uses -- so this must NOT reorder by index (see
    # apply_operations_to_structure docstring).
    logger.info(f" OPERATIONS: Applying {len(operations)} operations in recorded order")
    operations_applied, _, property_warnings = apply_operations_to_structure(
        structure, operations, strict_mode=True, allowed=allowed
    )
    if property_warnings:
        logger.warning(f"Property loss during atomic operations: {property_warnings}")
    logger.info(f"✅ OPERATIONS: Applied {operations_applied}/{len(operations)} operations - "
               f"Formula: {structure.formula}, Sites: {len(structure.sites)}")

    # Re-check the site cap after applying operations: supercell creation
    # already enforces MAX_TOTAL_SITES on its own scaling factor, but a
    # long enough run of 'insert' operations can independently push the
    # structure past that same limit.
    if len(structure.sites) > MAX_TOTAL_SITES:
        raise HTTPException(
            status_code=400,
            detail=f"Resulting structure would have {len(structure.sites)} atoms, "
                   f"exceeding the limit of {MAX_TOTAL_SITES}")

    return structure, operations_applied


@app.post("/api/apply-atomic-operations")
async def apply_atomic_operations(request: dict):
    """Apply atomic operations to the current structure in session"""
    try:
        session_id = request.get("session_id")
        operations = request.get("operations", [])

        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="Session ID is required and must be a string")

        if not isinstance(operations, list):
            raise HTTPException(status_code=400, detail="operations must be a list")

        if len(operations) > MAX_OPERATIONS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Too many operations in one request ({len(operations)}); "
                       f"the limit is {MAX_OPERATIONS_PER_REQUEST}")

        logger.info(f" OPERATIONS: Starting atomic operations for session: {session_id[:8]}...")
        logger.info(f" OPERATIONS: Received {len(operations)} operations: {operations}")

        # Get current structure from session
        current_structure = session_manager.get_current_structure(session_id)
        if current_structure is None:
            logger.error(f"❌ OPERATIONS: No structure found for session {session_id}")
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        logger.info(f"✅ OPERATIONS: Current structure found - Formula: {current_structure.formula}, Sites: {len(current_structure.sites)}")

        # Get session info to recreate structure from original + supercell + operations
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            logger.error(f"❌ OPERATIONS: Session info not found for {session_id}")
            raise HTTPException(status_code=404, detail=f"Session info not found for {session_id}")

        logger.info(f"✅ OPERATIONS: Session info retrieved, keys: {list(session_info.keys())}")

        # Serialize concurrent requests against this same session (see
        # _get_session_lock's docstring) before doing any of the heavy work.
        session_lock = _get_session_lock(session_info)
        async with session_lock:
            # Start with original structure and apply supercell + operations
            original_structure = session_info['original_structure']
            logger.info(f" OPERATIONS: Original structure - Formula: {original_structure.formula}, Sites: {len(original_structure.sites)}")

            supercell_size = session_info['supercell_size']
            try:
                structure, operations_applied = await asyncio.to_thread(
                    _apply_atomic_operations_sync, original_structure, supercell_size, operations,
                    _session_allowed_elements(session_info)
                )
            except ValueError as e:
                logger.error(f"❌ OPERATIONS: Validation failed - Rejecting {len(operations)} operations: {e}")
                raise HTTPException(status_code=400, detail=str(e))

            # The session may have been replaced wholesale (a brand new dict
            # for the same session_id -- e.g. via /api/create-supercell)
            # while the above ran; that new dict has its own lock, which does
            # not block this coroutine, so it must be detected explicitly
            # instead of relying on the lock alone. Refuse to resurrect this
            # now-stale result over whatever replaced it.
            if session_manager.sessions.get(session_id) is not session_info:
                raise HTTPException(
                    status_code=409,
                    detail="Session was modified concurrently; please retry")

            # Update session with modified structure and operations
            logger.info(f" OPERATIONS: Updating session with final structure - Formula: {structure.formula}, Sites: {len(structure.sites)}")
            session_manager.update_structure(session_id, structure, operations=operations)
            logger.info(f"✅ OPERATIONS: Session updated successfully")

        logger.info(f"Applied {len(operations)} operations to session {session_id[:8]}...")

        labels, unique_elements = build_element_labels(structure)

        return {
            "status": "success",
            "message": f"Applied {len(operations)} atomic operations",
            "num_sites": len(structure),
            "composition": str(structure.composition),  # kept for existing callers/tests
            # Authoritative structure state: the client overwrites its own
            # copy with these instead of recomputing formula/volume/density
            # from a parsed formula string client-side, which cannot be done
            # correctly -- deleting an atom does not shrink the lattice
            # volume, and the regex-based formula parser dropped elements
            # whose count was implicit (e.g. "Fe" rather than "Fe1").
            "formula": str(structure.formula),
            "volume": float(structure.volume),
            "density": float(structure.density),
            "labels": labels,
            "unique_elements": unique_elements,
        }

    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error applying atomic operations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
        logger.error(f"Error scanning sample CIF files: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/analyze-cif-sample")
async def analyze_sample_cif(data: dict):
    logger.info(f" ANALYZE_SAMPLE: Starting analysis of sample file: {data}")
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
        if not resolved_path.is_relative_to(base_path):
            raise ValueError("Path outside sample directory")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="CIF file not found")
        
        result = await analyze_cif_file(file_path)
        result["filename"] = safe_path_str

        # Analytics Logging
        try:
            analytics_db.log_event(
                event_type="sample_load",
                filename=safe_path_str,
                formula=result.get("formula"),
                num_atoms=result.get("num_atoms")
            )
        except Exception as e:
            logger.error(f"Failed to log sample load event: {e}")
        
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
async def analyze_uploaded_cif(request: Request, file: UploadFile = File(...)):
    logger.info(f" ANALYZE_UPLOAD: Starting analysis of uploaded file: {file.filename}")
    try:
        # Filename and size validation
        if not file.filename or not file.filename.lower().endswith('.cif'):
            raise HTTPException(status_code=400, detail="File must be a CIF file")

        # Cheap early rejection based on the client-supplied Content-Length,
        # before reading any of the body. This does not fully defend against
        # an attacker who omits Content-Length (the chunked read below is
        # the real backstop for that), but it avoids buffering an obviously
        # too-large upload at all in the common case.
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413,
                                         detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")
            except ValueError:
                pass  # Malformed header; fall through to the chunked read check below.

        # File size limit -- read in bounded chunks so a request far
        # larger than MAX_FILE_SIZE (e.g. a multi-GB POST, or one with no/
        # spoofed Content-Length) is rejected as soon as that much has
        # arrived, instead of first buffering the entire body (file.read()
        # with no size arg) and only checking afterward.
        chunk_size = 1024 * 1024  # 1MB
        chunks = []
        total_size = 0
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes")
            chunks.append(chunk)
        contents = b"".join(chunks)
        
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
        
        # Parse from a private temporary file rather than persisting into
        # uploads/: nothing reads an uploaded CIF back off disk later (the
        # app works entirely off structure_data/session state after this
        # request returns), so saving it into uploads/ only accumulated
        # files there until the periodic cleanup task eventually deleted
        # them. delete=False because analyze_cif_file/Structure.from_file
        # below need to reopen it by path; it is always removed in the
        # `finally` block below regardless of success or failure.
        import tempfile
        tmp_handle = tempfile.NamedTemporaryFile(
            mode='w', suffix='.cif', encoding='utf-8', delete=False
        )
        temp_path = Path(tmp_handle.name)
        try:
            tmp_handle.write(contents_str)
            tmp_handle.close()
            logger.info(f"✅ UPLOAD: Temp file written, size: {temp_path.stat().st_size} bytes")

            logger.info(f" UPLOAD: Starting analysis of uploaded file: {temp_path}")
            result = await analyze_cif_file(temp_path)
            safe_name = safe_filename(file.filename)
            result["filename"] = safe_name

            # Analytics Logging
            try:
                analytics_db.log_event(
                    event_type="upload",
                    filename=safe_name,
                    formula=result.get("formula"),
                    num_atoms=result.get("num_atoms")
                )
            except Exception as e:
                logger.error(f"Failed to log upload event: {e}")
            logger.info(f"✅ UPLOAD: Analysis completed, result keys: {list(result.keys())}")

            # Store the original structure for later use in createSupercell
            # Parse the structure and add it to the result
            try:
                from pymatgen.core import Structure
                logger.info(f" UPLOAD: Loading structure from file for storage")
                structure = Structure.from_file(str(temp_path))
                logger.info(f" UPLOAD: Structure loaded - Formula: {structure.formula}, Sites: {len(structure.sites)}")

                # Store structure as a dictionary for JSON serialization
                structure_dict = structure.as_dict()
                result["structure_data"] = structure_dict
                logger.info(f"✅ UPLOAD: Structure data stored, dict keys: {list(structure_dict.keys())}")
                logger.info(f"✅ UPLOAD: Successfully stored structure data for uploaded file: {safe_name}")
            except Exception as struct_error:
                logger.error(f"❌ UPLOAD: Failed to store structure data: {struct_error}")
                logger.error(f"❌ UPLOAD: Structure error type: {type(struct_error)}")
                import traceback
                logger.error(f"❌ UPLOAD: Structure error traceback: {traceback.format_exc()}")
                # Continue without structure data - fallback will handle this

            logger.info(f" UPLOAD: Returning result with keys: {list(result.keys())}")
            return result
        finally:
            # Always remove the temp file: the analysis above has already
            # extracted everything the rest of the app needs into `result`
            # (including structure_data), so nothing later reads this path.
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning(f"Failed to remove temp upload file {temp_path}: {cleanup_error}")

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

def analyze_cif_file_sync(file_path: Path) -> Dict:
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
                structures = parser.parse_structures(primitive=False)
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
        # Generic message: pymatgen/ASE exceptions routinely embed absolute
        # filesystem paths and library versions (already logged above).
        raise HTTPException(status_code=500, detail="Internal server error")

async def analyze_cif_file(file_path: Path) -> Dict:
    return await asyncio.to_thread(analyze_cif_file_sync, file_path)

def _load_and_build_supercell(crystal_data: dict, filename: str, a_mult: int, b_mult: int, c_mult: int) -> Tuple[Structure, Structure]:
    """Load the original structure and build the requested supercell. CPU-bound pymatgen work — call via asyncio.to_thread."""
    from pymatgen.io.cif import CifParser
    from pymatgen.core import Structure
    original_structure = None

    # Method 1: Try to load from stored structure_data (for uploaded files)
    logger.info(f" SUPERCELL: Starting structure loading for: {filename}")
    logger.info(f" SUPERCELL: Crystal data keys available: {list(crystal_data.keys())}")

    if "structure_data" in crystal_data:
        logger.info(f" SUPERCELL: Method 1 - Trying structure_data loading")
        try:
            from pymatgen.core import Structure
            structure_data = crystal_data["structure_data"]
            logger.info(f" SUPERCELL: Structure data type: {type(structure_data)}")

            if structure_data and isinstance(structure_data, dict):
                logger.info(f" SUPERCELL: Structure data keys: {list(structure_data.keys())}")
                # Validate client-supplied dict before Structure.from_dict.
                # Two layers of defense against monty.json.MontyDecoder's
                # dynamic @module/@class imports (see _reject_msonable_payload
                # docstring for the exact gadget this closes):
                if "@module" in structure_data or "@class" in structure_data:
                    if (structure_data.get("@module") != "pymatgen.core.structure"
                            or structure_data.get("@class") != "Structure"):
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid structure_data: only pymatgen.core.structure.Structure is allowed"
                        )
                if "lattice" not in structure_data or "sites" not in structure_data:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid structure_data: 'lattice' and 'sites' keys are required"
                    )
                # 1) Reject any nested MSONable marker anywhere in the
                #    payload (belt-and-suspenders; closes any decode path
                #    pymatgen may add in the future).
                _reject_msonable_payload(structure_data)
                # 2) Drop per-site `properties` outright: this is the one
                #    field pymatgen actually round-trips through
                #    MontyDecoder, and this app never needs it (CIF output
                #    always uses write_magmoms=False), so removing it is
                #    the real fix; (1) above is defense-in-depth only.
                for site in structure_data.get("sites", []):
                    if isinstance(site, dict):
                        site.pop("properties", None)
                original_structure = Structure.from_dict(structure_data)
                logger.info(f"✅ SUPERCELL: Structure loaded from structure_data - Formula: {original_structure.formula}, Sites: {len(original_structure.sites)}")

                # Validate the loaded structure
                if not original_structure.sites:
                    logger.warning(f"⚠️ SUPERCELL: Structure has no sites: {filename}")
                    original_structure = None
                else:
                    # Reject partial occupancy here too (the same check CIF
                    # uploads go through in analyze_cif_file_sync) -- without
                    # it, a structure_data payload with disordered sites
                    # reaches make_supercell()/MLIP later and fails there
                    # with an unhandled 500 instead of a clear 400. Converted
                    # to HTTPException explicitly (rather than letting the
                    # ValueError fall into the generic `except Exception`
                    # below, which would swallow it into a 500).
                    try:
                        validate_occupancy(original_structure)
                    except ValueError as occ_error:
                        raise HTTPException(status_code=400, detail=str(occ_error))
                    logger.info(f"✅ SUPERCELL: Structure validation passed for: {filename}")
            else:
                logger.warning(f"⚠️ SUPERCELL: Invalid structure_data format: {type(structure_data)}")
        except HTTPException:
            # Propagate validation errors (400) to the client
            raise
        except Exception as e:
            logger.error(f"❌ SUPERCELL: Failed to load from structure_data: {e}")
            logger.error(f"❌ SUPERCELL: Structure data content preview: {str(crystal_data.get('structure_data', 'None'))[:200]}...")
            import traceback
            logger.error(f"❌ SUPERCELL: Full traceback: {traceback.format_exc()}")
            original_structure = None
    else:
        logger.info(f" SUPERCELL: No structure_data found, will try file loading")

    # Method 2: Try to load from CIF file (for sample files and uploaded files)
    sample_parse_error = None
    if original_structure is None and filename != "unknown.cif":
        logger.info(f" SUPERCELL: Method 2 - Trying file-based loading for: {filename}")

        # Secure path validation (supports subdirectories)
        try:
            validated_filename = safe_path(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid filename: {e}")

        # First try sample directory
        cif_path = SAMPLE_CIF_DIR / validated_filename
        logger.info(f" SUPERCELL: Checking sample path: {cif_path}")

        if cif_path.exists():
            logger.info(f"✅ SUPERCELL: Found sample file, parsing...")
            try:
                parser = CifParser(str(cif_path))
                original_structure = parser.parse_structures(primitive=False)[0]
                logger.info(f"✅ SUPERCELL: Loaded structure from sample CIF - Formula: {original_structure.formula}, Sites: {len(original_structure.sites)}")
            except Exception as e:
                logger.error(f"❌ SUPERCELL: Failed to parse sample file: {e}")
                sample_parse_error = str(e)
        else:
            # Uploaded files are not looked up by filename on disk:
            # they are saved under a random UUID name (see
            # /api/analyze-cif-upload), which has no reliable mapping
            # back to the original filename. Uploaded structures must
            # come through structure_data (Method 1) instead; if that
            # was missing or invalid, this is a genuine error.
            logger.warning(f"⚠️ SUPERCELL: Sample file not found and no structure_data available for: {filename}")

    # If no structure available, this is a client-input problem (bad/missing
    # structure_data, an unparseable sample CIF, or an unknown sample
    # filename) -- not a server fault, so this must be a 400/404, not a 500.
    if original_structure is None:
        logger.error(f"❌ SUPERCELL: All structure loading methods failed for: {filename}")
        if "structure_data" in crystal_data:
            raise HTTPException(
                status_code=400,
                detail=f"Structure data was provided for {filename} but could not be parsed. "
                       f"Please ensure the CIF file is valid and properly uploaded.")
        if sample_parse_error is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse sample CIF file {filename}: {sample_parse_error}")
        raise HTTPException(status_code=404, detail=f"Sample CIF file not found: {filename}")

    # Create supercell
    logger.info(f" SUPERCELL: Creating supercell {a_mult}×{b_mult}×{c_mult} from structure: {original_structure.formula}")
    check_supercell_site_limit(len(original_structure.sites), [a_mult, b_mult, c_mult])
    supercell_structure = original_structure.copy()
    supercell_structure.make_supercell([a_mult, b_mult, c_mult])
    logger.info(f"✅ SUPERCELL: Supercell created - Formula: {supercell_structure.formula}, Sites: {len(supercell_structure.sites)}")
    return original_structure, supercell_structure

@app.post("/api/create-supercell")
async def create_supercell(request: Request, data: dict):
    try:
        client_host = request.client.host if request.client else None
        crystal_data = data.get("crystal_data")

        # Must be checked before any .get() call on crystal_data below: a
        # non-dict truthy value (e.g. a string or list) would otherwise
        # raise an unhandled AttributeError (500) the first time .get() is
        # called on it.
        if crystal_data is not None and not isinstance(crystal_data, dict):
            raise HTTPException(status_code=400, detail="crystal_data must be an object")

        supercell_size_raw = data.get("supercell_size", [1, 1, 1])
        session_id = data.get("session_id")

        logger.debug(f"create_supercell called with crystal_data type: {type(crystal_data)}")
        logger.debug(f"filename from crystal_data: {crystal_data.get('filename') if crystal_data else None}")

        if not crystal_data:
            raise HTTPException(status_code=400, detail="Crystal data is required")
        # session_id is optional here (unlike every other session-scoped
        # endpoint): this is the one call that may be minting a brand new
        # session. A missing/unrecognized value causes
        # session_manager.create_session() below to mint a fresh,
        # server-issued id instead of trusting a client-chosen one -- see
        # its docstring. The response always reports the id actually used
        # so the client can adopt it.

        # Validate supercell size
        supercell_size = validate_supercell_size(supercell_size_raw)

        # The model is chosen at load time and then fixed for this structure.
        # Unknown -> 400, unavailable -> 503 (the model is not loaded here).
        backend = _resolve_backend(_request_model_id(data))

        # Calculate supercell information
        a_mult, b_mult, c_mult = supercell_size
        scaling_factor = a_mult * b_mult * c_mult

        if not isinstance(crystal_data.get("volume"), (int, float)) or isinstance(crystal_data.get("volume"), bool):
            raise HTTPException(status_code=400, detail="crystal_data.volume is required and must be a number")
        if not isinstance(crystal_data.get("num_sites"), int) or isinstance(crystal_data.get("num_sites"), bool):
            raise HTTPException(status_code=400, detail="crystal_data.num_sites is required and must be an integer")
        if not isinstance(crystal_data.get("formula"), str) or not crystal_data.get("formula"):
            raise HTTPException(status_code=400, detail="crystal_data.formula is required and must be a non-empty string")
        raw_filename = crystal_data.get("filename")
        if raw_filename is not None and not isinstance(raw_filename, str):
            raise HTTPException(status_code=400, detail="crystal_data.filename must be a string")

        # Generate actual pymatgen Structure for 3D visualization. Any
        # failure here (bad structure_data, an unparseable/unknown sample
        # file, or a malformed filename) must surface as a client error
        # (400/404) rather than silently falling back to an approximate,
        # session-less "success" response -- see _load_and_build_supercell
        # for the specific error mapping.
        filename = crystal_data.get("filename", "unknown.cif")
        try:
            original_structure, supercell_structure = await asyncio.to_thread(
                _load_and_build_supercell, crystal_data, filename, a_mult, b_mult, c_mult
            )
        except HTTPException:
            raise
        except TypeError as e:
            # Belt-and-suspenders: a non-string filename that slipped past
            # the isinstance check above (or reaches safe_path()/safe_path-
            # like helpers some other way) raises TypeError there instead of
            # ValueError.
            logger.error(f"Invalid crystal_data for supercell creation: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid crystal_data: {e}")
        except Exception as e:
            logger.error(f"Could not create structure object: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail="Failed to create supercell structure")

        if supercell_structure is None or original_structure is None:
            # Should be unreachable (_load_and_build_supercell always either
            # returns two real Structures or raises), but never report a
            # "supercell_created" success with session_id: null.
            raise HTTPException(status_code=400, detail="Failed to create supercell structure")

        # Reject elements the chosen model cannot handle up front (422), so
        # an unsupported structure is never loaded into a session.
        _ensure_elements_supported(backend, _structure_symbols(supercell_structure))

        # Create or update session with structures. server_session_id is
        # the id actually stored under (see create_session's docstring);
        # it may differ from the request's session_id.
        server_session_id = session_manager.create_session(session_id, filename, original_structure,
                                                             owner_ip=client_host)
        session_manager.update_structure(server_session_id, supercell_structure,
                                       operations=[], supercell_size=supercell_size)
        # create_session always builds a fresh dict (also when reusing an
        # existing id), so the model from THIS request is the one stored.
        session_manager.sessions[server_session_id]['model'] = backend.id

        # Get structure dictionary for CIF generation (CPU-bound — offload)
        structure_dict = await asyncio.to_thread(supercell_structure.as_dict)
        logger.info(f"Successfully created supercell structure and session for {filename}")

        # Authoritative values from the actual pymatgen Structure, rather
        # than a formula-string regex scaling and a volume/site-count
        # multiplied out of the client-supplied original_data. Keeps the
        # very first structure state on the same footing as
        # /api/apply-atomic-operations' response (see build_element_labels
        # usage there).
        supercell_info = {
            "size": supercell_size,
            "volume": float(supercell_structure.volume),
            "num_sites": len(supercell_structure),
            "scaling_factor": scaling_factor,
            "formula": str(supercell_structure.formula),
            "density": float(supercell_structure.density),
        }

        return {
            "status": "supercell_created",
            # Authoritative session id: the client must adopt this value
            # (it may differ from the session_id sent in the request --
            # see SessionManager.create_session) for every subsequent
            # session-scoped call.
            "session_id": server_session_id,
            "model": backend.id,
            "original_data": crystal_data,
            "supercell_info": supercell_info,
            "structure_dict": structure_dict,  # Add structure_dict for 3D visualization
            "message": f"Supercell {a_mult}x{b_mult}x{c_mult} created successfully"
        }
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error creating supercell: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/get-element-labels")
async def get_element_labels(data: dict):
    """
    Get element labels directly from the actual Structure object to ensure correct ordering
    This guarantees labels match MLIP results ordering
    """
    try:
        session_id = data.get("session_id")
        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get the actual current structure from session (with supercell + operations applied)
        structure = session_manager.get_current_structure(session_id)
        if structure is None:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")

        # Generate labels directly from Structure sites in the exact same
        # order, ensuring perfect alignment with MLIP results.
        labels, unique_elements = build_element_labels(structure)

        logger.info(f"Generated structure-based labels for session {session_id}: {labels}")
        
        return {
            "status": "success", 
            "labels": labels,
            "unique_elements": unique_elements,
            "total_labels": len(labels),
            "method": "structure_based"  # Indicate this is the reliable method
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error getting element labels: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/models")
async def get_models():
    """List registered ML potentials (no model is loaded)."""
    return {"status": "success", "default": default_model_id(), "models": list_models()}


@app.get("/api/elements")
async def get_elements(model: Optional[str] = None):
    """
    Elements supported by a model, sorted by atomic number.

    ``model`` is a model id (default: the registry default). Unknown id -> 400.
    A registered but unavailable model (e.g. mace-torch not installed) returns
    200 with ``available: false`` and an empty list so the UI can render a
    disabled option. Does not load the model.
    """
    model_id = (model or "").strip() or default_model_id()
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_id}'. Available models: {sorted(MODEL_REGISTRY)}")
    try:
        available = _model_is_available(model_id)
        elements = []
        if available:
            elements = sorted(MODEL_REGISTRY[model_id]().supported_elements(),
                              key=lambda el: Element(el).Z)
        return {
            "status": "success",
            "model": model_id,
            "available": available,
            "elements": elements,
            "total_elements": len(elements),
            "source": "dynamic_detection" if available else "unavailable"
        }
    except Exception as e:
        logger.error(f"Error getting supported elements: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported elements")

def _generate_modified_structure_cif_sync(session_id: Optional[str], filename: str,
                                           supercell_size: List[int], operations: List[dict],
                                           allowed: Optional[Set[str]] = None
                                           ) -> "Tuple[Structure, int, List[str], List[str], str]":
    """
    CPU-bound: resolve the base supercell, replay `operations` against it
    (tolerant mode -- invalid operations are skipped, not fatal), apply CIF
    element labels, and render CIF text. Call via asyncio.to_thread.
    """
    from pymatgen.io.cif import CifWriter

    # Resolve the base supercell from the session when available (works
    # for uploaded files too, since the parsed Structure already lives
    # in memory), falling back to the sample directory by filename.
    structure = load_base_supercell(session_id, filename, supercell_size)
    logger.info(f"Base supercell: {structure.formula} ({len(structure.sites)} sites)")

    # Apply atomic operations with partial success tolerance (for CIF
    # generation), in the exact order the client recorded them.
    operations_applied, skipped_operations, property_warnings = apply_operations_to_structure(
        structure, operations, strict_mode=False, allowed=allowed
    )

    if skipped_operations:
        logger.warning(f"Skipping {len(skipped_operations)} invalid operations during CIF generation: {skipped_operations}")

    logger.info(f"Applied {operations_applied}/{len(operations)} operations successfully")
    logger.info(f"Final structure: {structure.formula} ({len(structure.sites)} sites)")

    # Generate CIF using pymatgen CifWriter
    apply_element_labels_to_structure(structure)
    cif_writer = CifWriter(
        structure,
        write_magmoms=False,
        significant_figures=6
    )

    cif_content = str(cif_writer)
    return structure, operations_applied, skipped_operations, property_warnings, cif_content


@app.post("/api/generate-modified-structure-cif")
async def generate_modified_structure_cif(request: dict):
    """
    Generate CIF from structure with applied atomic operations
    Supports complete atomic operation history replay
    """
    try:
        session_id = request.get("session_id")
        filename = request.get("filename")
        supercell_size = request.get("supercell_size", [1, 1, 1])
        operations = request.get("operations", [])

        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        filename = sanitize_display_filename(filename)
        supercell_size = validate_supercell_size(supercell_size)

        logger.info(f"Generating modified structure CIF for {filename}")
        logger.info(f"Supercell size: {supercell_size}")
        logger.info(f"Operations to apply: {len(operations)}")

        structure, operations_applied, skipped_operations, property_warnings, cif_content = await asyncio.to_thread(
            _generate_modified_structure_cif_sync, session_id, filename, supercell_size, operations,
            _session_allowed_elements(
                session_manager.sessions.get(session_id) if isinstance(session_id, str) else None)
        )

        # Add metadata header
        operations_summary = f"{operations_applied}/{len(operations)} operations applied"
        metadata_lines = [
            f"# Modified structure CIF generated by CrystalNexus",
            f"# Original file: {filename}",
            f"# Supercell size: {'x'.join(map(str, supercell_size))}",
            f"# Operations: {operations_summary}",
            f"# Final formula: {structure.formula}",
            f"# Number of atoms: {len(structure.sites)}",
            f"# Volume: {structure.volume:.2f} A^3",
        ]
        if skipped_operations:
            metadata_lines.append(f"# WARNING: {len(skipped_operations)} operation(s) skipped: {'; '.join(skipped_operations)}")
        if property_warnings:
            metadata_lines.append(
                f"# NOTE: {len(property_warnings)} substitution(s) dropped an oxidation state "
                f"and/or site properties (e.g. magmom): {'; '.join(property_warnings)}"
            )
        metadata_lines.append("")

        final_cif = "\n".join(metadata_lines) + cif_content

        logger.info(f"Modified CIF generated successfully, length: {len(final_cif)} characters")

        return Response(
            content=final_cif,
            media_type="chemical/x-cif",
            headers={
                "Content-Disposition": build_content_disposition(f"{filename.replace('.cif', '')}_modified.cif"),
                "Cache-Control": "no-cache",
                # The CIF body only exposes skips as a "# WARNING:" comment
                # line, which the client does not parse; this header lets it
                # surface a notification without reading CIF text.
                "X-Operations-Skipped": str(len(skipped_operations)),
            }
        )

    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = f"Failed to generate modified structure CIF: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def _resolve_and_expand_supercell_direct(session_structure: Optional[Structure],
                                          filename: str,
                                          supercell_size: List[int]) -> Tuple[Structure, str]:
    """
    CPU-bound: resolve the base structure (an already-loaded session
    structure, or by parsing a sample CIF file), enforce the total-site
    limit, then build the supercell and render its CIF text.
    Call via asyncio.to_thread.
    """
    from pymatgen.io.cif import CifParser, CifWriter

    original_structure = session_structure

    # Method 2: Try to load from CIF file (for sample files)
    if original_structure is None:
        # Secure path validation (supports subdirectories)
        try:
            validated_filename = safe_path(filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid filename: {e}")
        cif_path = SAMPLE_CIF_DIR / validated_filename
        if cif_path.exists():
            logger.debug(f"Reading CIF file: {cif_path}")
            parser = CifParser(str(cif_path))
            structures = parser.parse_structures(primitive=False)  # Keep original lattice - don't convert to primitive
            if structures:
                original_structure = structures[0]
                logger.info(f"Loaded structure from sample CIF file: {filename}")

    if not original_structure:
        raise HTTPException(status_code=404, detail=f"CIF file not found: {filename}")

    logger.info(f"Original structure loaded: {original_structure.formula}")

    # Create supercell
    check_supercell_site_limit(len(original_structure.sites), supercell_size)
    supercell_structure = original_structure.copy()
    supercell_structure.make_supercell(supercell_size)

    logger.info(f"Supercell created: {supercell_structure.formula}")
    logger.info(f"Number of sites: {len(supercell_structure.sites)}")

    # Generate CIF using pymatgen CifWriter
    apply_element_labels_to_structure(supercell_structure)
    cif_writer = CifWriter(
        supercell_structure,
        write_magmoms=False,
        significant_figures=6
    )

    return supercell_structure, str(cif_writer)

@app.post("/api/generate-supercell-cif-direct")
async def generate_supercell_cif_direct(request: dict):
    """
    Generate supercell CIF directly from structure data or CIF file using pymatgen
    Supports both uploaded files and sample files
    """
    try:
        # Extract request parameters
        filename = request.get("filename")
        supercell_size_raw = request.get("supercell_size", [1, 1, 1])
        session_id = request.get("session_id")

        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        filename = sanitize_display_filename(filename)
        supercell_size = validate_supercell_size(supercell_size_raw)

        logger.info(f"Generating supercell CIF for {filename} with size {supercell_size}")

        # Method 1: Try to get structure from session (for uploaded files).
        # This is a cheap dict lookup, so it stays on the event loop; the
        # CPU-bound CIF parsing / make_supercell / CifWriter work below is
        # offloaded to a thread.
        session_structure = None
        if session_id:
            try:
                session_info = session_manager.get_session_info(session_id)
                if session_info and 'original_structure' in session_info:
                    session_structure = session_info['original_structure']
                    logger.info(f"Loaded structure from session for uploaded file: {filename}")
            except Exception as e:
                logger.warning(f"Could not load structure from session: {e}")

        supercell_structure, cif_content = await asyncio.to_thread(
            _resolve_and_expand_supercell_direct, session_structure, filename, supercell_size
        )

        # Add metadata header
        size_str = "x".join(map(str, supercell_size))
        metadata_lines = [
            f"# Supercell CIF generated by CrystalNexus",
            f"# Original file: {filename}",
            f"# Supercell size: {size_str}",
            f"# Formula: {supercell_structure.formula}",
            f"# Number of atoms: {len(supercell_structure.sites)}",
            f"# Volume: {supercell_structure.volume:.2f} A^3",
            ""
        ]
        
        final_cif = "\n".join(metadata_lines) + cif_content
        
        logger.info(f"CIF generated successfully, length: {len(final_cif)} characters")
        
        return Response(
            content=final_cif,
            media_type="chemical/x-cif",
            headers={
                "Content-Disposition": build_content_disposition(f"{filename.replace('.cif', '')}_supercell_{size_str}.cif"),
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except ValueError as e:
        # e.g. validate_supercell_size() rejecting a malformed/oversized
        # supercell_size -- a client input error, not a server error.
        raise HTTPException(status_code=400, detail=str(e))
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

def _get_device_info(backend=None):
    """Report the device the model is actually running on.

    This used to return "cuda:N" whenever torch.cuda.is_available(), while
    the model was unconditionally loaded with use_device="cpu" -- so the
    reported device (shown in the analysis modal) could disagree with
    reality on any CUDA-capable machine. Report what was actually loaded.
    """
    return (backend or get_backend()).device or "cpu"


def _request_model_id(data) -> Optional[str]:
    """Optional ``model`` field of a request body (None -> registry default)."""
    model_id = data.get("model") if isinstance(data, dict) else None
    if model_id is None or model_id == "":
        return None
    if not isinstance(model_id, str):
        raise HTTPException(status_code=400, detail="model must be a string")
    return model_id.strip()


def _resolve_backend(model_id=None):
    """Registry lookup mapped to HTTP errors (unknown -> 400, unavailable -> 503)."""
    try:
        return get_backend(model_id)
    except UnknownModelError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))


def _ensure_elements_supported(backend, symbols) -> None:
    """422 listing the elements ``backend`` cannot handle (before any inference)."""
    supported = set(backend.supported_elements())
    unsupported = sorted({sym for sym in symbols if sym not in supported},
                         key=lambda sym: Element(sym).Z)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=f"Element(s) {', '.join(unsupported)} not supported by model "
                   f"'{backend.id}' ({backend.display_name}). Choose a different "
                   f"model or remove these elements.")


def _structure_symbols(structure) -> Set[str]:
    return {_site_element_symbol(site.specie) for site in structure}


def _session_model_id(session_info, requested_id=None) -> Optional[str]:
    """Model id to use for a session-scoped request.

    The model chosen when the structure was loaded is authoritative: a request
    that omits ``model`` uses it, a request naming a different one -> 409. A
    session without a stored model (e.g. created directly in tests) keeps the
    legacy behavior: the requested id, or None (= registry default)."""
    session_model = session_info.get('model') if isinstance(session_info, dict) else None
    if not session_model:
        return requested_id
    if requested_id is None or requested_id == session_model:
        return session_model
    raise HTTPException(
        status_code=409,
        detail=f"This structure was loaded with model '{session_model}'. To use "
               f"'{requested_id}', load the structure again.")


def _session_allowed_elements(session_info) -> Optional[Set[str]]:
    """Supported-element set of the session's model (None -> ALLOWED_ELEMENTS)."""
    session_model = session_info.get('model') if isinstance(session_info, dict) else None
    if not session_model or session_model not in MODEL_REGISTRY:
        return None
    return set(MODEL_REGISTRY[session_model]().supported_elements())


async def _load_or_503(backend) -> None:
    """Load the backend, mapping load failures to HTTP 503."""
    try:
        await backend.load()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# Property sets requested from backends (see mlip.base PROP_* names).
_PROPS_ENERGY = ("energy",)
_PROPS_PREDICT = ("energy", "forces", "stress", "magmom", "site_energies")
_PROPS_RELAX = _PROPS_PREDICT + ("atom_features", "crystal_features")


def _prediction_payload(pred: dict) -> dict:
    """Normalized backend prediction -> response dict.

    Keys the backend did not provide (None) are omitted, which keeps the
    response structure identical to the pre-abstraction safe_get_prediction().
    """
    return {k: v for k, v in pred.items() if v is not None}


def evaluate_convergence(trajectory, fmax):
    """
    Evaluate convergence from the final trajectory step only.

    Uses the recorder's max_forces (the force the optimizer/watcher use: with
    relax_cell=True this includes FrechetCellFilter cell-stress terms). Falls
    back to atomic forces for trajectories without max_forces (fixed-cell
    behavior is identical either way).
    """
    import numpy as np
    
    # Use an explicit length check: numpy arrays raise on ambiguous truthiness
    if trajectory is None or not hasattr(trajectory, 'forces') or trajectory.forces is None or len(trajectory.forces) == 0:
        return False
    
    # Check final step only - simplified approach
    recorded = getattr(trajectory, 'max_forces', None)
    if recorded is not None and len(recorded) > 0:
        max_force = float(recorded[-1])
    else:
        final_forces = np.array(trajectory.forces[-1])
        max_force = np.linalg.norm(final_forces, axis=1).max()
    
    converged = max_force < fmax
    logger.info(f"Convergence check: max_force={max_force:.4f} eV/A, fmax={fmax}, converged={converged}")

    return bool(converged)


class RelaxationAborted(Exception):
    """Raised from an optimizer observer to stop a run cooperatively.

    This is the only reliable way to stop an in-progress ASE relaxation:
    asyncio.wait_for() would abandon the awaiting coroutine while the
    executor thread kept running and kept the single MLIP worker
    (_inference_executor, max_workers=1) busy for the rest of its steps.
    """


def _get_magmoms_or_none(atoms):
    """Per-atom magnetic moments, or None if the calculator cannot provide them.

    A calculator may *list* "magmoms" in implemented_properties without
    computing it (ASE's EMT does), so a missing result is tolerated too.
    """
    from ase.calculators.calculator import PropertyNotImplementedError

    calculator = atoms.calc
    if "magmoms" not in (getattr(calculator, "implemented_properties", None) or ()):
        return None
    try:
        return atoms.get_magnetic_moments()
    except PropertyNotImplementedError:
        return None


class TrajectoryRecorder:
    """Records the same fields as chgnet.model.dynamics.TrajectoryObserver
    (energies, forces, stresses, magmoms, atom_positions, cells) so the
    existing evaluate_convergence() and trajectory_data extraction code
    (main.py) can consume it unchanged.
    """

    def __init__(self, atoms, force_source=None):
        self.atoms = atoms
        # Same object the optimizer/watcher judge convergence with: the
        # FrechetCellFilter when relax_cell=True, else the Atoms itself.
        self.force_source = force_source if force_source is not None else atoms
        # Per-frame max force norm of force_source (includes cell stress
        # rows when relax_cell=True); used by evaluate_convergence().
        self.max_forces = []
        self.energies = []
        self.forces = []
        self.stresses = []
        self.magmoms = []
        self.atom_positions = []
        self.cells = []

    def __call__(self):
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.max_forces.append(float(np.linalg.norm(self.force_source.get_forces(), axis=1).max()))
        try:
            self.stresses.append(self.atoms.get_stress())
        except Exception:
            self.stresses.append(None)
        self.magmoms.append(_get_magmoms_or_none(self.atoms))
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self):
        return len(self.energies)


def _sanitize_json_value(value):
    """
    Recursively replace non-finite floats (NaN, Infinity, -Infinity) with
    None throughout a nested dict/list structure.

    A MLIP numerical blow-up (e.g. a badly relaxed/exploded structure)
    can produce NaN/Infinity energies or forces. Python's json module
    happily serializes these as the bare literals `NaN`/`Infinity`, which
    is not valid JSON -- FastAPI's default JSONResponse then either raises
    on strict encoders or ships a payload that breaks a strict JSON.parse()
    on the client, and the same values written into an SSE progress event
    break that stream's JSON.parse() too. Apply this to a response/progress
    payload right before it is stored or returned.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    return value


class RelaxationWatcher:
    """ASE observer: publishes progress and enforces a wall-clock deadline.

    Attached with interval=1, so ASE calls it once per optimizer step plus
    once at step 0 (ase.optimize.optimize.Dynamics.call_observers). The
    force/energy reads below are served from the calculator's cache -- the
    optimizer has just computed them for its own convergence check -- so
    the observer triggers no extra MLIP evaluation.

    Raising RelaxationAborted here propagates out of optimizer.run(); see
    the class docstring for why that (rather than asyncio.wait_for) is the
    correct way to enforce a timeout.
    """

    def __init__(self, atoms, fmax, max_steps, deadline, progress_slot=None,
                 force_source=None):
        self.atoms = atoms  # the UNFILTERED Atoms object (energy, raw forces)
        # Object whose get_forces() ASE's convergence test actually uses: the
        # FrechetCellFilter when relax_cell=True (atomic forces + cell DOF
        # rows), else the Atoms itself. Its get_forces() reuses the cached
        # atoms.get_forces()/get_stress(), so no extra MLIP evaluation.
        self.force_source = force_source if force_source is not None else atoms
        self.fmax = fmax
        self.max_steps = max_steps
        self.deadline = deadline
        self.progress_slot = progress_slot
        self.started_at = time.monotonic()
        self.step = -1

    def __call__(self):
        self.step += 1
        max_force = float(np.linalg.norm(self.force_source.get_forces(), axis=1).max())
        if self.progress_slot is not None:
            energy = float(self.atoms.get_potential_energy())
            # Plain dict assignment is atomic under the GIL, so the executor
            # thread (writer) and the event loop (reader, via the progress
            # polling/SSE endpoint) need no lock between them. "history" is
            # append-only, so a reader that remembers how many entries it
            # has already sent can safely slice off the rest.
            self.progress_slot.setdefault("history", []).append(_sanitize_json_value({
                "step": self.step,
                "energy_eV": energy,
                "max_force_eV_per_A": max_force,
            }))
            self.progress_slot["state"] = _sanitize_json_value({
                "step": self.step,
                "max_steps": self.max_steps,
                "max_force_eV_per_A": max_force,
                "fmax": self.fmax,
                "energy_eV": energy,
                "elapsed_s": time.monotonic() - self.started_at,
            })
        if time.monotonic() > self.deadline:
            raise RelaxationAborted(f"time limit reached at step {self.step}")


def run_relaxation(calculator, structure, *, fmax, steps, optimizer_name,
                    timeout_s, relax_cell=True, progress_slot=None):
    """Run an ASE relaxation with full control over observers and stopping.

    Reimplements chgnet.model.dynamics.StructOptimizer.relax (which this
    replaces at the call site) because that method exposes neither the
    Optimizer object -- so no progress/timeout observer can be attached and
    ASE's own convergence verdict cannot be read -- nor a way to skip its
    duplicated final obs() call, which is why the old code had to subtract
    2 from the trajectory length to get the true step count.

    Returns a dict with:
        final_structure: the relaxed pymatgen Structure
        trajectory: a TrajectoryRecorder (same shape as chgnet's own)
        ase_converged: bool, ASE's own verdict from optimizer.run(). With
            relax_cell=True this is computed on the FrechetCellFilter's
            extended force array (atomic forces + cell stress terms), which
            now agree with evaluate_convergence(), which uses the same force.
        optimizer_steps: int, optimizer.get_number_of_steps() -- exact,
            unlike the old len(trajectory) - 2 approximation.
        aborted: bool, True if the wall-clock deadline was hit first.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch
    from pymatgen.io.ase import AseAtomsAdaptor

    optimizers = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS,
                  "LBFGSLineSearch": LBFGSLineSearch}
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer {optimizer_name!r}; "
                         f"must be one of {sorted(optimizers)}")

    # Start the clock here (on the executor thread) rather than at submit
    # time, so time spent queued behind other inference jobs is not counted.
    deadline = time.monotonic() + timeout_s

    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = calculator

    target = FrechetCellFilter(atoms) if relax_cell else atoms
    obs = TrajectoryRecorder(atoms, force_source=target)
    watcher = RelaxationWatcher(atoms, fmax, steps, deadline, progress_slot,
                                force_source=target)
    optimizer = optimizers[optimizer_name](target, logfile=None)
    optimizer.attach(obs, interval=1)
    optimizer.attach(watcher, interval=1)

    aborted = False
    try:
        # ASE 3.27's Optimizer.run() returns the convergence verdict
        # directly; Optimizer.converged() takes a mandatory gradient
        # argument and is not meant to be called standalone here.
        ase_converged = bool(optimizer.run(fmax=fmax, steps=steps))
    except RelaxationAborted as e:
        logger.warning(f"Relaxation aborted: {e}")
        aborted = True
        ase_converged = False

    final_structure = AseAtomsAdaptor.get_structure(atoms)
    magmoms = _get_magmoms_or_none(atoms)
    if magmoms is not None:
        final_structure.add_site_property("magmom", [float(m) for m in magmoms])

    return {
        "final_structure": final_structure,
        "trajectory": obs,
        "ase_converged": ase_converged,
        "optimizer_steps": int(optimizer.get_number_of_steps()),
        "aborted": aborted,
    }


def _validate_relax_params(request: dict) -> Tuple[float, int, str, bool, float]:
    """Validate and clamp relaxation parameters before any model loading.

    Raises HTTPException(400) on invalid input so bad requests are rejected
    without ever touching the (slow, singleton, serialized) MLIP model.
    Returns (fmax, max_steps, optimizer_name, elastic, elastic_strain_percent).
    """
    try:
        fmax = float(request.get("fmax", 0.01))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="fmax must be a number")
    try:
        max_steps = int(request.get("max_steps", 100))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_steps must be an integer")

    if not (MIN_RELAX_FMAX <= fmax <= MAX_RELAX_FMAX):
        raise HTTPException(
            status_code=400,
            detail=f"fmax must be between {MIN_RELAX_FMAX} and {MAX_RELAX_FMAX} eV/A")
    if not (1 <= max_steps <= MAX_RELAX_STEPS):
        raise HTTPException(
            status_code=400,
            detail=f"max_steps must be between 1 and {MAX_RELAX_STEPS}")

    optimizer_name = request.get("optimizer", RELAX_OPTIMIZER) or RELAX_OPTIMIZER
    if optimizer_name not in VALID_RELAX_OPTIMIZERS:
        raise HTTPException(
            status_code=400,
            detail=f"optimizer must be one of {VALID_RELAX_OPTIMIZERS}")

    elastic = request.get("elastic", False)
    if not isinstance(elastic, bool):
        raise HTTPException(status_code=400, detail="elastic must be a boolean")
    raw_strain = request.get("elastic_strain", 0.5)
    try:
        if isinstance(raw_strain, bool):
            raise TypeError
        elastic_strain = float(raw_strain)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="elastic_strain must be a number")
    if not (ELASTIC_STRAIN_MIN_PCT <= elastic_strain <= ELASTIC_STRAIN_MAX_PCT):
        raise HTTPException(
            status_code=400,
            detail=f"elastic_strain must be between {ELASTIC_STRAIN_MIN_PCT} and "
                   f"{ELASTIC_STRAIN_MAX_PCT} (percent)")

    return fmax, max_steps, optimizer_name, elastic, elastic_strain


def _run_elastic_sync(calculator, structure, *, strain_fraction, fmax, timeout_s,
                      progress_slot):
    """Executor-thread worker: build Atoms from `structure` and compute elastic data."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from mlip.elastic import compute_elastic

    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = calculator
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()

    def _progress(k, n):
        if progress_slot is None:
            return
        prev = progress_slot.get("state") or {}
        progress_slot["state"] = {
            **prev,
            "message": f"Elastic: strain {k}/{n}",
            "elapsed_s": time.monotonic() - started,
        }

    return compute_elastic(atoms, calculator, strain=strain_fraction, fmax=fmax,
                           deadline=deadline, progress_cb=_progress)


def _load_structure_for_predict(session_id: Optional[str], filename: str,
                                        supercell_size: List[int], operations: List[dict],
                                        allowed: Optional[Set[str]] = None
                                        ) -> "Tuple[Structure, int, List[str], List[str]]":
    """
    CPU-bound: resolve the base supercell (session or sample file), enforce
    MAX_PREDICT_ATOMS, and replay `operations` against it (tolerant mode).
    Call via asyncio.to_thread.

    Raises HTTPException(400) if the base supercell exceeds
    MAX_PREDICT_ATOMS -- propagates through asyncio.to_thread unchanged.
    """
    structure = load_base_supercell(session_id, filename, supercell_size)
    logger.info(f"✅ MLIP: Base supercell - Formula: {structure.formula}, Sites: {len(structure.sites)}")

    if len(structure) > MAX_PREDICT_ATOMS:
        raise HTTPException(
            status_code=400,
            detail=f"Structure has {len(structure)} atoms; the prediction limit is "
                   f"{MAX_PREDICT_ATOMS}. Reduce the supercell size.")

    # Apply operations for MLIP prediction (skip invalid ones), in the
    # exact order the client recorded them.
    operations_applied, skipped_operations, property_warnings = apply_operations_to_structure(
        structure, operations, strict_mode=False, allowed=allowed
    )
    if property_warnings:
        logger.warning(f"Property loss during MLIP prediction operations: {property_warnings}")

    if skipped_operations:
        logger.warning(f"Skipping {len(skipped_operations)} invalid operations for MLIP prediction: {skipped_operations}")

    return structure, operations_applied, skipped_operations, property_warnings


@app.post("/api/predict")
async def predict_structure(request: dict):
    """
    Predict structure properties using the ML potential, replaying an operation
    history against a filename + supercell_size instead of a session's
    current structure.

    Not called by the frontend (Auto mode screening uses the batched
    /api/evaluate-candidate-energies and /api/evaluate-insertion-energies
    instead; see main.py history around commit 755aae8). Kept as a
    standalone single-structure prediction endpoint and exercised by
    tests/test_main.py::test_predict.
    """
    try:
        session_id = request.get("session_id")
        filename = request.get("filename")
        operations = request.get("operations", [])
        supercell_size = request.get("supercell_size", [1, 1, 1])
        predict_session = (session_manager.sessions.get(session_id)
                           if isinstance(session_id, str) else None)
        model_id = _session_model_id(predict_session, _request_model_id(request))

        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        supercell_size = validate_supercell_size(supercell_size)

        logger.info(f"MLIP prediction for {filename} with {len(operations)} operations")

        # Resolve the base supercell (session or sample file) and replay
        # operations against it -- CPU-bound pymatgen work, offloaded.
        structure, operations_applied, skipped_operations, property_warnings = await asyncio.to_thread(
            _load_structure_for_predict, session_id, filename, supercell_size, operations,
            _session_allowed_elements(predict_session)
        )

        # Resolve the requested model, reject unsupported elements (422)
        # before any inference, then load it (singleton per backend).
        backend = _resolve_backend(model_id)
        _ensure_elements_supported(backend, _structure_symbols(structure))
        await _load_or_503(backend)

        # Predict structure properties (normalized by the backend)
        pred = await backend.predict_one(structure, _PROPS_PREDICT)
        results = _prediction_payload(pred)
        
        # Add structure information
        results.update({
            "formula": str(structure.formula),
            "num_sites": len(structure.sites),
            "volume": float(structure.volume),
            "density": float(structure.density),
            "operations_applied": operations_applied,
            "operations_skipped": skipped_operations,
            "supercell_size": supercell_size
        })
        
        logger.info(f"MLIP prediction completed: {results.get('energy_eV_per_atom', 'N/A')} eV/atom")
        
        return {
            "status": "success",
            "prediction": results,
            "model_info": {
                "id": backend.id,
                "display_name": backend.display_name,
                "version": backend.version(),
                "device": _get_device_info(backend),
                "parameters": backend.parameters()
            }
        }
        
    except ValueError as e:
        logger.error(f"MLIP prediction validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"MLIP prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/relax")
async def relax_structure(request: dict):
    """
    Relax structure using the ML potential - uses pre-prepared structure from session
    """
    try:
        session_id = request.get("session_id")
        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Validated and clamped before any model loading, so malformed or
        # out-of-range requests never touch the slow, serialized MLIP model.
        fmax, max_steps, optimizer_name, elastic, elastic_strain = _validate_relax_params(request)
        requested_model_id = _request_model_id(request)

        # Get the current structure from session (already has supercell + operations applied)
        structure = session_manager.get_current_structure(session_id)
        if structure is None:
            raise HTTPException(status_code=404, detail=f"No structure found for session {session_id}")
        # The session's model (chosen at load time) is authoritative.
        model_id = _session_model_id(session_manager.sessions.get(session_id), requested_model_id)

        if len(structure) > MAX_RELAX_ATOMS:
            raise HTTPException(
                status_code=413,
                detail=f"Structure has {len(structure)} atoms; the relaxation limit is "
                       f"{MAX_RELAX_ATOMS}. Reduce the supercell size.")

        # Model selection and element support are checked before taking the
        # semaphore or doing any inference (unknown -> 400, unavailable ->
        # 503, unsupported elements -> 422).
        backend = _resolve_backend(model_id)
        _ensure_elements_supported(backend, _structure_symbols(structure))

        session_info = session_manager.get_session_info(session_id)
        filename = session_info.get('filename', 'unknown') if session_info else 'unknown'
        # Captured now so the write-back below can detect whether this
        # session's structure was edited by another request while this
        # (possibly minutes-long) relaxation was still running.
        structure_version_at_start = session_info.get('structure_version', 0) if session_info else 0

        logger.info(f"MLIP relaxation for session {session_id[:8]}... ({filename}) "
                    f"with fmax={fmax}, max_steps={max_steps}, optimizer={optimizer_name}")
        logger.info(f"Structure: {structure.composition} ({len(structure)} sites)")

        # A relaxation can run for minutes; refuse a second concurrent one
        # immediately rather than silently queuing it behind the single
        # MLIP worker thread (which would also starve Auto-mode screening).
        # No await happens between this check and the acquire() call below,
        # so the check-then-acquire is atomic on the single-threaded event
        # loop -- acquire() only suspends when the semaphore is contended.
        if _relax_semaphore.locked():
            raise HTTPException(
                status_code=429,
                detail="Another relaxation is already running. Please wait for it to finish.")
        await _relax_semaphore.acquire()
        try:
            # Create the progress slot immediately (before the potentially slow model
            # load / initial prediction below) so a client that connects to
            # /api/relax-progress/{session_id} right away sees `session_id in
            # _relax_progress` and waits for state, instead of falling through to an
            # immediate "done" and needing the browser's ~3s SSE auto-reconnect.
            _relax_progress[session_id] = {}

            # Load MLIP model/calculator (using singleton pattern)
            try:
                calculator = await backend.get_calculator()
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))

            # Predict initial structure using optimized prediction
            try:
                initial_results = _prediction_payload(
                    await backend.predict_one(structure, _PROPS_RELAX))
            except Exception as e:
                logger.warning(f"Initial prediction failed: {e}")
                initial_results = {"energy_eV_per_atom": None}

            # MLIP structure relaxation
            logger.info(f"Starting MLIP relaxation: fmax={fmax}, max_steps={max_steps}, optimizer={optimizer_name}")
            progress_slot = _relax_progress.setdefault(session_id, {})
            progress_slot.clear()
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    _inference_executor,
                    functools.partial(run_relaxation, calculator, structure,
                                      fmax=fmax, steps=max_steps,
                                      optimizer_name=optimizer_name,
                                      timeout_s=RELAX_TIMEOUT_SECONDS,
                                      relax_cell=True,
                                      progress_slot=progress_slot))
            finally:
                # With elastic=true the slot stays alive so the SSE stream can
                # report the elastic step; the outer finally removes it.
                if not elastic:
                    _relax_progress.pop(session_id, None)

            final_structure = result.get("final_structure")
            if final_structure is None:
                raise RuntimeError("Relaxation failed: no final structure returned")
            trajectory = result.get("trajectory")
            aborted = result.get("aborted", False)
            if aborted:
                logger.warning(f"Relaxation for session {session_id[:8]}... hit the "
                               f"{RELAX_TIMEOUT_SECONDS}s time limit; returning the best "
                               f"structure found so far")

            # Predict relaxed structure using optimized prediction
            try:
                final_results = _prediction_payload(
                    await backend.predict_one(final_structure, _PROPS_RELAX))
            except Exception as e:
                logger.warning(f"Final prediction failed: {e}")
                final_results = {"energy_eV_per_atom": None}

            # Optional elastic constants on the relaxed structure. Failures
            # never fail the relaxation: they are reported in elastic_info.
            elastic_info = None
            if elastic:
                elastic_warnings = []
                if CAP_STRESS not in backend.capabilities:
                    elastic_info = {
                        "error": f"Model '{backend.id}' does not provide stress; "
                                 f"elastic constants are unavailable.",
                        "warnings": elastic_warnings}
                else:
                    if aborted:
                        elastic_warnings.append(
                            "Relaxation was aborted; elastic constants were computed "
                            "on a partially relaxed structure.")
                    elif not evaluate_convergence(trajectory, fmax):
                        elastic_warnings.append(
                            "Relaxation did not converge; elastic constants may be inaccurate.")
                    if fmax > ELASTIC_FMAX_WARN + 1e-12:
                        elastic_warnings.append(
                            f"Relaxation force tolerance ({fmax:g} eV/Å) is looser than "
                            f"{ELASTIC_FMAX_WARN:g} eV/Å; elastic constants may be inaccurate.")
                    elastic_started = time.monotonic()
                    try:
                        elastic_info = await asyncio.get_running_loop().run_in_executor(
                            _inference_executor,
                            functools.partial(_run_elastic_sync, calculator, final_structure,
                                              strain_fraction=elastic_strain / 100.0,
                                              fmax=ELASTIC_FMAX_WARN,
                                              timeout_s=ELASTIC_TIMEOUT_SECONDS,
                                              progress_slot=progress_slot))
                        elastic_info["execution_time_s"] = round(time.monotonic() - elastic_started, 2)
                        if not elastic_info.get("stable", True):
                            elastic_warnings.append(
                                "Elastic tensor is not positive definite (mechanically unstable).")
                        elastic_info["warnings"] = elastic_warnings
                    except Exception as e:
                        logger.warning(f"Elastic calculation failed: {e}")
                        elastic_info = {
                            "error": f"Elastic calculation failed: {e}",
                            "execution_time_s": round(time.monotonic() - elastic_started, 2),
                            "warnings": elastic_warnings}

            # Calculate energy difference in total eV
            energy_diff = None
            energy_diff_per_atom = None
            if (initial_results.get("total_energy_eV") is not None and
                final_results.get("total_energy_eV") is not None):
                energy_diff = final_results["total_energy_eV"] - initial_results["total_energy_eV"]
                energy_diff_per_atom = final_results["energy_eV_per_atom"] - initial_results["energy_eV_per_atom"]

            # Extract relaxation information. "converged" uses the same max
            # force as the optimizer/watcher (FrechetCellFilter forces when
            # relax_cell=True, atomic forces otherwise) from the final step.
            # "converged_optimizer" is ASE's own verdict from run_relaxation().
            steps = len(trajectory) if trajectory else 0

            converged = evaluate_convergence(trajectory, fmax)

            # Log convergence evaluation details using consistent force calculation method
            if trajectory and hasattr(trajectory, 'forces') and len(trajectory.forces) > 0:
                recorded_mf = getattr(trajectory, 'max_forces', None)
                if recorded_mf is not None and len(recorded_mf) > 0:
                    max_final_force = float(recorded_mf[-1])
                else:
                    max_final_force = np.linalg.norm(np.array(trajectory.forces[-1]), axis=1).max()
                logger.info(f"Convergence evaluation: max_force={max_final_force:.6f} eV/A, fmax={fmax}, converged={converged}")
            else:
                logger.info(f"Convergence evaluation: no force data available, converged={converged}")

            logger.info(f"Final relaxation status: converged={converged}, steps={steps}, aborted={aborted}")

            relaxation_info = {
                "converged": converged,
                "converged_optimizer": result.get("ase_converged", False),
                "steps": steps,
                "fmax": fmax,
                "max_steps": max_steps,
                "energy_change_eV": energy_diff,
                "energy_change_eV_per_atom": energy_diff_per_atom,
                # Exact now: optimizer.get_number_of_steps() from
                # run_relaxation(), instead of the old len(trajectory) - 2
                # approximation needed to undo chgnet's duplicated final
                # observer call.
                "optimizer_steps": result.get("optimizer_steps", 0),
                "trajectory_frames": steps,
                "optimizer": optimizer_name,
                "aborted": aborted,
                "abort_reason": "timeout" if aborted else None,
                # Complete per-step history for the live charts: the SSE
                # stream can end (slot removed) or be closed by the client
                # before the last steps are delivered, so the response
                # carries the authoritative list.
                "progress_history": list(progress_slot.get("history", [])),
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

            logger.info(f"MLIP relaxation completed: {relaxation_info['steps']} steps, converged: {relaxation_info['converged']}")

            # Generate relaxed structure info
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            # Symmetry analysis is informational only; a spglib failure must
            # not discard a relaxation that may have taken minutes.
            space_group = space_group_number = point_group = crystal_system = None
            try:
                analyzer = await asyncio.to_thread(SpacegroupAnalyzer, final_structure)
                space_group = analyzer.get_space_group_symbol()
                space_group_number = analyzer.get_space_group_number()
                point_group = analyzer.get_point_group_symbol()
                crystal_system = analyzer.get_crystal_system()
            except Exception as e:
                logger.warning(f"Symmetry analysis of relaxed structure failed: {e}")
                space_group = space_group_number = point_group = crystal_system = None

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
                "space_group": space_group,
                "space_group_number": space_group_number,
                "point_group": point_group,
                "crystal_system": crystal_system,
                "num_sites": len(final_structure.sites)
            }

            # Extract trajectory data for analysis modal
            # Use all steps as provided by MLIP - no unnecessary optimization
            trajectory_data = None
            if trajectory:
                # trajectory.energies (TrajectoryRecorder) stores TOTAL energy
                # per step (mirrors chgnet's own TrajectoryObserver interface,
                # so that class itself must not change), but the frontend
                # charts/labels this array as eV/atom, matching
                # energy_eV_per_atom reported elsewhere in this same response.
                # Convert here, at the response boundary, instead.
                num_atoms_for_energy = len(final_structure.sites)
                trajectory_data = {
                    "steps": len(trajectory.forces) if hasattr(trajectory, 'forces') else 0,
                    "energies": [float(e) / num_atoms_for_energy for e in trajectory.energies] if hasattr(trajectory, 'energies') and num_atoms_for_energy > 0 else [],
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

            # Save relaxed structure and MLIP result metadata to session for
            # later use -- but only if the session's structure hasn't been
            # edited (substitution, reset, etc.) since this relaxation started.
            # Otherwise this result is stale and must not resurrect
            # relaxed_structure/relax_result after update_structure() already
            # invalidated them for the edit (see update_structure's comment).
            if session_info.get('structure_version', 0) == structure_version_at_start:
                session_info['relaxed_structure'] = final_structure
                session_info['relax_result'] = {
                    'model': backend.id,
                    'fmax': fmax,
                    'converged': relaxation_info.get('converged', False),
                    'steps': relaxation_info.get('optimizer_steps', 0),
                    'optimizer': optimizer_name,
                }
                logger.info(f"Saved relaxed structure to session {session_id[:8]}...")
            else:
                logger.warning(
                    f"Session {session_id[:8]}... structure changed while this "
                    f"relaxation was running; discarding its now-stale result "
                    f"instead of saving relaxed_structure/relax_result."
                )

            # Analytics Logging - Log successful relaxation
            try:
                analytics_db.log_event(
                    event_type="relax",
                    filename=filename,
                    formula=final_formula, # Use final formula
                    num_atoms=final_num_sites,
                    execution_time=None, # We don't have precise execution time here easily without refactoring
                    parameters={
                        "fmax": fmax,
                        "steps": relaxation_info["optimizer_steps"],
                        "converged": converged,
                        "energy_change_eV": energy_diff,
                        "optimizer": optimizer_name,
                        "aborted": aborted,
                        "elastic": elastic,
                        "elastic_strain": elastic_strain,
                    },
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Failed to log relaxation event: {e}")

            response = {
                "status": "success",
                "initial_prediction": initial_results,
                "final_prediction": final_results,
                "relaxation_info": relaxation_info,
                "relaxed_structure_info": relaxed_structure_info,
                "trajectory_data": trajectory_data,
                "model_info": {
                    "id": backend.id,
                    "display_name": backend.display_name,
                    "version": backend.version(),
                    "device": _get_device_info(backend),
                    "parameters": backend.parameters()
                }
            }
            if elastic:
                response["elastic_info"] = elastic_info
            # Sanitize before returning: a MLIP blow-up can produce NaN/
            # Infinity energies or forces, which json.dumps() would emit as
            # bare (invalid-JSON) literals -- see _sanitize_json_value.
            return _sanitize_json_value(response)
        finally:
            _relax_progress.pop(session_id, None)
            _relax_semaphore.release()

    except ValueError as e:
        logger.error(f"MLIP relaxation validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"MLIP relaxation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

def _new_history_points(slot: dict, sent: int) -> list:
    """Return the history entries of a progress slot beyond the first `sent`.

    The watcher only ever appends, so slicing from a per-connection index
    yields exactly the points this client has not seen yet (the 0.5 s SSE
    poll would otherwise drop every intermediate optimizer step).
    """
    return list(slot.get("history", ())[sent:])

@app.get("/api/relax-progress/{session_id}")
async def stream_relax_progress(session_id: str):
    """Server-sent events for an in-flight relaxation.

    RelaxationWatcher (run inside the MLIP executor thread) publishes into
    _relax_progress[session_id] as the optimizer steps; a plain dict
    assignment is atomic under the GIL, so neither side needs a lock. The
    stream ends on its own once /api/relax's finally block removes
    the session's slot.
    """
    async def events():
        last = None
        sent = 0  # history entries already sent to THIS client
        slot = None
        while session_id in _relax_progress:
            slot = _relax_progress[session_id]
            state = slot.get("state")
            new_points = _new_history_points(slot, sent)
            if state is not None and (state != last or new_points):
                last = state
                sent += len(new_points)
                yield f"data: {json.dumps({**state, 'history': new_points})}\n\n"
            await asyncio.sleep(0.5)
        # The slot is removed as soon as the relaxation finishes; flush the
        # steps published since the last poll before signalling completion.
        if slot is not None and slot.get("state") is not None:
            tail = _new_history_points(slot, sent)
            if tail:
                yield f"data: {json.dumps({**slot['state'], 'history': tail})}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

def _reset_session_structure_sync(original_structure: Structure, supercell_size: List[int]) -> Structure:
    """
    CPU-bound: rebuild the supercell from `original_structure` at
    `supercell_size`. Call via asyncio.to_thread.
    """
    reset_structure = original_structure.copy()
    reset_structure.make_supercell(supercell_size)
    return reset_structure


@app.post("/api/reset-session-structure")
async def reset_session_structure(request: dict):
    """
    Reset session structure to original supercell state
    Ensures frontend and backend state consistency after Reset Operations
    """
    try:
        session_id = request.get("session_id")
        if not session_id or not isinstance(session_id, str):
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

        # Serialize concurrent requests against this same session (see
        # _get_session_lock's docstring) before doing any of the heavy work.
        session_lock = _get_session_lock(session_info)
        async with session_lock:
            # Recreate supercell from original structure
            reset_structure = await asyncio.to_thread(
                _reset_session_structure_sync, original_structure, supercell_size
            )

            # The session may have been replaced wholesale (a brand new dict
            # for the same session_id) while the above ran; see the same
            # check in apply_atomic_operations for why this is necessary.
            if session_manager.sessions.get(session_id) is not session_info:
                raise HTTPException(
                    status_code=409,
                    detail="Session was modified concurrently; please retry")

            # Update session with reset structure and clear operations. This also
            # discards any relaxed_structure/relax_result (update_structure
            # always invalidates them), so no separate cleanup is needed here.
            session_manager.update_structure(session_id, reset_structure, operations=[])

        logger.info(f"Session {session_id[:8]}... reset successfully")
        logger.info(f"Reset structure: {reset_structure.composition} ({len(reset_structure)} sites)")

        return {
            "status": "success",
            "message": "Session reset to original structure",
            "session_id": session_id,
            "structure_info": {
                # str(Structure.formula), not str(.composition): the client
                # compares this against supercell_info.formula (which has the
                # same "Cu32"-shaped form), and the two do not always render
                # identically -- a mismatch there falsely showed "(Note: Sync
                # issue detected)" on an otherwise-correct reset.
                "formula": str(reset_structure.formula),
                "num_sites": len(reset_structure),
                "volume": float(reset_structure.volume),
                "density": float(reset_structure.density),
                "supercell_size": supercell_size
            }
        }
        
    except ValueError as e:
        logger.error(f"Session reset validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Session reset error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


def _label_and_write_relaxed_cif(structure: Structure) -> "Tuple[Structure, str]":
    """
    CPU-bound: apply CIF element labels to `structure` (mutates it in
    place) and render CIF text. Call via asyncio.to_thread on a private
    copy of the structure -- never on a session's shared Structure object.
    """
    from pymatgen.io.cif import CifWriter
    apply_element_labels_to_structure(structure)
    cif_writer = CifWriter(
        structure,
        write_magmoms=False,
        significant_figures=6
    )
    return structure, str(cif_writer)


@app.post("/api/generate-relaxed-structure-cif")
async def generate_relaxed_structure_cif(request: dict):
    """
    Generate CIF from relaxed structure - uses session-managed relaxation results
    """
    try:
        session_id = request.get("session_id")

        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="Session ID is required")

        # Get session info to access the latest relaxed structure
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        
        filename = sanitize_display_filename(session_info.get('filename', 'unknown'))
        logger.info(f"Generating relaxed structure CIF for {filename}")

        # Optional model guard: reject a request naming a model other than the
        # one that produced the stored relaxation (omitted -> no check).
        requested_model_id = _request_model_id(request)
        # A request naming a model other than the session's -> 409.
        _session_model_id(session_info, requested_model_id)
        requested_backend = None
        if requested_model_id is not None:
            # Registry lookup only (no model load); unknown id -> 400.
            requested_backend = _resolve_backend(requested_model_id)
        
        # Check if relaxed structure exists in session. No fallback to the
        # unrelaxed current structure: silently downloading the pre-relaxation
        # structure as if it were "the relaxed structure" is misleading, so
        # this must fail loudly instead.
        relaxed_structure = session_info.get('relaxed_structure')
        if relaxed_structure is None:
            raise HTTPException(
                status_code=404,
                detail="No relaxed structure available for this session. Run a structure relaxation first."
            )
        logger.info("Using relaxed structure from session")

        # Never label/write on the session's own stored Structure object:
        # apply_element_labels_to_structure mutates in place, and running
        # that mutation in a worker thread on the shared session object
        # would race with any other request reading/writing it. Work on a
        # private copy instead.
        final_structure = relaxed_structure.copy()

        # Get operations and supercell info from session
        operations = session_info.get('operations', [])
        supercell_size = session_info.get('supercell_size', [1, 1, 1])

        # Get relaxation info from session
        relax_result = session_info.get('relax_result', {})
        fmax = relax_result.get('fmax', 'N/A')
        converged = relax_result.get('converged', 'N/A')
        steps = relax_result.get('steps', 'N/A')
        # Default to FIRE for CIFs from sessions relaxed before the
        # optimizer became selectable, so old sessions still read sensibly.
        optimizer_used = relax_result.get('optimizer', 'FIRE')
        # Sessions relaxed before the model field existed were CHGNet runs.
        model_id = relax_result.get('model', 'chgnet')
        model_cls = MODEL_REGISTRY.get(model_id)
        model_display_name = model_cls.display_name if model_cls else model_id

        if requested_backend is not None and requested_backend.id != model_id:
            raise HTTPException(
                status_code=409,
                detail=f"Relaxed structure was produced by '{model_display_name}'; "
                       f"re-run the relaxation with '{requested_backend.display_name}' first.")

        # Generate CIF using pymatgen CifWriter (CPU-bound -- offload)
        final_structure, cif_content = await asyncio.to_thread(
            _label_and_write_relaxed_cif, final_structure
        )

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
            f"# {model_display_name} relaxation: model={model_id}, fmax={fmax}, steps={steps}, converged={converged}, optimizer={optimizer_used}",
            f"# Final formula: {final_formula_str}",
            f"# Number of atoms: {final_num_atoms}",
            f"# Volume: {final_volume_val:.2f} A^3",
            ""
        ]
        
        final_cif = "\n".join(metadata_lines) + cif_content
        
        logger.info(f"Relaxed CIF generated successfully, length: {len(final_cif)} characters")
        
        return Response(
            content=final_cif,
            media_type="chemical/x-cif",
            headers={
                "Content-Disposition": build_content_disposition(f"{filename.replace('.cif', '')}_relaxed.cif"),
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        error_msg = f"Failed to generate relaxed structure CIF: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def find_interstitial_candidates(structure: Structure, min_dist: float = 0.9,
                                 clustering_tol: float = 0.5,
                                 fingerprint_cutoff: float = 6.0,
                                 max_candidates: int = MAX_VOID_CANDIDATES) -> List[Dict]:
    """
    Find symmetry-distinct interstitial candidate sites via Voronoi tessellation.

    This replaces pymatgen's VoronoiInterstitialGenerator, whose StructureMatcher-based
    symmetry grouping performs O(C^2) full structure comparisons over the C candidate
    sites (each comparison itself scales with the number of atoms). On a 3x3x3 Cu
    supercell (108 atoms) that generator takes ~7 minutes; this function returns the
    same sites in well under a second by grouping candidates with a local-environment
    fingerprint (an O(C*N) operation) instead of pairwise structure matching.

    Args:
        structure: host structure (without the inserted atom)
        min_dist: minimum allowed distance between a candidate site and any host atom
        clustering_tol: distance tolerance for merging nearby Voronoi nodes
        fingerprint_cutoff: neighbor search radius used to build the dedup fingerprint
        max_candidates: hard cap on the number of returned sites (protects the
            downstream per-site MLIP energy evaluation from low-symmetry structures
            that can otherwise produce very many candidates)

    Returns:
        List of dicts with "id", "label", "frac_coords", "min_dist", sorted by
        min_dist descending (largest voids first).
    """
    lattice = structure.lattice
    frac = structure.frac_coords
    if len(frac) == 0:
        return []

    # Voronoi polyhedra extend beyond the unit cell, so tessellate over the
    # periodic images too. Very small cells need a wider image range for an
    # accurate tessellation near the cell boundary (mirrors the same bump
    # pymatgen's TopographyAnalyzer applies for <5-atom cells).
    cell_range = [-2, -1, 0, 1, 2] if len(structure) < 5 else [-1, 0, 1]
    shifts = np.array(list(itertools.product(cell_range, cell_range, cell_range)))
    all_frac = (frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    cart_coords = lattice.get_cartesian_coords(all_frac)

    try:
        voro = Voronoi(cart_coords)
    except Exception as e:
        logger.error(f"Voronoi tessellation failed: {e}")
        return []

    vertex_frac = lattice.get_fractional_coords(voro.vertices)
    tol = 1e-4
    in_cell = np.all((vertex_frac >= -tol) & (vertex_frac < 1 + tol), axis=1)
    sites = vertex_frac[in_cell]
    if len(sites) == 0:
        return []

    # Reuse pymatgen's own helpers for collision removal and node clustering so
    # the geometric definition of a "site" matches the previous implementation.
    sites = remove_collisions(sites, structure=structure, min_dist=min_dist)
    if len(sites) == 0:
        return []
    sites = cluster_nodes(sites, lattice=lattice, tol=clustering_tol)

    # Group candidates by a local-environment fingerprint instead of full
    # structure matching. Sites with an identical multiset of (neighbor
    # element, rounded distance) pairs within fingerprint_cutoff are treated
    # as symmetry-equivalent and collapsed to one representative.
    seen = set()
    candidates = []
    empty_counter = 0
    for fc in sites:
        probe = PeriodicSite("X", fc, lattice)
        neighbors = structure.get_neighbors(probe, fingerprint_cutoff)
        if not neighbors:
            # No atoms within the cutoff (e.g. large-vacuum / molecular
            # structures): nothing to fingerprint against, so keep each such
            # site distinct rather than collapsing them all into one.
            key = ("__empty__", empty_counter)
            empty_counter += 1
            min_d = 0.0
        else:
            key = tuple(sorted((n.specie.symbol, round(n.nn_distance, 2)) for n in neighbors))
            min_d = min(n.nn_distance for n in neighbors)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "frac_coords": [float(x) for x in fc],
            "min_dist": round(float(min_d), 3),
        })

    # Largest voids first, then cap so a low-symmetry structure can't blow up
    # the number of per-site MLIP evaluations downstream.
    candidates.sort(key=lambda c: c["min_dist"], reverse=True)
    candidates = candidates[:max_candidates]
    for i, c in enumerate(candidates):
        c["id"] = i
        c["label"] = f"Site {i + 1}"
    return candidates

@app.post("/api/get-insertable-elements")
async def get_insertable_elements(data: dict):
    """
    Returns all allowed elements for insertion. 
    Geometric filtering is removed for performance optimization.
    """
    try:
        # We no longer need Voronoi analysis here as it's slow.
        # Just return the sorted list of allowed elements.
        
        # Optional model: restrict to that model's elements (default: union).
        sid = data.get("session_id")
        model_id = _session_model_id(
            session_manager.sessions.get(sid) if isinstance(sid, str) else None,
            _request_model_id(data))
        echoed_model = None
        if model_id is not None:
            backend = _resolve_backend(model_id)
            allowed = set(backend.supported_elements())
            echoed_model = backend.id
        else:
            allowed = ALLOWED_ELEMENTS
        insertable_elements = []
        for symbol in allowed:
            insertable_elements.append(symbol)
            
        # Sort by atomic number
        insertable_elements.sort(key=lambda x: Element(x).Z)
        
        return {
            "status": "success",
            "insertable_elements": insertable_elements,
            "max_void_radius": None,  # No longer calculated for speed
            "model": echoed_model,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting insertable elements: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/get-insertion-voids")
async def get_insertion_voids(data: dict):
    """
    Finds potential interstitial sites using Voronoi analysis.
    Does not perform energy calculation.

    Candidate sites do not depend on the element being inserted (pymatgen's own
    generator computes them the same way), so results are cached per-session and
    reused across element changes within the same structure state.
    """
    try:
        session_id = data.get("session_id")
        element_symbol = data.get("element")

        if not session_id or not isinstance(session_id, str) or not element_symbol:
            raise HTTPException(status_code=400, detail="Session ID and element are required")

        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Structure not found")
        structure = session_info["current_structure"]

        if len(structure) > MAX_PREDICT_ATOMS:
            raise HTTPException(
                status_code=400,
                detail=f"Structure has {len(structure)} atoms; the limit for insertion-void "
                       f"search is {MAX_PREDICT_ATOMS}. Reduce the supercell size.")

        # Cache key reflects the structure state actually seen by voids search:
        # structure_version (bumped on every structural change, including a
        # Reset, by SessionManager.update_structure) + supercell size. Using
        # len(operations) here previously collided across a Reset: e.g.
        # "delete atom 3" then, after a Reset, "delete atom 7" both produce a
        # 1-length operations list and would return stale voids that may now
        # overlap a real atom. Element is deliberately excluded since it does
        # not affect the candidate geometry.
        cache_key = (session_info.get("structure_version", 0),
                    tuple(session_info.get("supercell_size", [])))
        void_cache = session_info.setdefault("void_cache", {})

        if cache_key in void_cache:
            voids = void_cache[cache_key]
            logger.info(f"Using cached {len(voids)} insertion site(s) for session {session_id[:8]}...")
        else:
            try:
                voids = await asyncio.to_thread(find_interstitial_candidates, structure)
            except Exception as e:
                logger.error(f"Failed to find insertion sites: {e}")
                raise HTTPException(status_code=400, detail="Failed to find insertion voids for this structure")
            void_cache.clear()  # only the current structure state is worth keeping
            void_cache[cache_key] = voids
            logger.info(f"Found {len(voids)} potential insertion site(s) for session {session_id[:8]}...")

        return {
            "status": "success",
            "element": element_symbol,
            "voids": voids
        }
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error getting insertion voids: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/evaluate-insertion-energy")
async def evaluate_insertion_energy(data: dict):
    """
    Predicts the energy of a structure with a single inserted atom.
    Used for iterative progress display.
    """
    try:
        session_id = data.get("session_id")
        element_symbol = data.get("element")
        frac_coords = data.get("frac_coords")
        
        if not session_id or not isinstance(session_id, str) or not element_symbol or frac_coords is None:
            raise HTTPException(status_code=400, detail="Session ID, element, and coords are required")
        eval_session = session_manager.sessions.get(session_id)
        model_id = _session_model_id(eval_session, _request_model_id(data))
        try:
            element_symbol = validate_element(element_symbol, _session_allowed_elements(eval_session))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            frac_coords = validate_frac_coords(frac_coords, "frac_coords")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        structure = session_manager.get_current_structure(session_id)
        if not structure:
            raise HTTPException(status_code=404, detail="Structure not found")

        if len(structure) > MAX_PREDICT_ATOMS:
            raise HTTPException(
                status_code=400,
                detail=f"Structure has {len(structure)} atoms; the prediction limit is "
                       f"{MAX_PREDICT_ATOMS}. Reduce the supercell size.")

        # Create candidate structure
        cand_structure = structure.copy()
        cand_structure.append(element_symbol, frac_coords)

        # Predict energy. task="e" skips force/stress/magmom autograd, which are
        # unused for site screening and account for most of the per-site cost
        # (~5x speedup measured: 2.0s -> 0.4s per site on a 109-atom structure).
        backend = _resolve_backend(model_id)
        _ensure_elements_supported(backend, _structure_symbols(structure) | {element_symbol})
        await _load_or_503(backend)
        try:
            results = _prediction_payload(
                await backend.predict_one(cand_structure, _PROPS_ENERGY))
            per_atom_energy = results.get("energy_eV_per_atom")
            if per_atom_energy is None or not math.isfinite(per_atom_energy):
                raise ValueError("Prediction returned no finite energy value")
            total_energy_eV = per_atom_energy * len(cand_structure)
        except Exception as e:
            logger.error(f"Energy prediction failed: {e}")
            return {"status": "error", "energy": None, "detail": str(e)}

        return {
            "status": "success",
            "energy": total_energy_eV,           # total energy of the candidate structure in eV
            "energy_eV_per_atom": per_atom_energy,
            "model": backend.id,
        }
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error evaluating insertion energy: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/evaluate-insertion-energies")
async def evaluate_insertion_energies(data: dict):
    """
    Predicts insertion energies for a batch of candidate sites in one MLIP call.

    Used by the Auto-mode insertion sweep to replace the previous one-HTTP-request-
    per-site loop; sites are chunked client-side (MAX_INSERTION_BATCH per request)
    so a single request stays short enough for STOP to remain responsive.
    """
    try:
        session_id = data.get("session_id")
        element_symbol = data.get("element")
        sites = data.get("sites")

        if not session_id or not isinstance(session_id, str) or not element_symbol or not sites:
            raise HTTPException(status_code=400, detail="Session ID, element, and sites are required")
        eval_session = session_manager.sessions.get(session_id)
        model_id = _session_model_id(eval_session, _request_model_id(data))
        try:
            element_symbol = validate_element(element_symbol, _session_allowed_elements(eval_session))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not isinstance(sites, list):
            raise HTTPException(status_code=400, detail="sites must be a list")
        if len(sites) > MAX_INSERTION_BATCH:
            raise HTTPException(status_code=400,
                               detail=f"sites exceeds the maximum batch size of {MAX_INSERTION_BATCH}")

        structure = session_manager.get_current_structure(session_id)
        if not structure:
            raise HTTPException(status_code=404, detail="Structure not found")

        if len(structure) > MAX_PREDICT_ATOMS:
            raise HTTPException(
                status_code=400,
                detail=f"Structure has {len(structure)} atoms; the prediction limit is "
                       f"{MAX_PREDICT_ATOMS}. Reduce the supercell size.")

        cand_structures = []
        site_ids = []
        for site in sites:
            if not isinstance(site, dict):
                raise HTTPException(status_code=400, detail="Each site must be an object with 'id' and 'frac_coords'")
            site_id = site.get("id")
            frac_coords = site.get("frac_coords")
            if site_id is None or frac_coords is None:
                raise HTTPException(status_code=400, detail="Each site requires 'id' and 'frac_coords'")
            try:
                frac_coords = validate_frac_coords(frac_coords, "frac_coords")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"site {site_id}: {e}")
            cand_structure = structure.copy()
            cand_structure.append(element_symbol, frac_coords)
            cand_structures.append(cand_structure)
            site_ids.append(site_id)

        num_atoms = len(structure) + 1
        backend = _resolve_backend(model_id)
        _ensure_elements_supported(backend, _structure_symbols(structure) | {element_symbol})
        await _load_or_503(backend)
        pred_results = await backend.predict(cand_structures, _PROPS_ENERGY)

        results = []
        for site_id, pred_result in zip(site_ids, pred_results):
            if pred_result is None:
                results.append({"id": site_id, "energy": None, "energy_eV_per_atom": None,
                               "error": "Prediction failed"})
                continue
            try:
                parsed = _prediction_payload(pred_result)
                per_atom_energy = parsed.get("energy_eV_per_atom")
                if per_atom_energy is None or not math.isfinite(per_atom_energy):
                    raise ValueError("Prediction returned no finite energy value")
                results.append({
                    "id": site_id,
                    "energy": per_atom_energy * num_atoms,
                    "energy_eV_per_atom": per_atom_energy,
                    "error": None,
                })
            except Exception as e:
                logger.warning(f"Failed to parse prediction for site {site_id}: {e}")
                results.append({"id": site_id, "energy": None, "energy_eV_per_atom": None, "error": str(e)})

        return {"status": "success", "element": element_symbol, "model": backend.id,
                "results": results}
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error evaluating insertion energies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/evaluate-candidate-energies")
async def evaluate_candidate_energies(data: dict):
    """
    Predicts substitution/deletion energies for a batch of candidate atoms in
    one MLIP call.

    Mirrors /api/evaluate-insertion-energies (added for the equivalent
    insertion sweep): used by the Auto-mode substitution and deletion sweeps
    to replace the previous one-HTTP-request-per-atom loop, where each
    request re-parsed the CIF from disk, rebuilt the supercell, and replayed
    the full operation history just to score one candidate. Candidates are
    chunked client-side (MAX_CANDIDATE_BATCH per request) so a single
    request stays short enough for STOP to remain responsive, and each
    chunk is one native batched MLIP call.
    """
    try:
        session_id = data.get("session_id")
        candidates = data.get("candidates")

        if not session_id or not isinstance(session_id, str) or not candidates:
            raise HTTPException(status_code=400, detail="Session ID and candidates are required")
        if not isinstance(candidates, list):
            raise HTTPException(status_code=400, detail="candidates must be a list")
        if len(candidates) > MAX_CANDIDATE_BATCH:
            raise HTTPException(status_code=400,
                               detail=f"candidates exceeds the maximum batch size of {MAX_CANDIDATE_BATCH}")

        structure = session_manager.get_current_structure(session_id)
        if not structure:
            raise HTTPException(status_code=404, detail="Structure not found")
        cand_session = session_manager.sessions.get(session_id)
        model_id = _session_model_id(cand_session, _request_model_id(data))
        allowed_elements = _session_allowed_elements(cand_session)

        if len(structure) > MAX_PREDICT_ATOMS:
            raise HTTPException(
                status_code=400,
                detail=f"Structure has {len(structure)} atoms; the prediction limit is "
                       f"{MAX_PREDICT_ATOMS}. Reduce the supercell size.")

        # Built aligned to `candidates` order (one slot per candidate) so
        # same-element substitutions can be filled in immediately below,
        # without going through the batched MLIP call, while everything
        # else is filled in afterward at its own original slot.
        results: List[Optional[dict]] = [None] * len(candidates)

        cand_structures = []
        cand_ids = []
        cand_num_atoms = []
        cand_result_slots = []
        for slot, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise HTTPException(status_code=400, detail="Each candidate must be an object")
            cand_id = candidate.get("id")
            action = candidate.get("action")
            index = candidate.get("index")

            if cand_id is None or action not in ("substitute", "delete"):
                raise HTTPException(status_code=400,
                                   detail="Each candidate requires 'id' and action 'substitute' or 'delete'")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(structure):
                raise HTTPException(status_code=400,
                                   detail=f"candidate {cand_id}: index {index} out of range (max: {len(structure) - 1})")

            cand_structure = structure.copy()
            if action == "substitute":
                new_element = candidate.get("to")
                try:
                    # Use the stripped value validate_element() returns, so a
                    # symbol with leading/trailing whitespace doesn't reach
                    # pymatgen's Element() unstripped and raise ValueError (500).
                    new_element = validate_element(new_element, allowed_elements)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=f"candidate {cand_id}: {e}")

                old_symbol = _site_element_symbol(structure[index].specie)
                if new_element == old_symbol:
                    # Nothing to predict -- fill this slot directly instead of
                    # sending a no-op substitution through the MLIP batch.
                    results[slot] = {"id": cand_id, "energy": None, "energy_eV_per_atom": None,
                                     "error": "candidate element is unchanged"}
                    continue

                substitute_site(cand_structure, index, new_element)
            else:  # delete
                cand_structure.remove_sites([index])

            cand_structures.append(cand_structure)
            cand_ids.append(cand_id)
            cand_num_atoms.append(len(cand_structure))
            cand_result_slots.append(slot)

        backend = _resolve_backend(model_id)
        _ensure_elements_supported(
            backend, _structure_symbols(structure) | {c.get("to") for c in candidates
                                                      if c.get("action") == "substitute"
                                                      and isinstance(c.get("to"), str)})
        pred_results = []
        if cand_structures:
            await _load_or_503(backend)
            pred_results = await backend.predict(cand_structures, _PROPS_ENERGY)

        for slot, cand_id, num_atoms, pred_result in zip(cand_result_slots, cand_ids, cand_num_atoms, pred_results):
            if pred_result is None:
                results[slot] = {"id": cand_id, "energy": None, "energy_eV_per_atom": None,
                                 "error": "Prediction failed"}
                continue
            try:
                parsed = _prediction_payload(pred_result)
                per_atom_energy = parsed.get("energy_eV_per_atom")
                if per_atom_energy is None or not math.isfinite(per_atom_energy):
                    raise ValueError("Prediction returned no finite energy value")
                results[slot] = {
                    "id": cand_id,
                    "energy": per_atom_energy * num_atoms,
                    "energy_eV_per_atom": per_atom_energy,
                    "error": None,
                }
            except Exception as e:
                logger.warning(f"Failed to parse prediction for candidate {cand_id}: {e}")
                results[slot] = {"id": cand_id, "energy": None, "energy_eV_per_atom": None, "error": str(e)}

        return {"status": "success", "model": backend.id, "results": results}
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error evaluating candidate energies: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/equivalent-sites")
async def equivalent_sites(data: dict):
    """
    Groups a structure's sites of one element into symmetry-equivalent
    classes, so the Auto-mode substitution/deletion sweep
    (runOptimalCandidateSweep in templates/index.html) can evaluate just one
    representative per class with MLIP instead of every site: candidates
    within a class are guaranteed to produce symmetrically identical
    structures, so scoring more than one per class is redundant work (see
    plan doc / commit message for measurements). This is a pure
    optimization -- the client falls back to the full, unreduced candidate
    list on any failure of this endpoint.

    Response `groups` is a partition of every site of `element`: each inner
    list is one symmetry-equivalent class (its indices sorted ascending),
    and the outer list is sorted by each class's smallest index ascending.
    This endpoint intentionally returns the full partition rather than just
    one representative index per class, so the client decides which index
    within a class to use as its representative (currently: the smallest)
    without a server round-trip if that choice ever changes.
    """
    try:
        session_id = data.get("session_id")
        element = data.get("element")

        if not session_id or not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="Session ID is required")

        structure = session_manager.get_current_structure(session_id)
        if not structure:
            raise HTTPException(status_code=404, detail="Structure not found")

        try:
            element = validate_element(element, _session_allowed_elements(
                session_manager.sessions.get(session_id)))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if len(structure) > MAX_PREDICT_ATOMS:
            raise HTTPException(
                status_code=400,
                detail=f"Structure has {len(structure)} atoms; the prediction limit is "
                       f"{MAX_PREDICT_ATOMS}. Reduce the supercell size.")

        # CPU-bound pymatgen symmetry analysis, offloaded off the event loop.
        equivalent_groups = await asyncio.to_thread(_compute_equivalent_indices, structure)
        if equivalent_groups is None:
            logger.warning(
                f"Symmetry analysis failed for session {session_id[:8]}...; "
                f"falling back to singleton equivalence groups for /api/equivalent-sites")
            equivalent_groups = [[i] for i in range(len(structure))]

        groups = []
        for group in equivalent_groups:
            sorted_group = sorted(group)
            if not sorted_group:
                continue
            representative_element = _site_element_symbol(structure[sorted_group[0]].specie)
            if representative_element == element:
                groups.append(sorted_group)
        groups.sort(key=lambda g: g[0])

        return {"status": "success", "element": element, "groups": groups}
    except HTTPException:
        # Re-raise HTTPExceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error computing equivalent sites: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Analytics API Routes ---

def require_analytics_access(request: Request) -> None:
    """
    Access gate for the analytics dashboard and its APIs (S-3 in the
    security review): with no gate at all, anyone who can reach this
    server could read every visitor's IP/User-Agent history and every
    uploaded filename/formula via /api/analytics/recent.

    If CRYSTALNEXUS_ANALYTICS_TOKEN is set, a request must present a
    matching token (X-Analytics-Token header, or a ?token= query
    parameter for the plain <a href> dashboard link), compared with a
    constant-time comparison to avoid a timing side channel.

    If it is not set, fail safe by allowing only loopback requests -- the
    same default posture as HOST (main.py defaults to 127.0.0.1), so a
    fresh install is never silently open even on a LAN-facing deployment.
    """
    if CRYSTALNEXUS_ANALYTICS_TOKEN:
        supplied = request.headers.get("x-analytics-token") or request.query_params.get("token")
        # Encode both sides to bytes before comparing: secrets.compare_digest
        # requires its two arguments be the same type, and raises TypeError
        # (surfacing as an unhandled 500) if `supplied` is a `str` containing
        # non-ASCII characters compared against another `str` in some
        # implementations -- encoding explicitly sidesteps that entirely.
        if not supplied or not secrets.compare_digest(
            supplied.encode("utf-8", errors="ignore"),
            CRYSTALNEXUS_ANALYTICS_TOKEN.encode("utf-8"),
        ):
            raise HTTPException(status_code=401, detail="Analytics access requires a valid token")
        return

    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=401,
            detail=(
                "Analytics dashboard is only accessible from localhost "
                "unless CRYSTALNEXUS_ANALYTICS_TOKEN is set"
            )
        )

@app.get("/analytics", response_class=HTMLResponse, dependencies=[Depends(require_analytics_access)])
async def analytics_dashboard(request: Request):
    """Serve the analytics dashboard page"""
    return templates.TemplateResponse(request, "analytics.html")

@app.get("/api/analytics/summary", dependencies=[Depends(require_analytics_access)])
async def get_analytics_summary():
    """Get summary statistics for dashboard"""
    try:
        daily_counts = analytics_db.get_daily_access_counts(days=14)
        feature_usage = analytics_db.get_feature_usage()
        return {
            "daily_access": daily_counts,
            "feature_usage": feature_usage
        }
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")

@app.get("/api/analytics/ranking", dependencies=[Depends(require_analytics_access)])
async def get_analytics_ranking():
    """Get popular samples ranking"""
    try:
        popular_samples = analytics_db.get_popular_samples(limit=10)
        return {
            "popular_samples": popular_samples
        }
    except Exception as e:
        logger.error(f"Failed to get analytics ranking: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics ranking")

@app.get("/api/analytics/recent", dependencies=[Depends(require_analytics_access)])
async def get_analytics_recent():
    """Get recent activity log"""
    try:
        recent_events = analytics_db.get_recent_events(limit=50)
        return {
            "recent_events": recent_events
        }
    except Exception as e:
        logger.error(f"Failed to get recent events: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recent events")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
