"""CHGNet backend (ported from the former main.CHGNetModelManager)."""
import asyncio
import logging
import os
import platform
from typing import List, Optional, Set

from pymatgen.core import Element

from .base import (CAP_BATCH, CAP_ENERGY, CAP_FEATURES, CAP_FORCES, CAP_MAGMOM,
                   CAP_SITE_ENERGIES, CAP_STRESS, PROP_ATOM_FEATURES,
                   PROP_CRYSTAL_FEATURES, PROP_ENERGY, PROP_SITE_ENERGIES,
                   MLIPBackend)
from .executor import run_in_inference_executor

logger = logging.getLogger(__name__)

WINDOWS_PLATFORM = platform.system() == "Windows"

# CHGnet import with Windows compatibility
try:
    import torch
    # Windows-specific NumPy/PyTorch compatibility settings
    if WINDOWS_PLATFORM:
        # Set NumPy to use compatible dtypes for Windows
        os.environ["NPY_NO_DEPRECATED_API"] = "NPY_1_7_API_VERSION"
        # Ensure PyTorch uses CPU on Windows to avoid CUDA issues
        torch.set_default_dtype(torch.float32)
        logger.info("Windows platform detected - applied compatibility settings")

    from chgnet.model import CHGNet  # noqa: F401
    CHGNET_AVAILABLE = True
    logger.info(f"CHGNet successfully loaded (Platform: {platform.system()})")
except ImportError:
    CHGNET_AVAILABLE = False
    logger.warning(f"CHGNet not available on {platform.system()}: Install with: pip install chgnet")
except Exception as e:
    CHGNET_AVAILABLE = False
    if WINDOWS_PLATFORM and "Buffer dtype mismatch" in str(e):
        logger.error(f"Windows CHGNet compatibility issue detected: {e}")
        logger.error("Solution: Install Microsoft Visual C++ Redistributable")
    else:
        logger.error(f"CHGNet loading error: {e}")

# Configurable batch size for CPU processing (env name kept for now)
MLIP_BATCH_SIZE = int(os.getenv('MLIP_BATCH_SIZE', '4'))

# CHGNet model settings
MAX_ATOMIC_NUMBER = int(os.getenv('MAX_ATOMIC_NUMBER', '94'))
MIN_SUPPORTED_ELEMENTS = int(os.getenv('MIN_SUPPORTED_ELEMENTS', '50'))


# ---------------------------------------------------------------------------
# Supported elements (moved from main.py)
# ---------------------------------------------------------------------------

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
    """Get element list supported by CHGNet (atomic-number range with exclusions)"""
    try:
        # Note: no model load needed here; the supported set is derived from the
        # known CHGNet atomic-number range, avoiding a redundant CHGNet.load().
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
        # Periods 1-2
        'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
        # Period 3
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
        # Period 4
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br',
        # Period 5
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I',
        # Period 6 (main)
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
        'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
        'Au', 'Hg', 'Tl', 'Pb', 'Bi',
        # Period 7 (stable)
        'Th', 'U'
    }


# ---------------------------------------------------------------------------
# Normalization (moved from main.safe_get_prediction)
# ---------------------------------------------------------------------------

_HANDLED_RAW_KEYS = ("energy", "e", "forces", "f", "stress", "s", "magmom", "m",
                     "site_energies", "crystal_fea", "atom_fea")


def normalize_chgnet_prediction(pred, num_atoms=None) -> dict:
    """Map CHGNet's raw predict_structure output to the normalized keys.

    Keys CHGNet did not return are None. Unknown raw keys are passed through
    unchanged (as the former safe_get_prediction did).
    """
    out = {
        "energy_eV_per_atom": None,
        "total_energy_eV": None,
        "forces_eV_per_A": None,
        "stress_GPa": None,
        "magmoms_muB": None,
        "site_energies_eV": None,
        "crystal_features": None,
        "atom_features": None,
    }
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
            out["forces_eV_per_A"] = [list(a) for a in pred[k].tolist()]
            break
    for k in ("stress", "s", "virial"):
        if k in pred:
            out["stress_GPa"] = [list(a) for a in pred[k].tolist()]
            break
    for k in ("magmom", "m", "magmoms"):
        if k in pred:
            out["magmoms_muB"] = [float(x) for x in pred[k].tolist()]
            break
    for k in ("site_energies", "site_energy"):
        if k in pred:
            out["site_energies_eV"] = [float(x) for x in pred[k].tolist()]
            break
    for k in ("crystal_fea", "crystal_features"):
        if k in pred:
            out["crystal_features"] = [float(x) for x in pred[k].tolist()]
            break
    for k in ("atom_fea", "atom_features"):
        if k in pred:
            out["atom_features"] = [[float(x) for x in atom] for atom in pred[k].tolist()]
            break

    # Other fields
    for k, v in pred.items():
        if k in _HANDLED_RAW_KEYS:
            continue
        try:
            out[k] = v.tolist() if hasattr(v, "tolist") else v
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def _predict_kwargs(properties) -> dict:
    """Translate normalized property names to CHGNet predict_structure kwargs."""
    props = set(properties)
    if props <= {PROP_ENERGY}:
        # task="e" skips force/stress/magmom autograd (~5x faster).
        return {"task": "e"}
    return {
        "return_site_energies": PROP_SITE_ENERGIES in props,
        "return_atom_feas": PROP_ATOM_FEATURES in props,
        "return_crystal_feas": PROP_CRYSTAL_FEATURES in props,
    }


class CHGNetBackend(MLIPBackend):
    """CHGNet (MPtrj-trained universal potential), CPU, lazily loaded."""

    id = "chgnet"
    display_name = "CHGNet"
    capabilities = frozenset({CAP_ENERGY, CAP_FORCES, CAP_STRESS, CAP_MAGMOM,
                              CAP_SITE_ENERGIES, CAP_FEATURES, CAP_BATCH})
    unavailable_message = "CHGNet not available"
    batch_size = MLIP_BATCH_SIZE

    def _init_state(self):
        super()._init_state()
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        return CHGNET_AVAILABLE

    def version(self) -> str:
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

    def supported_elements(self) -> List[str]:
        return sorted(get_chgnet_supported_elements(), key=lambda s: Element(s).Z)

    def parameters(self) -> Optional[int]:
        return getattr(self._model, "n_params", None) if self._model is not None else None

    # -- loading ------------------------------------------------------
    async def get_model(self):
        """Get CHGNet model instance (lazy loading)"""
        if self._model is None:
            async with self._lock:
                if self._model is None:  # Double-check locking
                    await self._load_model()
        return self._model

    async def load(self):
        return await self.get_model()

    async def _load_model(self):
        """Internal method to load CHGNet model"""
        if not CHGNET_AVAILABLE:
            raise RuntimeError("CHGNet not available")

        try:
            from chgnet.model.model import CHGNet

            self._model = CHGNet.load(use_device=self._device, verbose=False)
            logger.info(f"CHGNet model loaded and cached (device={self._device})")

        except Exception as load_error:
            if WINDOWS_PLATFORM and "Buffer dtype mismatch" in str(load_error):
                raise RuntimeError("Windows CHGNet compatibility issue. Solution: Install Microsoft Visual C++ Redistributable")
            else:
                raise RuntimeError(f"Failed to load CHGNet model: {load_error}")

    async def _build_calculator(self):
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(model=self._model, use_device=self._device)

    async def get_calculator(self):
        """Get the shared CHGNetCalculator instance (lazy loading).

        A single calculator is reused across every optimizer choice --
        switching optimizers (main.py run_relaxation()) is just a different
        ASE Optimizer wrapped around the same calculator.
        """
        # Resolved (and, on a cold start, loaded) BEFORE this method takes
        # its own lock below: get_model() acquires the same self._lock
        # internally, and asyncio.Lock is not reentrant, so calling it from
        # inside an already-held `async with self._lock` block here would
        # deadlock the very first request in a fresh process.
        await self.get_model()
        if self._calculator is None:
            async with self._lock:
                if self._calculator is None:  # Double-check locking
                    self._calculator = await self._build_calculator()
                    logger.info("CHGNet calculator created and cached")
        return self._calculator

    # -- prediction ---------------------------------------------------
    async def predict(self, structures, properties) -> List[Optional[dict]]:
        """
        Batch prediction of multiple structures for improved performance.

        CHGNet's predict_structure natively accepts a list of structures plus a
        batch_size argument, so each chunk is submitted as a single executor
        call. This matters a lot for the atom insertion energy screening,
        where many candidate structures differ by only one site.

        Returns normalized predictions (None for entries that failed).
        """
        if not structures:
            return []

        model = await self.get_model()
        kwargs = _predict_kwargs(properties)
        results = []

        # Chunk to bound peak memory; each chunk is one native batched call.
        for i in range(0, len(structures), self.batch_size):
            batch = structures[i:i + self.batch_size]
            try:
                batch_pred = await run_in_inference_executor(
                    model.predict_structure, batch, batch_size=len(batch), **kwargs)
                # A single-element list can come back as a bare dict rather than
                # a list of dicts; normalize so downstream code always sees a list.
                if isinstance(batch_pred, dict):
                    batch_pred = [batch_pred]
                if len(batch_pred) != len(batch):
                    raise ValueError(
                        f"Batch prediction returned {len(batch_pred)} results "
                        f"for {len(batch)} structures")
                results.extend(normalize_chgnet_prediction(p, len(s))
                               for p, s in zip(batch_pred, batch))
            except Exception as e:
                logger.warning(f"Native batch prediction failed for chunk starting at {i}, "
                               f"falling back to per-structure calls: {e}")
                for structure in batch:
                    try:
                        pred = await run_in_inference_executor(
                            model.predict_structure, structure, **kwargs)
                        results.append(normalize_chgnet_prediction(pred, len(structure)))
                    except Exception as inner_e:
                        logger.warning(f"Prediction failed for structure {len(results)}: {inner_e}")
                        results.append(None)

        return results

    async def predict_one(self, structure, properties) -> dict:
        """
        Optimized single structure prediction with memory management.

        Raises on failure. If the detailed keyword arguments are rejected
        (TypeError), retries once with predict_structure's defaults.
        """
        model = await self.get_model()

        # Memory optimization for large structures
        if len(structure) > 100:
            # Force garbage collection before large structure prediction
            import gc
            gc.collect()
            logger.debug(f"Garbage collection performed for large structure ({len(structure)} atoms)")

        try:
            try:
                pred = await run_in_inference_executor(
                    model.predict_structure, structure, **_predict_kwargs(properties))
            except TypeError:
                # Fallback if detailed prediction fails
                pred = await run_in_inference_executor(model.predict_structure, structure)
        except Exception as e:
            logger.error(f"Optimized prediction failed: {e}")
            raise
        return normalize_chgnet_prediction(pred, len(structure))
