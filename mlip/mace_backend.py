"""MACE-MP-0 (medium) backend.

Uses ``mace.calculators.mace_mp`` as the ASE calculator. Nothing from mace or
torch is imported at module import time, so the application starts (and
reports this model as unavailable) when ``mace-torch`` is not installed.

Facts verified in the Phase 0 spike:
  * ``model="medium"`` must be passed explicitly (the library default is a
    different checkpoint, medium-mpa-0).
  * ``mace_mp`` does not change torch's global default dtype.
  * Stress: ``atoms.get_stress(voigt=False) / ase.units.GPa`` with no sign
    flip, identical to CHGNet.
  * MACE has no magnetic moments, and ``node_energy`` is not an exact
    decomposition of the total energy, so site energies are not offered.
"""
import asyncio
import importlib.util
import logging
import os
from typing import List, Optional

from pymatgen.core import Element

from .base import (CAP_ENERGY, CAP_FORCES, CAP_STRESS, MPTRJ_ATOMIC_NUMBERS,
                   MLIPBackend, ModelUnavailableError)

logger = logging.getLogger(__name__)

# float64 is the recommended precision for geometry optimization; float32 is
# faster and uses less memory.
MACE_DTYPE = os.getenv("MACE_DTYPE", "float64").strip().lower()
if MACE_DTYPE not in ("float32", "float64"):
    logger.warning(f"Invalid MACE_DTYPE '{MACE_DTYPE}', using float64")
    MACE_DTYPE = "float64"

# Elements covered by MACE-MP-0 (shared MPtrj set). Used until the model is
# loaded; a slow test checks it equals calc.models[0].atomic_numbers.
_STATIC_ATOMIC_NUMBERS = MPTRJ_ATOMIC_NUMBERS

_DOWNLOAD_HINT = (
    "MACE-MP-0 weights are not available locally and could not be downloaded. "
    "They are fetched on first use into ~/.cache/mace, so an internet connection "
    "is needed once."
)


class MACEBackend(MLIPBackend):
    """MACE-MP-0 medium (MPtrj-trained foundation model), CPU, lazily loaded."""

    id = "mace-mp-0-medium"
    display_name = "MACE-MP-0 (medium)"
    capabilities = frozenset({CAP_ENERGY, CAP_FORCES, CAP_STRESS})
    unavailable_message = "MACE not available (install with: pip install mace-torch)"

    @classmethod
    def is_available(cls) -> bool:
        try:
            return importlib.util.find_spec("mace") is not None
        except (ImportError, ValueError):
            return False

    def version(self) -> str:
        try:
            from importlib.metadata import version
            return version("mace-torch")
        except Exception:
            return "unknown"

    def supported_elements(self) -> List[str]:
        if self._calculator is not None:
            zs = [int(z) for z in self._calculator.models[0].atomic_numbers]
        else:
            zs = _STATIC_ATOMIC_NUMBERS
        return [Element.from_Z(z).symbol for z in sorted(zs)]

    def parameters(self) -> Optional[int]:
        if self._calculator is None:
            return None
        try:
            return int(sum(p.numel() for p in self._calculator.models[0].parameters()))
        except Exception:
            return None

    def _load_sync(self):
        try:
            from mace.calculators import mace_mp
        except (ImportError, AttributeError, ValueError, OSError) as e:
            # Broken installs (torch/e3nn mismatch, missing DLL) fail at import
            # with more than ImportError; report them all as "unavailable".
            raise ModelUnavailableError(f"{self.unavailable_message}: {e}")
        try:
            return mace_mp(model="medium", device=self._device,
                           default_dtype=MACE_DTYPE)
        except RuntimeError as e:
            if "download failed" in str(e).lower():
                raise ModelUnavailableError(_DOWNLOAD_HINT)
            raise
        except OSError as e:
            raise ModelUnavailableError(f"{_DOWNLOAD_HINT} ({e})")
        except (ImportError, AttributeError, ValueError) as e:
            raise ModelUnavailableError(f"{self.unavailable_message}: {e}")

    async def _build_calculator(self):
        # Weights may be downloaded on first use; keep that off the event loop
        # and off the shared inference thread.
        calc = await asyncio.to_thread(self._load_sync)
        logger.info(f"MACE-MP-0 medium loaded (device={self._device}, dtype={MACE_DTYPE})")
        return calc
