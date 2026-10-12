"""Backend abstraction for machine-learning interatomic potentials (MLIPs).

The boundary between the web layer and a model is an ASE ``Calculator``
(``get_calculator``) plus a normalized ``predict`` that always returns the
same set of keys. Features a model cannot provide are ``None``.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, Iterable, List, Optional

from .executor import run_in_inference_executor

logger = logging.getLogger(__name__)

# Capabilities a backend may declare.
CAP_ENERGY = "energy"
CAP_FORCES = "forces"
CAP_STRESS = "stress"
CAP_MAGMOM = "magmom"
CAP_SITE_ENERGIES = "site_energies"
CAP_FEATURES = "features"
CAP_BATCH = "batch"

# Property names accepted by ``predict(structures, properties)``.
PROP_ENERGY = "energy"
PROP_FORCES = "forces"
PROP_STRESS = "stress"
PROP_MAGMOM = "magmom"
PROP_SITE_ENERGIES = "site_energies"
PROP_ATOM_FEATURES = "atom_features"
PROP_CRYSTAL_FEATURES = "crystal_features"

# Normalized result keys (always present in a prediction dict).
RESULT_KEYS = (
    "energy_eV_per_atom",
    "total_energy_eV",
    "forces_eV_per_A",
    "stress_GPa",
    "magmoms_muB",
    "site_energies_eV",
    "crystal_features",
    "atom_features",
)


class MLIPError(Exception):
    """Base class for registry/backend selection errors."""


class UnknownModelError(MLIPError):
    """The requested model id is not registered (endpoints map this to 400)."""


class ModelUnavailableError(MLIPError, RuntimeError):
    """The model cannot be used: dependencies missing or weights unobtainable.

    Endpoints map this to 503. It is also a RuntimeError so that the existing
    ``except RuntimeError`` load guards produce the same 503.
    """


def ase_stress_to_gpa(stress):
    """Convert an ASE stress (eV/A^3; Voigt-6 or 3x3) to GPa.

    No sign flip is applied: CHGNet's ``predict_structure`` stress equals
    ``calc.get_stress(voigt=False) / ase.units.GPa`` (verified in Phase 0).
    Returns nested lists / a list of floats with the input's shape.
    """
    import numpy as np
    from ase import units

    return (np.asarray(stress, dtype=float) / units.GPa).tolist()


def make_prediction(num_atoms: Optional[int] = None, *, energy_per_atom=None,
                    forces=None, stress=None, magmoms=None, site_energies=None,
                    crystal_features=None, atom_features=None) -> Dict[str, object]:
    """Build a normalized prediction dict; anything not provided is ``None``."""
    total = None
    if energy_per_atom is not None and num_atoms is not None:
        total = energy_per_atom * num_atoms
    return {
        "energy_eV_per_atom": energy_per_atom,
        "total_energy_eV": total,
        "forces_eV_per_A": forces,
        "stress_GPa": stress,
        "magmoms_muB": magmoms,
        "site_energies_eV": site_energies,
        "crystal_features": crystal_features,
        "atom_features": atom_features,
    }


class MLIPBackend(ABC):
    """One loadable model. Instances are singletons per backend class."""

    id: str = ""
    display_name: str = ""
    capabilities: FrozenSet[str] = frozenset()
    unavailable_message: str = "Model not available"

    _instances: Dict[type, "MLIPBackend"] = {}

    def __new__(cls, *args, **kwargs):
        inst = MLIPBackend._instances.get(cls)
        if inst is None:
            inst = super().__new__(cls)
            inst._init_state()
            MLIPBackend._instances[cls] = inst
        return inst

    def _init_state(self):
        self._calculator = None
        self._lock = asyncio.Lock()
        self._device = "cpu"

    # -- metadata -----------------------------------------------------
    @classmethod
    def is_available(cls) -> bool:
        """True if the model's dependencies can be imported."""
        return True

    @property
    def device(self) -> str:
        return self._device

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def supported_elements(self) -> List[str]: ...

    def parameters(self) -> Optional[int]:
        """Number of model parameters if known (after load), else None."""
        return None

    # -- loading ------------------------------------------------------
    async def load(self):
        """Ensure the model is loaded (raises RuntimeError on failure)."""
        return await self.get_calculator()

    @abstractmethod
    async def _build_calculator(self):
        """Create the ASE calculator (called once, under the lock)."""

    async def get_calculator(self):
        """Shared ASE calculator (lazy, double-checked asyncio.Lock)."""
        if self._calculator is None:
            async with self._lock:
                if self._calculator is None:
                    self._calculator = await self._build_calculator()
        return self._calculator

    # -- prediction ---------------------------------------------------
    def _sync_predict_one(self, calculator, structure, properties) -> dict:
        """Calculator-based single-structure prediction (blocking)."""
        import numpy as np
        from pymatgen.io.ase import AseAtomsAdaptor

        props = set(properties)
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.calc = calculator
        n = len(atoms)
        kwargs = {"num_atoms": n,
                  "energy_per_atom": float(atoms.get_potential_energy()) / n}
        if PROP_FORCES in props and CAP_FORCES in self.capabilities:
            kwargs["forces"] = [list(map(float, f)) for f in atoms.get_forces()]
        if PROP_STRESS in props and CAP_STRESS in self.capabilities:
            kwargs["stress"] = ase_stress_to_gpa(atoms.get_stress(voigt=False))
        if PROP_MAGMOM in props and CAP_MAGMOM in self.capabilities:
            kwargs["magmoms"] = [float(m) for m in np.asarray(atoms.get_magnetic_moments())]
        return make_prediction(**kwargs)

    async def predict(self, structures, properties: Iterable[str]) -> List[Optional[dict]]:
        """Normalized predictions, one per structure (None where it failed).

        Default: sequential ASE-calculator evaluation on the shared executor.
        Backends with CAP_BATCH override this.
        """
        calculator = await self.get_calculator()
        results: List[Optional[dict]] = []
        for structure in structures:
            try:
                results.append(await run_in_inference_executor(
                    self._sync_predict_one, calculator, structure, properties))
            except Exception as e:
                logger.warning(f"Prediction failed for structure {len(results)}: {e}")
                results.append(None)
        return results

    async def predict_one(self, structure, properties: Iterable[str]) -> dict:
        """Single-structure prediction; raises on failure."""
        calculator = await self.get_calculator()
        return await run_in_inference_executor(
            self._sync_predict_one, calculator, structure, properties)
