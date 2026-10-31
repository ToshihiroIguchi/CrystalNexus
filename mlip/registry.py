"""Model registry: maps model ids to backend classes."""
import logging
import os
from typing import Dict, List, Optional, Type

from .base import MLIPBackend, ModelUnavailableError, UnknownModelError
from .chgnet_backend import CHGNetBackend
from .mace_backend import MACEBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL_FALLBACK = "chgnet"

MODEL_REGISTRY: Dict[str, Type[MLIPBackend]] = {
    "chgnet": CHGNetBackend,
    "mace-mp-0-medium": MACEBackend,
}


_warned_invalid_defaults = set()


def default_model_id() -> str:
    """Default model id (env CRYSTALNEXUS_DEFAULT_MODEL, else 'chgnet')."""
    mid = os.getenv("CRYSTALNEXUS_DEFAULT_MODEL", "").strip()
    if not mid:
        return DEFAULT_MODEL_FALLBACK
    if mid not in MODEL_REGISTRY:
        if mid not in _warned_invalid_defaults:
            _warned_invalid_defaults.add(mid)
            logger.warning(
                f"CRYSTALNEXUS_DEFAULT_MODEL='{mid}' is not a registered model "
                f"({sorted(MODEL_REGISTRY)}); falling back to '{DEFAULT_MODEL_FALLBACK}'")
        return DEFAULT_MODEL_FALLBACK
    return mid


def is_available(model_id: str) -> bool:
    """True if ``model_id`` is registered and its dependencies import."""
    cls = MODEL_REGISTRY.get(model_id)
    return cls is not None and cls.is_available()


def get_backend(model_id: Optional[str] = None) -> MLIPBackend:
    """Return the singleton backend for ``model_id`` (default model if None).

    Raises UnknownModelError for an unregistered id and ModelUnavailableError
    when the model's dependencies cannot be imported.
    """
    mid = model_id or default_model_id()
    cls = MODEL_REGISTRY.get(mid)
    if cls is None:
        raise UnknownModelError(
            f"Unknown model '{mid}'. Available models: {sorted(MODEL_REGISTRY)}")
    if not cls.is_available():
        raise ModelUnavailableError(cls.unavailable_message)
    return cls()


def list_models() -> List[dict]:
    """Describe every registered model (no model loading)."""
    default = default_model_id()
    models = []
    for mid, cls in MODEL_REGISTRY.items():
        available = cls.is_available()
        models.append({
            "id": mid,
            "display_name": cls.display_name,
            "available": available,
            "capabilities": sorted(cls.capabilities),
            "version": cls().version() if available else None,
            "is_default": mid == default,
        })
    return models
