"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import pkgutil
import importlib
import logging
from starVLA.model.tools import FRAMEWORK_REGISTRY

logger = logging.getLogger(__name__)


def _import_framework_modules() -> None:
    """Populate the registry only when the training factory is requested."""

    for _, module_name, _ in pkgutil.iter_modules(__path__):
        try:
            importlib.import_module(f"{__name__}.{module_name}")
        except Exception as exc:
            logger.warning(
                "Failed to auto-import framework submodule '%s': %s",
                module_name,
                exc,
            )


def build_framework(cfg):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"):
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml


    _import_framework_modules()
    # auto detect from registry
    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented. Plz, python yourframework_py to specify framework module.")

    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
