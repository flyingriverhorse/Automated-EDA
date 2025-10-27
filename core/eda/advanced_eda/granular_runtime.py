from __future__ import annotations

"""Compatibility shim that proxies to the modular granular runtime package."""

from importlib import util
from pathlib import Path
import sys
from types import ModuleType

_PACKAGE_PATH = Path(__file__).with_suffix("") / "granular_runtime" / "__init__.py"


def _load_runtime_package() -> ModuleType:
    if not _PACKAGE_PATH.exists():
        raise ImportError(
            "Granular runtime package is missing; ensure 'core/eda/advanced_eda/granular_runtime/' exists."
        )

    spec = util.spec_from_file_location("core.eda.advanced_eda._granular_runtime", _PACKAGE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("Unable to import granular runtime package")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_runtime_module = _load_runtime_package()

# Re-export public API from the package so existing imports continue to work.
for name in getattr(_runtime_module, "__all__", []):
    globals()[name] = getattr(_runtime_module, name)

__all__ = list(getattr(_runtime_module, "__all__", []))

# Ensure downstream imports resolve to the package implementation.
sys.modules[__name__] = _runtime_module
