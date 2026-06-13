"""Method registry with dynamic discovery.

Committed methods live here (``proposed``, ``baseline_open3d``). Experimental
variants go in ``experimental/`` (gitignored): drop a module there and it is
auto-registered and shows up in the benchmark. Once the best one is chosen it is
promoted to ``proposed.py`` and the experimental folder can be discarded.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

from .base import BaseRegistration

REGISTRY: Dict[str, Type[BaseRegistration]] = {}


def register_method(name: str):
    def deco(cls: Type[BaseRegistration]) -> Type[BaseRegistration]:
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


def _discover() -> None:
    """Import every method module so its @register_method runs (idempotent)."""
    pkg_dir = Path(__file__).parent
    # Committed methods directly under methods/
    for mod in pkgutil.iter_modules([str(pkg_dir)]):
        if mod.name in ("base", "__init__"):
            continue
        importlib.import_module(f"{__name__}.{mod.name}")
    # Experimental methods under methods/experimental/ (may be absent)
    exp_dir = pkg_dir / "experimental"
    if exp_dir.is_dir():
        for mod in pkgutil.iter_modules([str(exp_dir)]):
            if mod.name == "__init__":
                continue
            importlib.import_module(f"{__name__}.experimental.{mod.name}")


def available_methods() -> List[str]:
    _discover()
    return sorted(REGISTRY.keys())


def get_method(name: str) -> BaseRegistration:
    _discover()
    if name not in REGISTRY:
        raise KeyError(f"unknown method '{name}'. available: {available_methods()}")
    return REGISTRY[name]()
