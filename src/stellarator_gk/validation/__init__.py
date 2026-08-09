"""Opt-in external-reference and benchmark validation API.

Import focused modules for new code. Attribute access at this namespace remains
lazy and provides a compatibility bridge while ``benchmarks.py`` is split.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from ._lazy import benchmark_symbol

__all__ = ["cyclone_gkw", "fixture_io", "geometry_parity", "w7x"]
_FOCUSED_MODULES = frozenset(__all__)


def __getattr__(name: str) -> Any | ModuleType:
    if name in _FOCUSED_MODULES:
        return import_module(f"{__name__}.{name}")
    return benchmark_symbol(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _FOCUSED_MODULES)
