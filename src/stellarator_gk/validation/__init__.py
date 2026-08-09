"""Opt-in external-reference and benchmark validation API.

Import focused modules for new code. Attribute access at this namespace remains
lazy and provides a compatibility bridge while ``benchmarks.py`` is split.
"""

from __future__ import annotations

from typing import Any

from . import cyclone_gkw, fixture_io, geometry_parity, w7x
from ._lazy import benchmark_symbol

__all__ = ["cyclone_gkw", "fixture_io", "geometry_parity", "w7x"]


def __getattr__(name: str) -> Any:
    return benchmark_symbol(name)
