"""Lazy access to the legacy benchmark implementation during its module split."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def benchmark_symbol(name: str, allowed: frozenset[str] | None = None) -> Any:
    """Load one validation symbol without importing benchmarks with the solver API."""
    if allowed is not None and name not in allowed:
        raise AttributeError(name)
    module = import_module("stellarator_gk.benchmarks")
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(name) from exc
