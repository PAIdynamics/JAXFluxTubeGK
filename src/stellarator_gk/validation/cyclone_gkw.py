"""Cyclone, GKW, Gyaradax, and mode-structure validation workflows."""

# ruff: noqa: F822 - exports are provided lazily by module ``__getattr__``.

from __future__ import annotations

from typing import Any

from ._lazy import benchmark_symbol

__all__ = [
    "BenchmarkGateResult",
    "BenchmarkTarget",
    "CycloneKyScanGateReport",
    "CycloneTermParityReport",
    "CycloneTrace",
    "CycloneTraceComparisonReport",
    "PerKyModeStructureComparisonReport",
    "PerKyModeStructureFixture",
    "compare_cyclone_base_case_traces",
    "compare_gkw_state_trace_to_source_term_trace",
    "compare_per_ky_mode_structure_fixtures",
    "evaluate_benchmark_gate",
    "evaluate_cyclone_ky_scan_gate",
    "run_cyclone_base_case_ky_scan_gate",
    "run_cyclone_base_case_mode_structure_fixture",
    "run_cyclone_base_case_term_parity_audit",
    "run_cyclone_base_case_trace",
    "run_production_cyclone_base_case_gate",
    "run_reduced_cyclone_base_case_gate",
]

_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    return benchmark_symbol(name, _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS)
