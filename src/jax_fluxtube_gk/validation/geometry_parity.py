"""Geometry conversion, parity, and independent-provider validation workflows."""

# ruff: noqa: F822 - exports are provided lazily by module ``__getattr__``.

from __future__ import annotations

from typing import Any

from ._lazy import benchmark_symbol

__all__ = [
    "ExternalEikProducerReport",
    "GxEikGeometryParityReport",
    "GxEikGeometryReference",
    "ModeBoundaryContractReport",
    "StellaratorGeometryPreflightReport",
    "build_desc_gx_eik_reference_from_path",
    "build_flux_tube_geometry_from_gx_eik_reference",
    "compare_geometry_to_gx_eik_reference",
    "geometry_to_gx_eik_reference",
    "gx_eik_kperp2",
    "run_desc_gx_eik_external_geometry_gate",
    "run_geometry_to_gx_eik_export_gate",
    "run_gx_eik_geometry_gate",
    "run_gx_gist_external_eik_suite_gate",
    "run_independent_external_eik_producer_gate",
    "run_independent_external_eik_producer_report",
    "run_mode_boundary_contract",
    "run_solver_geometry_to_gx_eik_gate",
    "run_stellarator_geometry_preflight",
]

_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    return benchmark_symbol(name, _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS)
