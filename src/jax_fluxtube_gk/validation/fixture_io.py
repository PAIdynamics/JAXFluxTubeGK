"""External fixture records, readers, resampling, and compact writers."""

# ruff: noqa: F822 - exports are provided lazily by module ``__getattr__``.

from __future__ import annotations

from typing import Any

from ._lazy import benchmark_symbol

__all__ = [
    "GkwIghInputTrace",
    "GkwIghMatrixTrace",
    "GkwSelectedModeRhsTrace",
    "GkwStateTrace",
    "GkwVelocitySpaceSlice",
    "GkwVelocitySpaceSliceSeries",
    "GxCycloneInputReference",
    "EikGeometryReference",
    "GxGrowthRateReference",
    "PerKyModeStructureFixture",
    "SelectedModeStateTrace",
    "load_cyclone_trace_csv",
    "load_gkw_igh_input_trace",
    "load_gkw_igh_matrix_trace",
    "load_gkw_parallel_phi_trace",
    "load_gkw_selected_mode_rhs_trace",
    "load_gkw_selected_mode_state_trace",
    "load_gkw_state_trace",
    "load_gkw_time_dat_trace",
    "load_gkw_velocity_space_slice",
    "load_gkw_velocity_space_slice_series",
    "load_gx_cyclone_input_reference",
    "load_eik_geometry_reference",
    "load_gx_growth_rate_reference",
    "load_gx_mode_structure_fixture",
    "load_per_ky_mode_structure_fixture_csv",
    "load_stella_mode_structure_fixture",
    "resample_eik_geometry_reference",
    "resample_per_ky_mode_structure_fixture",
    "write_cyclone_source_term_trace_csv",
    "write_cyclone_trace_csv",
    "write_per_ky_mode_structure_fixture_csv",
]

_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    return benchmark_symbol(name, _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _EXPORTS)
