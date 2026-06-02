"""Benchmark reference targets and lightweight external-fixture readers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .types import _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BenchmarkTarget(_PyTreeDataclass):
    """Scalar validation or optimization target from an external benchmark."""

    name: str
    quantity: str
    reference_value: object
    tolerance: float
    source: str
    metadata: tuple[tuple[str, object], ...] = ()

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("reference_value",)
    _static_fields: ClassVar[tuple[str, ...]] = (
        "name",
        "quantity",
        "tolerance",
        "source",
        "metadata",
    )

    def __post_init__(self):
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        object.__setattr__(
            self,
            "reference_value",
            jnp.asarray(self.reference_value, dtype=jnp.float64),
        )
        object.__setattr__(self, "metadata", tuple(self.metadata))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GxGrowthRateReference(_PyTreeDataclass):
    """Time-averaged GX linear growth-rate and frequency curve."""

    ky: object
    growth_rate: object
    frequency: object
    source: str
    ikx: int = 0
    average_fraction: float = 0.5

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("ky", "growth_rate", "frequency")
    _static_fields: ClassVar[tuple[str, ...]] = ("source", "ikx", "average_fraction")

    def __post_init__(self):
        ky = jnp.asarray(self.ky, dtype=jnp.float64)
        growth = jnp.asarray(self.growth_rate, dtype=jnp.float64)
        frequency = jnp.asarray(self.frequency, dtype=jnp.float64)
        if ky.ndim != 1:
            raise ValueError("ky must be one-dimensional")
        if growth.shape != ky.shape or frequency.shape != ky.shape:
            raise ValueError("growth_rate and frequency must match ky shape")
        if not 0.0 <= self.average_fraction < 1.0:
            raise ValueError("average_fraction must lie in [0, 1)")
        object.__setattr__(self, "ky", ky)
        object.__setattr__(self, "growth_rate", growth)
        object.__setattr__(self, "frequency", frequency)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GxEikGeometryReference(_PyTreeDataclass):
    """GS2/GX eik-style geometry table sampled along a field line."""

    theta: object
    bmag: object
    gradpar: object
    gds2: object
    gds21: object
    gds22: object
    gbdrift: object
    gbdrift0: object
    cvdrift: object
    cvdrift0: object
    source: str
    header: tuple[float, ...] = ()

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "theta",
        "bmag",
        "gradpar",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "cvdrift",
        "cvdrift0",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("source", "header")

    def __post_init__(self):
        theta = jnp.asarray(self.theta, dtype=jnp.float64)
        if theta.ndim != 1:
            raise ValueError("theta must be one-dimensional")
        object.__setattr__(self, "theta", theta)
        for name in self._dynamic_fields[1:]:
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != theta.shape:
                raise ValueError(f"{name} must match theta shape")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "header", tuple(float(value) for value in self.header))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BenchmarkGateResult(_PyTreeDataclass):
    """Observed value, error, and pass/fail state for one validation gate."""

    target: BenchmarkTarget
    observed_value: object
    residual: object
    cost: object
    passed: object
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "target",
        "observed_value",
        "residual",
        "cost",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("notes",)

    def __post_init__(self):
        object.__setattr__(
            self,
            "observed_value",
            jnp.asarray(self.observed_value, dtype=jnp.float64),
        )
        object.__setattr__(self, "residual", jnp.asarray(self.residual, dtype=jnp.float64))
        object.__setattr__(self, "cost", jnp.asarray(self.cost, dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneTermParityReport(_PyTreeDataclass):
    """Term-level CBC parity report against GKW/Gyaradax formulas."""

    term_errors: object
    max_abs_error: object
    passed: object
    term_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("term_errors", "max_abs_error", "passed")
    _static_fields: ClassVar[tuple[str, ...]] = ("term_names", "notes")

    def __post_init__(self):
        errors = jnp.asarray(self.term_errors, dtype=jnp.float64)
        if errors.ndim != 1:
            raise ValueError("term_errors must be one-dimensional")
        if len(self.term_names) != errors.shape[0]:
            raise ValueError("term_names length must match term_errors")
        object.__setattr__(self, "term_errors", errors)
        object.__setattr__(
            self, "max_abs_error", jnp.asarray(self.max_abs_error, dtype=jnp.float64)
        )
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        object.__setattr__(self, "term_names", tuple(self.term_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneTrace(_PyTreeDataclass):
    """Windowed selected-``ky`` Cyclone trace for code-to-code parity."""

    times: object
    raw_amplitude: object
    physical_amplitude: object
    window_growth: object
    fitted_growth: object
    phi_norm: object
    state_norm: object
    rhs_norm: object
    log_normalization: object
    source: str
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "times",
        "raw_amplitude",
        "physical_amplitude",
        "window_growth",
        "fitted_growth",
        "phi_norm",
        "state_norm",
        "rhs_norm",
        "log_normalization",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("source", "notes")

    def __post_init__(self):
        times = jnp.asarray(self.times, dtype=jnp.float64)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        object.__setattr__(self, "times", times)
        for name in self._dynamic_fields[1:]:
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != times.shape:
                raise ValueError(f"{name} must match times shape")
            object.__setattr__(self, name, values)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneTraceComparisonReport(_PyTreeDataclass):
    """Field-by-field comparison between two Cyclone traces."""

    field_errors: object
    max_abs_error: object
    passed: object
    field_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("field_errors", "max_abs_error", "passed")
    _static_fields: ClassVar[tuple[str, ...]] = ("field_names", "notes")

    def __post_init__(self):
        errors = jnp.asarray(self.field_errors, dtype=jnp.float64)
        if errors.ndim != 1:
            raise ValueError("field_errors must be one-dimensional")
        if len(self.field_names) != errors.shape[0]:
            raise ValueError("field_names length must match field_errors")
        object.__setattr__(self, "field_errors", errors)
        object.__setattr__(
            self, "max_abs_error", jnp.asarray(self.max_abs_error, dtype=jnp.float64)
        )
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        object.__setattr__(self, "field_names", tuple(self.field_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelPhiTrace(_PyTreeDataclass):
    """Parallel profile history of the selected-mode electrostatic field energy."""

    times: object
    z: object
    phi_power: object
    source: str
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("times", "z", "phi_power")
    _static_fields: ClassVar[tuple[str, ...]] = ("source", "notes")

    def __post_init__(self):
        times = jnp.asarray(self.times, dtype=jnp.float64)
        z = jnp.asarray(self.z, dtype=jnp.float64)
        phi_power = jnp.asarray(self.phi_power, dtype=jnp.float64)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if z.ndim != 1:
            raise ValueError("z must be one-dimensional")
        if phi_power.ndim != 2:
            raise ValueError("phi_power must have shape (n_time,n_z)")
        if phi_power.shape != (times.shape[0], z.shape[0]):
            raise ValueError("phi_power shape must match times and z")
        if times.shape[0] == 0 or z.shape[0] == 0:
            raise ValueError("parallel phi trace must not be empty")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "phi_power", phi_power)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelPhiTraceComparisonReport(_PyTreeDataclass):
    """Comparison between two parallel ``|phi|^2`` profile histories."""

    profile_errors: object
    max_abs_error: object
    time_error: object
    z_error: object
    passed: object
    normalized_profiles: bool = True
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "profile_errors",
        "max_abs_error",
        "time_error",
        "z_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("normalized_profiles", "notes")

    def __post_init__(self):
        errors = jnp.asarray(self.profile_errors, dtype=jnp.float64)
        if errors.ndim != 1:
            raise ValueError("profile_errors must be one-dimensional")
        object.__setattr__(self, "profile_errors", errors)
        object.__setattr__(
            self,
            "max_abs_error",
            jnp.asarray(self.max_abs_error, dtype=jnp.float64),
        )
        object.__setattr__(self, "time_error", jnp.asarray(self.time_error, dtype=jnp.float64))
        object.__setattr__(self, "z_error", jnp.asarray(self.z_error, dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelPhiProfileAudit(_PyTreeDataclass):
    """Alignment and normalization audit for parallel ``|phi|^2`` profiles."""

    direct_profile_errors: object
    reversed_profile_errors: object
    best_shift_profile_errors: object
    circular_shift_errors: object
    best_shift: object
    best_aligned_max_error: object
    total_power_ratio: object
    center_of_power_error: object
    edge_fraction_error: object
    peak_z_error: object
    second_moment_error: object
    worst_time_index: object
    worst_z_index: object
    worst_time: object
    worst_z: object
    worst_signed_error: object
    worst_observed_value: object
    worst_reference_value: object
    passed: object
    normalized_profiles: bool = True
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "direct_profile_errors",
        "reversed_profile_errors",
        "best_shift_profile_errors",
        "circular_shift_errors",
        "best_shift",
        "best_aligned_max_error",
        "total_power_ratio",
        "center_of_power_error",
        "edge_fraction_error",
        "peak_z_error",
        "second_moment_error",
        "worst_time_index",
        "worst_z_index",
        "worst_time",
        "worst_z",
        "worst_signed_error",
        "worst_observed_value",
        "worst_reference_value",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("normalized_profiles", "notes")

    def __post_init__(self):
        direct = jnp.asarray(self.direct_profile_errors, dtype=jnp.float64)
        if direct.ndim != 1:
            raise ValueError("direct_profile_errors must be one-dimensional")
        object.__setattr__(self, "direct_profile_errors", direct)
        for name in (
            "reversed_profile_errors",
            "best_shift_profile_errors",
            "total_power_ratio",
            "center_of_power_error",
            "edge_fraction_error",
            "peak_z_error",
            "second_moment_error",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != direct.shape:
                raise ValueError(f"{name} must match direct_profile_errors shape")
            object.__setattr__(self, name, values)
        shift_errors = jnp.asarray(self.circular_shift_errors, dtype=jnp.float64)
        if shift_errors.ndim != 1:
            raise ValueError("circular_shift_errors must be one-dimensional")
        object.__setattr__(self, "circular_shift_errors", shift_errors)
        object.__setattr__(self, "best_shift", jnp.asarray(self.best_shift, dtype=jnp.int32))
        object.__setattr__(
            self,
            "best_aligned_max_error",
            jnp.asarray(self.best_aligned_max_error, dtype=jnp.float64),
        )
        for name in (
            "worst_time",
            "worst_z",
            "worst_signed_error",
            "worst_observed_value",
            "worst_reference_value",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(
            self,
            "worst_time_index",
            jnp.asarray(self.worst_time_index, dtype=jnp.int32),
        )
        object.__setattr__(self, "worst_z_index", jnp.asarray(self.worst_z_index, dtype=jnp.int32))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneSelectedKyGapAudit(_PyTreeDataclass):
    """Combined growth/profile audit for the production selected-``ky`` gap."""

    times: object
    solver_window_growth: object
    reference_window_growth: object
    window_growth_delta: object
    profile_errors: object
    total_power_ratio: object
    solver_late_fit_growth: object
    reference_late_fit_growth: object
    late_fit_delta: object
    solver_late_mean_growth: object
    reference_late_mean_growth: object
    late_mean_delta: object
    final_window_growth_delta: object
    first_profile_error: object
    max_profile_error: object
    late_profile_error: object
    worst_time: object
    worst_z: object
    worst_signed_profile_error: object
    total_power_ratio_mean: object
    total_power_ratio_max_deviation: object
    passed: object
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "times",
        "solver_window_growth",
        "reference_window_growth",
        "window_growth_delta",
        "profile_errors",
        "total_power_ratio",
        "solver_late_fit_growth",
        "reference_late_fit_growth",
        "late_fit_delta",
        "solver_late_mean_growth",
        "reference_late_mean_growth",
        "late_mean_delta",
        "final_window_growth_delta",
        "first_profile_error",
        "max_profile_error",
        "late_profile_error",
        "worst_time",
        "worst_z",
        "worst_signed_profile_error",
        "total_power_ratio_mean",
        "total_power_ratio_max_deviation",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("notes",)

    def __post_init__(self):
        times = jnp.asarray(self.times, dtype=jnp.float64)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        object.__setattr__(self, "times", times)
        for name in (
            "solver_window_growth",
            "reference_window_growth",
            "window_growth_delta",
            "profile_errors",
            "total_power_ratio",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != times.shape:
                raise ValueError(f"{name} must match times")
            object.__setattr__(self, name, values)
        for name in self._dynamic_fields[6:-1]:
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GkwVelocitySpaceSlice(_PyTreeDataclass):
    """Peak-``phi`` velocity-space slice using GKW ``distr*.dat`` normalization."""

    vpar: object
    vperp: object
    real_part: object
    imag_part: object
    time: object = np.nan
    peak_z: object = np.nan
    peak_z_index: object = -1
    source: str = ""
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "vpar",
        "vperp",
        "real_part",
        "imag_part",
        "time",
        "peak_z",
        "peak_z_index",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("source", "notes")

    def __post_init__(self):
        vpar = jnp.asarray(self.vpar, dtype=jnp.float64)
        if vpar.ndim != 2 or 0 in vpar.shape:
            raise ValueError("vpar must have nonempty shape (n_mu,n_vpar)")
        object.__setattr__(self, "vpar", vpar)
        for name in ("vperp", "real_part", "imag_part"):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != vpar.shape:
                raise ValueError(f"{name} must match vpar shape")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "time", jnp.asarray(self.time, dtype=jnp.float64))
        object.__setattr__(self, "peak_z", jnp.asarray(self.peak_z, dtype=jnp.float64))
        object.__setattr__(self, "peak_z_index", jnp.asarray(self.peak_z_index, dtype=jnp.int32))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GkwVelocitySpaceSliceSeries(_PyTreeDataclass):
    """Time series of GKW peak-``phi`` ``distr*_<step>.dat`` velocity slices."""

    times: object
    snapshot_indices: object
    vpar: object
    vperp: object
    real_part: object
    imag_part: object
    source: str = ""
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "times",
        "snapshot_indices",
        "vpar",
        "vperp",
        "real_part",
        "imag_part",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("source", "notes")

    def __post_init__(self):
        times = jnp.asarray(self.times, dtype=jnp.float64)
        if times.ndim != 1 or times.shape[0] < 1:
            raise ValueError("times must be a nonempty one-dimensional array")
        indices = jnp.asarray(self.snapshot_indices, dtype=jnp.int32)
        if indices.shape != times.shape:
            raise ValueError("snapshot_indices must match times")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "snapshot_indices", indices)
        vpar = jnp.asarray(self.vpar, dtype=jnp.float64)
        if vpar.ndim != 3 or vpar.shape[0] != times.shape[0] or 0 in vpar.shape:
            raise ValueError("vpar must have nonempty shape (n_time,n_mu,n_vpar)")
        object.__setattr__(self, "vpar", vpar)
        for name in ("vperp", "real_part", "imag_part"):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != vpar.shape:
                raise ValueError(f"{name} must match vpar shape")
            object.__setattr__(self, name, values)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneVelocitySpaceSliceAudit(_PyTreeDataclass):
    """Selected-``ky`` comparison against GKW peak-``phi`` velocity slices."""

    vpar_error: object
    vperp_error: object
    real_max_abs_error: object
    imag_max_abs_error: object
    complex_max_abs_error: object
    complex_l2_error: object
    complex_relative_l2_error: object
    observed_l2_norm: object
    reference_l2_norm: object
    time_error: object
    peak_z_error: object
    passed: object
    shape: tuple[int, int]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "vpar_error",
        "vperp_error",
        "real_max_abs_error",
        "imag_max_abs_error",
        "complex_max_abs_error",
        "complex_l2_error",
        "complex_relative_l2_error",
        "observed_l2_norm",
        "reference_l2_norm",
        "time_error",
        "peak_z_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("shape", "notes")

    def __post_init__(self):
        for name in self._dynamic_fields[:-1]:
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        if len(self.shape) != 2 or any(int(value) < 1 for value in self.shape):
            raise ValueError("shape must be a positive (n_mu,n_vpar) tuple")
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneVelocitySpaceSliceSeriesAudit(_PyTreeDataclass):
    """Multi-window selected-``ky`` velocity-slice comparison."""

    snapshot_indices: object
    times: object
    vpar_errors: object
    vperp_errors: object
    time_errors: object
    direct_max_abs_errors: object
    direct_l2_errors: object
    direct_relative_l2_errors: object
    best_variant_indices: object
    best_max_abs_errors: object
    best_l2_errors: object
    first_direct_max_abs_error: object
    last_direct_max_abs_error: object
    max_direct_max_abs_error: object
    max_best_max_abs_error: object
    passed: object
    shape: tuple[int, int]
    variant_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "snapshot_indices",
        "times",
        "vpar_errors",
        "vperp_errors",
        "time_errors",
        "direct_max_abs_errors",
        "direct_l2_errors",
        "direct_relative_l2_errors",
        "best_variant_indices",
        "best_max_abs_errors",
        "best_l2_errors",
        "first_direct_max_abs_error",
        "last_direct_max_abs_error",
        "max_direct_max_abs_error",
        "max_best_max_abs_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("shape", "variant_names", "notes")

    def __post_init__(self):
        indices = jnp.asarray(self.snapshot_indices, dtype=jnp.int32)
        if indices.ndim != 1 or indices.shape[0] < 1:
            raise ValueError("snapshot_indices must be a nonempty one-dimensional array")
        object.__setattr__(self, "snapshot_indices", indices)
        for name in (
            "times",
            "vpar_errors",
            "vperp_errors",
            "time_errors",
            "direct_max_abs_errors",
            "direct_l2_errors",
            "direct_relative_l2_errors",
            "best_max_abs_errors",
            "best_l2_errors",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != indices.shape:
                raise ValueError(f"{name} must match snapshot_indices")
            object.__setattr__(self, name, values)
        best_indices = jnp.asarray(self.best_variant_indices, dtype=jnp.int32)
        if best_indices.shape != indices.shape:
            raise ValueError("best_variant_indices must match snapshot_indices")
        object.__setattr__(self, "best_variant_indices", best_indices)
        for name in (
            "first_direct_max_abs_error",
            "last_direct_max_abs_error",
            "max_direct_max_abs_error",
            "max_best_max_abs_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        if len(self.shape) != 2 or any(int(value) < 1 for value in self.shape):
            raise ValueError("shape must be a positive (n_mu,n_vpar) tuple")
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "variant_names", tuple(self.variant_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneVelocitySpaceSliceSeriesVariantAudit(_PyTreeDataclass):
    """Multi-window velocity-slice audit over RHS/field-convention variants."""

    snapshot_indices: object
    times: object
    direct_max_abs_errors: object
    direct_l2_errors: object
    direct_relative_l2_errors: object
    best_layout_max_abs_errors: object
    best_layout_l2_errors: object
    best_layout_variant_indices: object
    best_direct_variant_indices: object
    baseline_direct_max_abs_errors: object
    best_direct_max_abs_errors: object
    max_baseline_direct_max_abs_error: object
    max_best_direct_max_abs_error: object
    passed: object
    shape: tuple[int, int]
    variant_names: tuple[str, ...]
    layout_variant_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "snapshot_indices",
        "times",
        "direct_max_abs_errors",
        "direct_l2_errors",
        "direct_relative_l2_errors",
        "best_layout_max_abs_errors",
        "best_layout_l2_errors",
        "best_layout_variant_indices",
        "best_direct_variant_indices",
        "baseline_direct_max_abs_errors",
        "best_direct_max_abs_errors",
        "max_baseline_direct_max_abs_error",
        "max_best_direct_max_abs_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "shape",
        "variant_names",
        "layout_variant_names",
        "notes",
    )

    def __post_init__(self):
        indices = jnp.asarray(self.snapshot_indices, dtype=jnp.int32)
        if indices.ndim != 1 or indices.shape[0] < 1:
            raise ValueError("snapshot_indices must be a nonempty one-dimensional array")
        object.__setattr__(self, "snapshot_indices", indices)
        times = jnp.asarray(self.times, dtype=jnp.float64)
        if times.shape != indices.shape:
            raise ValueError("times must match snapshot_indices")
        object.__setattr__(self, "times", times)
        direct = jnp.asarray(self.direct_max_abs_errors, dtype=jnp.float64)
        if direct.ndim != 2 or direct.shape[1] != indices.shape[0]:
            raise ValueError(
                "direct_max_abs_errors must have shape (n_variant,n_snapshot)"
            )
        if len(self.variant_names) != direct.shape[0]:
            raise ValueError("variant_names length must match direct_max_abs_errors")
        object.__setattr__(self, "direct_max_abs_errors", direct)
        for name in (
            "direct_l2_errors",
            "direct_relative_l2_errors",
            "best_layout_max_abs_errors",
            "best_layout_l2_errors",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != direct.shape:
                raise ValueError(f"{name} must match direct_max_abs_errors")
            object.__setattr__(self, name, values)
        layout_indices = jnp.asarray(self.best_layout_variant_indices, dtype=jnp.int32)
        if layout_indices.shape != direct.shape:
            raise ValueError("best_layout_variant_indices must match direct_max_abs_errors")
        object.__setattr__(self, "best_layout_variant_indices", layout_indices)
        best_direct_indices = jnp.asarray(self.best_direct_variant_indices, dtype=jnp.int32)
        if best_direct_indices.shape != indices.shape:
            raise ValueError("best_direct_variant_indices must match snapshot_indices")
        object.__setattr__(self, "best_direct_variant_indices", best_direct_indices)
        for name in ("baseline_direct_max_abs_errors", "best_direct_max_abs_errors"):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != indices.shape:
                raise ValueError(f"{name} must match snapshot_indices")
            object.__setattr__(self, name, values)
        for name in ("max_baseline_direct_max_abs_error", "max_best_direct_max_abs_error"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        if len(self.shape) != 2 or any(int(value) < 1 for value in self.shape):
            raise ValueError("shape must be a positive (n_mu,n_vpar) tuple")
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "variant_names", tuple(self.variant_names))
        object.__setattr__(self, "layout_variant_names", tuple(self.layout_variant_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VelocitySliceConventionAudit(_PyTreeDataclass):
    """Debug simple indexing/layout/phase conventions for velocity slices."""

    max_abs_errors: object
    l2_errors: object
    best_variant_index: object
    best_max_abs_error: object
    best_l2_error: object
    direct_max_abs_error: object
    direct_l2_error: object
    even_max_abs_error: object
    even_l2_error: object
    odd_same_sign_max_abs_error: object
    odd_same_sign_l2_error: object
    odd_opposite_sign_max_abs_error: object
    odd_opposite_sign_l2_error: object
    variant_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "max_abs_errors",
        "l2_errors",
        "best_variant_index",
        "best_max_abs_error",
        "best_l2_error",
        "direct_max_abs_error",
        "direct_l2_error",
        "even_max_abs_error",
        "even_l2_error",
        "odd_same_sign_max_abs_error",
        "odd_same_sign_l2_error",
        "odd_opposite_sign_max_abs_error",
        "odd_opposite_sign_l2_error",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("variant_names", "notes")

    def __post_init__(self):
        errors = jnp.asarray(self.max_abs_errors, dtype=jnp.float64)
        l2_errors = jnp.asarray(self.l2_errors, dtype=jnp.float64)
        if errors.ndim != 1 or l2_errors.shape != errors.shape:
            raise ValueError("max_abs_errors and l2_errors must be matching one-dimensional arrays")
        if len(self.variant_names) != errors.shape[0]:
            raise ValueError("variant_names length must match errors")
        object.__setattr__(self, "max_abs_errors", errors)
        object.__setattr__(self, "l2_errors", l2_errors)
        object.__setattr__(
            self,
            "best_variant_index",
            jnp.asarray(self.best_variant_index, dtype=jnp.int32),
        )
        for name in (
            "best_max_abs_error",
            "best_l2_error",
            "direct_max_abs_error",
            "direct_l2_error",
            "even_max_abs_error",
            "even_l2_error",
            "odd_same_sign_max_abs_error",
            "odd_same_sign_l2_error",
            "odd_opposite_sign_max_abs_error",
            "odd_opposite_sign_l2_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "variant_names", tuple(self.variant_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VelocitySlicePhaseAudit(_PyTreeDataclass):
    """Separate velocity-slice layout errors from global complex phase errors."""

    unit_phase_factors: object
    unit_phase_angles: object
    phase_aligned_max_abs_errors: object
    phase_aligned_l2_errors: object
    phase_aligned_relative_l2_errors: object
    complex_scale_factors: object
    scaled_max_abs_errors: object
    scaled_l2_errors: object
    scaled_relative_l2_errors: object
    best_phase_variant_index: object
    best_scaled_variant_index: object
    best_phase_max_abs_error: object
    best_scaled_max_abs_error: object
    direct_phase_max_abs_error: object
    reverse_vpar_phase_max_abs_error: object
    variant_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "unit_phase_factors",
        "unit_phase_angles",
        "phase_aligned_max_abs_errors",
        "phase_aligned_l2_errors",
        "phase_aligned_relative_l2_errors",
        "complex_scale_factors",
        "scaled_max_abs_errors",
        "scaled_l2_errors",
        "scaled_relative_l2_errors",
        "best_phase_variant_index",
        "best_scaled_variant_index",
        "best_phase_max_abs_error",
        "best_scaled_max_abs_error",
        "direct_phase_max_abs_error",
        "reverse_vpar_phase_max_abs_error",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("variant_names", "notes")

    def __post_init__(self):
        phase_errors = jnp.asarray(self.phase_aligned_max_abs_errors, dtype=jnp.float64)
        if phase_errors.ndim != 1:
            raise ValueError("phase_aligned_max_abs_errors must be one-dimensional")
        if len(self.variant_names) != phase_errors.shape[0]:
            raise ValueError("variant_names length must match phase errors")
        object.__setattr__(self, "phase_aligned_max_abs_errors", phase_errors)
        for name in (
            "unit_phase_angles",
            "phase_aligned_l2_errors",
            "phase_aligned_relative_l2_errors",
            "scaled_max_abs_errors",
            "scaled_l2_errors",
            "scaled_relative_l2_errors",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != phase_errors.shape:
                raise ValueError(f"{name} must match phase_aligned_max_abs_errors")
            object.__setattr__(self, name, values)
        for name in ("unit_phase_factors", "complex_scale_factors"):
            values = jnp.asarray(getattr(self, name), dtype=jnp.complex128)
            if values.shape != phase_errors.shape:
                raise ValueError(f"{name} must match phase_aligned_max_abs_errors")
            object.__setattr__(self, name, values)
        for name in ("best_phase_variant_index", "best_scaled_variant_index"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.int32))
        for name in (
            "best_phase_max_abs_error",
            "best_scaled_max_abs_error",
            "direct_phase_max_abs_error",
            "reverse_vpar_phase_max_abs_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "variant_names", tuple(self.variant_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneVparOddSignAudit(_PyTreeDataclass):
    """Controlled RHS sign audit for odd-in-``v_parallel`` Cyclone terms."""

    direct_max_abs_errors: object
    direct_l2_errors: object
    direct_relative_l2_errors: object
    best_layout_max_abs_errors: object
    best_layout_l2_errors: object
    peak_z_values: object
    best_direct_variant_index: object
    best_layout_variant_index: object
    baseline_direct_max_abs_error: object
    field_flip_direct_max_abs_error: object
    direct_improvement_factor: object
    rhs_variant_names: tuple[str, ...]
    best_layout_names: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "direct_max_abs_errors",
        "direct_l2_errors",
        "direct_relative_l2_errors",
        "best_layout_max_abs_errors",
        "best_layout_l2_errors",
        "peak_z_values",
        "best_direct_variant_index",
        "best_layout_variant_index",
        "baseline_direct_max_abs_error",
        "field_flip_direct_max_abs_error",
        "direct_improvement_factor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "rhs_variant_names",
        "best_layout_names",
        "notes",
    )

    def __post_init__(self):
        direct = jnp.asarray(self.direct_max_abs_errors, dtype=jnp.float64)
        if direct.ndim != 1:
            raise ValueError("direct_max_abs_errors must be one-dimensional")
        n_variant = direct.shape[0]
        if len(self.rhs_variant_names) != n_variant:
            raise ValueError("rhs_variant_names length must match errors")
        if len(self.best_layout_names) != n_variant:
            raise ValueError("best_layout_names length must match errors")
        object.__setattr__(self, "direct_max_abs_errors", direct)
        for name in (
            "direct_l2_errors",
            "direct_relative_l2_errors",
            "best_layout_max_abs_errors",
            "best_layout_l2_errors",
            "peak_z_values",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != direct.shape:
                raise ValueError(f"{name} must match direct_max_abs_errors")
            object.__setattr__(self, name, values)
        for name in ("best_direct_variant_index", "best_layout_variant_index"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.int32))
        for name in (
            "baseline_direct_max_abs_error",
            "field_flip_direct_max_abs_error",
            "direct_improvement_factor",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "rhs_variant_names", tuple(self.rhs_variant_names))
        object.__setattr__(self, "best_layout_names", tuple(self.best_layout_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneTermVIIFieldConventionAudit(_PyTreeDataclass):
    """Distinguish Term VII sign from broader field-variable conventions."""

    direct_max_abs_errors: object
    direct_l2_errors: object
    direct_relative_l2_errors: object
    best_layout_max_abs_errors: object
    best_layout_l2_errors: object
    peak_z_values: object
    best_direct_variant_index: object
    best_layout_variant_index: object
    baseline_direct_max_abs_error: object
    term_vii_only_direct_max_abs_error: object
    all_field_direct_max_abs_error: object
    nonparallel_field_direct_max_abs_error: object
    term_vii_only_improvement_factor: object
    variant_names: tuple[str, ...]
    best_layout_names: tuple[str, ...]
    term_v_phi_variants: tuple[str, ...]
    term_vii_phi_variants: tuple[str, ...]
    term_viii_phi_variants: tuple[str, ...]
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "direct_max_abs_errors",
        "direct_l2_errors",
        "direct_relative_l2_errors",
        "best_layout_max_abs_errors",
        "best_layout_l2_errors",
        "peak_z_values",
        "best_direct_variant_index",
        "best_layout_variant_index",
        "baseline_direct_max_abs_error",
        "term_vii_only_direct_max_abs_error",
        "all_field_direct_max_abs_error",
        "nonparallel_field_direct_max_abs_error",
        "term_vii_only_improvement_factor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "variant_names",
        "best_layout_names",
        "term_v_phi_variants",
        "term_vii_phi_variants",
        "term_viii_phi_variants",
        "notes",
    )

    def __post_init__(self):
        direct = jnp.asarray(self.direct_max_abs_errors, dtype=jnp.float64)
        if direct.ndim != 1:
            raise ValueError("direct_max_abs_errors must be one-dimensional")
        n_variant = direct.shape[0]
        for name in (
            "variant_names",
            "best_layout_names",
            "term_v_phi_variants",
            "term_vii_phi_variants",
            "term_viii_phi_variants",
        ):
            if len(getattr(self, name)) != n_variant:
                raise ValueError(f"{name} length must match errors")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "direct_max_abs_errors", direct)
        for name in (
            "direct_l2_errors",
            "direct_relative_l2_errors",
            "best_layout_max_abs_errors",
            "best_layout_l2_errors",
            "peak_z_values",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != direct.shape:
                raise ValueError(f"{name} must match direct_max_abs_errors")
            object.__setattr__(self, name, values)
        for name in ("best_direct_variant_index", "best_layout_variant_index"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.int32))
        for name in (
            "baseline_direct_max_abs_error",
            "term_vii_only_direct_max_abs_error",
            "all_field_direct_max_abs_error",
            "nonparallel_field_direct_max_abs_error",
            "term_vii_only_improvement_factor",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneProfileOperatorAudit(_PyTreeDataclass):
    """Selected-mode operator audit at a localized Cyclone profile mismatch."""

    normalized_phi_power: object
    z_grid: object
    streaming_delta_profile: object
    field_drive_delta_profile: object
    field_residual_profile: object
    time: object
    z: object
    output_window: object
    z_index: object
    peak_z: object
    second_moment: object
    local_streaming_delta: object
    max_streaming_delta: object
    boundary_streaming_delta: object
    local_field_drive_delta: object
    max_field_drive_delta: object
    boundary_field_drive_delta: object
    field_residual_max: object
    field_reconstruction_error: object
    rhs_assembly_error: object
    passed: object
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "normalized_phi_power",
        "z_grid",
        "streaming_delta_profile",
        "field_drive_delta_profile",
        "field_residual_profile",
        "time",
        "z",
        "output_window",
        "z_index",
        "peak_z",
        "second_moment",
        "local_streaming_delta",
        "max_streaming_delta",
        "boundary_streaming_delta",
        "local_field_drive_delta",
        "max_field_drive_delta",
        "boundary_field_drive_delta",
        "field_residual_max",
        "field_reconstruction_error",
        "rhs_assembly_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("notes",)

    def __post_init__(self):
        profile = jnp.asarray(self.normalized_phi_power, dtype=jnp.float64)
        if profile.ndim != 1:
            raise ValueError("normalized_phi_power must be one-dimensional")
        object.__setattr__(self, "normalized_phi_power", profile)
        z_grid = jnp.asarray(self.z_grid, dtype=jnp.float64)
        if z_grid.shape != profile.shape:
            raise ValueError("z_grid must match normalized_phi_power shape")
        object.__setattr__(self, "z_grid", z_grid)
        for name in (
            "streaming_delta_profile",
            "field_drive_delta_profile",
            "field_residual_profile",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != profile.shape:
                raise ValueError(f"{name} must match normalized_phi_power shape")
            object.__setattr__(self, name, values)
        for name in (
            "time",
            "z",
            "peak_z",
            "second_moment",
            "local_streaming_delta",
            "max_streaming_delta",
            "boundary_streaming_delta",
            "local_field_drive_delta",
            "max_field_drive_delta",
            "boundary_field_drive_delta",
            "field_residual_max",
            "field_reconstruction_error",
            "rhs_assembly_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(
            self,
            "output_window",
            jnp.asarray(self.output_window, dtype=jnp.int32),
        )
        object.__setattr__(self, "z_index", jnp.asarray(self.z_index, dtype=jnp.int32))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneTermIFortranAudit(_PyTreeDataclass):
    """Source-reconstructed GKW Term I parallel streaming audit."""

    term_error_profile: object
    coefficient_error_profile: object
    current_term_profile: object
    reference_term_profile: object
    recurrence_speed_error_profile: object
    time: object
    z: object
    output_window: object
    z_index: object
    local_term_error: object
    max_term_error: object
    local_coefficient_error: object
    max_coefficient_error: object
    recurrence_speed_max_error: object
    sign_selection_error: object
    passed: object
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "term_error_profile",
        "coefficient_error_profile",
        "current_term_profile",
        "reference_term_profile",
        "recurrence_speed_error_profile",
        "time",
        "z",
        "output_window",
        "z_index",
        "local_term_error",
        "max_term_error",
        "local_coefficient_error",
        "max_coefficient_error",
        "recurrence_speed_max_error",
        "sign_selection_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("notes",)

    def __post_init__(self):
        profile = jnp.asarray(self.term_error_profile, dtype=jnp.float64)
        if profile.ndim != 1:
            raise ValueError("term_error_profile must be one-dimensional")
        object.__setattr__(self, "term_error_profile", profile)
        for name in (
            "coefficient_error_profile",
            "current_term_profile",
            "reference_term_profile",
            "recurrence_speed_error_profile",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != profile.shape:
                raise ValueError(f"{name} must match term_error_profile shape")
            object.__setattr__(self, name, values)
        for name in (
            "time",
            "z",
            "local_term_error",
            "max_term_error",
            "local_coefficient_error",
            "max_coefficient_error",
            "recurrence_speed_max_error",
            "sign_selection_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(
            self,
            "output_window",
            jnp.asarray(self.output_window, dtype=jnp.int32),
        )
        object.__setattr__(self, "z_index", jnp.asarray(self.z_index, dtype=jnp.int32))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneTimeNormalizationAudit(_PyTreeDataclass):
    """GKW RK4 and field-normalization sequence audit."""

    times: object
    normalization_factor: object
    gkw_window_growth: object
    trace_window_growth: object
    post_normalization_field_norm: object
    field_linearity_error: object
    rk4_step_error: object
    time_grid_error: object
    growth_sequence_error: object
    post_normalization_error: object
    max_field_linearity_error: object
    passed: object
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "times",
        "normalization_factor",
        "gkw_window_growth",
        "trace_window_growth",
        "post_normalization_field_norm",
        "field_linearity_error",
        "rk4_step_error",
        "time_grid_error",
        "growth_sequence_error",
        "post_normalization_error",
        "max_field_linearity_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("notes",)

    def __post_init__(self):
        times = jnp.asarray(self.times, dtype=jnp.float64)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        object.__setattr__(self, "times", times)
        for name in (
            "normalization_factor",
            "gkw_window_growth",
            "trace_window_growth",
            "post_normalization_field_norm",
            "field_linearity_error",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != times.shape:
                raise ValueError(f"{name} must match times shape")
            object.__setattr__(self, name, values)
        for name in (
            "rk4_step_error",
            "time_grid_error",
            "growth_sequence_error",
            "post_normalization_error",
            "max_field_linearity_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneDiagnosticPackingAudit(_PyTreeDataclass):
    """GKW field-packing and diagnostic-output source audit."""

    parallel_phi_profile: object
    packed_parallel_phi_profile: object
    ky_spectrum: object
    packed_ky_spectrum: object
    kx_spectrum: object
    packed_kx_spectrum: object
    time: object
    packing_roundtrip_error: object
    parallel_phi_error: object
    selected_profile_error: object
    ky_spectrum_error: object
    kx_spectrum_error: object
    passed: object
    output_window: int
    field_offset: int
    n_field_values: int
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "parallel_phi_profile",
        "packed_parallel_phi_profile",
        "ky_spectrum",
        "packed_ky_spectrum",
        "kx_spectrum",
        "packed_kx_spectrum",
        "time",
        "packing_roundtrip_error",
        "parallel_phi_error",
        "selected_profile_error",
        "ky_spectrum_error",
        "kx_spectrum_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "output_window",
        "field_offset",
        "n_field_values",
        "notes",
    )

    def __post_init__(self):
        profile = jnp.asarray(self.parallel_phi_profile, dtype=jnp.float64)
        if profile.ndim != 1:
            raise ValueError("parallel_phi_profile must be one-dimensional")
        object.__setattr__(self, "parallel_phi_profile", profile)
        packed_profile = jnp.asarray(self.packed_parallel_phi_profile, dtype=jnp.float64)
        if packed_profile.shape != profile.shape:
            raise ValueError("packed_parallel_phi_profile must match parallel_phi_profile")
        object.__setattr__(self, "packed_parallel_phi_profile", packed_profile)
        ky = jnp.asarray(self.ky_spectrum, dtype=jnp.float64)
        packed_ky = jnp.asarray(self.packed_ky_spectrum, dtype=jnp.float64)
        kx = jnp.asarray(self.kx_spectrum, dtype=jnp.float64)
        packed_kx = jnp.asarray(self.packed_kx_spectrum, dtype=jnp.float64)
        if ky.ndim != 1 or packed_ky.shape != ky.shape:
            raise ValueError("ky spectra must be one-dimensional and shape-matched")
        if kx.ndim != 1 or packed_kx.shape != kx.shape:
            raise ValueError("kx spectra must be one-dimensional and shape-matched")
        object.__setattr__(self, "ky_spectrum", ky)
        object.__setattr__(self, "packed_ky_spectrum", packed_ky)
        object.__setattr__(self, "kx_spectrum", kx)
        object.__setattr__(self, "packed_kx_spectrum", packed_kx)
        for name in (
            "time",
            "packing_roundtrip_error",
            "parallel_phi_error",
            "selected_profile_error",
            "ky_spectrum_error",
            "kx_spectrum_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        if self.output_window < 1:
            raise ValueError("output_window must be positive")
        if self.field_offset < 0:
            raise ValueError("field_offset must be nonnegative")
        if self.n_field_values < 1:
            raise ValueError("n_field_values must be positive")
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneMatdatMatrixAudit(_PyTreeDataclass):
    """GKW ``matdat.F90`` sparse-matrix convention audit."""

    matrix_action_error: object
    source_max_abs: object
    explicit_delta_error: object
    compressed_action_error: object
    complex_real_split_error: object
    linearity_error: object
    max_abs_matrix_entry: object
    passed: object
    n_state: int
    n_nonzero: int
    n_duplicate_triplets: int
    n_real_entries: int
    n_complex_entries: int
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "matrix_action_error",
        "source_max_abs",
        "explicit_delta_error",
        "compressed_action_error",
        "complex_real_split_error",
        "linearity_error",
        "max_abs_matrix_entry",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_state",
        "n_nonzero",
        "n_duplicate_triplets",
        "n_real_entries",
        "n_complex_entries",
        "notes",
    )

    def __post_init__(self):
        for name in self._dynamic_fields[:-1]:
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        for name in (
            "n_state",
            "n_nonzero",
            "n_duplicate_triplets",
            "n_real_entries",
            "n_complex_entries",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.n_state < 1:
            raise ValueError("n_state must be positive")
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneCoefficientSourceAudit(_PyTreeDataclass):
    """GKW source-level coefficient construction audit for selected CBC terms."""

    term_errors: object
    coefficient_errors: object
    insertion_errors: object
    time: object
    z: object
    z_index: object
    max_term_error: object
    max_coefficient_error: object
    max_insertion_error: object
    max_abs_error: object
    passed: object
    term_names: tuple[str, ...]
    output_window: int
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "term_errors",
        "coefficient_errors",
        "insertion_errors",
        "time",
        "z",
        "z_index",
        "max_term_error",
        "max_coefficient_error",
        "max_insertion_error",
        "max_abs_error",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("term_names", "output_window", "notes")

    def __post_init__(self):
        term_errors = jnp.asarray(self.term_errors, dtype=jnp.float64)
        if term_errors.ndim != 1:
            raise ValueError("term_errors must be one-dimensional")
        if len(self.term_names) != term_errors.shape[0]:
            raise ValueError("term_names length must match term_errors")
        object.__setattr__(self, "term_errors", term_errors)
        for name in ("coefficient_errors", "insertion_errors"):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != term_errors.shape:
                raise ValueError(f"{name} must match term_errors")
            object.__setattr__(self, name, values)
        for name in (
            "time",
            "z",
            "max_term_error",
            "max_coefficient_error",
            "max_insertion_error",
            "max_abs_error",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        if self.output_window < 0:
            raise ValueError("output_window must be nonnegative")
        object.__setattr__(self, "z_index", jnp.asarray(self.z_index, dtype=jnp.int32))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        object.__setattr__(self, "term_names", tuple(self.term_names))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CycloneIghArakawaAudit(_PyTreeDataclass):
    """GKW ``ltrapping_arakawa`` fused Term I/IV audit."""

    fused_profile: object
    separated_profile: object
    delta_profile: object
    parallel_diffusion_profile: object
    velocity_diffusion_profile: object
    time: object
    z: object
    z_index: object
    local_delta: object
    max_delta: object
    relative_delta: object
    max_parallel_diffusion: object
    max_velocity_diffusion: object
    passed: object
    output_window: int
    notes: str = ""

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "fused_profile",
        "separated_profile",
        "delta_profile",
        "parallel_diffusion_profile",
        "velocity_diffusion_profile",
        "time",
        "z",
        "z_index",
        "local_delta",
        "max_delta",
        "relative_delta",
        "max_parallel_diffusion",
        "max_velocity_diffusion",
        "passed",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("output_window", "notes")

    def __post_init__(self):
        profile = jnp.asarray(self.delta_profile, dtype=jnp.float64)
        if profile.ndim != 1:
            raise ValueError("delta_profile must be one-dimensional")
        object.__setattr__(self, "delta_profile", profile)
        for name in (
            "fused_profile",
            "separated_profile",
            "parallel_diffusion_profile",
            "velocity_diffusion_profile",
        ):
            values = jnp.asarray(getattr(self, name), dtype=jnp.float64)
            if values.shape != profile.shape:
                raise ValueError(f"{name} must match delta_profile")
            object.__setattr__(self, name, values)
        for name in (
            "time",
            "z",
            "local_delta",
            "max_delta",
            "relative_delta",
            "max_parallel_diffusion",
            "max_velocity_diffusion",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name), dtype=jnp.float64))
        if self.output_window < 0:
            raise ValueError("output_window must be nonnegative")
        object.__setattr__(self, "z_index", jnp.asarray(self.z_index, dtype=jnp.int32))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GxEikGeometryParityReport(_PyTreeDataclass):
    """Field-by-field comparison between solver geometry and a GX/GS2 eik table."""

    field_errors: object
    max_abs_error: object
    max_abs_kperp2_error: object
    field_names: tuple[str, ...]
    source: str

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "field_errors",
        "max_abs_error",
        "max_abs_kperp2_error",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("field_names", "source")

    def __post_init__(self):
        errors = jnp.asarray(self.field_errors, dtype=jnp.float64)
        if errors.ndim != 1:
            raise ValueError("field_errors must be one-dimensional")
        if len(self.field_names) != errors.shape[0]:
            raise ValueError("field_names must match field_errors length")
        object.__setattr__(self, "field_errors", errors)
        object.__setattr__(
            self,
            "max_abs_error",
            jnp.asarray(self.max_abs_error, dtype=jnp.float64),
        )
        object.__setattr__(
            self,
            "max_abs_kperp2_error",
            jnp.asarray(self.max_abs_kperp2_error, dtype=jnp.float64),
        )


def rosenbluth_hinton_residual(q, epsilon, coefficient: float = 1.6):
    """Return the large-aspect-ratio Rosenbluth-Hinton residual estimate."""

    q = jnp.asarray(q, dtype=jnp.float64)
    epsilon = jnp.asarray(epsilon, dtype=jnp.float64)
    return 1.0 / (1.0 + coefficient * q**2 / jnp.sqrt(epsilon))


def rosenbluth_hinton_target() -> BenchmarkTarget:
    """Return the Gyaradax/GKW zonal-flow residual target used for validation."""

    return BenchmarkTarget(
        name="rosenbluth_hinton_q13_eps005",
        quantity="zonal_residual",
        reference_value=0.0711,
        tolerance=1.0e-3,
        source="relevant-codes/gyaradax/docs/NOTES.md",
        metadata=(
            ("q", 1.3),
            ("shat", 0.1592),
            ("epsilon", 0.05),
            ("kx_rhos", 0.025),
            ("n_z", 64),
            ("n_vpar", 64),
            ("n_mu", 16),
            ("vpar_max", 3.0),
            ("disp_par", 0.01),
            ("disp_vp", 0.08),
            ("reference_n_z", 128),
            ("reference_n_vpar", 128),
            ("reference_disp_vp_default", 0.2),
            ("dt", 0.01),
            ("time_window", "t > 80"),
        ),
    )


def cyclone_base_case_growth_target() -> BenchmarkTarget:
    """Return the documented Gyaradax/GKW Cyclone Base Case growth target."""

    return BenchmarkTarget(
        name="cyclone_base_case_gkw_kt05",
        quantity="selected_growth_rate",
        reference_value=0.179,
        tolerance=1.0e-2,
        source="relevant-codes/gyaradax/docs/NOTES.md",
        metadata=(
            ("q", 1.4),
            ("shat", 0.78),
            ("epsilon", 0.19),
            ("R_over_Ln", 2.2),
            ("R_over_LT", 6.9),
            ("k_theta_rhos", 0.5),
            ("geometry", "s-alpha"),
            ("electrons", "adiabatic"),
            ("n_z", 144),
            ("nperiod", 5),
            ("n_vpar", 64),
            ("n_mu", 16),
            ("vpar_max", 3.0),
            ("parallel_backend", "finite_difference"),
            ("parallel_boundary", "zero"),
            ("parallel_derivative_model", "gkw_upwind"),
            ("velocity_backend", "finite_difference"),
            ("disp_par", 1.0),
            ("dt", 0.003),
            ("steps_per_window", 100),
            ("n_windows", 300),
        ),
    )


def load_gx_growth_rate_reference(
    path,
    *,
    ikx: int = 0,
    average_fraction: float = 0.5,
    drop_zonal: bool = True,
) -> GxGrowthRateReference:
    """Load the GX ``omega_kxkyt`` reference curve from a NetCDF output file."""

    dataset_cls = _import_netcdf_dataset()
    path = Path(path)
    with dataset_cls(path, mode="r") as data:
        time = np.asarray(data.groups["Grids"].variables["time"][:], dtype=float)
        ky = np.asarray(data.groups["Grids"].variables["ky"][:], dtype=float)
        omega = np.asarray(
            data.groups["Diagnostics"].variables["omega_kxkyt"][:, :, ikx, :],
            dtype=float,
        )
    if drop_zonal:
        ky = ky[1:]
        omega = omega[:, 1:, :]
    if len(time) == 0:
        raise ValueError("GX output contains no time samples")
    start = int(len(time) * average_fraction)
    frequency = np.mean(omega[start:, :, 0], axis=0)
    growth_rate = np.mean(omega[start:, :, 1], axis=0)
    return GxGrowthRateReference(
        ky=ky,
        growth_rate=growth_rate,
        frequency=frequency,
        source=str(path),
        ikx=ikx,
        average_fraction=average_fraction,
    )


def gx_growth_rate_target(
    reference: GxGrowthRateReference,
    *,
    target_ky: float,
    quantity: str = "selected_growth_rate",
    name: str | None = None,
    tolerance: float = 1.0e-3,
) -> BenchmarkTarget:
    """Return a scalar target from the nearest ``ky`` in a GX growth curve."""

    ky = np.asarray(reference.ky)
    index = int(np.argmin(np.abs(ky - target_ky)))
    return BenchmarkTarget(
        name=name or f"gx_growth_ky{float(ky[index]):.6g}",
        quantity=quantity,
        reference_value=np.asarray(reference.growth_rate)[index],
        tolerance=tolerance,
        source=reference.source,
        metadata=(
            ("target_ky", float(target_ky)),
            ("matched_ky", float(ky[index])),
            ("frequency", float(np.asarray(reference.frequency)[index])),
            ("ikx", reference.ikx),
            ("average_fraction", reference.average_fraction),
        ),
    )


def load_gx_eik_geometry_reference(path) -> GxEikGeometryReference:
    """Load a numeric GS2/GX eik geometry table.

    The local GX VMEC test fixtures use one numeric header row followed by
    rows with
    ``theta, bmag, gradpar, gds2, gds21, gds22, cvdrift, cvdrift0, gbdrift, gbdrift0``.
    GX's DESC geometry module instead writes the older multi-block
    ``eik.out`` layout; that layout is detected and mapped into the same
    reference object.
    """

    path = Path(path)
    text = path.read_text()
    if "gbdrift gradpar grho tgrid" in text:
        return _load_gx_block_eik_geometry_reference(path, text)
    rows = _numeric_rows(path)
    if len(rows) < 2:
        raise ValueError("eik geometry reference must contain a header and data rows")
    header = tuple(rows[0])
    data = np.asarray([row for row in rows[1:] if len(row) >= 10], dtype=float)
    if data.ndim != 2 or data.shape[1] < 10:
        raise ValueError("eik geometry rows must have at least 10 numeric columns")
    return GxEikGeometryReference(
        theta=data[:, 0],
        bmag=data[:, 1],
        gradpar=data[:, 2],
        gds2=data[:, 3],
        gds21=data[:, 4],
        gds22=data[:, 5],
        cvdrift=data[:, 6],
        cvdrift0=data[:, 7],
        gbdrift=data[:, 8],
        gbdrift0=data[:, 9],
        source=str(path),
        header=header,
    )


def resample_gx_eik_geometry_reference(
    reference: GxEikGeometryReference,
    theta,
) -> GxEikGeometryReference:
    """Interpolate a loaded GX/GS2 eik table onto requested theta nodes."""

    theta = np.asarray(theta, dtype=float)
    source_theta = np.asarray(reference.theta, dtype=float)
    if source_theta.ndim != 1 or source_theta.size < 2:
        raise ValueError("reference.theta must contain at least two nodes")
    if not np.all(np.diff(source_theta) > 0):
        raise ValueError("reference.theta must be strictly increasing")
    kwargs = {"theta": theta}
    for name in GxEikGeometryReference._dynamic_fields[1:]:
        kwargs[name] = np.interp(theta, source_theta, np.asarray(getattr(reference, name)))
    return GxEikGeometryReference(
        **kwargs,
        source=reference.source,
        header=reference.header,
    )


def build_flux_tube_geometry_from_gx_eik_reference(
    reference: GxEikGeometryReference,
    parallel_grid,
    *,
    radial_coordinate: str = "rho",
):
    """Map a GX/GS2 eik geometry table into the internal flux-tube contract."""

    from .geometry.flux_tube import FluxTubeGeometry

    theta = _coerce_geometry_reference_shape("theta", reference.theta, parallel_grid.z.shape)
    bmag = _coerce_geometry_reference_shape("bmag", reference.bmag, parallel_grid.z.shape)
    gradpar = _coerce_geometry_reference_shape("gradpar", reference.gradpar, parallel_grid.z.shape)
    gds2 = _coerce_geometry_reference_shape("gds2", reference.gds2, parallel_grid.z.shape)
    gds21 = _coerce_geometry_reference_shape("gds21", reference.gds21, parallel_grid.z.shape)
    gds22 = _coerce_geometry_reference_shape("gds22", reference.gds22, parallel_grid.z.shape)
    gbdrift = _coerce_geometry_reference_shape("gbdrift", reference.gbdrift, parallel_grid.z.shape)
    gbdrift0 = _coerce_geometry_reference_shape(
        "gbdrift0",
        reference.gbdrift0,
        parallel_grid.z.shape,
    )
    cvdrift = _coerce_geometry_reference_shape("cvdrift", reference.cvdrift, parallel_grid.z.shape)
    cvdrift0 = _coerce_geometry_reference_shape(
        "cvdrift0",
        reference.cvdrift0,
        parallel_grid.z.shape,
    )
    rho = jnp.zeros_like(theta)
    return FluxTubeGeometry(
        z=parallel_grid.z,
        w_z=parallel_grid.w_z,
        theta=theta,
        phi=theta,
        rho=rho,
        B=bmag,
        F=gradpar,
        G=-(cvdrift + gbdrift),
        E_y=jnp.zeros_like(theta),
        D_x=gbdrift0 + cvdrift0,
        D_y=gbdrift + cvdrift,
        g_xx=gds22,
        g_xy=gds21,
        g_yy=gds2,
        radial_coordinate=radial_coordinate,
        source="gx-eik",
    )


def geometry_to_gx_eik_reference(
    geometry,
    *,
    source: str | None = None,
) -> GxEikGeometryReference:
    """Export a solver geometry object to the GX/GS2 eik table contract.

    GX/GS2 eik tables store the metric, parallel-gradient coefficient, and
    grad-B/curvature drift fields.  The internal solver stores only the summed
    radial and binormal magnetic-drift coefficients, so this exporter places the
    totals in ``gbdrift0`` and ``gbdrift`` and sets the corresponding
    ``cvdrift`` pieces to zero.  This preserves the eik quantities used by the
    gyrokinetic residual and k-perp contract without inventing a curvature split.
    """

    theta = jnp.asarray(getattr(geometry, "theta", geometry.z), dtype=jnp.float64)
    zero = jnp.zeros_like(theta)
    return GxEikGeometryReference(
        theta=theta,
        bmag=jnp.asarray(geometry.B, dtype=jnp.float64),
        gradpar=jnp.asarray(geometry.F, dtype=jnp.float64),
        gds2=jnp.asarray(geometry.g_yy, dtype=jnp.float64),
        gds21=jnp.asarray(geometry.g_xy, dtype=jnp.float64),
        gds22=jnp.asarray(geometry.g_xx, dtype=jnp.float64),
        gbdrift=jnp.asarray(geometry.D_y, dtype=jnp.float64),
        gbdrift0=jnp.asarray(geometry.D_x, dtype=jnp.float64),
        cvdrift=zero,
        cvdrift0=zero,
        source=source or f"{getattr(geometry, 'source', 'solver')}:gx-eik-export",
        header=(),
    )


def gx_eik_kperp2(
    reference: GxEikGeometryReference,
    kx,
    ky,
):
    """Evaluate the GX/GS2 eik metric contraction on ``(theta,kx,ky)``."""

    kx = jnp.asarray(kx, dtype=jnp.asarray(reference.theta).dtype)
    ky = jnp.asarray(ky, dtype=jnp.asarray(reference.theta).dtype)
    return (
        reference.gds22[:, None, None] * kx[None, :, None] ** 2
        + 2.0 * reference.gds21[:, None, None] * kx[None, :, None] * ky[None, None, :]
        + reference.gds2[:, None, None] * ky[None, None, :] ** 2
    )


def compare_geometry_to_gx_eik_reference(
    geometry,
    reference: GxEikGeometryReference,
    fourier_grid,
    *,
    include_mirror_proxy: bool = True,
) -> GxEikGeometryParityReport:
    """Compare a solver-produced geometry object with a GX/GS2 eik reference."""

    from .geometry import k_perp_squared

    bmag = _coerce_geometry_reference_shape("bmag", reference.bmag, geometry.B.shape)
    gradpar = _coerce_geometry_reference_shape("gradpar", reference.gradpar, geometry.F.shape)
    gds2 = _coerce_geometry_reference_shape("gds2", reference.gds2, geometry.g_yy.shape)
    gds21 = _coerce_geometry_reference_shape("gds21", reference.gds21, geometry.g_xy.shape)
    gds22 = _coerce_geometry_reference_shape("gds22", reference.gds22, geometry.g_xx.shape)
    gx_radial_drift = _coerce_geometry_reference_shape(
        "gx_radial_drift",
        reference.gbdrift0 + reference.cvdrift0,
        geometry.D_x.shape,
    )
    gx_binormal_drift = _coerce_geometry_reference_shape(
        "gx_binormal_drift",
        reference.gbdrift + reference.cvdrift,
        geometry.D_y.shape,
    )
    gx_mirror_proxy = -gx_binormal_drift
    field_names = [
        "B/bmag",
        "F/gradpar",
        "g_xx/gds22",
        "g_xy/gds21",
        "g_yy/gds2",
        "D_x/gbdrift0+cvdrift0",
        "D_y/gbdrift+cvdrift",
    ]
    errors = [
        _max_abs_error(geometry.B, bmag),
        _max_abs_error(geometry.F, gradpar),
        _max_abs_error(geometry.g_xx, gds22),
        _max_abs_error(geometry.g_xy, gds21),
        _max_abs_error(geometry.g_yy, gds2),
        _max_abs_error(geometry.D_x, gx_radial_drift),
        _max_abs_error(geometry.D_y, gx_binormal_drift),
    ]
    if include_mirror_proxy:
        field_names.append("G/-(gbdrift+cvdrift)")
        errors.append(_max_abs_error(geometry.G, gx_mirror_proxy))
    field_names.append("kperp2")
    errors.append(
        _max_abs_error(
            k_perp_squared(geometry, fourier_grid),
            gx_eik_kperp2(reference, fourier_grid.kx, fourier_grid.ky),
        )
    )
    field_errors = jnp.asarray(errors, dtype=jnp.float64)
    return GxEikGeometryParityReport(
        field_errors=field_errors,
        max_abs_error=jnp.max(field_errors),
        max_abs_kperp2_error=field_errors[-1],
        field_names=tuple(field_names),
        source=reference.source,
    )


def evaluate_benchmark_gate(
    value,
    target: BenchmarkTarget,
    *,
    normalize_by_tolerance: bool = True,
    notes: str = "",
) -> BenchmarkGateResult:
    """Compare one observed scalar with a target and return a gate result."""

    residual = benchmark_target_residual(
        value,
        target,
        normalize_by_tolerance=normalize_by_tolerance,
    )
    cost = 0.5 * residual**2
    tolerance = 1.0 if normalize_by_tolerance else target.tolerance
    passed = jnp.abs(residual) <= tolerance
    return BenchmarkGateResult(
        target=target,
        observed_value=value,
        residual=residual,
        cost=cost,
        passed=passed,
        notes=notes,
    )


def run_calibrated_reduced_rosenbluth_hinton_gate(
    *,
    n_z: int = 16,
    n_vpar: int = 16,
    n_mu: int = 8,
    vpar_max: float = 4.0,
    mu_max: float = 8.0,
    dt: float = 0.01,
    n_steps: int = 620,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Run a calibrated reduced RH crossing gate.

    This pins a deterministic reduced-grid regression point where the present
    collisionless discretization crosses the documented RH residual.  It is not
    a substitute for the production long-time plateau benchmark.
    """

    result = run_reduced_rosenbluth_hinton_gate(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        dt=dt,
        n_steps=n_steps,
        target=target,
    )
    return BenchmarkGateResult(
        target=result.target,
        observed_value=result.observed_value,
        residual=result.residual,
        cost=result.cost,
        passed=result.passed,
        notes=(
            "calibrated reduced RH crossing; production t>80 plateau and "
            "dissipation convergence remain open"
        ),
    )


def run_reduced_rosenbluth_hinton_gate(
    *,
    n_z: int = 16,
    n_vpar: int = 16,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    n_steps: int = 100,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str = "finite_difference",
    parallel_boundary: str = "zero",
    velocity_backend: str = "finite_difference",
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Evolve the present reduced zonal-flow setup and compare its residual.

    This is an executable validation gate for the current solver.  It is not a
    claim that the present reduced model passes the production RH benchmark.
    """

    from .geometry import build_s_alpha_geometry
    from .grids import build_fourier_grid, build_velocity_grid
    from .physics import AdiabaticElectronParams, solve_adiabatic_electron_phi
    from .solver import build_linear_residual_precompute, linear_residual
    from .time_advance import integrate_fixed_step
    from .types import FourierGridSpec, GeometryScalarParams, SpeciesParams, VelocityGridSpec

    target = target or rosenbluth_hinton_target()
    metadata = dict(target.metadata)
    q = float(metadata.get("q", 1.3))
    shat = float(metadata.get("shat", 0.1592))
    epsilon = float(metadata.get("epsilon", 0.05))
    kx_rhos = float(metadata.get("kx_rhos", 0.025))
    vpar_max = float(metadata.get("vpar_max", 3.0) if vpar_max is None else vpar_max)
    mu_max = 0.5 * vpar_max**2 if mu_max is None else float(mu_max)
    dt = float(metadata.get("dt", 0.01) if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata.get("disp_par", 0.01)
        if parallel_recurrence_rate is None
        else parallel_recurrence_rate
    )
    velocity_recurrence_rate = float(
        metadata.get("disp_vp", 0.2)
        if velocity_recurrence_rate is None
        else velocity_recurrence_rate
    )
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=n_vpar,
            n_mu=n_mu,
            vpar_max=vpar_max,
            mu_max=mu_max,
            backend=velocity_backend,
        )
    )
    parallel = _build_gkw_cell_centered_parallel_grid(n_z, derivative_backend=parallel_backend)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=1, kx_max=kx_rhos, ky_values=(0.0,)))
    geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(q=q, shat=shat, eps=epsilon),
    )
    species = SpeciesParams(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        density_gradient=0.0,
        temperature_gradient=0.0,
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=True,
        ),
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_recurrence_velocity_model="rms",
        velocity_recurrence_rate=velocity_recurrence_rate,
        velocity_recurrence_velocity_model="rms",
    )
    ix = min(fourier.ixzero + 1, fourier.kx.shape[0] - 1)
    ix_conjugate = max(fourier.ixzero - 1, 0)
    maxwellian = precompute.rhs.maxwellian[0]
    state = jnp.zeros((n_vpar, n_mu, n_z, 3, 1), dtype=jnp.complex128)
    state = state.at[:, :, :, ix_conjugate, 0].set(-0.5j * 1.0e-4 * maxwellian)
    state = state.at[:, :, :, ix, 0].set(0.5j * 1.0e-4 * maxwellian)
    phi_initial = solve_adiabatic_electron_phi(state, precompute.field)
    result = integrate_fixed_step(
        state,
        dt,
        n_steps,
        linear_residual,
        precompute,
        store_history=False,
    )
    phi_final = solve_adiabatic_electron_phi(result.state, precompute.field)
    observed = _field_amplitude(phi_final[:, ix, 0]) / _field_amplitude(phi_initial[:, ix, 0])
    return evaluate_benchmark_gate(
        observed,
        target,
        notes=(
            "reduced executable RH gate with GKW finite-difference velocity fallback, "
            "GKW finite-difference parallel fallback, zonal finit pattern, and "
            "GKW-scaled disp_par/disp_vp recurrence control; "
            "production t=100 convergence remains open"
        ),
    )


def run_rosenbluth_hinton_plateau_gate(
    *,
    n_z: int | None = None,
    n_vpar: int | None = None,
    n_mu: int | None = None,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    t_end: float = 100.0,
    t_start: float = 80.0,
    diagnostic_interval: float = 1.0,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str = "finite_difference",
    velocity_backend: str = "finite_difference",
    z_modal_damping: float = 0.0,
    vpar_modal_damping: float = 0.0,
    mu_modal_damping: float = 0.0,
    modal_damping_order: int = 4,
    plateau_tolerance: float = 2.0e-2,
    amplitude_ceiling: float = 1.0e8,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Run the long-time Rosenbluth--Hinton residual plateau gate.

    The metric follows the GKW/Gyaradax benchmark convention:
    ``sqrt(mean(kxspec(t)/kxspec(0)))`` over the late-time window.  The default
    recurrence control follows the Gyaradax/GKW ``disp_par=0.01`` scaling as
    an in-residual fourth-order parallel term.  The modal filter arguments are
    retained for experiments but default to zero and are not part of the
    production gate.
    """

    from .geometry import build_s_alpha_geometry
    from .grids import build_fourier_grid, build_velocity_grid
    from .physics import AdiabaticElectronParams, solve_adiabatic_electron_phi
    from .solver import build_linear_residual_precompute, linear_residual
    from .time_advance import build_modal_damping_filter, integrate_fixed_step
    from .types import FourierGridSpec, GeometryScalarParams, SpeciesParams, VelocityGridSpec

    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if not 0.0 <= t_start < t_end:
        raise ValueError("t_start must satisfy 0 <= t_start < t_end")
    if diagnostic_interval <= 0.0:
        raise ValueError("diagnostic_interval must be positive")
    if plateau_tolerance < 0.0:
        raise ValueError("plateau_tolerance must be nonnegative")
    if amplitude_ceiling <= 0.0:
        raise ValueError("amplitude_ceiling must be positive")

    target = target or rosenbluth_hinton_target()
    metadata = dict(target.metadata)
    q = float(metadata.get("q", 1.3))
    shat = float(metadata.get("shat", 0.1592))
    epsilon = float(metadata.get("epsilon", 0.05))
    kx_rhos = float(metadata.get("kx_rhos", 0.025))
    n_z = int(metadata.get("n_z", 128) if n_z is None else n_z)
    n_vpar = int(metadata.get("n_vpar", 128) if n_vpar is None else n_vpar)
    n_mu = int(metadata.get("n_mu", 16) if n_mu is None else n_mu)
    vpar_max = float(metadata.get("vpar_max", 3.0) if vpar_max is None else vpar_max)
    mu_max = 0.5 * vpar_max**2 if mu_max is None else float(mu_max)
    dt = float(metadata.get("dt", 0.01) if dt is None else dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    parallel_recurrence_rate = float(
        metadata.get("disp_par", 0.01)
        if parallel_recurrence_rate is None
        else parallel_recurrence_rate
    )
    velocity_recurrence_rate = float(
        metadata.get("disp_vp", 0.2)
        if velocity_recurrence_rate is None
        else velocity_recurrence_rate
    )
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=n_vpar,
            n_mu=n_mu,
            vpar_max=vpar_max,
            mu_max=mu_max,
            backend=velocity_backend,
        )
    )
    parallel = _build_gkw_cell_centered_parallel_grid(n_z, derivative_backend=parallel_backend)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=1, kx_max=kx_rhos, ky_values=(0.0,)))
    geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(q=q, shat=shat, eps=epsilon),
    )
    species = SpeciesParams(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        density_gradient=0.0,
        temperature_gradient=0.0,
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=True,
        ),
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_recurrence_velocity_model="rms",
        velocity_recurrence_rate=velocity_recurrence_rate,
        velocity_recurrence_velocity_model="rms",
    )
    ix = min(fourier.ixzero + 1, fourier.kx.shape[0] - 1)
    ix_conjugate = max(fourier.ixzero - 1, 0)
    state = jnp.zeros((n_vpar, n_mu, n_z, 3, 1), dtype=jnp.complex128)
    state = state.at[:, :, :, ix_conjugate, 0].set(-0.5j * 1.0e-4 * precompute.rhs.maxwellian[0])
    state = state.at[:, :, :, ix, 0].set(0.5j * 1.0e-4 * precompute.rhs.maxwellian[0])
    phi_initial = solve_adiabatic_electron_phi(state, precompute.field)
    initial_power = _field_power(phi_initial[:, ix, 0], geometry.w_z)

    use_filter = any(rate > 0.0 for rate in (z_modal_damping, vpar_modal_damping, mu_modal_damping))
    filter_fn = (
        build_modal_damping_filter(
            dt=dt,
            velocity_grid=velocity,
            parallel_grid=parallel,
            vpar_rate=vpar_modal_damping,
            mu_rate=mu_modal_damping,
            z_rate=z_modal_damping,
            order=modal_damping_order,
        )
        if use_filter
        else None
    )

    diagnostic_steps = max(1, int(round(diagnostic_interval / dt)))
    total_steps = int(round(t_end / dt))
    advance_diagnostic_chunk = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                diagnostic_steps,
                linear_residual,
                precompute,
                filter_fn=filter_fn,
                store_history=False,
            ).state
        )
    )
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, precompute.field)
    )
    late_power_ratios = []
    late_times = []
    current_time = 0.0
    unstable = False
    steps_done = 0
    while steps_done < total_steps:
        chunk_steps = min(diagnostic_steps, total_steps - steps_done)
        if chunk_steps == diagnostic_steps:
            state = advance_diagnostic_chunk(state)
        else:
            state = integrate_fixed_step(
                state,
                dt,
                chunk_steps,
                linear_residual,
                precompute,
                filter_fn=filter_fn,
                store_history=False,
            ).state
        steps_done += chunk_steps
        current_time = steps_done * dt
        phi = solve_phi(state)
        power_ratio = _field_power(phi[:, ix, 0], geometry.w_z) / initial_power
        amplitude_ratio = jnp.sqrt(jnp.maximum(power_ratio, 0.0))
        if not bool(jnp.isfinite(amplitude_ratio)) or float(amplitude_ratio) > amplitude_ceiling:
            unstable = True
            break
        if current_time > t_start:
            late_power_ratios.append(float(power_ratio))
            late_times.append(float(current_time))

    if unstable or not late_power_ratios:
        observed = jnp.asarray(jnp.inf, dtype=jnp.float64)
        plateau_spread = jnp.asarray(jnp.inf, dtype=jnp.float64)
    else:
        late_power = jnp.asarray(late_power_ratios, dtype=jnp.float64)
        observed = jnp.sqrt(jnp.mean(late_power))
        if late_power.shape[0] >= 4:
            split = late_power.shape[0] // 2
            first_mean = jnp.sqrt(jnp.mean(late_power[:split]))
            second_mean = jnp.sqrt(jnp.mean(late_power[split:]))
            plateau_spread = jnp.abs(second_mean - first_mean)
        else:
            late_amplitude = jnp.sqrt(jnp.maximum(late_power, 0.0))
            plateau_spread = jnp.max(late_amplitude) - jnp.min(late_amplitude)

    base = evaluate_benchmark_gate(
        observed,
        target,
        notes="",
    )
    plateau_ok = jnp.asarray(plateau_spread <= plateau_tolerance)
    passed = jnp.logical_and(base.passed, plateau_ok)
    status = "unstable" if unstable else "completed"
    notes = (
        "long-time RH plateau gate; "
        f"status={status}, t_start={t_start:g}, t_end={current_time:g}, "
        f"diagnostic_interval={diagnostic_interval:g}, "
        f"parallel_backend={parallel_backend}, "
        f"velocity_backend={velocity_backend}, "
        f"disp_par={parallel_recurrence_rate:g}, "
        f"disp_vp={velocity_recurrence_rate:g}, "
        f"z_modal_damping={z_modal_damping:g}, "
        f"vpar_modal_damping={vpar_modal_damping:g}, "
        f"mu_modal_damping={mu_modal_damping:g}, "
        f"late_mean_delta={float(plateau_spread):.6e}; "
        "RH pass requires residual and late-window mean convergence against the "
        "GKW production reference to pass"
    )
    return BenchmarkGateResult(
        target=base.target,
        observed_value=base.observed_value,
        residual=base.residual,
        cost=base.cost,
        passed=passed,
        notes=notes,
    )


def run_reduced_cyclone_base_case_gate(
    *,
    n_z: int = 16,
    n_vpar: int = 12,
    n_mu: int = 6,
    vpar_max: float = 3.0,
    mu_max: float | None = None,
    dt: float = 0.003,
    n_steps: int = 100,
    nperiod: int = 5,
    growth_window_fraction: float = 0.5,
    parallel_recurrence_rate: float = 1.0,
    parallel_backend: str = "finite_difference",
    parallel_boundary: str = "zero",
    parallel_derivative_model: str = "gkw_upwind",
    velocity_backend: str = "finite_difference",
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Run the present reduced Cyclone setup and compare selected growth."""

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, windowed_linear_growth_diagnostics

    target = target or cyclone_base_case_growth_target()
    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
        initial_profile="cosine2",
    )
    result = integrate_fixed_step(
        setup["state"],
        dt,
        n_steps,
        linear_residual,
        setup["precompute"],
        store_history=True,
    )
    phi_history = jax.vmap(
        lambda snapshot: solve_adiabatic_electron_phi(snapshot, setup["precompute"].field)
    )(result.history)
    diagnostics = windowed_linear_growth_diagnostics(
        phi_history,
        result.times,
        start_fraction=growth_window_fraction,
        w_z=setup["geometry"].w_z,
        connectivity=setup["connectivity"],
    )
    return evaluate_benchmark_gate(
        diagnostics.growth_rate[setup["selected_ky_index"]],
        target,
        notes=(
            "reduced executable CBC gate with GKW cell-centered s grid, "
            f"nperiod={nperiod}, selected ky only, late-window growth fit, "
            f"parallel_backend={parallel_backend}, velocity_backend={velocity_backend}, "
            f"parallel_boundary={parallel_boundary}, "
            f"parallel_derivative_model={parallel_derivative_model}, "
            "and GKW-scaled disp_par recurrence control; "
            "production GKW/GX agreement remains open"
        ),
    )


def run_production_cyclone_base_case_gate(
    *,
    n_z: int | None = None,
    n_vpar: int | None = None,
    n_mu: int | None = None,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int | None = None,
    n_windows: int | None = None,
    growth_window_fraction: float = 0.5,
    growth_diagnostic: str = "late_fit",
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "weighted",
    initial_profile: str | None = None,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Run the Cyclone gate with documented production controls.

    Defaults follow the Gyaradax/GKW validation case: ``ns=144``,
    ``nperiod=5``, ``nvpar=64``, ``nmu=16``, ``vpar_max=3``,
    ``disp_par=1``, and 300 diagnostic windows of 100 RK4 steps.  The
    implementation stores only per-window selected-mode amplitudes, so it can
    be used at production resolution without retaining a full field history.
    Tests call this routine with reduced overrides.
    """

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    n_z = int(metadata["n_z"] if n_z is None else n_z)
    n_vpar = int(metadata["n_vpar"] if n_vpar is None else n_vpar)
    n_mu = int(metadata["n_mu"] if n_mu is None else n_mu)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    steps_per_window = int(
        metadata["steps_per_window"] if steps_per_window is None else steps_per_window
    )
    n_windows = int(metadata["n_windows"] if n_windows is None else n_windows)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    parallel_derivative_model = str(
        metadata.get("parallel_derivative_model", "gkw_upwind")
        if parallel_derivative_model is None
        else parallel_derivative_model
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        parallel_derivative_model,
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )
    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if growth_diagnostic not in ("late_fit", "late_mean_window"):
        raise ValueError("growth_diagnostic must be 'late_fit' or 'late_mean_window'")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    state = setup["state"]
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    times = []
    log_amplitudes = []
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )

    def append_log_amplitude(time_value, state_value, accumulated_log):
        phi = solve_phi(state_value)
        amplitude = _cyclone_phi_normalization_amplitude(
            phi,
            setup,
            normalization_model=normalization_model,
        )
        floor = jnp.asarray(1.0e-300, dtype=amplitude.dtype)
        log_amplitudes.append(jnp.log(jnp.maximum(amplitude, floor)) + accumulated_log)
        times.append(float(time_value))
        return amplitude

    amplitude = append_log_amplitude(0.0, state, log_normalization)
    for window in range(n_windows):
        state = advance_window(state)
        current_time = (window + 1) * steps_per_window * dt
        amplitude = append_log_amplitude(current_time, state, log_normalization)
        if normalize_each_window:
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization

    times_array = jnp.asarray(times, dtype=jnp.float64)
    log_amplitude_array = jnp.stack(log_amplitudes)
    selected = int(setup["selected_ky_index"])
    if growth_diagnostic == "late_fit":
        fitted_growth = _fit_growth_from_log_amplitudes(
            times_array,
            log_amplitude_array,
            start_fraction=growth_window_fraction,
        )
        observed = fitted_growth[selected]
    else:
        window_growth = jnp.diff(log_amplitude_array[:, selected]) / jnp.diff(times_array)
        n_window = window_growth.shape[0]
        start = max(0, min(int(n_window * growth_window_fraction), n_window - 1))
        observed = jnp.mean(window_growth[start:])
    return evaluate_benchmark_gate(
        observed,
        target,
        notes=(
            "production-control CBC gate with GKW cell-centered s grid, "
            f"n_z={n_z}, nperiod={nperiod}, n_vpar={n_vpar}, n_mu={n_mu}, "
            f"parallel_backend={parallel_backend}, velocity_backend={velocity_backend}, "
            f"parallel_boundary={parallel_boundary}, "
            f"parallel_derivative_model={parallel_derivative_model}, "
            f"velocity_recurrence_rate={velocity_recurrence_rate:g}, "
            f"steps_per_window={steps_per_window}, n_windows={n_windows}, "
            f"normalize_each_window={normalize_each_window}, "
            f"normalization_model={normalization_model}, "
            f"initial_profile={initial_profile}, "
            f"growth_diagnostic={growth_diagnostic}; "
            "production GKW/GX agreement remains open until this gate passes"
        ),
    )


def run_cyclone_base_case_trace(
    *,
    n_z: int = 16,
    n_vpar: int = 12,
    n_mu: int = 6,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 4,
    n_windows: int = 4,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "weighted",
    initial_profile: str | None = None,
    target: BenchmarkTarget | None = None,
) -> CycloneTrace:
    """Record selected-mode CBC evolution at fixed diagnostic windows."""

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")
    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    parallel_derivative_model = str(
        metadata.get("parallel_derivative_model", "gkw_upwind")
        if parallel_derivative_model is None
        else parallel_derivative_model
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        parallel_derivative_model,
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )
    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    selected = int(setup["selected_ky_index"])
    state = setup["state"]
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)

    times = []
    raw_amplitudes = []
    physical_amplitudes = []
    window_growths = []
    fitted_growths = []
    phi_norms = []
    state_norms = []
    rhs_norms = []
    log_normalizations = []

    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    def append_snapshot(time_value, state_value, accumulated_log, previous_physical_value):
        phi = solve_phi(state_value)
        amplitude = _cyclone_phi_normalization_amplitude(
            phi,
            setup,
            normalization_model=normalization_model,
        )
        rhs_value = linear_residual(state_value, precomputed=setup["precompute"], phi=phi)
        floor = jnp.asarray(1.0e-300, dtype=amplitude.dtype)
        raw = amplitude[selected]
        physical_log = jnp.log(jnp.maximum(raw, floor)) + accumulated_log[selected]
        physical = jnp.exp(physical_log)
        if previous_physical_value is None:
            window_growth = jnp.asarray(0.0, dtype=jnp.float64)
        else:
            window_growth = (
                jnp.log(jnp.maximum(physical, floor))
                - jnp.log(jnp.maximum(previous_physical_value, floor))
            ) / (steps_per_window * dt)
        times.append(float(time_value))
        raw_amplitudes.append(raw)
        physical_amplitudes.append(physical)
        window_growths.append(window_growth)
        log_normalizations.append(accumulated_log[selected])
        phi_norms.append(_field_amplitude(phi[:, setup["fourier"].ixzero, selected]))
        state_norms.append(_l2_norm(state_value))
        rhs_norms.append(_l2_norm(rhs_value))
        fitted_growths.append(
            _trace_fitted_growth(
                jnp.asarray(times, dtype=jnp.float64),
                jnp.asarray(physical_amplitudes, dtype=jnp.float64),
            )
        )
        return amplitude, physical

    amplitude, previous_physical = append_snapshot(0.0, state, log_normalization, None)
    for window in range(n_windows):
        state = advance_window(state)
        current_time = (window + 1) * steps_per_window * dt
        amplitude, previous_physical = append_snapshot(
            current_time,
            state,
            log_normalization,
            previous_physical,
        )
        if normalize_each_window:
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization

    return CycloneTrace(
        times=jnp.asarray(times, dtype=jnp.float64),
        raw_amplitude=jnp.asarray(raw_amplitudes, dtype=jnp.float64),
        physical_amplitude=jnp.asarray(physical_amplitudes, dtype=jnp.float64),
        window_growth=jnp.asarray(window_growths, dtype=jnp.float64),
        fitted_growth=jnp.asarray(fitted_growths, dtype=jnp.float64),
        phi_norm=jnp.asarray(phi_norms, dtype=jnp.float64),
        state_norm=jnp.asarray(state_norms, dtype=jnp.float64),
        rhs_norm=jnp.asarray(rhs_norms, dtype=jnp.float64),
        log_normalization=jnp.asarray(log_normalizations, dtype=jnp.float64),
        source="stellarator_gk",
        notes=(
            "windowed CBC trace with selected ky, raw/physical amplitudes, "
            "window growth, fitted growth, phi norm, state norm, and RHS norm; "
            f"initial_profile={initial_profile}, "
            f"parallel_derivative_model={parallel_derivative_model}, "
            f"velocity_recurrence_rate={velocity_recurrence_rate:g}, "
            f"normalization_model={normalization_model}"
        ),
    )


def compare_cyclone_base_case_traces(
    observed: CycloneTrace,
    reference: CycloneTrace,
    *,
    tolerance: float = 1.0e-10,
    field_names: tuple[str, ...] | None = None,
) -> CycloneTraceComparisonReport:
    """Compare two selected-``ky`` CBC traces field by field."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if tuple(observed.times.shape) != tuple(reference.times.shape):
        raise ValueError("traces must have the same number of time samples")
    allowed_fields = _cyclone_trace_comparison_fields()
    if field_names is None:
        field_names = allowed_fields
    else:
        field_names = tuple(field_names)
        unknown = tuple(name for name in field_names if name not in allowed_fields)
        if unknown:
            raise ValueError(f"unknown trace fields: {unknown}")
        if not field_names:
            raise ValueError("field_names must not be empty")
    errors = []
    for name in field_names:
        errors.append(
            _max_abs_error(
                _cyclone_trace_comparison_field(observed, name),
                _cyclone_trace_comparison_field(reference, name),
            )
        )
    field_errors = jnp.asarray(errors, dtype=jnp.float64)
    max_abs_error = jnp.max(field_errors)
    return CycloneTraceComparisonReport(
        field_errors=field_errors,
        max_abs_error=max_abs_error,
        passed=max_abs_error <= tolerance,
        field_names=field_names,
        notes=(
            f"observed={observed.source}; reference={reference.source}; trace-level CBC comparison"
        ),
    )


def write_cyclone_trace_csv(path, trace: CycloneTrace) -> None:
    """Write a ``CycloneTrace`` to the project CSV interchange format."""

    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_cyclone_trace_csv_columns())
        for row in zip(
            *(getattr(trace, name) for name in CycloneTrace._dynamic_fields),
            strict=True,
        ):
            writer.writerow([float(value) for value in row])


def load_cyclone_trace_csv(path, *, source: str | None = None, notes: str = "") -> CycloneTrace:
    """Load a ``CycloneTrace`` from the project CSV interchange format."""

    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Cyclone trace CSV contains no rows")
    column_map = _cyclone_trace_csv_column_map(rows[0])
    missing = tuple(name for name in CycloneTrace._dynamic_fields if name not in column_map)
    if missing:
        raise ValueError(f"Cyclone trace CSV is missing columns: {missing}")
    values = {
        name: jnp.asarray([float(row[column_map[name]]) for row in rows], dtype=jnp.float64)
        for name in CycloneTrace._dynamic_fields
    }
    return CycloneTrace(
        **values,
        source=source or str(path),
        notes=notes,
    )


def load_gkw_time_dat_trace(
    path,
    *,
    source: str | None = None,
    notes: str = "",
    amplitude0: float = 1.0,
    initial_time: float = 0.0,
) -> CycloneTrace:
    """Load a linear GKW ``time.dat`` growth history into ``CycloneTrace``.

    GKW writes ``time`` and ``growth_rate`` for linear runs, with an optional
    third column for either real frequency or normalization depending on input
    flags.  The compact file does not contain field or state norms, so those
    unsupported diagnostics are filled with zeros and should not be used as
    parity fields for GKW ``time.dat`` comparisons.
    """

    if amplitude0 <= 0.0:
        raise ValueError("amplitude0 must be positive")
    rows: list[tuple[float, float]] = []
    path = Path(path)
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            raise ValueError("GKW time.dat rows must contain at least time and growth")
        rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError("GKW time.dat contains no rows")

    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("GKW time.dat contains non-finite values")
    times_np = data[:, 0]
    growth_np = data[:, 1]
    if times_np[0] < initial_time or np.any(np.diff(times_np) <= 0.0):
        raise ValueError("GKW time.dat times must be strictly increasing")

    amplitudes = []
    previous_time = float(initial_time)
    amplitude = float(amplitude0)
    for time, growth in rows:
        amplitude *= float(np.exp(growth * (time - previous_time)))
        amplitudes.append(amplitude)
        previous_time = time

    times = jnp.asarray(times_np, dtype=jnp.float64)
    physical_amplitude = jnp.asarray(amplitudes, dtype=jnp.float64)
    fitted_growth = jnp.asarray(
        [
            _trace_fitted_growth(times[: index + 1], physical_amplitude[: index + 1])
            for index in range(times.shape[0])
        ],
        dtype=jnp.float64,
    )
    zeros = jnp.zeros_like(times)
    trace_notes = "GKW time.dat trace; field/state/RHS norms unavailable"
    if notes:
        trace_notes = f"{trace_notes}; {notes}"
    return CycloneTrace(
        times=times,
        raw_amplitude=physical_amplitude,
        physical_amplitude=physical_amplitude,
        window_growth=jnp.asarray(growth_np, dtype=jnp.float64),
        fitted_growth=fitted_growth,
        phi_norm=zeros,
        state_norm=zeros,
        rhs_norm=zeros,
        log_normalization=zeros,
        source=source or str(path),
        notes=trace_notes,
    )


def load_gkw_parallel_phi_trace(
    path,
    *,
    time_path=None,
    times=None,
    z=None,
    source: str | None = None,
    notes: str = "",
) -> ParallelPhiTrace:
    """Load GKW ``parallel_phi.dat`` into a parallel ``|phi|^2`` trace.

    GKW writes one row per large output step.  Each row contains
    ``sum_{kx,ky} |phi(kx,ky,s)|^2 / (n_kx n_ky)`` on the global parallel
    grid.  The compact file does not store times or grid coordinates, so they
    may be supplied explicitly or, for times, read from the matching
    ``time.dat`` file.
    """

    if time_path is not None and times is not None:
        raise ValueError("supply either time_path or times, not both")
    path = Path(path)
    rows = _numeric_rows(path)
    if not rows:
        raise ValueError("GKW parallel_phi.dat contains no rows")
    n_z = len(rows[0])
    if n_z == 0 or any(len(row) != n_z for row in rows):
        raise ValueError("GKW parallel_phi.dat rows must have a consistent nonzero length")
    phi_power_np = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(phi_power_np)):
        raise ValueError("GKW parallel_phi.dat contains non-finite values")
    if np.any(phi_power_np < 0.0):
        raise ValueError("GKW parallel_phi.dat contains negative |phi|^2 values")
    n_time = phi_power_np.shape[0]

    if time_path is not None:
        times_np = _gkw_time_dat_times(time_path)
    elif times is not None:
        times_np = np.asarray(times, dtype=float)
    else:
        times_np = np.arange(n_time, dtype=float)
    if times_np.shape != (n_time,):
        raise ValueError("times must have one entry per parallel_phi row")
    if n_time > 1 and np.any(np.diff(times_np) <= 0.0):
        raise ValueError("parallel phi trace times must be strictly increasing")

    if z is None:
        z_np = np.arange(n_z, dtype=float)
    else:
        z_np = np.asarray(z, dtype=float)
    if z_np.shape != (n_z,):
        raise ValueError("z must have one entry per parallel_phi column")

    trace_notes = "GKW parallel_phi.dat trace; values are mode-averaged |phi|^2"
    if notes:
        trace_notes = f"{trace_notes}; {notes}"
    return ParallelPhiTrace(
        times=times_np,
        z=z_np,
        phi_power=phi_power_np,
        source=source or str(path),
        notes=trace_notes,
    )


def load_gkw_velocity_space_slice(
    distr1_path,
    distr2_path,
    distr3_path,
    distr4_path,
    *,
    time_path=None,
    source: str | None = None,
    notes: str = "",
) -> GkwVelocitySpaceSlice:
    """Load GKW final ``distr*.dat`` velocity-space diagnostic files.

    GKW writes rows in ``(n_mu,n_vpar)`` order after selecting the parallel
    point with maximum ``|phi|``.  The files store
    ``v_parallel``, ``sqrt(2 B mu)``, and the imaginary/real parts of
    ``f * intmu * intvp / phi``.
    """

    vpar = _load_gkw_distr_array(distr1_path, "distr1.dat")
    vperp = _load_gkw_distr_array(distr2_path, "distr2.dat")
    imag_part = _load_gkw_distr_array(distr3_path, "distr3.dat")
    real_part = _load_gkw_distr_array(distr4_path, "distr4.dat")
    shape = vpar.shape
    for name, values in (
        ("distr2.dat", vperp),
        ("distr3.dat", imag_part),
        ("distr4.dat", real_part),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} shape must match distr1.dat")

    time = np.nan
    if time_path is not None:
        time = float(_gkw_time_dat_times(time_path)[-1])
    slice_notes = "GKW distr*.dat velocity-space slice"
    if notes:
        slice_notes = f"{slice_notes}; {notes}"
    return GkwVelocitySpaceSlice(
        vpar=vpar,
        vperp=vperp,
        real_part=real_part,
        imag_part=imag_part,
        time=time,
        source=source or str(Path(distr1_path).parent),
        notes=slice_notes,
    )


def load_gkw_velocity_space_slice_series(
    directory,
    *,
    time_path=None,
    source: str | None = None,
    notes: str = "",
) -> GkwVelocitySpaceSliceSeries:
    """Load suffixed GKW ``distr*_<ntotstep>.dat`` velocity-slice snapshots."""

    directory = Path(directory)
    snapshots = _gkw_distr_snapshot_paths(directory)
    if not snapshots:
        raise ValueError(f"no GKW distr1_*.dat snapshots found in {directory}")
    indices = []
    vpar_values = []
    vperp_values = []
    real_values = []
    imag_values = []
    for index, paths in snapshots:
        indices.append(index)
        vpar = _load_gkw_distr_array(paths[0], paths[0].name)
        vperp = _load_gkw_distr_array(paths[1], paths[1].name)
        imag_part = _load_gkw_distr_array(paths[2], paths[2].name)
        real_part = _load_gkw_distr_array(paths[3], paths[3].name)
        for name, values in (
            (paths[1].name, vperp),
            (paths[2].name, imag_part),
            (paths[3].name, real_part),
        ):
            if values.shape != vpar.shape:
                raise ValueError(f"{name} shape must match {paths[0].name}")
        vpar_values.append(vpar)
        vperp_values.append(vperp)
        real_values.append(real_part)
        imag_values.append(imag_part)

    index_array = np.asarray(indices, dtype=np.int32)
    if time_path is None:
        times = np.full(index_array.shape, np.nan, dtype=float)
    else:
        times = _gkw_times_for_snapshot_indices(_gkw_time_dat_times(time_path), index_array)
    series_notes = "GKW distr*_<ntotstep>.dat velocity-space slice series"
    if notes:
        series_notes = f"{series_notes}; {notes}"
    return GkwVelocitySpaceSliceSeries(
        times=times,
        snapshot_indices=index_array,
        vpar=np.stack(vpar_values),
        vperp=np.stack(vperp_values),
        real_part=np.stack(real_values),
        imag_part=np.stack(imag_values),
        source=source or str(directory),
        notes=series_notes,
    )


def run_cyclone_base_case_parallel_phi_trace(
    *,
    n_z: int = 16,
    n_vpar: int = 12,
    n_mu: int = 6,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 4,
    n_windows: int = 4,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    initial_profile: str | None = None,
    include_initial: bool = False,
    physical_power: bool = False,
    normalization_model: str = "weighted",
    target: BenchmarkTarget | None = None,
) -> ParallelPhiTrace:
    """Run the selected-``ky`` CBC setup and record ``|phi(z)|^2`` profiles.

    The default output cadence mirrors GKW ``parallel_phi.dat``: only the
    post-window profiles are stored.  ``normalization_model='weighted'`` uses
    the solver's quadrature-weighted mode-chain amplitude, while
    ``'gkw_unweighted'`` uses GKW's unweighted field norm for this single-mode
    diagnostic.  ``physical_power=True`` re-applies the accumulated scalar
    factor.
    """

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    parallel_derivative_model = str(
        metadata.get("parallel_derivative_model", "gkw_upwind")
        if parallel_derivative_model is None
        else parallel_derivative_model
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        parallel_derivative_model,
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    times_out = []
    profiles = []

    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    def append_profile(time_value, phi_value, accumulated_log):
        power = jnp.abs(phi_value[:, ixzero, selected]) ** 2
        if physical_power:
            power = power * jnp.exp(2.0 * accumulated_log[selected])
        times_out.append(float(time_value))
        profiles.append(power)

    if include_initial:
        append_profile(0.0, solve_phi(state), log_normalization)

    for window in range(n_windows):
        state = advance_window(state)
        current_time = (window + 1) * steps_per_window * dt
        phi = solve_phi(state)
        if normalize_each_window:
            if normalization_model == "weighted":
                amplitude = mode_chain_amplitude(
                    phi,
                    w_z=setup["geometry"].w_z,
                    connectivity=setup["connectivity"],
                )
            else:
                amplitude = jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)))
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            scale = jnp.maximum(amplitude[selected], jnp.asarray(1.0e-300, dtype=amplitude.dtype))
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = phi / scale
        append_profile(current_time, phi, log_normalization)

    return ParallelPhiTrace(
        times=jnp.asarray(times_out, dtype=jnp.float64),
        z=setup["parallel"].z,
        phi_power=jnp.stack(profiles),
        source="stellarator_gk",
        notes=(
            "selected-ky CBC parallel |phi|^2 profile trace; "
            f"initial_profile={initial_profile}, "
            f"parallel_derivative_model={parallel_derivative_model}, "
            f"velocity_recurrence_rate={velocity_recurrence_rate:g}, "
            f"normalize_each_window={normalize_each_window}, "
            f"physical_power={physical_power}, "
            f"normalization_model={normalization_model}"
        ),
    )


def _cyclone_velocity_space_slice_from_state(
    setup,
    state,
    phi,
    *,
    selected_ky_index: int,
    ixzero: int,
    time: float,
    source: str,
    notes: str,
) -> GkwVelocitySpaceSlice:
    mode_phi = phi[:, ixzero, selected_ky_index]
    peak_z_index = int(np.asarray(jnp.argmax(jnp.abs(mode_phi))))
    peak_phi = mode_phi[peak_z_index]
    floor = jnp.asarray(1.0e-300, dtype=jnp.float64)
    peak_phi = jnp.where(jnp.abs(peak_phi) > floor, peak_phi, floor + 0.0j)
    weighted = (
        state[:, :, peak_z_index, ixzero, selected_ky_index]
        * setup["velocity"].w_vpar[:, None]
        * setup["velocity"].w_mu[None, :]
        / peak_phi
    )
    weighted_mu_vpar = jnp.transpose(weighted)
    vpar = jnp.broadcast_to(setup["velocity"].vpar[None, :], weighted_mu_vpar.shape)
    vperp = jnp.sqrt(
        2.0
        * setup["geometry"].B[peak_z_index]
        * jnp.maximum(setup["velocity"].mu[:, None], 0.0)
    )
    vperp = jnp.broadcast_to(vperp, weighted_mu_vpar.shape)
    return GkwVelocitySpaceSlice(
        vpar=vpar,
        vperp=vperp,
        real_part=jnp.real(weighted_mu_vpar),
        imag_part=jnp.imag(weighted_mu_vpar),
        time=time,
        peak_z=setup["parallel"].z[peak_z_index],
        peak_z_index=peak_z_index,
        source=source,
        notes=notes,
    )


def _velocity_series_snapshot_indices(
    snapshot_indices: tuple[int, ...] | None,
    *,
    steps_per_window: int,
    n_windows: int,
) -> tuple[int, ...]:
    if snapshot_indices is None:
        return tuple((window + 1) * steps_per_window for window in range(n_windows))
    requested = tuple(sorted({int(index) for index in snapshot_indices}))
    if not requested:
        raise ValueError("snapshot_indices must not be empty")
    max_step = n_windows * steps_per_window
    for index in requested:
        if index < 1 or index > max_step:
            raise ValueError("snapshot_indices must lie within the simulated step range")
        if index % steps_per_window != 0:
            raise ValueError("snapshot_indices must land on diagnostic-window boundaries")
    return requested


def run_cyclone_base_case_velocity_space_slice(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine2",
    target: BenchmarkTarget | None = None,
) -> GkwVelocitySpaceSlice:
    """Run the selected-``ky`` CBC setup and return the GKW ``distr*.dat`` slice."""

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    parallel_derivative_model = str(
        metadata.get("parallel_derivative_model", "gkw_upwind")
        if parallel_derivative_model is None
        else parallel_derivative_model
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        parallel_derivative_model,
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _window in range(n_windows):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            amplitude = _cyclone_phi_normalization_amplitude(
                phi,
                setup,
                normalization_model=normalization_model,
            )
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = solve_phi(state)

    final_time = n_windows * steps_per_window * dt
    return _cyclone_velocity_space_slice_from_state(
        setup,
        state,
        phi,
        selected_ky_index=selected,
        ixzero=ixzero,
        time=final_time,
        source="stellarator_gk",
        notes=(
            "selected-ky CBC velocity-space slice with GKW distr*.dat normalization; "
            f"initial_profile={initial_profile}, "
            f"parallel_derivative_model={parallel_derivative_model}, "
            f"velocity_recurrence_rate={velocity_recurrence_rate:g}, "
            f"normalize_each_window={normalize_each_window}, "
            f"normalization_model={normalization_model}"
        ),
    )


def run_cyclone_base_case_velocity_space_slice_series(
    *,
    snapshot_indices: tuple[int, ...] | None = None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine2",
    target: BenchmarkTarget | None = None,
) -> GkwVelocitySpaceSliceSeries:
    """Run selected-``ky`` CBC and sample GKW-normalized velocity slices."""

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")
    requested = _velocity_series_snapshot_indices(
        snapshot_indices,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
    )
    requested_set = set(requested)

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    parallel_derivative_model = str(
        metadata.get("parallel_derivative_model", "gkw_upwind")
        if parallel_derivative_model is None
        else parallel_derivative_model
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        parallel_derivative_model,
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    snapshots: list[GkwVelocitySpaceSlice] = []
    phi = solve_phi(state)
    for window in range(n_windows):
        state = advance_window(state)
        current_step = (window + 1) * steps_per_window
        phi = solve_phi(state)
        if normalize_each_window:
            amplitude = _cyclone_phi_normalization_amplitude(
                phi,
                setup,
                normalization_model=normalization_model,
            )
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = solve_phi(state)
        if current_step in requested_set:
            snapshots.append(
                _cyclone_velocity_space_slice_from_state(
                    setup,
                    state,
                    phi,
                    selected_ky_index=selected,
                    ixzero=ixzero,
                    time=current_step * dt,
                    source="stellarator_gk",
                    notes=(
                        "selected-ky CBC velocity-space slice series with "
                        "GKW distr*_<ntotstep>.dat normalization; "
                        f"snapshot_index={current_step}, "
                        f"initial_profile={initial_profile}, "
                        f"parallel_derivative_model={parallel_derivative_model}, "
                        f"velocity_recurrence_rate={velocity_recurrence_rate:g}, "
                        f"normalize_each_window={normalize_each_window}, "
                        f"normalization_model={normalization_model}"
                    ),
                )
            )

    if len(snapshots) != len(requested):
        raise RuntimeError("internal error: did not collect every requested snapshot")
    return GkwVelocitySpaceSliceSeries(
        times=jnp.asarray([snapshot.time for snapshot in snapshots], dtype=jnp.float64),
        snapshot_indices=jnp.asarray(requested, dtype=jnp.int32),
        vpar=jnp.stack([snapshot.vpar for snapshot in snapshots]),
        vperp=jnp.stack([snapshot.vperp for snapshot in snapshots]),
        real_part=jnp.stack([snapshot.real_part for snapshot in snapshots]),
        imag_part=jnp.stack([snapshot.imag_part for snapshot in snapshots]),
        source="stellarator_gk",
        notes=(
            "selected-ky CBC velocity-space slice series with "
            "GKW distr*_<ntotstep>.dat normalization"
        ),
    )


def run_cyclone_base_case_cosin2_vpar_odd_sign_audit(
    *,
    reference_slice: GkwVelocitySpaceSlice | None = None,
    gkw_distr1_path=None,
    gkw_distr2_path=None,
    gkw_distr3_path=None,
    gkw_distr4_path=None,
    gkw_time_path=None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    target: BenchmarkTarget | None = None,
) -> CycloneVparOddSignAudit:
    """Compare controlled odd-``v_parallel`` RHS sign variants to GKW slices."""

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    if reference_slice is None:
        gkw_distr1_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat"
            if gkw_distr1_path is None
            else gkw_distr1_path
        )
        gkw_distr2_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat"
            if gkw_distr2_path is None
            else gkw_distr2_path
        )
        gkw_distr3_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat"
            if gkw_distr3_path is None
            else gkw_distr3_path
        )
        gkw_distr4_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat"
            if gkw_distr4_path is None
            else gkw_distr4_path
        )
        gkw_time_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"
            if gkw_time_path is None
            else gkw_time_path
        )
        reference_slice = load_gkw_velocity_space_slice(
            gkw_distr1_path,
            gkw_distr2_path,
            gkw_distr3_path,
            gkw_distr4_path,
            time_path=gkw_time_path,
            source="gkw:cosin2:distr*.dat",
            notes="patched selected-ky production-control cosin2 final output",
        )

    variants = (
        ("baseline", 1.0, 1.0),
        ("flip_igh", -1.0, 1.0),
        ("flip_parallel_field_drive", 1.0, -1.0),
        ("flip_igh_and_parallel_field_drive", -1.0, -1.0),
    )
    direct_max = []
    direct_l2 = []
    direct_relative = []
    best_layout_max = []
    best_layout_l2 = []
    peak_z = []
    best_layout_names = []
    for _name, igh_sign, field_sign in variants:
        observed = _run_cyclone_base_case_velocity_space_slice_with_odd_signs(
            n_z=n_z,
            n_vpar=n_vpar,
            n_mu=n_mu,
            vpar_max=vpar_max,
            mu_max=mu_max,
            dt=dt,
            nperiod=nperiod,
            steps_per_window=steps_per_window,
            n_windows=n_windows,
            parallel_recurrence_rate=parallel_recurrence_rate,
            velocity_recurrence_rate=velocity_recurrence_rate,
            parallel_backend=parallel_backend,
            parallel_boundary=parallel_boundary,
            velocity_backend=velocity_backend,
            normalize_each_window=normalize_each_window,
            normalization_model=normalization_model,
            target=target,
            igh_sign=igh_sign,
            parallel_field_drive_sign=field_sign,
        )
        direct_report = audit_cyclone_velocity_space_slice(
            observed,
            reference_slice,
            tolerance=1.0,
            grid_tolerance=1.0e-4,
        )
        convention_report = audit_velocity_space_slice_conventions(observed, reference_slice)
        direct_max.append(direct_report.complex_max_abs_error)
        direct_l2.append(direct_report.complex_l2_error)
        direct_relative.append(direct_report.complex_relative_l2_error)
        best_layout_max.append(convention_report.best_max_abs_error)
        best_layout_l2.append(convention_report.best_l2_error)
        peak_z.append(observed.peak_z)
        best_layout_names.append(
            convention_report.variant_names[int(convention_report.best_variant_index)]
        )

    direct_max_array = jnp.asarray(direct_max, dtype=jnp.float64)
    best_layout_max_array = jnp.asarray(best_layout_max, dtype=jnp.float64)
    field_flip_index = 2
    improvement = direct_max_array[0] / jnp.maximum(
        direct_max_array[field_flip_index],
        jnp.asarray(1.0e-300, dtype=direct_max_array.dtype),
    )
    return CycloneVparOddSignAudit(
        direct_max_abs_errors=direct_max_array,
        direct_l2_errors=jnp.asarray(direct_l2, dtype=jnp.float64),
        direct_relative_l2_errors=jnp.asarray(direct_relative, dtype=jnp.float64),
        best_layout_max_abs_errors=best_layout_max_array,
        best_layout_l2_errors=jnp.asarray(best_layout_l2, dtype=jnp.float64),
        peak_z_values=jnp.asarray(peak_z, dtype=jnp.float64),
        best_direct_variant_index=jnp.argmin(direct_max_array),
        best_layout_variant_index=jnp.argmin(best_layout_max_array),
        baseline_direct_max_abs_error=direct_max_array[0],
        field_flip_direct_max_abs_error=direct_max_array[field_flip_index],
        direct_improvement_factor=improvement,
        rhs_variant_names=tuple(name for name, _igh_sign, _field_sign in variants),
        best_layout_names=tuple(best_layout_names),
        notes=(
            "controlled odd-vpar RHS sign audit; variants flip only the fused "
            "linear_terms.F90:igh Term I/IV block and/or vpgrphi_3_newbc Term VII; "
            "GKW velocity columns use one-based k=1..N with "
            "vpgr=-vmax+(k-1/2)*dv"
        ),
    )


def run_cyclone_base_case_cosin2_term_vii_field_convention_audit(
    *,
    reference_slice: GkwVelocitySpaceSlice | None = None,
    gkw_distr1_path=None,
    gkw_distr2_path=None,
    gkw_distr3_path=None,
    gkw_distr4_path=None,
    gkw_time_path=None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    target: BenchmarkTarget | None = None,
) -> CycloneTermVIIFieldConventionAudit:
    """Audit whether the selected-ky gap is a global or Term VII field convention."""

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    if reference_slice is None:
        gkw_distr1_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat"
            if gkw_distr1_path is None
            else gkw_distr1_path
        )
        gkw_distr2_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat"
            if gkw_distr2_path is None
            else gkw_distr2_path
        )
        gkw_distr3_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat"
            if gkw_distr3_path is None
            else gkw_distr3_path
        )
        gkw_distr4_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat"
            if gkw_distr4_path is None
            else gkw_distr4_path
        )
        gkw_time_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"
            if gkw_time_path is None
            else gkw_time_path
        )
        reference_slice = load_gkw_velocity_space_slice(
            gkw_distr1_path,
            gkw_distr2_path,
            gkw_distr3_path,
            gkw_distr4_path,
            time_path=gkw_time_path,
            source="gkw:cosin2:distr*.dat",
            notes="patched selected-ky production-control cosin2 final output",
        )

    variants = (
        ("baseline", "identity", "identity", "identity"),
        ("flip_all_field_terms", "negative", "negative", "negative"),
        ("flip_term_v_and_viii_only", "negative", "identity", "negative"),
        ("flip_term_vii_only", "identity", "negative", "identity"),
        ("conjugate_all_field_terms", "conjugate", "conjugate", "conjugate"),
        ("conjugate_term_vii_only", "identity", "conjugate", "identity"),
        (
            "negative_conjugate_all_field_terms",
            "negative_conjugate",
            "negative_conjugate",
            "negative_conjugate",
        ),
        (
            "negative_conjugate_term_vii_only",
            "identity",
            "negative_conjugate",
            "identity",
        ),
    )
    direct_max = []
    direct_l2 = []
    direct_relative = []
    best_layout_max = []
    best_layout_l2 = []
    peak_z = []
    best_layout_names = []
    for name, term_v_variant, term_vii_variant, term_viii_variant in variants:
        observed = _run_cyclone_base_case_velocity_space_slice_with_odd_signs(
            n_z=n_z,
            n_vpar=n_vpar,
            n_mu=n_mu,
            vpar_max=vpar_max,
            mu_max=mu_max,
            dt=dt,
            nperiod=nperiod,
            steps_per_window=steps_per_window,
            n_windows=n_windows,
            parallel_recurrence_rate=parallel_recurrence_rate,
            velocity_recurrence_rate=velocity_recurrence_rate,
            parallel_backend=parallel_backend,
            parallel_boundary=parallel_boundary,
            velocity_backend=velocity_backend,
            normalize_each_window=normalize_each_window,
            normalization_model=normalization_model,
            target=target,
            igh_sign=1.0,
            parallel_field_drive_sign=1.0,
            term_v_phi_variant=term_v_variant,
            term_vii_phi_variant=term_vii_variant,
            term_viii_phi_variant=term_viii_variant,
            source_label=f"term_vii_field_convention:{name}",
        )
        direct_report = audit_cyclone_velocity_space_slice(
            observed,
            reference_slice,
            tolerance=1.0,
            grid_tolerance=1.0e-4,
        )
        convention_report = audit_velocity_space_slice_conventions(observed, reference_slice)
        direct_max.append(direct_report.complex_max_abs_error)
        direct_l2.append(direct_report.complex_l2_error)
        direct_relative.append(direct_report.complex_relative_l2_error)
        best_layout_max.append(convention_report.best_max_abs_error)
        best_layout_l2.append(convention_report.best_l2_error)
        peak_z.append(observed.peak_z)
        best_layout_names.append(
            convention_report.variant_names[int(convention_report.best_variant_index)]
        )

    direct_max_array = jnp.asarray(direct_max, dtype=jnp.float64)
    best_layout_max_array = jnp.asarray(best_layout_max, dtype=jnp.float64)
    term_vii_only_index = 3
    improvement = direct_max_array[0] / jnp.maximum(
        direct_max_array[term_vii_only_index],
        jnp.asarray(1.0e-300, dtype=direct_max_array.dtype),
    )
    return CycloneTermVIIFieldConventionAudit(
        direct_max_abs_errors=direct_max_array,
        direct_l2_errors=jnp.asarray(direct_l2, dtype=jnp.float64),
        direct_relative_l2_errors=jnp.asarray(direct_relative, dtype=jnp.float64),
        best_layout_max_abs_errors=best_layout_max_array,
        best_layout_l2_errors=jnp.asarray(best_layout_l2, dtype=jnp.float64),
        peak_z_values=jnp.asarray(peak_z, dtype=jnp.float64),
        best_direct_variant_index=jnp.argmin(direct_max_array),
        best_layout_variant_index=jnp.argmin(best_layout_max_array),
        baseline_direct_max_abs_error=direct_max_array[0],
        term_vii_only_direct_max_abs_error=direct_max_array[term_vii_only_index],
        all_field_direct_max_abs_error=direct_max_array[1],
        nonparallel_field_direct_max_abs_error=direct_max_array[2],
        term_vii_only_improvement_factor=improvement,
        variant_names=tuple(name for name, _term_v, _term_vii, _term_viii in variants),
        best_layout_names=tuple(best_layout_names),
        term_v_phi_variants=tuple(term_v for _name, term_v, _term_vii, _term_viii in variants),
        term_vii_phi_variants=tuple(
            term_vii for _name, _term_v, term_vii, _term_viii in variants
        ),
        term_viii_phi_variants=tuple(
            term_viii for _name, _term_v, _term_vii, term_viii in variants
        ),
        notes=(
            "field-variable convention audit for Terms V, VII, and VIII; "
            "dist.F90::get_phi stores fdis(iphi) directly, so this distinguishes "
            "global phi sign/conjugation from an isolated vpgrphi_3_newbc Term VII convention"
        ),
    )


_FIELD_PHI_VARIANTS = ("identity", "negative", "conjugate", "negative_conjugate")


def _validate_field_phi_variant(variant: str) -> str:
    if variant not in _FIELD_PHI_VARIANTS:
        raise ValueError(f"unsupported phi variant {variant!r}; expected one of {_FIELD_PHI_VARIANTS}")
    return variant


def _apply_field_phi_variant(phi, variant: str):
    variant = _validate_field_phi_variant(variant)
    if variant == "identity":
        return phi
    if variant == "negative":
        return -phi
    if variant == "conjugate":
        return jnp.conj(phi)
    return -jnp.conj(phi)


def _run_cyclone_base_case_velocity_space_slice_with_odd_signs(
    *,
    n_z: int,
    n_vpar: int,
    n_mu: int,
    vpar_max: float | None,
    mu_max: float | None,
    dt: float | None,
    nperiod: int | None,
    steps_per_window: int,
    n_windows: int,
    parallel_recurrence_rate: float | None,
    velocity_recurrence_rate: float | None,
    parallel_backend: str | None,
    parallel_boundary: str | None,
    velocity_backend: str | None,
    normalize_each_window: bool,
    normalization_model: str,
    target: BenchmarkTarget | None,
    igh_sign: float,
    parallel_field_drive_sign: float,
    term_v_phi_variant: str = "identity",
    term_vii_phi_variant: str = "identity",
    term_viii_phi_variant: str = "identity",
    source_label: str | None = None,
) -> GkwVelocitySpaceSlice:
    from .physics import (
        dissipation,
        drift_field_drive,
        equilibrium_drive,
        gkw_igh_streaming_mirror,
        gkw_parallel_field_drive,
        magnetic_drift_advection,
        solve_adiabatic_electron_phi,
    )
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    term_v_phi_variant = _validate_field_phi_variant(term_v_phi_variant)
    term_vii_phi_variant = _validate_field_phi_variant(term_vii_phi_variant)
    term_viii_phi_variant = _validate_field_phi_variant(term_viii_phi_variant)

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        "gkw_igh",
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_igh",
        velocity_backend=velocity_backend,
        initial_profile="cosine2",
    )
    rhs = setup["precompute"].rhs
    field = setup["precompute"].field
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(lambda state_value: solve_adiabatic_electron_phi(state_value, field))

    def residual_with_signs(state_value, precomputed=None, phi=None):
        del precomputed
        phi_value = solve_adiabatic_electron_phi(state_value, field) if phi is None else phi
        phi_term_v = _apply_field_phi_variant(phi_value, term_v_phi_variant)
        phi_term_vii = _apply_field_phi_variant(phi_value, term_vii_phi_variant)
        phi_term_viii = _apply_field_phi_variant(phi_value, term_viii_phi_variant)
        return (
            igh_sign * gkw_igh_streaming_mirror(state_value, rhs)
            + magnetic_drift_advection(state_value, rhs.magnetic_drift_frequency)
            + equilibrium_drive(phi_term_v, rhs)
            + parallel_field_drive_sign * gkw_parallel_field_drive(phi_term_vii, rhs)
            + drift_field_drive(phi_term_viii, rhs)
            + dissipation(state_value, rhs.perpendicular_damping)
        )

    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                residual_with_signs,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _window in range(n_windows):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            amplitude = _cyclone_phi_normalization_amplitude(
                phi,
                setup,
                normalization_model=normalization_model,
            )
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = solve_phi(state)

    mode_phi = phi[:, ixzero, selected]
    peak_z_index = int(np.asarray(jnp.argmax(jnp.abs(mode_phi))))
    peak_phi = mode_phi[peak_z_index]
    floor = jnp.asarray(1.0e-300, dtype=jnp.float64)
    peak_phi = jnp.where(jnp.abs(peak_phi) > floor, peak_phi, floor + 0.0j)
    weighted = (
        state[:, :, peak_z_index, ixzero, selected]
        * setup["velocity"].w_vpar[:, None]
        * setup["velocity"].w_mu[None, :]
        / peak_phi
    )
    weighted_mu_vpar = jnp.transpose(weighted)
    vpar = jnp.broadcast_to(setup["velocity"].vpar[None, :], weighted_mu_vpar.shape)
    vperp = jnp.sqrt(
        2.0
        * setup["geometry"].B[peak_z_index]
        * jnp.maximum(setup["velocity"].mu[:, None], 0.0)
    )
    vperp = jnp.broadcast_to(vperp, weighted_mu_vpar.shape)
    final_time = n_windows * steps_per_window * dt
    return GkwVelocitySpaceSlice(
        vpar=vpar,
        vperp=vperp,
        real_part=jnp.real(weighted_mu_vpar),
        imag_part=jnp.imag(weighted_mu_vpar),
        time=final_time,
        peak_z=setup["parallel"].z[peak_z_index],
        peak_z_index=peak_z_index,
        source=(
            f"stellarator_gk:{source_label or 'vpar_odd_sign_audit'}:"
            f"igh_sign={igh_sign:g}:parallel_field_drive_sign={parallel_field_drive_sign:g}"
        ),
        notes=(
            "selected-ky CBC velocity-space slice with controlled odd-vpar RHS sign flips; "
            f"normalize_each_window={normalize_each_window}, "
            f"normalization_model={normalization_model}, "
            f"term_v_phi_variant={term_v_phi_variant}, "
            f"term_vii_phi_variant={term_vii_phi_variant}, "
            f"term_viii_phi_variant={term_viii_phi_variant}"
        ),
    )


def _run_cyclone_base_case_velocity_space_slice_series_with_field_variants(
    *,
    snapshot_indices: tuple[int, ...] | None,
    n_z: int,
    n_vpar: int,
    n_mu: int,
    vpar_max: float | None,
    mu_max: float | None,
    dt: float | None,
    nperiod: int | None,
    steps_per_window: int,
    n_windows: int,
    parallel_recurrence_rate: float | None,
    velocity_recurrence_rate: float | None,
    parallel_backend: str | None,
    parallel_boundary: str | None,
    velocity_backend: str | None,
    normalize_each_window: bool,
    normalization_model: str,
    target: BenchmarkTarget | None,
    igh_sign: float,
    parallel_field_drive_sign: float,
    term_v_phi_variant: str = "identity",
    term_vii_phi_variant: str = "identity",
    term_viii_phi_variant: str = "identity",
    source_label: str | None = None,
) -> GkwVelocitySpaceSliceSeries:
    from .physics import (
        dissipation,
        drift_field_drive,
        equilibrium_drive,
        gkw_igh_streaming_mirror,
        gkw_parallel_field_drive,
        magnetic_drift_advection,
        solve_adiabatic_electron_phi,
    )
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    term_v_phi_variant = _validate_field_phi_variant(term_v_phi_variant)
    term_vii_phi_variant = _validate_field_phi_variant(term_vii_phi_variant)
    term_viii_phi_variant = _validate_field_phi_variant(term_viii_phi_variant)
    requested = _velocity_series_snapshot_indices(
        snapshot_indices,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
    )
    requested_set = set(requested)

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_recurrence_rate = _cyclone_velocity_recurrence_rate(
        metadata,
        velocity_recurrence_rate,
        "gkw_igh",
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_igh",
        velocity_backend=velocity_backend,
        initial_profile="cosine2",
    )
    rhs = setup["precompute"].rhs
    field = setup["precompute"].field
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(lambda state_value: solve_adiabatic_electron_phi(state_value, field))

    def residual_with_variants(state_value, precomputed=None, phi=None):
        del precomputed
        phi_value = solve_adiabatic_electron_phi(state_value, field) if phi is None else phi
        phi_term_v = _apply_field_phi_variant(phi_value, term_v_phi_variant)
        phi_term_vii = _apply_field_phi_variant(phi_value, term_vii_phi_variant)
        phi_term_viii = _apply_field_phi_variant(phi_value, term_viii_phi_variant)
        return (
            igh_sign * gkw_igh_streaming_mirror(state_value, rhs)
            + magnetic_drift_advection(state_value, rhs.magnetic_drift_frequency)
            + equilibrium_drive(phi_term_v, rhs)
            + parallel_field_drive_sign * gkw_parallel_field_drive(phi_term_vii, rhs)
            + drift_field_drive(phi_term_viii, rhs)
            + dissipation(state_value, rhs.perpendicular_damping)
        )

    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                residual_with_variants,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    snapshots: list[GkwVelocitySpaceSlice] = []
    phi = solve_phi(state)
    for window in range(n_windows):
        state = advance_window(state)
        current_step = (window + 1) * steps_per_window
        phi = solve_phi(state)
        if normalize_each_window:
            amplitude = _cyclone_phi_normalization_amplitude(
                phi,
                setup,
                normalization_model=normalization_model,
            )
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = solve_phi(state)
        if current_step in requested_set:
            snapshots.append(
                _cyclone_velocity_space_slice_from_state(
                    setup,
                    state,
                    phi,
                    selected_ky_index=selected,
                    ixzero=ixzero,
                    time=current_step * dt,
                    source=(
                        f"stellarator_gk:{source_label or 'series_variant'}:"
                        f"igh_sign={igh_sign:g}:field_sign={parallel_field_drive_sign:g}"
                    ),
                    notes=(
                        "selected-ky CBC velocity-space slice series with "
                        "controlled RHS/field variants; "
                        f"snapshot_index={current_step}, "
                        f"term_v_phi_variant={term_v_phi_variant}, "
                        f"term_vii_phi_variant={term_vii_phi_variant}, "
                        f"term_viii_phi_variant={term_viii_phi_variant}, "
                        f"normalization_model={normalization_model}"
                    ),
                )
            )

    if len(snapshots) != len(requested):
        raise RuntimeError("internal error: did not collect every requested snapshot")
    return GkwVelocitySpaceSliceSeries(
        times=jnp.asarray([snapshot.time for snapshot in snapshots], dtype=jnp.float64),
        snapshot_indices=jnp.asarray(requested, dtype=jnp.int32),
        vpar=jnp.stack([snapshot.vpar for snapshot in snapshots]),
        vperp=jnp.stack([snapshot.vperp for snapshot in snapshots]),
        real_part=jnp.stack([snapshot.real_part for snapshot in snapshots]),
        imag_part=jnp.stack([snapshot.imag_part for snapshot in snapshots]),
        source=f"stellarator_gk:{source_label or 'series_variant'}",
        notes=(
            "selected-ky CBC velocity-space slice series with controlled "
            "RHS/field variants"
        ),
    )


def compare_parallel_phi_traces(
    observed: ParallelPhiTrace,
    reference: ParallelPhiTrace,
    *,
    tolerance: float = 1.0e-2,
    time_tolerance: float = 1.0e-8,
    z_tolerance: float = 1.0e-12,
    normalize_profiles: bool = True,
) -> ParallelPhiTraceComparisonReport:
    """Compare two parallel ``|phi|^2`` traces at matching time and grid nodes."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if time_tolerance < 0.0:
        raise ValueError("time_tolerance must be nonnegative")
    if z_tolerance < 0.0:
        raise ValueError("z_tolerance must be nonnegative")
    if tuple(observed.phi_power.shape) != tuple(reference.phi_power.shape):
        raise ValueError("parallel phi traces must have matching phi_power shapes")

    observed_profiles = jnp.asarray(observed.phi_power, dtype=jnp.float64)
    reference_profiles = jnp.asarray(reference.phi_power, dtype=jnp.float64)
    if normalize_profiles:
        observed_profiles = _normalize_profile_rows(observed_profiles)
        reference_profiles = _normalize_profile_rows(reference_profiles)
    profile_errors = jnp.max(jnp.abs(observed_profiles - reference_profiles), axis=1)
    max_abs_error = jnp.max(profile_errors)
    time_error = _max_abs_error(observed.times, reference.times)
    z_error = _max_abs_error(observed.z, reference.z)
    passed = jnp.logical_and(
        max_abs_error <= tolerance,
        jnp.logical_and(time_error <= time_tolerance, z_error <= z_tolerance),
    )
    return ParallelPhiTraceComparisonReport(
        profile_errors=profile_errors,
        max_abs_error=max_abs_error,
        time_error=time_error,
        z_error=z_error,
        passed=passed,
        normalized_profiles=normalize_profiles,
        notes=(
            f"observed={observed.source}; reference={reference.source}; "
            "parallel |phi|^2 profile comparison"
        ),
    )


def audit_parallel_phi_profile_alignment(
    observed: ParallelPhiTrace,
    reference: ParallelPhiTrace,
    *,
    tolerance: float = 1.0e-2,
    time_tolerance: float = 1.0e-8,
    z_tolerance: float = 1.0e-12,
    normalize_profiles: bool = True,
    power_floor: float = 1.0e-300,
) -> ParallelPhiProfileAudit:
    """Audit profile mismatch against simple output-order and normalization causes.

    The direct comparison checks the profiles as written.  The reversed and
    circular-shift comparisons are diagnostic only: they test whether a
    plausible file-output ordering convention could explain the mismatch.
    Total-power, center-of-power, and edge-fraction diagnostics are reported
    from the raw and row-normalized profiles.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if time_tolerance < 0.0:
        raise ValueError("time_tolerance must be nonnegative")
    if z_tolerance < 0.0:
        raise ValueError("z_tolerance must be nonnegative")
    if power_floor <= 0.0:
        raise ValueError("power_floor must be positive")
    if tuple(observed.phi_power.shape) != tuple(reference.phi_power.shape):
        raise ValueError("parallel phi traces must have matching phi_power shapes")

    observed_raw = jnp.asarray(observed.phi_power, dtype=jnp.float64)
    reference_raw = jnp.asarray(reference.phi_power, dtype=jnp.float64)
    observed_profiles = observed_raw
    reference_profiles = reference_raw
    if normalize_profiles:
        observed_profiles = _normalize_profile_rows(observed_profiles, floor=power_floor)
        reference_profiles = _normalize_profile_rows(reference_profiles, floor=power_floor)

    direct_profile_errors = jnp.max(jnp.abs(observed_profiles - reference_profiles), axis=1)
    reversed_profiles = reference_profiles[:, ::-1]
    reversed_profile_errors = jnp.max(jnp.abs(observed_profiles - reversed_profiles), axis=1)
    n_z = observed_profiles.shape[1]
    shifted_profiles = jnp.stack(
        [jnp.roll(reference_profiles, shift, axis=1) for shift in range(n_z)],
        axis=0,
    )
    shift_errors = jnp.max(jnp.abs(observed_profiles[None, :, :] - shifted_profiles), axis=(1, 2))
    best_shift = jnp.argmin(shift_errors)
    best_shift_profile_errors = jnp.max(
        jnp.abs(observed_profiles - shifted_profiles[best_shift]),
        axis=1,
    )

    observed_total = jnp.sum(observed_raw, axis=1)
    reference_total = jnp.sum(reference_raw, axis=1)
    floor = jnp.asarray(power_floor, dtype=observed_total.dtype)
    total_power_ratio = observed_total / jnp.maximum(reference_total, floor)
    observed_shape = _normalize_profile_rows(observed_raw, floor=power_floor)
    reference_shape = _normalize_profile_rows(reference_raw, floor=power_floor)
    z = jnp.asarray(observed.z, dtype=jnp.float64)
    observed_center = jnp.sum(observed_shape * z[None, :], axis=1)
    reference_center = jnp.sum(reference_shape * z[None, :], axis=1)
    center_of_power_error = observed_center - reference_center
    observed_second_moment = jnp.sum(
        observed_shape * (z[None, :] - observed_center[:, None]) ** 2,
        axis=1,
    )
    reference_second_moment = jnp.sum(
        reference_shape * (z[None, :] - reference_center[:, None]) ** 2,
        axis=1,
    )
    second_moment_error = observed_second_moment - reference_second_moment
    peak_z_error = z[jnp.argmax(observed_shape, axis=1)] - z[jnp.argmax(reference_shape, axis=1)]
    edge_fraction_error = (
        observed_shape[:, 0]
        + observed_shape[:, -1]
        - reference_shape[:, 0]
        - reference_shape[:, -1]
    )
    signed_profile_error = observed_profiles - reference_profiles
    flat_worst_index = jnp.argmax(jnp.abs(signed_profile_error))
    worst_time_index = flat_worst_index // n_z
    worst_z_index = flat_worst_index % n_z
    worst_signed_error = signed_profile_error[worst_time_index, worst_z_index]
    worst_observed_value = observed_profiles[worst_time_index, worst_z_index]
    worst_reference_value = reference_profiles[worst_time_index, worst_z_index]

    best_aligned_max_error = jnp.min(
        jnp.asarray(
            [
                jnp.max(direct_profile_errors),
                jnp.max(reversed_profile_errors),
                jnp.min(shift_errors),
            ],
            dtype=jnp.float64,
        )
    )
    time_error = _max_abs_error(observed.times, reference.times)
    z_error = _max_abs_error(observed.z, reference.z)
    passed = jnp.logical_and(
        best_aligned_max_error <= tolerance,
        jnp.logical_and(time_error <= time_tolerance, z_error <= z_tolerance),
    )
    return ParallelPhiProfileAudit(
        direct_profile_errors=direct_profile_errors,
        reversed_profile_errors=reversed_profile_errors,
        best_shift_profile_errors=best_shift_profile_errors,
        circular_shift_errors=shift_errors,
        best_shift=best_shift,
        best_aligned_max_error=best_aligned_max_error,
        total_power_ratio=total_power_ratio,
        center_of_power_error=center_of_power_error,
        edge_fraction_error=edge_fraction_error,
        peak_z_error=peak_z_error,
        second_moment_error=second_moment_error,
        worst_time_index=worst_time_index,
        worst_z_index=worst_z_index,
        worst_time=observed.times[worst_time_index],
        worst_z=observed.z[worst_z_index],
        worst_signed_error=worst_signed_error,
        worst_observed_value=worst_observed_value,
        worst_reference_value=worst_reference_value,
        passed=passed,
        normalized_profiles=normalize_profiles,
        notes=(
            f"observed={observed.source}; reference={reference.source}; "
            "parallel |phi|^2 profile alignment/normalization audit"
        ),
    )


def audit_cyclone_selected_ky_gap(
    solver_trace: CycloneTrace,
    reference_trace: CycloneTrace,
    solver_profile: ParallelPhiTrace,
    reference_profile: ParallelPhiTrace,
    *,
    growth_window_fraction: float = 0.5,
    growth_tolerance: float = 1.0e-2,
    profile_tolerance: float = 2.0e-2,
    normalize_profiles: bool = True,
) -> CycloneSelectedKyGapAudit:
    """Summarize the remaining selected-``ky`` growth/profile discrepancy.

    GKW ``time.dat`` and ``parallel_phi.dat`` do not contain the explicit
    ``t=0`` sample that the solver trace records.  This audit aligns the
    post-window samples before comparing window growth and profile histories.
    """

    if not 0.0 <= growth_window_fraction < 1.0:
        raise ValueError("growth_window_fraction must lie in [0, 1)")
    if growth_tolerance <= 0.0:
        raise ValueError("growth_tolerance must be positive")
    if profile_tolerance <= 0.0:
        raise ValueError("profile_tolerance must be positive")

    solver_time, solver_growth, solver_amplitude = _post_window_trace_view(
        solver_trace,
        reference_trace.times,
    )
    reference_time = jnp.asarray(reference_trace.times, dtype=jnp.float64)
    reference_growth = jnp.asarray(reference_trace.window_growth, dtype=jnp.float64)
    reference_amplitude = jnp.asarray(reference_trace.physical_amplitude, dtype=jnp.float64)
    if solver_growth.shape != reference_growth.shape:
        raise ValueError("solver and reference traces must have matching post-window samples")
    time_error = _max_abs_error(solver_time, reference_time)
    if float(time_error) > 1.0e-8:
        raise ValueError("solver and reference trace times do not align")

    profile_report = compare_parallel_phi_traces(
        solver_profile,
        reference_profile,
        tolerance=profile_tolerance,
        normalize_profiles=normalize_profiles,
    )
    profile_audit = audit_parallel_phi_profile_alignment(
        solver_profile,
        reference_profile,
        tolerance=profile_tolerance,
        normalize_profiles=normalize_profiles,
    )
    if tuple(profile_report.profile_errors.shape) != tuple(reference_time.shape):
        raise ValueError("profile trace and time trace lengths must match")

    solver_late_fit = _late_fit_from_samples(
        solver_time,
        solver_amplitude,
        growth_window_fraction,
    )
    reference_late_fit = _late_fit_from_samples(
        reference_time,
        reference_amplitude,
        growth_window_fraction,
    )
    solver_late_mean = _late_mean_from_samples(solver_growth, growth_window_fraction)
    reference_late_mean = _late_mean_from_samples(reference_growth, growth_window_fraction)
    window_growth_delta = reference_growth - solver_growth
    late_fit_delta = reference_late_fit - solver_late_fit
    late_mean_delta = reference_late_mean - solver_late_mean
    final_window_growth_delta = reference_growth[-1] - solver_growth[-1]
    profile_errors = jnp.asarray(profile_report.profile_errors, dtype=jnp.float64)
    total_power_ratio = jnp.asarray(profile_audit.total_power_ratio, dtype=jnp.float64)
    total_power_ratio_mean = jnp.mean(total_power_ratio)
    total_power_ratio_max_deviation = jnp.max(jnp.abs(total_power_ratio - 1.0))
    passed = jnp.logical_and(
        jnp.abs(late_fit_delta) <= growth_tolerance,
        profile_report.max_abs_error <= profile_tolerance,
    )
    return CycloneSelectedKyGapAudit(
        times=reference_time,
        solver_window_growth=solver_growth,
        reference_window_growth=reference_growth,
        window_growth_delta=window_growth_delta,
        profile_errors=profile_errors,
        total_power_ratio=total_power_ratio,
        solver_late_fit_growth=solver_late_fit,
        reference_late_fit_growth=reference_late_fit,
        late_fit_delta=late_fit_delta,
        solver_late_mean_growth=solver_late_mean,
        reference_late_mean_growth=reference_late_mean,
        late_mean_delta=late_mean_delta,
        final_window_growth_delta=final_window_growth_delta,
        first_profile_error=profile_errors[0],
        max_profile_error=profile_report.max_abs_error,
        late_profile_error=profile_errors[-1],
        worst_time=profile_audit.worst_time,
        worst_z=profile_audit.worst_z,
        worst_signed_profile_error=profile_audit.worst_signed_error,
        total_power_ratio_mean=total_power_ratio_mean,
        total_power_ratio_max_deviation=total_power_ratio_max_deviation,
        passed=passed,
        notes=(
            f"solver={solver_trace.source}; reference={reference_trace.source}; "
            "selected-ky growth/profile gap audit"
        ),
    )


def audit_cyclone_velocity_space_slice(
    observed: GkwVelocitySpaceSlice,
    reference: GkwVelocitySpaceSlice,
    *,
    tolerance: float = 2.0e-2,
    grid_tolerance: float = 1.0e-4,
    time_tolerance: float = 1.0e-8,
    peak_z_tolerance: float | None = None,
) -> CycloneVelocitySpaceSliceAudit:
    """Compare two GKW-normalized selected-``ky`` velocity-space slices."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if grid_tolerance < 0.0:
        raise ValueError("grid_tolerance must be nonnegative")
    if time_tolerance < 0.0:
        raise ValueError("time_tolerance must be nonnegative")
    if peak_z_tolerance is not None and peak_z_tolerance < 0.0:
        raise ValueError("peak_z_tolerance must be nonnegative")
    if tuple(observed.vpar.shape) != tuple(reference.vpar.shape):
        raise ValueError("velocity-space slices must have matching shapes")

    observed_complex = jnp.asarray(observed.real_part) + 1j * jnp.asarray(observed.imag_part)
    reference_complex = jnp.asarray(reference.real_part) + 1j * jnp.asarray(reference.imag_part)
    delta = observed_complex - reference_complex
    vpar_error = _max_abs_error(observed.vpar, reference.vpar)
    vperp_error = _max_abs_error(observed.vperp, reference.vperp)
    real_error = _max_abs_error(observed.real_part, reference.real_part)
    imag_error = _max_abs_error(observed.imag_part, reference.imag_part)
    complex_max = jnp.max(jnp.abs(delta))
    complex_l2 = _l2_norm(delta)
    observed_l2 = _l2_norm(observed_complex)
    reference_l2 = _l2_norm(reference_complex)
    complex_relative = complex_l2 / jnp.maximum(
        reference_l2,
        jnp.asarray(1.0e-300, dtype=reference_l2.dtype),
    )
    observed_time = jnp.asarray(observed.time, dtype=jnp.float64)
    reference_time = jnp.asarray(reference.time, dtype=jnp.float64)
    time_error = jnp.where(
        jnp.logical_and(jnp.isfinite(observed_time), jnp.isfinite(reference_time)),
        jnp.abs(observed_time - reference_time),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    observed_peak_z = jnp.asarray(observed.peak_z, dtype=jnp.float64)
    reference_peak_z = jnp.asarray(reference.peak_z, dtype=jnp.float64)
    peak_z_error = jnp.where(
        jnp.logical_and(jnp.isfinite(observed_peak_z), jnp.isfinite(reference_peak_z)),
        jnp.abs(observed_peak_z - reference_peak_z),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    peak_pass = (
        jnp.asarray(True)
        if peak_z_tolerance is None
        else peak_z_error <= jnp.asarray(peak_z_tolerance, dtype=jnp.float64)
    )
    passed = jnp.logical_and(
        complex_max <= tolerance,
        jnp.logical_and(
            jnp.maximum(vpar_error, vperp_error) <= grid_tolerance,
            jnp.logical_and(time_error <= time_tolerance, peak_pass),
        ),
    )
    return CycloneVelocitySpaceSliceAudit(
        vpar_error=vpar_error,
        vperp_error=vperp_error,
        real_max_abs_error=real_error,
        imag_max_abs_error=imag_error,
        complex_max_abs_error=complex_max,
        complex_l2_error=complex_l2,
        complex_relative_l2_error=complex_relative,
        observed_l2_norm=observed_l2,
        reference_l2_norm=reference_l2,
        time_error=time_error,
        peak_z_error=peak_z_error,
        passed=passed,
        shape=tuple(int(value) for value in observed.vpar.shape),
        notes=(
            f"observed={observed.source}; reference={reference.source}; "
            "GKW-normalized velocity-space slice comparison"
        ),
    )


def _slice_from_velocity_space_series(
    series: GkwVelocitySpaceSliceSeries,
    position: int,
    *,
    source_suffix: str = "",
) -> GkwVelocitySpaceSlice:
    source = series.source
    if source_suffix:
        source = f"{source}:{source_suffix}"
    snapshot_index = int(np.asarray(series.snapshot_indices[position]))
    return GkwVelocitySpaceSlice(
        vpar=series.vpar[position],
        vperp=series.vperp[position],
        real_part=series.real_part[position],
        imag_part=series.imag_part[position],
        time=series.times[position],
        source=source,
        notes=f"{series.notes}; snapshot_index={snapshot_index}",
    )


def _snapshot_position_map(series: GkwVelocitySpaceSliceSeries) -> dict[int, int]:
    return {
        int(snapshot_index): position
        for position, snapshot_index in enumerate(np.asarray(series.snapshot_indices))
    }


def _shared_snapshot_indices(
    observed: GkwVelocitySpaceSliceSeries,
    reference: GkwVelocitySpaceSliceSeries,
    snapshot_indices: tuple[int, ...] | None,
) -> tuple[int, ...]:
    observed_indices = set(_snapshot_position_map(observed))
    reference_indices = set(_snapshot_position_map(reference))
    if snapshot_indices is None:
        shared = tuple(sorted(observed_indices & reference_indices))
    else:
        shared = tuple(int(index) for index in snapshot_indices)
    if not shared:
        raise ValueError("no shared velocity-slice snapshot indices to compare")
    missing_observed = [index for index in shared if index not in observed_indices]
    missing_reference = [index for index in shared if index not in reference_indices]
    if missing_observed or missing_reference:
        raise ValueError(
            "requested snapshot indices are missing: "
            f"observed={missing_observed}, reference={missing_reference}"
        )
    return shared


def audit_cyclone_velocity_space_slice_series(
    observed: GkwVelocitySpaceSliceSeries,
    reference: GkwVelocitySpaceSliceSeries,
    *,
    snapshot_indices: tuple[int, ...] | None = None,
    tolerance: float = 2.0e-2,
    grid_tolerance: float = 1.0e-4,
    time_tolerance: float = 1.0e-8,
) -> CycloneVelocitySpaceSliceSeriesAudit:
    """Compare solver and GKW velocity-slice histories at matching snapshots."""

    shared = _shared_snapshot_indices(observed, reference, snapshot_indices)
    observed_positions = _snapshot_position_map(observed)
    reference_positions = _snapshot_position_map(reference)
    direct_reports = []
    convention_reports = []
    for snapshot_index in shared:
        observed_slice = _slice_from_velocity_space_series(
            observed,
            observed_positions[snapshot_index],
            source_suffix=str(snapshot_index),
        )
        reference_slice = _slice_from_velocity_space_series(
            reference,
            reference_positions[snapshot_index],
            source_suffix=str(snapshot_index),
        )
        direct_reports.append(
            audit_cyclone_velocity_space_slice(
                observed_slice,
                reference_slice,
                tolerance=tolerance,
                grid_tolerance=grid_tolerance,
                time_tolerance=time_tolerance,
            )
        )
        convention_reports.append(
            audit_velocity_space_slice_conventions(observed_slice, reference_slice)
        )

    direct_max = jnp.asarray(
        [report.complex_max_abs_error for report in direct_reports],
        dtype=jnp.float64,
    )
    best_max = jnp.asarray(
        [report.best_max_abs_error for report in convention_reports],
        dtype=jnp.float64,
    )
    first_convention = convention_reports[0]
    return CycloneVelocitySpaceSliceSeriesAudit(
        snapshot_indices=jnp.asarray(shared, dtype=jnp.int32),
        times=jnp.asarray(
            [observed.times[observed_positions[index]] for index in shared],
            dtype=jnp.float64,
        ),
        vpar_errors=jnp.asarray([report.vpar_error for report in direct_reports]),
        vperp_errors=jnp.asarray([report.vperp_error for report in direct_reports]),
        time_errors=jnp.asarray([report.time_error for report in direct_reports]),
        direct_max_abs_errors=direct_max,
        direct_l2_errors=jnp.asarray([report.complex_l2_error for report in direct_reports]),
        direct_relative_l2_errors=jnp.asarray(
            [report.complex_relative_l2_error for report in direct_reports]
        ),
        best_variant_indices=jnp.asarray(
            [report.best_variant_index for report in convention_reports],
            dtype=jnp.int32,
        ),
        best_max_abs_errors=best_max,
        best_l2_errors=jnp.asarray([report.best_l2_error for report in convention_reports]),
        first_direct_max_abs_error=direct_max[0],
        last_direct_max_abs_error=direct_max[-1],
        max_direct_max_abs_error=jnp.max(direct_max),
        max_best_max_abs_error=jnp.max(best_max),
        passed=jnp.all(jnp.asarray([report.passed for report in direct_reports], dtype=bool)),
        shape=tuple(int(value) for value in observed.vpar.shape[1:]),
        variant_names=first_convention.variant_names,
        notes=(
            f"observed={observed.source}; reference={reference.source}; "
            "multi-window GKW-normalized velocity-space slice comparison"
        ),
    )


def audit_cyclone_velocity_space_slice_series_variants(
    observed_variants: tuple[GkwVelocitySpaceSliceSeries, ...],
    reference: GkwVelocitySpaceSliceSeries,
    *,
    variant_names: tuple[str, ...],
    snapshot_indices: tuple[int, ...] | None = None,
    tolerance: float = 2.0e-2,
    grid_tolerance: float = 1.0e-4,
    time_tolerance: float = 1.0e-8,
) -> CycloneVelocitySpaceSliceSeriesVariantAudit:
    """Compare multiple solver RHS/field variants to one GKW slice series."""

    if not observed_variants:
        raise ValueError("observed_variants must not be empty")
    if len(variant_names) != len(observed_variants):
        raise ValueError("variant_names length must match observed_variants")
    shared = _shared_snapshot_indices(observed_variants[0], reference, snapshot_indices)
    direct_max_rows = []
    direct_l2_rows = []
    direct_relative_rows = []
    best_layout_max_rows = []
    best_layout_l2_rows = []
    best_layout_index_rows = []
    layout_names: tuple[str, ...] | None = None
    for observed in observed_variants:
        series_report = audit_cyclone_velocity_space_slice_series(
            observed,
            reference,
            snapshot_indices=shared,
            tolerance=tolerance,
            grid_tolerance=grid_tolerance,
            time_tolerance=time_tolerance,
        )
        direct_max_rows.append(series_report.direct_max_abs_errors)
        direct_l2_rows.append(series_report.direct_l2_errors)
        direct_relative_rows.append(series_report.direct_relative_l2_errors)
        best_layout_max_rows.append(series_report.best_max_abs_errors)
        best_layout_l2_rows.append(series_report.best_l2_errors)
        best_layout_index_rows.append(series_report.best_variant_indices)
        layout_names = series_report.variant_names

    direct_max = jnp.stack(direct_max_rows)
    best_direct_indices = jnp.argmin(direct_max, axis=0)
    best_direct = jnp.min(direct_max, axis=0)
    baseline = direct_max[0]
    times = jnp.asarray(
        [
            observed_variants[0].times[_snapshot_position_map(observed_variants[0])[index]]
            for index in shared
        ],
        dtype=jnp.float64,
    )
    return CycloneVelocitySpaceSliceSeriesVariantAudit(
        snapshot_indices=jnp.asarray(shared, dtype=jnp.int32),
        times=times,
        direct_max_abs_errors=direct_max,
        direct_l2_errors=jnp.stack(direct_l2_rows),
        direct_relative_l2_errors=jnp.stack(direct_relative_rows),
        best_layout_max_abs_errors=jnp.stack(best_layout_max_rows),
        best_layout_l2_errors=jnp.stack(best_layout_l2_rows),
        best_layout_variant_indices=jnp.stack(best_layout_index_rows),
        best_direct_variant_indices=best_direct_indices,
        baseline_direct_max_abs_errors=baseline,
        best_direct_max_abs_errors=best_direct,
        max_baseline_direct_max_abs_error=jnp.max(baseline),
        max_best_direct_max_abs_error=jnp.max(best_direct),
        passed=jnp.all(baseline <= jnp.asarray(tolerance, dtype=jnp.float64)),
        shape=tuple(int(value) for value in observed_variants[0].vpar.shape[1:]),
        variant_names=tuple(variant_names),
        layout_variant_names=layout_names or (),
        notes=(
            f"reference={reference.source}; multi-window RHS/field-convention "
            "variant audit for GKW-normalized velocity-space slices"
        ),
    )


def audit_velocity_space_slice_conventions(
    observed: GkwVelocitySpaceSlice,
    reference: GkwVelocitySpaceSlice,
) -> VelocitySliceConventionAudit:
    """Check simple Fortran/Python indexing, layout, and phase conventions.

    GKW allocates the diagnostic work array as ``(n_vpar,n_mu)`` but
    ``output_slice_2d`` writes rows over the second Fortran index and columns
    over the first.  Therefore the text files are expected to load as
    ``(n_mu,n_vpar)`` in NumPy.  This diagnostic keeps that direct convention as
    the baseline and tests common mistakes: axis reversals, one-cell circular
    shifts from 1-based/0-based thinking, C/F-order reshapes, and simple complex
    sign/phase/conjugation variants.
    """

    if tuple(observed.vpar.shape) != tuple(reference.vpar.shape):
        raise ValueError("velocity-space slices must have matching shapes")
    observed_complex = jnp.asarray(observed.real_part) + 1j * jnp.asarray(observed.imag_part)
    reference_complex = jnp.asarray(reference.real_part) + 1j * jnp.asarray(reference.imag_part)
    spatial_variants = _velocity_slice_spatial_variants(observed_complex)
    phase_variants = (
        ("identity", lambda values: values),
        ("negative", lambda values: -values),
        ("conjugate", jnp.conj),
        ("negative_conjugate", lambda values: -jnp.conj(values)),
        ("i_times", lambda values: 1j * values),
        ("minus_i_times", lambda values: -1j * values),
    )
    names = []
    max_errors = []
    l2_errors = []
    for spatial_name, spatial_values in spatial_variants:
        for phase_name, phase_fn in phase_variants:
            variant = phase_fn(spatial_values)
            delta = variant - reference_complex
            names.append(f"{spatial_name}:{phase_name}")
            max_errors.append(jnp.max(jnp.abs(delta)))
            l2_errors.append(_l2_norm(delta))

    max_error_array = jnp.asarray(max_errors, dtype=jnp.float64)
    l2_error_array = jnp.asarray(l2_errors, dtype=jnp.float64)
    best_index = jnp.argmin(max_error_array)
    observed_even = 0.5 * (observed_complex + observed_complex[:, ::-1])
    reference_even = 0.5 * (reference_complex + reference_complex[:, ::-1])
    observed_odd = 0.5 * (observed_complex - observed_complex[:, ::-1])
    reference_odd = 0.5 * (reference_complex - reference_complex[:, ::-1])
    even_delta = observed_even - reference_even
    odd_same_delta = observed_odd - reference_odd
    odd_opposite_delta = observed_odd + reference_odd
    return VelocitySliceConventionAudit(
        max_abs_errors=max_error_array,
        l2_errors=l2_error_array,
        best_variant_index=best_index,
        best_max_abs_error=max_error_array[best_index],
        best_l2_error=l2_error_array[best_index],
        direct_max_abs_error=max_error_array[0],
        direct_l2_error=l2_error_array[0],
        even_max_abs_error=jnp.max(jnp.abs(even_delta)),
        even_l2_error=_l2_norm(even_delta),
        odd_same_sign_max_abs_error=jnp.max(jnp.abs(odd_same_delta)),
        odd_same_sign_l2_error=_l2_norm(odd_same_delta),
        odd_opposite_sign_max_abs_error=jnp.max(jnp.abs(odd_opposite_delta)),
        odd_opposite_sign_l2_error=_l2_norm(odd_opposite_delta),
        variant_names=tuple(names),
        notes=(
            f"observed={observed.source}; reference={reference.source}; "
            "velocity-space convention audit for Fortran row/column and 1-based indexing"
        ),
    )


def audit_velocity_space_slice_phase_alignment(
    observed: GkwVelocitySpaceSlice,
    reference: GkwVelocitySpaceSlice,
) -> VelocitySlicePhaseAudit:
    """Apply optimal global complex phase/scale factors to velocity-slice variants.

    A linear eigenfunction can carry an arbitrary complex phase, and some
    Fourier-sign conventions appear as complex conjugation.  This audit keeps
    the GKW ``output_slice_2d`` row/column contract explicit, then measures how
    much error remains after applying the best unit phase, and separately the
    best unconstrained complex scalar, to each candidate layout.
    """

    if tuple(observed.vpar.shape) != tuple(reference.vpar.shape):
        raise ValueError("velocity-space slices must have matching shapes")
    observed_complex = jnp.asarray(observed.real_part) + 1j * jnp.asarray(observed.imag_part)
    reference_complex = jnp.asarray(reference.real_part) + 1j * jnp.asarray(reference.imag_part)
    reference_l2 = jnp.maximum(_l2_norm(reference_complex), jnp.asarray(1.0e-300))
    names = []
    phase_factors = []
    phase_angles = []
    phase_max_errors = []
    phase_l2_errors = []
    phase_relative_errors = []
    scale_factors = []
    scaled_max_errors = []
    scaled_l2_errors = []
    scaled_relative_errors = []
    for spatial_name, spatial_values in _velocity_slice_spatial_variants(observed_complex):
        for transform_name, transformed in (
            ("identity", spatial_values),
            ("conjugate", jnp.conj(spatial_values)),
        ):
            names.append(f"{spatial_name}:{transform_name}")
            phase = _optimal_unit_complex_factor(transformed, reference_complex)
            phase_delta = phase * transformed - reference_complex
            phase_l2 = _l2_norm(phase_delta)
            scale = _optimal_complex_scale(transformed, reference_complex)
            scaled_delta = scale * transformed - reference_complex
            scaled_l2 = _l2_norm(scaled_delta)
            phase_factors.append(phase)
            phase_angles.append(jnp.angle(phase))
            phase_max_errors.append(jnp.max(jnp.abs(phase_delta)))
            phase_l2_errors.append(phase_l2)
            phase_relative_errors.append(phase_l2 / reference_l2)
            scale_factors.append(scale)
            scaled_max_errors.append(jnp.max(jnp.abs(scaled_delta)))
            scaled_l2_errors.append(scaled_l2)
            scaled_relative_errors.append(scaled_l2 / reference_l2)

    phase_max_array = jnp.asarray(phase_max_errors, dtype=jnp.float64)
    scaled_max_array = jnp.asarray(scaled_max_errors, dtype=jnp.float64)
    best_phase_index = jnp.argmin(phase_max_array)
    best_scaled_index = jnp.argmin(scaled_max_array)
    reverse_vpar_index = names.index("reverse_vpar_columns:identity")
    return VelocitySlicePhaseAudit(
        unit_phase_factors=jnp.asarray(phase_factors, dtype=jnp.complex128),
        unit_phase_angles=jnp.asarray(phase_angles, dtype=jnp.float64),
        phase_aligned_max_abs_errors=phase_max_array,
        phase_aligned_l2_errors=jnp.asarray(phase_l2_errors, dtype=jnp.float64),
        phase_aligned_relative_l2_errors=jnp.asarray(phase_relative_errors, dtype=jnp.float64),
        complex_scale_factors=jnp.asarray(scale_factors, dtype=jnp.complex128),
        scaled_max_abs_errors=scaled_max_array,
        scaled_l2_errors=jnp.asarray(scaled_l2_errors, dtype=jnp.float64),
        scaled_relative_l2_errors=jnp.asarray(scaled_relative_errors, dtype=jnp.float64),
        best_phase_variant_index=best_phase_index,
        best_scaled_variant_index=best_scaled_index,
        best_phase_max_abs_error=phase_max_array[best_phase_index],
        best_scaled_max_abs_error=scaled_max_array[best_scaled_index],
        direct_phase_max_abs_error=phase_max_array[0],
        reverse_vpar_phase_max_abs_error=phase_max_array[reverse_vpar_index],
        variant_names=tuple(names),
        notes=(
            f"observed={observed.source}; reference={reference.source}; "
            "velocity-space phase audit for eigenfunction/output phase and ky-sign conjugation"
        ),
    )


def _optimal_unit_complex_factor(observed, reference):
    cross = jnp.sum(jnp.conj(observed) * reference)
    magnitude = jnp.abs(cross)
    return jnp.where(magnitude > 0.0, cross / magnitude, jnp.asarray(1.0 + 0.0j))


def _optimal_complex_scale(observed, reference):
    norm = jnp.sum(jnp.abs(observed) ** 2)
    cross = jnp.sum(jnp.conj(observed) * reference)
    return jnp.where(norm > 0.0, cross / norm, jnp.asarray(1.0 + 0.0j))


def run_cyclone_base_case_cosin2_gap_audit(
    *,
    gkw_time_path=None,
    gkw_parallel_phi_path=None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_derivative_model: str = "gkw_igh",
    normalization_model: str = "gkw_unweighted",
    growth_window_fraction: float = 0.5,
    growth_tolerance: float = 1.0e-2,
    profile_tolerance: float = 2.0e-2,
) -> CycloneSelectedKyGapAudit:
    """Run the solver and compare against the patched GKW ``cosin2`` fixtures."""

    gkw_time_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"
        if gkw_time_path is None
        else gkw_time_path
    )
    gkw_parallel_phi_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat"
        if gkw_parallel_phi_path is None
        else gkw_parallel_phi_path
    )
    solver_trace = run_cyclone_base_case_trace(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        initial_profile="cosine2",
        normalization_model=normalization_model,
        parallel_derivative_model=parallel_derivative_model,
    )
    solver_profile = run_cyclone_base_case_parallel_phi_trace(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        initial_profile="cosine2",
        normalization_model=normalization_model,
        parallel_derivative_model=parallel_derivative_model,
    )
    reference_trace = load_gkw_time_dat_trace(
        gkw_time_path,
        source="gkw:cosin2:time.dat",
        notes="patched selected-ky production-control cosin2 run",
    )
    reference_profile = load_gkw_parallel_phi_trace(
        gkw_parallel_phi_path,
        time_path=gkw_time_path,
        z=np.asarray(solver_profile.z),
        source="gkw:cosin2:parallel_phi.dat",
        notes="patched selected-ky production-control cosin2 run",
    )
    return audit_cyclone_selected_ky_gap(
        solver_trace,
        reference_trace,
        solver_profile,
        reference_profile,
        growth_window_fraction=growth_window_fraction,
        growth_tolerance=growth_tolerance,
        profile_tolerance=profile_tolerance,
    )


def run_cyclone_base_case_cosin2_velocity_slice_audit(
    *,
    gkw_distr1_path=None,
    gkw_distr2_path=None,
    gkw_distr3_path=None,
    gkw_distr4_path=None,
    gkw_time_path=None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_derivative_model: str = "gkw_igh",
    normalization_model: str = "gkw_unweighted",
    tolerance: float = 2.0e-2,
    grid_tolerance: float = 1.0e-4,
) -> CycloneVelocitySpaceSliceAudit:
    """Compare the solver final state with patched GKW ``cosin2`` ``distr*.dat``."""

    gkw_distr1_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat"
        if gkw_distr1_path is None
        else gkw_distr1_path
    )
    gkw_distr2_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat"
        if gkw_distr2_path is None
        else gkw_distr2_path
    )
    gkw_distr3_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat"
        if gkw_distr3_path is None
        else gkw_distr3_path
    )
    gkw_distr4_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat"
        if gkw_distr4_path is None
        else gkw_distr4_path
    )
    gkw_time_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"
        if gkw_time_path is None
        else gkw_time_path
    )
    observed = run_cyclone_base_case_velocity_space_slice(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        initial_profile="cosine2",
        normalization_model=normalization_model,
        parallel_derivative_model=parallel_derivative_model,
    )
    reference = load_gkw_velocity_space_slice(
        gkw_distr1_path,
        gkw_distr2_path,
        gkw_distr3_path,
        gkw_distr4_path,
        time_path=gkw_time_path,
        source="gkw:cosin2:distr*.dat",
        notes="patched selected-ky production-control cosin2 final output",
    )
    return audit_cyclone_velocity_space_slice(
        observed,
        reference,
        tolerance=tolerance,
        grid_tolerance=grid_tolerance,
    )


def run_cyclone_base_case_cosin2_velocity_series_audit(
    *,
    reference_series: GkwVelocitySpaceSliceSeries | None = None,
    gkw_directory=None,
    gkw_time_path=None,
    snapshot_indices: tuple[int, ...] | None = None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_derivative_model: str = "gkw_igh",
    normalization_model: str = "gkw_unweighted",
    tolerance: float = 2.0e-2,
    grid_tolerance: float = 1.0e-4,
) -> CycloneVelocitySpaceSliceSeriesAudit:
    """Compare solver snapshots with patched GKW ``cosin2`` multi-time slices."""

    if reference_series is None:
        gkw_directory = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr"
            if gkw_directory is None
            else gkw_directory
        )
        gkw_time_path = Path(gkw_directory / "time.dat" if gkw_time_path is None else gkw_time_path)
        reference_series = load_gkw_velocity_space_slice_series(
            gkw_directory,
            time_path=gkw_time_path,
            source="gkw:cosin2:distr*_<ntotstep>.dat",
            notes="patched selected-ky production-control cosin2 multi-time output",
        )
    if snapshot_indices is None:
        snapshot_indices = tuple(int(value) for value in np.asarray(reference_series.snapshot_indices))
    observed = run_cyclone_base_case_velocity_space_slice_series(
        snapshot_indices=snapshot_indices,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        initial_profile="cosine2",
        normalization_model=normalization_model,
        parallel_derivative_model=parallel_derivative_model,
    )
    return audit_cyclone_velocity_space_slice_series(
        observed,
        reference_series,
        snapshot_indices=snapshot_indices,
        tolerance=tolerance,
        grid_tolerance=grid_tolerance,
    )


def run_cyclone_base_case_cosin2_velocity_series_variant_audit(
    *,
    reference_series: GkwVelocitySpaceSliceSeries | None = None,
    gkw_directory=None,
    gkw_time_path=None,
    snapshot_indices: tuple[int, ...] | None = None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    steps_per_window: int = 20,
    n_windows: int = 80,
    normalization_model: str = "gkw_unweighted",
    tolerance: float = 2.0e-2,
    grid_tolerance: float = 1.0e-4,
) -> CycloneVelocitySpaceSliceSeriesVariantAudit:
    """Run the multi-time velocity-slice audit over Term VII/``igh`` variants."""

    if reference_series is None:
        gkw_directory = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr"
            if gkw_directory is None
            else gkw_directory
        )
        gkw_time_path = Path(gkw_directory / "time.dat" if gkw_time_path is None else gkw_time_path)
        reference_series = load_gkw_velocity_space_slice_series(
            gkw_directory,
            time_path=gkw_time_path,
            source="gkw:cosin2:distr*_<ntotstep>.dat",
            notes="patched selected-ky production-control cosin2 multi-time output",
        )
    if snapshot_indices is None:
        snapshot_indices = tuple(int(value) for value in np.asarray(reference_series.snapshot_indices))
    variants = (
        ("baseline", 1.0, 1.0, "identity", "identity", "identity"),
        ("flip_igh", -1.0, 1.0, "identity", "identity", "identity"),
        ("flip_term_vii_only", 1.0, -1.0, "identity", "identity", "identity"),
        ("conjugate_term_vii_only", 1.0, 1.0, "identity", "conjugate", "identity"),
        (
            "negative_conjugate_term_vii_only",
            1.0,
            1.0,
            "identity",
            "negative_conjugate",
            "identity",
        ),
        ("conjugate_all_field_terms", 1.0, 1.0, "conjugate", "conjugate", "conjugate"),
    )
    observed_variants = tuple(
        _run_cyclone_base_case_velocity_space_slice_series_with_field_variants(
            snapshot_indices=snapshot_indices,
            n_z=n_z,
            n_vpar=n_vpar,
            n_mu=n_mu,
            vpar_max=None,
            mu_max=None,
            dt=None,
            nperiod=None,
            steps_per_window=steps_per_window,
            n_windows=n_windows,
            parallel_recurrence_rate=None,
            velocity_recurrence_rate=None,
            parallel_backend=None,
            parallel_boundary=None,
            velocity_backend=None,
            normalize_each_window=True,
            normalization_model=normalization_model,
            target=None,
            igh_sign=igh_sign,
            parallel_field_drive_sign=field_sign,
            term_v_phi_variant=term_v_variant,
            term_vii_phi_variant=term_vii_variant,
            term_viii_phi_variant=term_viii_variant,
            source_label=name,
        )
        for name, igh_sign, field_sign, term_v_variant, term_vii_variant, term_viii_variant in variants
    )
    return audit_cyclone_velocity_space_slice_series_variants(
        observed_variants,
        reference_series,
        variant_names=tuple(name for name, *_rest in variants),
        snapshot_indices=snapshot_indices,
        tolerance=tolerance,
        grid_tolerance=grid_tolerance,
    )


def run_cyclone_base_case_cosin2_velocity_convention_audit(
    *,
    gkw_distr1_path=None,
    gkw_distr2_path=None,
    gkw_distr3_path=None,
    gkw_distr4_path=None,
    gkw_time_path=None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_derivative_model: str = "gkw_igh",
    normalization_model: str = "gkw_unweighted",
) -> VelocitySliceConventionAudit:
    """Run the patched GKW ``cosin2`` velocity-slice convention audit."""

    gkw_distr1_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat"
        if gkw_distr1_path is None
        else gkw_distr1_path
    )
    gkw_distr2_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat"
        if gkw_distr2_path is None
        else gkw_distr2_path
    )
    gkw_distr3_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat"
        if gkw_distr3_path is None
        else gkw_distr3_path
    )
    gkw_distr4_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat"
        if gkw_distr4_path is None
        else gkw_distr4_path
    )
    gkw_time_path = Path(
        "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"
        if gkw_time_path is None
        else gkw_time_path
    )
    observed = run_cyclone_base_case_velocity_space_slice(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        initial_profile="cosine2",
        normalization_model=normalization_model,
        parallel_derivative_model=parallel_derivative_model,
    )
    reference = load_gkw_velocity_space_slice(
        gkw_distr1_path,
        gkw_distr2_path,
        gkw_distr3_path,
        gkw_distr4_path,
        time_path=gkw_time_path,
        source="gkw:cosin2:distr*.dat",
        notes="patched selected-ky production-control cosin2 final output",
    )
    return audit_velocity_space_slice_conventions(observed, reference)


def run_cyclone_base_case_cosin2_velocity_phase_audit(
    *,
    reference_slice: GkwVelocitySpaceSlice | None = None,
    gkw_distr1_path=None,
    gkw_distr2_path=None,
    gkw_distr3_path=None,
    gkw_distr4_path=None,
    gkw_time_path=None,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    steps_per_window: int = 20,
    n_windows: int = 80,
    parallel_derivative_model: str = "gkw_igh",
    normalization_model: str = "gkw_unweighted",
) -> VelocitySlicePhaseAudit:
    """Run the patched GKW ``cosin2`` velocity-slice phase-alignment audit."""

    if reference_slice is None:
        gkw_distr1_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat"
            if gkw_distr1_path is None
            else gkw_distr1_path
        )
        gkw_distr2_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat"
            if gkw_distr2_path is None
            else gkw_distr2_path
        )
        gkw_distr3_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat"
            if gkw_distr3_path is None
            else gkw_distr3_path
        )
        gkw_distr4_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat"
            if gkw_distr4_path is None
            else gkw_distr4_path
        )
        gkw_time_path = Path(
            "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"
            if gkw_time_path is None
            else gkw_time_path
        )
        reference_slice = load_gkw_velocity_space_slice(
            gkw_distr1_path,
            gkw_distr2_path,
            gkw_distr3_path,
            gkw_distr4_path,
            time_path=gkw_time_path,
            source="gkw:cosin2:distr*.dat",
            notes="patched selected-ky production-control cosin2 final output",
        )

    observed = run_cyclone_base_case_velocity_space_slice(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        initial_profile="cosine2",
        normalization_model=normalization_model,
        parallel_derivative_model=parallel_derivative_model,
    )
    return audit_velocity_space_slice_phase_alignment(observed, reference_slice)


def run_cyclone_base_case_profile_operator_audit(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    output_window: int = 62,
    target_z: float = 0.09375,
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-11,
    target: BenchmarkTarget | None = None,
) -> CycloneProfileOperatorAudit:
    """Audit selected-mode parallel operators at the localized CBC profile gap.

    The default targets the row and grid point identified by the matched GKW
    ``parallel_phi.dat`` comparison: output window 62, ``t=3.72``, and
    ``z=0.09375`` on the 48-point GKW cell-centered grid.  The diagnostic is
    intentionally a solver-internal consistency check: it confirms whether the
    field solve and RHS assembly are exact at that state, while measuring the
    local matrix-versus-GKW-upwind parallel operator deltas.
    """

    from .physics import (
        adiabatic_density_numerator,
        adiabatic_quasineutrality_residual_from_density,
        dissipation,
        drift_field_drive,
        equilibrium_drive,
        gkw_parallel_field_drive,
        gkw_parallel_streaming,
        magnetic_drift_advection,
        mirror_force,
        parallel_field_drive,
        parallel_streaming,
        solve_adiabatic_electron_phi,
        solve_adiabatic_electron_phi_from_density,
        velocity_recurrence_control,
    )
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if output_window < 1:
        raise ValueError("output_window must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    rhs = setup["precompute"].rhs
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _ in range(output_window):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            if normalization_model == "weighted":
                amplitude = mode_chain_amplitude(
                    phi,
                    w_z=setup["geometry"].w_z,
                    connectivity=setup["connectivity"],
                )
            else:
                amplitude = jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)))
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            scale = jnp.maximum(amplitude[selected], jnp.asarray(1.0e-300, dtype=amplitude.dtype))
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = phi / scale

    z_index = int(jnp.argmin(jnp.abs(setup["parallel"].z - target_z)))
    power = jnp.abs(phi[:, ixzero, selected]) ** 2
    normalized_power = power / jnp.maximum(jnp.sum(power), jnp.asarray(1.0e-300, dtype=power.dtype))
    center = jnp.sum(normalized_power * setup["parallel"].z)
    second_moment = jnp.sum(normalized_power * (setup["parallel"].z - center) ** 2)

    matrix_parallel = parallel_streaming(state, rhs.D_z, rhs.parallel_streaming_coeff)
    gkw_parallel = gkw_parallel_streaming(state, rhs)
    matrix_field = parallel_field_drive(phi, rhs.D_z, rhs)
    gkw_field = gkw_parallel_field_drive(phi, rhs)
    streaming_delta_profile = _selected_mode_rms_profile(
        gkw_parallel - matrix_parallel,
        ixzero,
        selected,
    )
    field_drive_delta_profile = _selected_mode_rms_profile(
        gkw_field - matrix_field,
        ixzero,
        selected,
    )

    numerator = adiabatic_density_numerator(state, setup["precompute"].field)
    reconstructed_phi = solve_adiabatic_electron_phi_from_density(
        numerator, setup["precompute"].field
    )
    field_residual = adiabatic_quasineutrality_residual_from_density(
        phi,
        numerator,
        setup["precompute"].field,
    )
    field_residual_profile = jnp.abs(field_residual[:, ixzero, selected])

    manual = (
        gkw_parallel
        + magnetic_drift_advection(state, rhs.magnetic_drift_frequency)
        + mirror_force(state, rhs.D_vpar, rhs.mirror_force_coeff)
        + equilibrium_drive(phi, rhs)
        + gkw_field
        + drift_field_drive(phi, rhs)
        + dissipation(state, rhs.perpendicular_damping)
        + velocity_recurrence_control(
            state,
            rhs.velocity_recurrence_operator,
            rhs.velocity_recurrence_coeff,
        )
    )
    assembled = linear_residual(state, precomputed=setup["precompute"], phi=phi)
    rhs_assembly_error = _max_abs_error(assembled, manual)
    field_reconstruction_error = _max_abs_error(reconstructed_phi, phi)
    field_residual_max = jnp.max(jnp.abs(field_residual))
    boundary_streaming_delta = jnp.max(
        jnp.asarray([streaming_delta_profile[0], streaming_delta_profile[-1]])
    )
    boundary_field_drive_delta = jnp.max(
        jnp.asarray([field_drive_delta_profile[0], field_drive_delta_profile[-1]])
    )
    passed = (
        (field_residual_max <= tolerance)
        & (field_reconstruction_error <= tolerance)
        & (rhs_assembly_error <= tolerance)
    )

    return CycloneProfileOperatorAudit(
        normalized_phi_power=normalized_power,
        z_grid=setup["parallel"].z,
        streaming_delta_profile=streaming_delta_profile,
        field_drive_delta_profile=field_drive_delta_profile,
        field_residual_profile=field_residual_profile,
        time=output_window * steps_per_window * dt,
        z=setup["parallel"].z[z_index],
        output_window=output_window,
        z_index=z_index,
        peak_z=setup["parallel"].z[jnp.argmax(normalized_power)],
        second_moment=second_moment,
        local_streaming_delta=streaming_delta_profile[z_index],
        max_streaming_delta=jnp.max(streaming_delta_profile),
        boundary_streaming_delta=boundary_streaming_delta,
        local_field_drive_delta=field_drive_delta_profile[z_index],
        max_field_drive_delta=jnp.max(field_drive_delta_profile),
        boundary_field_drive_delta=boundary_field_drive_delta,
        field_residual_max=field_residual_max,
        field_reconstruction_error=field_reconstruction_error,
        rhs_assembly_error=rhs_assembly_error,
        passed=passed,
        notes=(
            "central profile operator audit; "
            f"initial_profile={initial_profile}, "
            f"normalization_model={normalization_model}, "
            f"target_z={target_z}"
        ),
    )


def run_cyclone_base_case_term_i_fortran_audit(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    output_window: int = 62,
    target_z: float = 0.09375,
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-12,
    target: BenchmarkTarget | None = None,
) -> CycloneTermIFortranAudit:
    """Compare the selected-mode Term I operator with GKW Fortran formulas.

    This reconstructs the ``linear_terms.F90::vpar_grad_df_4d_testnewbc``
    fourth-order, sign-dependent boundary coefficients from the Fortran
    formulas, including the ``idisp=2`` ``vpgr_rms`` recurrence-control
    convention, then compares that source reconstruction with the current JAX
    GKW-upwind implementation.
    """

    from .physics import gkw_parallel_streaming, solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if output_window < 1:
        raise ValueError("output_window must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    rhs = setup["precompute"].rhs
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _ in range(output_window):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            if normalization_model == "weighted":
                amplitude = mode_chain_amplitude(
                    phi,
                    w_z=setup["geometry"].w_z,
                    connectivity=setup["connectivity"],
                )
            else:
                amplitude = jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)))
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            scale = jnp.maximum(amplitude[selected], jnp.asarray(1.0e-300, dtype=amplitude.dtype))
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = phi / scale

    z_index = int(jnp.argmin(jnp.abs(setup["parallel"].z - target_z)))
    current = gkw_parallel_streaming(state, rhs)
    reference, source_coefficients, current_coefficients, expected_recurrence = (
        _gkw_fortran_term_i_reference(
            state,
            rhs.parallel_streaming_coeff[0],
            rhs.parallel_recurrence_coeff[0],
            rhs.gkw_parallel_stencil,
            setup["velocity"].vpar,
            parallel_recurrence_rate,
        )
    )
    term_error_profile = _selected_mode_rms_profile(current - reference, ixzero, selected)
    coefficient_error_profile = jnp.max(
        jnp.abs(current_coefficients - source_coefficients),
        axis=(0, 2),
    )
    current_term_profile = _selected_mode_rms_profile(current, ixzero, selected)
    reference_term_profile = _selected_mode_rms_profile(reference, ixzero, selected)
    recurrence_speed_error_profile = jnp.max(
        jnp.abs(rhs.parallel_recurrence_coeff[0] - expected_recurrence),
        axis=0,
    )
    sign_selection_error = _max_abs_error(
        jnp.sign(-rhs.parallel_streaming_coeff[0]),
        jnp.sign(-rhs.parallel_streaming_coeff[0]),
    )
    max_term_error = jnp.max(term_error_profile)
    max_coefficient_error = jnp.max(coefficient_error_profile)
    recurrence_speed_max_error = jnp.max(recurrence_speed_error_profile)
    passed = (
        (max_term_error <= tolerance)
        & (max_coefficient_error <= tolerance)
        & (recurrence_speed_max_error <= tolerance)
        & (sign_selection_error <= tolerance)
    )

    return CycloneTermIFortranAudit(
        term_error_profile=term_error_profile,
        coefficient_error_profile=coefficient_error_profile,
        current_term_profile=current_term_profile,
        reference_term_profile=reference_term_profile,
        recurrence_speed_error_profile=recurrence_speed_error_profile,
        time=output_window * steps_per_window * dt,
        z=setup["parallel"].z[z_index],
        output_window=output_window,
        z_index=z_index,
        local_term_error=term_error_profile[z_index],
        max_term_error=max_term_error,
        local_coefficient_error=coefficient_error_profile[z_index],
        max_coefficient_error=max_coefficient_error,
        recurrence_speed_max_error=recurrence_speed_max_error,
        sign_selection_error=sign_selection_error,
        passed=passed,
        notes=(
            "GKW Fortran Term I source reconstruction; "
            "linear_terms.F90:vpar_grad_df_4d_testnewbc; "
            f"initial_profile={initial_profile}, "
            f"normalization_model={normalization_model}, "
            f"target_z={target_z}"
        ),
    )


def run_cyclone_base_case_time_normalization_audit(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    n_windows: int = 8,
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-12,
    target: BenchmarkTarget | None = None,
) -> CycloneTimeNormalizationAudit:
    """Audit the selected-mode CBC RK4 and GKW normalization sequence.

    GKW's explicit linear path stores RK increments as ``dtim * RHS`` inside
    ``exp_integration.F90::rk4``.  At the first call, and after each diagnostic
    window, ``normalise.F90::normalize(2)`` computes an unweighted field norm,
    divides the full state by that factor, and reports
    ``log(factor) / (time-last_time)`` as the window growth rate.  This audit
    replays that source sequence and checks it against the solver's window
    trace when the same normalization model is selected.
    """

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude, rk4_step

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    selected = int(setup["selected_ky_index"])
    state = setup["state"]
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    window_duration = steps_per_window * dt

    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )

    def rhs_fn(state_value):
        return linear_residual(state_value, precomputed=setup["precompute"])

    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )
    reference_rk4_step = jax.jit(
        lambda state_value: _gkw_rk4_step_reference(state_value, dt, rhs_fn)
    )
    solver_rk4_step = jax.jit(lambda state_value: rk4_step(state_value, dt, rhs_fn))

    times = [0.0]
    factors = []
    source_growth = [jnp.asarray(0.0, dtype=jnp.float64)]
    post_norms = []
    field_linearity_errors = []

    phi = solve_phi(state)
    amplitude = _cyclone_phi_normalization_amplitude(
        phi,
        setup,
        normalization_model=normalization_model,
    )
    normalized = normalize_by_ky_amplitude(
        state,
        amplitude,
        log_normalization=log_normalization,
    )
    scale = jnp.maximum(amplitude[selected], jnp.asarray(1.0e-300, dtype=amplitude.dtype))
    state = normalized.state
    log_normalization = normalized.log_normalization
    normalized_phi = solve_phi(state)
    factors.append(scale)
    post_norms.append(
        _cyclone_phi_normalization_amplitude(
            normalized_phi,
            setup,
            normalization_model=normalization_model,
        )[selected]
    )
    field_linearity_errors.append(_max_abs_error(normalized_phi, phi / scale))

    rk4_step_error = _max_abs_error(reference_rk4_step(state), solver_rk4_step(state))

    for window in range(n_windows):
        state = advance_window(state)
        phi = solve_phi(state)
        amplitude = _cyclone_phi_normalization_amplitude(
            phi,
            setup,
            normalization_model=normalization_model,
        )
        scale = jnp.maximum(amplitude[selected], jnp.asarray(1.0e-300, dtype=amplitude.dtype))
        source_growth.append(jnp.log(scale) / window_duration)
        normalized = normalize_by_ky_amplitude(
            state,
            amplitude,
            log_normalization=log_normalization,
        )
        state = normalized.state
        log_normalization = normalized.log_normalization
        normalized_phi = solve_phi(state)
        factors.append(scale)
        post_norms.append(
            _cyclone_phi_normalization_amplitude(
                normalized_phi,
                setup,
                normalization_model=normalization_model,
            )[selected]
        )
        field_linearity_errors.append(_max_abs_error(normalized_phi, phi / scale))
        times.append((window + 1) * window_duration)

    trace = run_cyclone_base_case_trace(
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        dt=dt,
        nperiod=nperiod,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        normalize_each_window=True,
        normalization_model=normalization_model,
        initial_profile=initial_profile,
        target=target,
    )

    times_array = jnp.asarray(times, dtype=jnp.float64)
    expected_times = window_duration * jnp.arange(n_windows + 1, dtype=jnp.float64)
    factors_array = jnp.asarray(factors, dtype=jnp.float64)
    source_growth_array = jnp.asarray(source_growth, dtype=jnp.float64)
    post_norms_array = jnp.asarray(post_norms, dtype=jnp.float64)
    field_linearity_array = jnp.asarray(field_linearity_errors, dtype=jnp.float64)
    time_grid_error = _max_abs_error(times_array, expected_times)
    growth_sequence_error = _max_abs_error(source_growth_array, trace.window_growth)
    post_normalization_error = _max_abs_error(post_norms_array, jnp.ones_like(post_norms_array))
    max_field_linearity_error = jnp.max(field_linearity_array)
    passed = (
        (rk4_step_error <= tolerance)
        & (time_grid_error <= tolerance)
        & (growth_sequence_error <= tolerance)
        & (post_normalization_error <= tolerance)
        & (max_field_linearity_error <= tolerance)
    )

    return CycloneTimeNormalizationAudit(
        times=times_array,
        normalization_factor=factors_array,
        gkw_window_growth=source_growth_array,
        trace_window_growth=trace.window_growth,
        post_normalization_field_norm=post_norms_array,
        field_linearity_error=field_linearity_array,
        rk4_step_error=rk4_step_error,
        time_grid_error=time_grid_error,
        growth_sequence_error=growth_sequence_error,
        post_normalization_error=post_normalization_error,
        max_field_linearity_error=max_field_linearity_error,
        passed=passed,
        notes=(
            "GKW exp_integration.F90/rk4 and normalise.F90 sequence audit; "
            f"steps_per_window={steps_per_window}, n_windows={n_windows}, "
            f"initial_profile={initial_profile}, "
            f"normalization_model={normalization_model}"
        ),
    )


def run_cyclone_base_case_diagnostic_packing_audit(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    output_window: int = 62,
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-12,
    target: BenchmarkTarget | None = None,
) -> CycloneDiagnosticPackingAudit:
    """Audit GKW ``get_phi`` packing and field diagnostics for CBC.

    GKW stores electrostatic field values after the kinetic distribution in
    ``fdisi`` using the default ``index_function.F90`` order
    ``ix`` fastest, then ``imod``, then ``s``.  ``diagnostic.F90`` then calls
    ``get_phi`` and forms ``parallel_phi.dat``, ``kyspec``, and ``kxspec`` by
    summing ``abs(phi(imod,ix,is))**2`` with fixed normalizing counts.  This
    audit packs the solver field into that source layout, unpacks it through
    the same index contract, and compares the resulting diagnostics.
    """

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if output_window < 1:
        raise ValueError("output_window must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _ in range(output_window):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            amplitude = _cyclone_phi_normalization_amplitude(
                phi,
                setup,
                normalization_model=normalization_model,
            )
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            scale = jnp.maximum(amplitude[selected], jnp.asarray(1.0e-300, dtype=amplitude.dtype))
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = phi / scale

    n_z_actual, n_kx, n_ky = phi.shape
    field_offset = int(state.size)
    n_field_values = int(n_z_actual * n_kx * n_ky)
    packed = _pack_gkw_phi_field(phi, field_offset=field_offset)
    unpacked = _unpack_gkw_phi_field(
        packed,
        field_offset=field_offset,
        n_z=n_z_actual,
        n_kx=n_kx,
        n_ky=n_ky,
    )
    parallel_phi = _gkw_parallel_phi_diagnostic(phi)
    packed_parallel_phi = _gkw_parallel_phi_diagnostic(unpacked)
    ky_spectrum = _gkw_ky_spectrum_diagnostic(phi)
    packed_ky_spectrum = _gkw_ky_spectrum_diagnostic(unpacked)
    kx_spectrum = _gkw_kx_spectrum_diagnostic(phi)
    packed_kx_spectrum = _gkw_kx_spectrum_diagnostic(unpacked)
    selected_profile = jnp.abs(phi[:, ixzero, selected]) ** 2

    packing_roundtrip_error = _max_abs_error(unpacked, phi)
    parallel_phi_error = _max_abs_error(packed_parallel_phi, parallel_phi)
    selected_profile_error = _max_abs_error(parallel_phi, selected_profile)
    ky_spectrum_error = _max_abs_error(packed_ky_spectrum, ky_spectrum)
    kx_spectrum_error = _max_abs_error(packed_kx_spectrum, kx_spectrum)
    passed = (
        (packing_roundtrip_error <= tolerance)
        & (parallel_phi_error <= tolerance)
        & (selected_profile_error <= tolerance)
        & (ky_spectrum_error <= tolerance)
        & (kx_spectrum_error <= tolerance)
    )

    return CycloneDiagnosticPackingAudit(
        parallel_phi_profile=parallel_phi,
        packed_parallel_phi_profile=packed_parallel_phi,
        ky_spectrum=ky_spectrum,
        packed_ky_spectrum=packed_ky_spectrum,
        kx_spectrum=kx_spectrum,
        packed_kx_spectrum=packed_kx_spectrum,
        time=output_window * steps_per_window * dt,
        packing_roundtrip_error=packing_roundtrip_error,
        parallel_phi_error=parallel_phi_error,
        selected_profile_error=selected_profile_error,
        ky_spectrum_error=ky_spectrum_error,
        kx_spectrum_error=kx_spectrum_error,
        passed=passed,
        output_window=output_window,
        field_offset=field_offset,
        n_field_values=n_field_values,
        notes=(
            "GKW dist.F90/get_phi and diagnostic.F90 field-output audit; "
            f"steps_per_window={steps_per_window}, output_window={output_window}, "
            f"initial_profile={initial_profile}, "
            f"normalization_model={normalization_model}"
        ),
    )


def run_cyclone_base_case_matdat_matrix_audit(
    *,
    n_z: int = 8,
    n_vpar: int = 6,
    n_mu: int = 4,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-11,
    nonzero_threshold: float = 1.0e-14,
    max_size: int = 4096,
    target: BenchmarkTarget | None = None,
) -> CycloneMatdatMatrixAudit:
    """Audit reduced CBC residuals against GKW ``matdat.F90`` conventions.

    GKW stores linear terms as sparse triplets through ``put_element``, keeps
    any inhomogeneous term in ``source``, compresses duplicate ``(ii,jj)``
    entries, and the explicit integrator applies
    ``dtim * (source + mat * fdis_tmp)``.  This reduced audit builds the dense
    matrix-free residual matrix, reconstructs those sparse conventions, and
    verifies that the compressed and ``complex-real`` split actions reproduce
    the residual.
    """

    from .operators import dense_matrix_from_action, flatten_state, unflatten_state
    from .solver import linear_residual

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if nonzero_threshold < 0.0:
        raise ValueError("nonzero_threshold must be nonnegative")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    state = setup["state"]
    zero_state = jnp.zeros_like(state)

    def action(state_value):
        return linear_residual(state_value, precomputed=setup["precompute"])

    matrix = dense_matrix_from_action(action, state, max_size=max_size)
    vector = flatten_state(state)
    residual = action(state)
    residual_from_matrix = unflatten_state(matrix @ vector, state.shape)
    source = action(zero_state)
    source_vector = flatten_state(source)
    dt_value = jnp.asarray(dt, dtype=jnp.float64)
    explicit_delta = dt_value * (source_vector + matrix @ vector)
    expected_delta = dt_value * flatten_state(residual)

    probe = _deterministic_complex_probe(state.shape, dtype=state.dtype)
    combo = 0.37 * state - 0.21j * probe
    linearity_reference = 0.37 * action(state) - 0.21j * action(probe)
    linearity_error = _max_abs_error(action(combo), linearity_reference)

    triplets = _gkw_triplets_from_dense_matrix(np.asarray(matrix), threshold=nonzero_threshold)
    compressed = _gkw_compress_triplets(*triplets, n_rows=int(vector.shape[0]))
    compressed_action = _gkw_apply_compressed_triplets(compressed, np.asarray(vector))
    split_action, n_real_entries, n_complex_entries = _gkw_complex_real_split_action(
        np.asarray(matrix),
        np.asarray(vector),
        threshold=nonzero_threshold,
    )

    matrix_action_error = _max_abs_error(residual_from_matrix, residual)
    source_max_abs = jnp.max(jnp.abs(source_vector))
    explicit_delta_error = _max_abs_error(explicit_delta, expected_delta)
    compressed_action_error = _max_abs_error(compressed_action, np.asarray(matrix @ vector))
    complex_real_split_error = _max_abs_error(split_action, np.asarray(matrix @ vector))
    max_abs_matrix_entry = jnp.max(jnp.abs(matrix))
    passed = (
        (matrix_action_error <= tolerance)
        & (source_max_abs <= tolerance)
        & (explicit_delta_error <= tolerance)
        & (compressed_action_error <= tolerance)
        & (complex_real_split_error <= tolerance)
        & (linearity_error <= tolerance)
    )

    return CycloneMatdatMatrixAudit(
        matrix_action_error=matrix_action_error,
        source_max_abs=source_max_abs,
        explicit_delta_error=explicit_delta_error,
        compressed_action_error=compressed_action_error,
        complex_real_split_error=complex_real_split_error,
        linearity_error=linearity_error,
        max_abs_matrix_entry=max_abs_matrix_entry,
        passed=passed,
        n_state=int(vector.shape[0]),
        n_nonzero=int(triplets[0].shape[0]),
        n_duplicate_triplets=int(2 * triplets[0].shape[0]),
        n_real_entries=int(n_real_entries),
        n_complex_entries=int(n_complex_entries),
        notes=(
            "GKW matdat.F90 sparse matrix/source convention audit; "
            f"n_z={n_z}, n_vpar={n_vpar}, n_mu={n_mu}, "
            f"initial_profile={initial_profile}, "
            f"parallel_backend={parallel_backend}, velocity_backend={velocity_backend}"
        ),
    )


def run_cyclone_base_case_coefficient_source_audit(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    output_window: int = 62,
    target_z: float = 0.09375,
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-11,
    target: BenchmarkTarget | None = None,
) -> CycloneCoefficientSourceAudit:
    """Audit GKW source coefficient formulas for Terms II, IV, V, VII, and VIII."""

    from .physics import (
        drift_field_drive,
        equilibrium_drive,
        gkw_parallel_field_drive,
        magnetic_drift_advection,
        mirror_force,
        solve_adiabatic_electron_phi,
    )
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if output_window < 0:
        raise ValueError("output_window must be nonnegative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    rhs = setup["precompute"].rhs
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(
            state_value,
            setup["precompute"].field,
        )
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _ in range(output_window):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            if normalization_model == "weighted":
                amplitude = mode_chain_amplitude(
                    phi,
                    w_z=setup["geometry"].w_z,
                    connectivity=setup["connectivity"],
                )
            else:
                amplitude = jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)))
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            scale = jnp.maximum(
                amplitude[selected],
                jnp.asarray(1.0e-300, dtype=amplitude.dtype),
            )
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = phi / scale

    source = _cyclone_source_term_arrays(setup, metadata)
    term_ii = -1j * source["drift_frequency"] * state
    term_iv = source["mirror_coeff"][None, :, :, None, None] * _apply_first_axis_matrix(
        rhs.D_vpar, state
    )
    term_v = source["equilibrium_drive_coeff"] * phi[None, None, :, :, :]
    term_vii = _gkw_fortran_term_vii_reference(
        phi,
        source["parallel_field_coeff"],
        rhs.flr_factors.bessel_j0[0],
        rhs.gkw_parallel_stencil,
    )
    term_viii = source["drift_field_coeff"] * phi[None, None, :, :, :]

    current_term_ii = magnetic_drift_advection(state, rhs.magnetic_drift_frequency)
    current_term_iv = mirror_force(state, rhs.D_vpar, rhs.mirror_force_coeff)
    current_term_v = equilibrium_drive(phi, rhs)
    current_term_vii = gkw_parallel_field_drive(phi, rhs)
    current_term_viii = drift_field_drive(phi, rhs)

    term_errors = jnp.asarray(
        [
            _max_abs_error(current_term_ii, term_ii),
            _max_abs_error(current_term_iv, term_iv),
            _max_abs_error(current_term_v, term_v),
            _max_abs_error(current_term_vii, term_vii),
            _max_abs_error(current_term_viii, term_viii),
        ],
        dtype=jnp.float64,
    )
    coefficient_errors = jnp.asarray(
        [
            _max_abs_error(rhs.magnetic_drift_frequency[0], source["drift_frequency"]),
            _max_abs_error(rhs.mirror_force_coeff[0], source["mirror_coeff"]),
            _max_abs_error(source["equilibrium_drive_coeff"], source["current_drive_coeff"]),
            _max_abs_error(source["parallel_field_coeff"], source["current_field_coeff"]),
            _max_abs_error(source["drift_field_coeff"], source["current_drift_field_coeff"]),
        ],
        dtype=jnp.float64,
    )
    insertion_errors = jnp.asarray(
        [
            _max_abs_error(term_ii, current_term_ii),
            _max_abs_error(term_iv, current_term_iv),
            _max_abs_error(term_v, current_term_v),
            _max_abs_error(term_vii, current_term_vii),
            _max_abs_error(term_viii, current_term_viii),
        ],
        dtype=jnp.float64,
    )
    max_term_error = jnp.max(term_errors)
    max_coefficient_error = jnp.max(coefficient_errors)
    max_insertion_error = jnp.max(insertion_errors)
    max_abs_error = jnp.max(
        jnp.asarray(
            [max_term_error, max_coefficient_error, max_insertion_error],
            dtype=jnp.float64,
        )
    )
    z_index = int(jnp.argmin(jnp.abs(setup["parallel"].z - target_z)))
    term_names = (
        "Term II vdgradf",
        "Term IV trapdf",
        "Term V ve_grad_fm",
        "Term VII vpgrphi",
        "Term VIII vd_grad_phi_fm",
    )
    return CycloneCoefficientSourceAudit(
        term_errors=term_errors,
        coefficient_errors=coefficient_errors,
        insertion_errors=insertion_errors,
        time=output_window * steps_per_window * dt,
        z=setup["parallel"].z[z_index],
        z_index=z_index,
        max_term_error=max_term_error,
        max_coefficient_error=max_coefficient_error,
        max_insertion_error=max_insertion_error,
        max_abs_error=max_abs_error,
        passed=max_abs_error <= tolerance,
        term_names=term_names,
        output_window=output_window,
        notes=(
            "GKW source-level coefficient audit; "
            "linear_terms.F90:vdgradf, trapdf_4d, ve_grad_fm, "
            "vpgrphi_3_newbc, vd_grad_phi_fm; "
            f"initial_profile={initial_profile}, "
            f"normalization_model={normalization_model}, "
            f"target_z={target_z}"
        ),
    )


def run_cyclone_base_case_igh_arakawa_audit(
    *,
    n_z: int = 48,
    n_vpar: int = 32,
    n_mu: int = 8,
    vpar_max: float | None = None,
    mu_max: float | None = None,
    dt: float | None = None,
    nperiod: int | None = None,
    steps_per_window: int = 20,
    output_window: int = 62,
    target_z: float = 0.09375,
    parallel_recurrence_rate: float | None = None,
    velocity_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    normalization_model: str = "gkw_unweighted",
    initial_profile: str | None = "cosine",
    tolerance: float = 5.0e-11,
    target: BenchmarkTarget | None = None,
) -> CycloneIghArakawaAudit:
    """Compare GKW's fused ``igh`` Term I/IV path with the separated fallback."""

    from .physics import (
        gkw_parallel_streaming,
        mirror_force,
        solve_adiabatic_electron_phi,
        velocity_recurrence_control,
    )
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if output_window < 0:
        raise ValueError("output_window must be nonnegative")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if normalization_model not in ("weighted", "gkw_unweighted"):
        raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    vpar_max = float(metadata["vpar_max"] if vpar_max is None else vpar_max)
    nperiod = int(metadata["nperiod"] if nperiod is None else nperiod)
    dt = float(metadata["dt"] if dt is None else dt)
    parallel_recurrence_rate = float(
        metadata["disp_par"] if parallel_recurrence_rate is None else parallel_recurrence_rate
    )
    velocity_recurrence_rate = float(
        metadata.get("disp_vp", 0.2)
        if velocity_recurrence_rate is None
        else velocity_recurrence_rate
    )
    parallel_backend = str(
        metadata.get("parallel_backend", "finite_difference")
        if parallel_backend is None
        else parallel_backend
    )
    parallel_boundary = str(
        metadata.get("parallel_boundary", "zero")
        if parallel_boundary is None
        else parallel_boundary
    )
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    initial_profile = str(
        metadata.get("initial_profile", "cosine2") if initial_profile is None else initial_profile
    )

    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=vpar_max,
        mu_max=mu_max,
        nperiod=nperiod,
        parallel_recurrence_rate=parallel_recurrence_rate,
        velocity_recurrence_rate=velocity_recurrence_rate,
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model="gkw_upwind",
        velocity_backend=velocity_backend,
        initial_profile=initial_profile,
    )
    rhs = setup["precompute"].rhs
    state = setup["state"]
    selected = int(setup["selected_ky_index"])
    ixzero = int(setup["fourier"].ixzero)
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    solve_phi = jax.jit(
        lambda state_value: solve_adiabatic_electron_phi(
            state_value,
            setup["precompute"].field,
        )
    )
    advance_window = jax.jit(
        lambda state_value: (
            integrate_fixed_step(
                state_value,
                dt,
                steps_per_window,
                linear_residual,
                setup["precompute"],
                store_history=False,
            ).state
        )
    )

    phi = solve_phi(state)
    for _ in range(output_window):
        state = advance_window(state)
        phi = solve_phi(state)
        if normalize_each_window:
            if normalization_model == "weighted":
                amplitude = mode_chain_amplitude(
                    phi,
                    w_z=setup["geometry"].w_z,
                    connectivity=setup["connectivity"],
                )
            else:
                amplitude = jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)))
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            scale = jnp.maximum(
                amplitude[selected],
                jnp.asarray(1.0e-300, dtype=amplitude.dtype),
            )
            state = normalized.state
            log_normalization = normalized.log_normalization
            phi = phi / scale

    fused_total, fused_hamiltonian, parallel_diffusion, velocity_diffusion = (
        _gkw_fortran_igh_reference(
            state,
            setup,
            disp_par=parallel_recurrence_rate,
            disp_vp=velocity_recurrence_rate,
        )
    )
    separated = (
        gkw_parallel_streaming(state, rhs)
        + mirror_force(state, rhs.D_vpar, rhs.mirror_force_coeff)
        + velocity_recurrence_control(
            state,
            rhs.velocity_recurrence_operator,
            rhs.velocity_recurrence_coeff,
        )
    )
    delta = fused_total - separated
    fused_profile = _selected_mode_rms_profile(fused_total, ixzero, selected)
    separated_profile = _selected_mode_rms_profile(separated, ixzero, selected)
    delta_profile = _selected_mode_rms_profile(delta, ixzero, selected)
    parallel_diffusion_profile = _selected_mode_rms_profile(
        parallel_diffusion,
        ixzero,
        selected,
    )
    velocity_diffusion_profile = _selected_mode_rms_profile(
        velocity_diffusion,
        ixzero,
        selected,
    )
    max_delta = jnp.max(delta_profile)
    fused_scale = jnp.max(
        jnp.maximum(fused_profile, jnp.asarray(1.0e-300, dtype=fused_profile.dtype))
    )
    relative_delta = max_delta / fused_scale
    z_index = int(jnp.argmin(jnp.abs(setup["parallel"].z - target_z)))

    return CycloneIghArakawaAudit(
        fused_profile=fused_profile,
        separated_profile=separated_profile,
        delta_profile=delta_profile,
        parallel_diffusion_profile=parallel_diffusion_profile,
        velocity_diffusion_profile=velocity_diffusion_profile,
        time=output_window * steps_per_window * dt,
        z=setup["parallel"].z[z_index],
        z_index=z_index,
        local_delta=delta_profile[z_index],
        max_delta=max_delta,
        relative_delta=relative_delta,
        max_parallel_diffusion=jnp.max(parallel_diffusion_profile),
        max_velocity_diffusion=jnp.max(velocity_diffusion_profile),
        passed=max_delta <= tolerance,
        output_window=output_window,
        notes=(
            "GKW ltrapping_arakawa fused Term I/IV audit; "
            "linear_terms.F90:igh, jhg_interior, igh_zero_two, igh_two, diffus; "
            f"disp_par={parallel_recurrence_rate:g}, "
            f"disp_vp={velocity_recurrence_rate:g}, "
            f"initial_profile={initial_profile}, "
            f"normalization_model={normalization_model}, "
            f"target_z={target_z}; "
            "passed means the separated solver fallback matches the fused GKW path"
        ),
    )


def _cyclone_trace_csv_columns() -> tuple[str, ...]:
    return ("time", *CycloneTrace._dynamic_fields[1:])


def _cyclone_trace_comparison_fields() -> tuple[str, ...]:
    return (
        *CycloneTrace._dynamic_fields,
        "physical_phi_norm",
        "physical_state_norm",
        "physical_rhs_norm",
    )


def _cyclone_trace_comparison_field(trace: CycloneTrace, name: str):
    if name == "physical_phi_norm":
        return trace.phi_norm * jnp.exp(trace.log_normalization)
    if name == "physical_state_norm":
        return trace.state_norm * jnp.exp(trace.log_normalization)
    if name == "physical_rhs_norm":
        return trace.rhs_norm * jnp.exp(trace.log_normalization)
    return getattr(trace, name)


def _cyclone_trace_csv_column_map(header: dict[str, object]) -> dict[str, str]:
    names = set(header)
    column_map = {}
    for name in CycloneTrace._dynamic_fields:
        if name in names:
            column_map[name] = name
    if "times" not in column_map and "time" in names:
        column_map["times"] = "time"
    return column_map


def _cyclone_velocity_recurrence_rate(
    metadata: dict[str, object],
    requested: float | None,
    parallel_derivative_model: str,
) -> float:
    if requested is not None:
        return float(requested)
    if parallel_derivative_model == "gkw_igh":
        return float(metadata.get("disp_vp", 0.2))
    return 0.0


def run_cyclone_base_case_term_parity_audit(
    *,
    n_z: int = 16,
    n_vpar: int = 12,
    n_mu: int = 6,
    tolerance: float = 5.0e-13,
    target: BenchmarkTarget | None = None,
) -> CycloneTermParityReport:
    """Audit CBC drift, drive, field-drive, boundary, and grid conventions."""

    from .physics import (
        dissipation,
        drift_field_drive,
        equilibrium_drive,
        gkw_parallel_field_drive,
        gkw_parallel_streaming,
        magnetic_drift_advection,
        mirror_force,
        parallel_field_drive,
        parallel_streaming,
        solve_adiabatic_electron_phi,
        velocity_recurrence_control,
    )
    from .solver import linear_residual

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    setup = _build_cyclone_base_case_setup(
        target,
        n_z=n_z,
        n_vpar=n_vpar,
        n_mu=n_mu,
        vpar_max=float(metadata.get("vpar_max", 3.0)),
        mu_max=None,
        nperiod=int(metadata.get("nperiod", 5)),
        parallel_recurrence_rate=float(metadata.get("disp_par", 1.0)),
        parallel_backend=str(metadata.get("parallel_backend", "finite_difference")),
        parallel_boundary=str(metadata.get("parallel_boundary", "zero")),
        parallel_derivative_model="gkw_upwind",
        velocity_backend=str(metadata.get("velocity_backend", "finite_difference")),
        initial_profile="cosine2",
    )
    rhs = setup["precompute"].rhs
    state = setup["state"]
    phi = solve_adiabatic_electron_phi(state, setup["precompute"].field)

    vpar = setup["velocity"].vpar[:, None, None, None, None]
    mu = setup["velocity"].mu[None, :, None, None, None]
    B = setup["geometry"].B[None, None, :, None, None]
    kx = setup["fourier"].kx[None, None, None, :, None]
    ky = setup["fourier"].ky[None, None, None, None, :]
    D_x = setup["geometry"].D_x[None, None, :, None, None]
    D_y = setup["geometry"].D_y[None, None, :, None, None]
    expected_drift = (vpar**2 + mu * B) * (kx * D_x + ky * D_y)
    drift_error = _max_abs_error(rhs.magnetic_drift_frequency[0], expected_drift)

    gyro_phi = rhs.flr_factors.bessel_j0[0] * phi[None, :, :, :]
    expected_drive = (
        1j
        * rhs.E_y[None, None, :, None, None]
        * rhs.ky[None, None, None, None, :]
        * gyro_phi[None, :, :, :, :]
        * rhs.maxwellian[0][..., None, None]
        * rhs.drive_factor[0][..., None, None]
    )
    equilibrium_drive_error = _max_abs_error(equilibrium_drive(phi, rhs), expected_drive)

    expected_drift_field = (
        -1j
        * rhs.charge_over_temperature[0]
        * rhs.magnetic_drift_frequency[0]
        * rhs.maxwellian[0][..., None, None]
        * gyro_phi[None, :, :, :, :]
    )
    drift_field_error = _max_abs_error(drift_field_drive(phi, rhs), expected_drift_field)

    matrix_parallel = parallel_streaming(state, rhs.D_z, rhs.parallel_streaming_coeff)
    gkw_parallel = gkw_parallel_streaming(state, rhs)
    matrix_field = parallel_field_drive(phi, rhs.D_z, rhs)
    gkw_field = gkw_parallel_field_drive(phi, rhs)
    boundary_delta = jnp.maximum(
        _max_abs_error(gkw_parallel, matrix_parallel),
        _max_abs_error(gkw_field, matrix_field),
    )
    boundary_map_error = _cyclone_boundary_map_error(setup)
    normalization_error = _cyclone_normalization_error(setup, metadata)

    manual = (
        gkw_parallel
        + magnetic_drift_advection(state, rhs.magnetic_drift_frequency)
        + mirror_force(state, rhs.D_vpar, rhs.mirror_force_coeff)
        + equilibrium_drive(phi, rhs)
        + gkw_field
        + drift_field_drive(phi, rhs)
        + dissipation(state, rhs.perpendicular_damping)
        + velocity_recurrence_control(
            state,
            rhs.velocity_recurrence_operator,
            rhs.velocity_recurrence_coeff,
        )
    )
    assembled = linear_residual(state, precomputed=setup["precompute"], phi=phi)
    assembly_error = _max_abs_error(assembled, manual)

    term_names = (
        "drift_frequency",
        "equilibrium_drive",
        "drift_field_drive",
        "boundary_map",
        "grid_normalization",
        "rhs_assembly",
    )
    term_errors = jnp.asarray(
        [
            drift_error,
            equilibrium_drive_error,
            drift_field_error,
            boundary_map_error,
            normalization_error,
            assembly_error,
        ],
        dtype=jnp.float64,
    )
    max_abs_error = jnp.max(term_errors)
    return CycloneTermParityReport(
        term_errors=term_errors,
        max_abs_error=max_abs_error,
        passed=max_abs_error <= tolerance,
        term_names=term_names,
        notes=(
            "CBC term audit against GKW/Gyaradax algebraic conventions; "
            f"matrix_vs_gkw_parallel_boundary_delta={float(boundary_delta):.6e}; "
            "the growth-rate gate remains separate"
        ),
    )


def run_solver_geometry_to_gx_eik_gate(
    geometry,
    reference: GxEikGeometryReference,
    fourier_grid,
    *,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Validate solver-produced geometry arrays against a GX/GS2 eik table."""

    target = target or BenchmarkTarget(
        name="gx_eik_solver_geometry_parity",
        quantity="max_abs_geometry_error",
        reference_value=0.0,
        tolerance=1.0e-10,
        source=reference.source,
    )
    report = compare_geometry_to_gx_eik_reference(geometry, reference, fourier_grid)
    return evaluate_benchmark_gate(
        report.max_abs_error,
        target,
        normalize_by_tolerance=False,
        notes="solver-produced geometry arrays compared field-by-field to GX/GS2 eik contract",
    )


def run_geometry_to_gx_eik_export_gate(
    geometry,
    fourier_grid,
    *,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Validate that a solver geometry exports to the GX/GS2 eik contract.

    This is an adapter-contract gate for stellarator geometry produced by DESC
    or another upstream equilibrium code.  It checks the eik-compatible fields
    that the solver can export directly: ``B``, ``gradpar``, metric elements,
    summed radial/binormal magnetic drifts, and ``k_perp^2``.  The internal
    mirror-force coefficient ``G`` is not included because standard eik tables
    do not carry it as an independent field.
    """

    reference = geometry_to_gx_eik_reference(geometry)
    target = target or BenchmarkTarget(
        name=f"{getattr(geometry, 'source', 'solver')}_gx_eik_export_contract",
        quantity="max_abs_eik_export_error",
        reference_value=0.0,
        tolerance=1.0e-12,
        source=reference.source,
    )
    report = compare_geometry_to_gx_eik_reference(
        geometry,
        reference,
        fourier_grid,
        include_mirror_proxy=False,
    )
    return evaluate_benchmark_gate(
        report.max_abs_error,
        target,
        normalize_by_tolerance=False,
        notes=(
            "solver-produced stellarator geometry exported to GX/GS2 eik-compatible "
            "fields; internal mirror coefficient G is tracked separately"
        ),
    )


def build_desc_gx_eik_reference_from_path(
    desc_path,
    *,
    ntheta: int = 32,
    npol: int = 1,
    rho: float = 0.5,
    alpha: float = 0.0,
    zeta_center: float = 0.0,
    shift_grad_alpha: bool = True,
    file_format: str | None = None,
    index: int = -1,
    loader=None,
) -> GxEikGeometryReference:
    """Evaluate DESC geometry using the GX DESC ``eik.out`` convention.

    This mirrors the field-line normalization in
    ``relevant-codes/gx/geometry_modules/desc/gx_desc_geo.py`` while using the
    current DESC coordinate API.  It is intentionally separate from the raw
    physical-array DESC adapter: this path produces the exact GS2/GX fields
    used by external ``eik.out`` parity tests.
    """

    if ntheta < 2 or ntheta % 2:
        raise ValueError("ntheta must be an even integer at least 2")
    if npol < 1:
        raise ValueError("npol must be at least 1")
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1]")

    from .geometry.desc_adapter import load_desc_equilibrium

    linear_grid_cls, get_rtz_grid = _import_desc_coordinate_helpers()
    eq = load_desc_equilibrium(
        desc_path,
        file_format=file_format,
        index=index,
        loader=loader,
    )
    profile_grid = linear_grid_cls(
        rho=np.asarray([rho]),
        theta=np.asarray([0.0]),
        zeta=np.asarray([0.0]),
    )
    boundary_grid = linear_grid_cls(
        rho=np.asarray([1.0]),
        theta=np.asarray([0.0]),
        zeta=np.asarray([0.0]),
    )
    profile = eq.compute(["iota", "iota_r", "a"], grid=profile_grid)
    boundary = eq.compute(["psi"], grid=boundary_grid)
    iota = float(np.ravel(np.asarray(profile["iota"]))[0])
    shear = float(np.ravel(np.asarray(profile["iota_r"]))[0])
    minor_radius = float(np.ravel(np.asarray(profile["a"]))[0])
    psib = float(np.ravel(np.asarray(boundary["psi"]))[0])
    if iota == 0.0:
        raise ValueError("DESC iota must be nonzero for GX eik field-line sampling")
    if psib == 0.0:
        raise ValueError("DESC boundary psi must be nonzero for GX eik normalization")

    effective_zeta_center = float(zeta_center) if shift_grad_alpha else 0.0
    nzgrid = ntheta // 2
    zeta = np.linspace(
        (-np.pi * npol - alpha) / abs(iota),
        (np.pi * npol - alpha) / abs(iota),
        2 * nzgrid + 1,
    )
    grid = get_rtz_grid(eq, rho, alpha, zeta, coordinates="raz", iota=iota)
    data = eq.compute(list(_DESC_GX_EIK_COMPUTE_KEYS), grid=grid)
    reference = _desc_gx_eik_reference_from_data(
        data,
        zeta=zeta,
        rho=float(rho),
        alpha=float(alpha),
        zeta_center=effective_zeta_center,
        iota=iota,
        shear=shear,
        psib=psib,
        minor_radius=minor_radius,
        nzgrid=nzgrid,
        npol=int(npol),
        source=str(desc_path),
    )
    return reference


def run_desc_gx_eik_external_geometry_gate(
    desc_path,
    eik_path,
    *,
    rho: float = 0.5,
    alpha: float = 0.0,
    zeta_center: float = 0.0,
    shift_grad_alpha: bool = True,
    file_format: str | None = None,
    index: int = -1,
    fourier_grid=None,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Compare DESC-produced GX-convention geometry with an external eik file."""

    from .grids import build_fourier_grid
    from .types import FourierGridSpec

    external = load_gx_eik_geometry_reference(eik_path)
    ntheta, npol = _infer_gx_eik_dimensions(external)
    solver_reference = build_desc_gx_eik_reference_from_path(
        desc_path,
        ntheta=ntheta,
        npol=npol,
        rho=rho,
        alpha=alpha,
        zeta_center=zeta_center,
        shift_grad_alpha=shift_grad_alpha,
        file_format=file_format,
        index=index,
    )
    parallel = _parallel_grid_from_eik_theta(solver_reference.theta)
    fourier = fourier_grid or build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    geometry = build_flux_tube_geometry_from_gx_eik_reference(solver_reference, parallel)
    report = compare_geometry_to_gx_eik_reference(geometry, external, fourier)
    target = target or BenchmarkTarget(
        name="desc_gx_external_eik_geometry_parity",
        quantity="max_abs_geometry_error",
        reference_value=0.0,
        tolerance=2.0e-6,
        source=str(eik_path),
        metadata=(
            ("desc_path", str(desc_path)),
            ("ntheta", int(ntheta)),
            ("npol", int(npol)),
            ("rho", float(rho)),
            ("alpha", float(alpha)),
        ),
    )
    return evaluate_benchmark_gate(
        report.max_abs_error,
        target,
        normalize_by_tolerance=False,
        notes=(
            "solver-produced DESC geometry evaluated in GX eik convention and "
            "compared field-by-field to a matched external eik.out fixture"
        ),
    )


def run_gx_gist_external_eik_suite_gate(
    paths,
    *,
    n_theta: int = 33,
    fourier_grid=None,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Run external GIST/GS2 eik fixtures through the solver geometry contract."""

    from .grids import build_fourier_grid, build_parallel_grid
    from .types import FourierGridSpec, ParallelGridSpec

    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError("at least one eik reference path is required")
    if n_theta < 2:
        raise ValueError("n_theta must be at least 2")
    fourier = fourier_grid or build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )
    errors = []
    for path in paths:
        reference = load_gx_eik_geometry_reference(path)
        sampled = resample_gx_eik_geometry_reference(reference, theta)
        geometry = build_flux_tube_geometry_from_gx_eik_reference(sampled, parallel)
        report = compare_geometry_to_gx_eik_reference(geometry, sampled, fourier)
        errors.append(report.max_abs_error)
    observed = jnp.max(jnp.asarray(errors, dtype=jnp.float64))
    target = target or BenchmarkTarget(
        name="gx_gist_external_eik_suite",
        quantity="max_abs_geometry_error",
        reference_value=0.0,
        tolerance=1.0e-10,
        source=";".join(str(path) for path in paths),
        metadata=(("n_references", len(paths)), ("n_theta", int(n_theta))),
    )
    return evaluate_benchmark_gate(
        observed,
        target,
        normalize_by_tolerance=False,
        notes=(
            "independent GX/VMEC GIST eik fixtures mapped into solver geometry "
            "and compared field-by-field"
        ),
    )


def run_gx_eik_geometry_gate(
    reference: GxEikGeometryReference,
    parallel_grid,
    fourier_grid,
    *,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Validate the imported GX/GS2 eik metric against the solver kperp contract."""

    target = target or BenchmarkTarget(
        name="gx_eik_kperp_contract",
        quantity="max_abs_kperp2_error",
        reference_value=0.0,
        tolerance=1.0e-12,
        source=reference.source,
    )
    geometry = build_flux_tube_geometry_from_gx_eik_reference(reference, parallel_grid)
    report = compare_geometry_to_gx_eik_reference(geometry, reference, fourier_grid)
    observed = report.max_abs_kperp2_error
    return evaluate_benchmark_gate(
        observed,
        target,
        normalize_by_tolerance=False,
        notes="GX/GS2 eik metric imported into solver kperp contract",
    )


def benchmark_target_residual(value, target: BenchmarkTarget, *, normalize_by_tolerance=True):
    """Return a signed residual between an observed scalar and a benchmark target."""

    residual = jnp.asarray(value) - target.reference_value
    if normalize_by_tolerance:
        residual = residual / jnp.asarray(target.tolerance, dtype=residual.dtype)
    return residual


def benchmark_target_cost(value, target: BenchmarkTarget, *, normalize_by_tolerance=True):
    """Return ``0.5 * residual**2`` for a scalar benchmark target."""

    residual = benchmark_target_residual(
        value,
        target,
        normalize_by_tolerance=normalize_by_tolerance,
    )
    return 0.5 * residual**2


_DESC_GX_EIK_COMPUTE_KEYS = (
    "B",
    "|B|",
    "lambda",
    "lambda_r",
    "lambda_t",
    "lambda_z",
    "|grad(rho)|",
    "g^rr",
    "g^tt",
    "g^zz",
    "g^rt",
    "g^rz",
    "g^tz",
    "g_tz",
    "g_tt",
    "g_zz",
    "B_theta",
    "B_zeta",
    "B_rho",
    "|B|_t",
    "|B|_z",
    "|B|_r",
    "B^theta",
    "B^zeta_r",
    "B^theta_r",
    "B^zeta",
    "e_theta",
    "e_theta_r",
    "e_zeta_r",
    "e_zeta",
    "p_r",
    "grad(psi)",
    "sqrt(g)",
)


def _import_netcdf_dataset():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError(
            "load_gx_growth_rate_reference requires netCDF4; install DESC/GX "
            "analysis dependencies or add netCDF4 to the environment"
        ) from exc
    return Dataset


def _import_desc_coordinate_helpers():
    try:
        from desc.equilibrium.coords import get_rtz_grid
        from desc.grid import LinearGrid
    except ImportError as exc:
        raise ImportError(
            "DESC is required for DESC/GX eik parity. Install desc-opt or add "
            "relevant-codes/DESC to PYTHONPATH."
        ) from exc
    return LinearGrid, get_rtz_grid


def _numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append([float(value) for value in stripped.split()])
        except ValueError:
            continue
    return rows


def _load_gkw_distr_array(path, name: str) -> np.ndarray:
    rows = _numeric_rows(Path(path))
    if not rows:
        raise ValueError(f"GKW {name} contains no numeric rows")
    n_column = len(rows[0])
    if n_column == 0 or any(len(row) != n_column for row in rows):
        raise ValueError(f"GKW {name} rows must have a consistent nonzero length")
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"GKW {name} must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"GKW {name} contains non-finite values")
    return values


def _gkw_distr_snapshot_paths(directory: Path) -> list[tuple[int, tuple[Path, Path, Path, Path]]]:
    snapshots = []
    for path in sorted(directory.glob("distr1_*.dat")):
        suffix = path.stem.removeprefix("distr1_")
        try:
            index = int(suffix)
        except ValueError:
            continue
        paths = tuple(directory / f"distr{component}_{suffix}.dat" for component in range(1, 5))
        if not all(candidate.is_file() for candidate in paths):
            missing = [candidate.name for candidate in paths if not candidate.is_file()]
            raise FileNotFoundError(f"incomplete GKW distr snapshot {suffix}: missing {missing}")
        snapshots.append((index, paths))
    return snapshots


def _gkw_times_for_snapshot_indices(times: np.ndarray, snapshot_indices: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    snapshot_indices = np.asarray(snapshot_indices, dtype=np.int32)
    if snapshot_indices.ndim != 1 or snapshot_indices.shape[0] < 1:
        raise ValueError("snapshot_indices must be nonempty and one-dimensional")
    if times.shape[0] == snapshot_indices.shape[0]:
        return times
    first_index = int(snapshot_indices[0])
    if first_index <= 0:
        raise ValueError("cannot align snapshot indices with time.dat rows")
    output_rows = snapshot_indices / first_index
    rounded_rows = np.rint(output_rows).astype(np.int32)
    expected = first_index * np.arange(1, snapshot_indices.shape[0] + 1, dtype=np.int32)
    if not np.array_equal(snapshot_indices, expected):
        raise ValueError("snapshot indices are not a regular diagnostic-window sequence")
    if np.any(rounded_rows < 1) or int(rounded_rows[-1]) > times.shape[0]:
        raise ValueError("time.dat does not contain enough rows for distr snapshots")
    return times[rounded_rows - 1]


def _velocity_slice_spatial_variants(values):
    values = jnp.asarray(values)
    numpy_values = np.asarray(values)
    fortran_flat_c_shape = jnp.asarray(
        np.reshape(np.ravel(numpy_values, order="F"), numpy_values.shape, order="C"),
        dtype=values.dtype,
    )
    c_flat_fortran_shape = jnp.asarray(
        np.reshape(np.ravel(numpy_values, order="C"), numpy_values.shape, order="F"),
        dtype=values.dtype,
    )
    return (
        ("direct_mu_rows_vpar_columns", values),
        ("reverse_vpar_columns", values[:, ::-1]),
        ("reverse_mu_rows", values[::-1, :]),
        ("reverse_mu_rows_vpar_columns", values[::-1, ::-1]),
        ("roll_vpar_minus_1", jnp.roll(values, -1, axis=1)),
        ("roll_vpar_plus_1", jnp.roll(values, 1, axis=1)),
        ("roll_mu_minus_1", jnp.roll(values, -1, axis=0)),
        ("roll_mu_plus_1", jnp.roll(values, 1, axis=0)),
        ("fortran_flatten_c_reshape", fortran_flat_c_shape),
        ("c_flatten_fortran_reshape", c_flat_fortran_shape),
    )


def _gkw_time_dat_times(path) -> np.ndarray:
    rows = _numeric_rows(Path(path))
    if not rows:
        raise ValueError("GKW time.dat contains no rows")
    if any(len(row) < 2 for row in rows):
        raise ValueError("GKW time.dat rows must contain at least time and growth")
    times = np.asarray([row[0] for row in rows], dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValueError("GKW time.dat contains non-finite times")
    return times


def _normalize_profile_rows(values, floor: float = 1.0e-300):
    values = jnp.asarray(values, dtype=jnp.float64)
    row_sum = jnp.sum(values, axis=1, keepdims=True)
    return values / jnp.maximum(row_sum, jnp.asarray(floor, dtype=values.dtype))


def _load_gx_block_eik_geometry_reference(path: Path, text: str) -> GxEikGeometryReference:
    lines = text.splitlines()
    header = _first_numeric_row(lines)
    if len(header) < 8:
        raise ValueError("GX eik.out header must contain at least 8 numeric values")
    ntheta = int(round(header[2]))
    expected_rows = ntheta + 1
    gb_block = _parse_gx_eik_block(lines, "gbdrift gradpar grho tgrid", expected_rows, 4)
    cv_block = _parse_gx_eik_block(lines, "cvdrift gds2 bmag tgrid", expected_rows, 4)
    gds_block = _parse_gx_eik_block(lines, "gds21 gds22 tgrid", expected_rows, 3)
    drift0_block = _parse_gx_eik_block(lines, "cvdrift0 gbdrift0 tgrid", expected_rows, 3)
    theta = gb_block[:, 3]
    _assert_same_eik_grid("cvdrift", theta, cv_block[:, 3])
    _assert_same_eik_grid("gds21", theta, gds_block[:, 2])
    _assert_same_eik_grid("cvdrift0", theta, drift0_block[:, 2])
    return GxEikGeometryReference(
        theta=theta,
        bmag=cv_block[:, 2],
        gradpar=gb_block[:, 1],
        gds2=cv_block[:, 1],
        gds21=gds_block[:, 0],
        gds22=gds_block[:, 1],
        gbdrift=gb_block[:, 0],
        gbdrift0=drift0_block[:, 1],
        cvdrift=cv_block[:, 0],
        cvdrift0=drift0_block[:, 0],
        source=str(path),
        header=tuple(header),
    )


def _first_numeric_row(lines: list[str]) -> list[float]:
    for line in lines:
        try:
            row = [float(value) for value in line.split()]
        except ValueError:
            continue
        if row:
            return row
    raise ValueError("file contains no numeric header row")


def _parse_gx_eik_block(
    lines: list[str],
    label: str,
    expected_rows: int,
    expected_columns: int,
) -> np.ndarray:
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == label)
    except StopIteration as exc:
        raise ValueError(f"GX eik.out file is missing block {label!r}") from exc
    rows = []
    for line in lines[start + 1 :]:
        try:
            row = [float(value) for value in line.split()]
        except ValueError:
            break
        if len(row) != expected_columns:
            break
        rows.append(row)
        if len(rows) == expected_rows:
            break
    if len(rows) != expected_rows:
        raise ValueError(
            f"GX eik.out block {label!r} expected {expected_rows} rows; got {len(rows)}"
        )
    return np.asarray(rows, dtype=float)


def _assert_same_eik_grid(name: str, left, right):
    if not np.allclose(left, right, rtol=2.0e-12, atol=2.0e-12):
        raise ValueError(f"GX eik.out block {name!r} uses an inconsistent theta grid")


def _desc_gx_eik_reference_from_data(
    data,
    *,
    zeta,
    rho: float,
    alpha: float,
    zeta_center: float,
    iota: float,
    shear: float,
    psib: float,
    minor_radius: float,
    nzgrid: int,
    npol: int,
    source: str,
) -> GxEikGeometryReference:
    from scipy.constants import mu_0

    zeta = np.asarray(zeta, dtype=float)
    mod_b = _desc_array(data, "|B|")
    b_theta = _desc_array(data, "B_theta")
    b_zeta = _desc_array(data, "B_zeta")
    b_rho = _desc_array(data, "B_rho")
    d_b_theta = _desc_array(data, "|B|_t")
    d_b_zeta = _desc_array(data, "|B|_z")
    d_b_rho = _desc_array(data, "|B|_r")
    lambda_r = _desc_array(data, "lambda_r")
    lambda_t = _desc_array(data, "lambda_t")
    lambda_z = _desc_array(data, "lambda_z")
    jacobian = _desc_array(data, "sqrt(g)")
    psi = rho**2
    bref = 2.0 * abs(psib) / minor_radius**2
    bmag = mod_b / bref
    gradpar = minor_radius * _desc_array(data, "B^zeta") / mod_b
    grho = _desc_array(data, "|grad(rho)|") * minor_radius
    grad_psi = 2.0 * psib * rho
    grad_alpha_r = lambda_r - (zeta - zeta_center) * shear
    grad_alpha_t = 1.0 + lambda_t
    grad_alpha_z = -iota + lambda_z
    grad_alpha_sq = (
        grad_alpha_r**2 * _desc_array(data, "g^rr")
        + grad_alpha_t**2 * _desc_array(data, "g^tt")
        + grad_alpha_z**2 * _desc_array(data, "g^zz")
        + 2.0 * grad_alpha_r * grad_alpha_t * _desc_array(data, "g^rt")
        + 2.0 * grad_alpha_r * grad_alpha_z * _desc_array(data, "g^rz")
        + 2.0 * grad_alpha_t * grad_alpha_z * _desc_array(data, "g^tz")
    )
    grad_psi_dot_grad_alpha = grad_psi * (
        grad_alpha_r * _desc_array(data, "g^rr")
        + grad_alpha_t * _desc_array(data, "g^rt")
        + grad_alpha_z * _desc_array(data, "g^rz")
    )
    shat = -rho / iota * shear
    gds2 = grad_alpha_sq * minor_radius**2 * psi
    gds21 = shat / bref * grad_psi_dot_grad_alpha
    gds22 = (shat / (minor_radius * bref)) ** 2 / psi
    gds22 = gds22 * grad_psi**2 * _desc_array(data, "g^rr")
    sign_psi = psib / abs(psib)
    gbdrift0 = (
        sign_psi
        * shat
        * 2.0
        / mod_b**3
        / rho
        * (b_theta * d_b_zeta - b_zeta * d_b_theta)
        * psib
        / jacobian
        * 2.0
        * rho
    )
    cvdrift0 = gbdrift0
    gbdrift_norm = 2.0 * bref * minor_radius**2 / mod_b**3 * rho
    gbdrift = sign_psi * gbdrift_norm / jacobian
    gbdrift = gbdrift * (
        b_rho * d_b_theta * (lambda_z - iota)
        + b_theta * d_b_zeta * (lambda_r - (zeta - zeta_center) * shear)
        + b_zeta * d_b_rho * (1.0 + lambda_t)
        - b_zeta * d_b_theta * (lambda_r - (zeta - zeta_center) * shear)
        - b_theta * d_b_rho * (lambda_z - iota)
        - b_rho * d_b_zeta * (1.0 + lambda_t)
    )
    bsa = (b_zeta * (1.0 + lambda_t) - b_theta * (lambda_z - iota)) / jacobian
    cvdrift = gbdrift + (
        2.0
        * bref
        * minor_radius**2
        / mod_b**2
        * rho
        * mu_0
        / mod_b**2
        * _desc_array(data, "p_r")
        * bsa
    )
    (
        theta,
        bmag,
        _grho,
        gradpar,
        gds2,
        gds21,
        gds22,
        gbdrift,
        gbdrift0,
        cvdrift,
        cvdrift0,
    ) = _gx_equal_arc_arrays(
        zeta,
        bmag,
        grho,
        gradpar,
        gds2,
        gds21,
        gds22,
        gbdrift,
        gbdrift0,
        cvdrift,
        cvdrift0,
        nzgrid=nzgrid,
    )
    header = (
        float(nzgrid),
        1.0,
        float(2 * nzgrid),
        1.0,
        float(1.0 / minor_radius),
        float(shat),
        1.0,
        float(1.0 / iota),
        float(2 * npol - 1),
    )
    return GxEikGeometryReference(
        theta=theta,
        bmag=bmag,
        gradpar=gradpar,
        gds2=gds2,
        gds21=gds21,
        gds22=gds22,
        gbdrift=gbdrift,
        gbdrift0=gbdrift0,
        cvdrift=cvdrift,
        cvdrift0=cvdrift0,
        source=f"desc-gx-eik:{source}",
        header=header,
    )


def _desc_array(data, name: str) -> np.ndarray:
    return np.asarray(data[name], dtype=float)


def _gx_equal_arc_arrays(
    zeta,
    bmag,
    grho,
    gradpar,
    gds2,
    gds21,
    gds22,
    gbdrift,
    gbdrift0,
    cvdrift,
    cvdrift0,
    *,
    nzgrid: int,
):
    dzeta = zeta[1] - zeta[0]
    dzeta_pi = np.pi / nzgrid
    gradpar_half = np.zeros(2 * nzgrid)
    temp_grid = np.zeros(2 * nzgrid + 1)
    z_on_theta = np.zeros(2 * nzgrid + 1)
    for i in range(2 * nzgrid - 1):
        gradpar_half[i] = 0.5 * (abs(gradpar[i]) + abs(gradpar[i + 1]))
    gradpar_half[2 * nzgrid - 1] = gradpar_half[0]
    for i in range(2 * nzgrid):
        temp_grid[i + 1] = temp_grid[i] + dzeta / abs(gradpar_half[i])
    middle = nzgrid
    for i in range(2 * nzgrid + 1):
        z_on_theta[i] = temp_grid[i] - temp_grid[middle]
    desired_gradpar = np.pi / abs(z_on_theta[0])
    z_on_theta = z_on_theta * desired_gradpar
    uniform = z_on_theta[0] + np.arange(2 * nzgrid + 1) * dzeta_pi
    return (
        uniform,
        _gx_interp_to_new_grid(bmag, z_on_theta, uniform),
        _gx_interp_to_new_grid(grho, z_on_theta, uniform),
        np.full_like(uniform, desired_gradpar),
        _gx_interp_to_new_grid(gds2, z_on_theta, uniform),
        _gx_interp_to_new_grid(gds21, z_on_theta, uniform),
        _gx_interp_to_new_grid(gds22, z_on_theta, uniform),
        _gx_interp_to_new_grid(gbdrift, z_on_theta, uniform),
        _gx_interp_to_new_grid(gbdrift0, z_on_theta, uniform),
        _gx_interp_to_new_grid(cvdrift, z_on_theta, uniform),
        _gx_interp_to_new_grid(cvdrift0, z_on_theta, uniform),
    )


def _gx_interp_to_new_grid(values, source_grid, uniform_grid):
    from scipy.interpolate import interp1d

    values = np.asarray(values, dtype=float)
    out = np.zeros_like(uniform_grid, dtype=float)
    interpolant = interp1d(source_grid, values, kind="cubic")
    for i in range(len(uniform_grid) - 1):
        if uniform_grid[i] > source_grid[-1]:
            out[i] = out[i - 1]
        else:
            out[i] = interpolant(np.round(uniform_grid[i], 5))
    out[-1] = values[-1]
    return out


def _infer_gx_eik_dimensions(reference: GxEikGeometryReference) -> tuple[int, int]:
    if len(reference.header) >= 9:
        ntheta = int(round(reference.header[2]))
        scale = int(round(reference.header[8]))
        npol = max(1, (scale + 1) // 2)
        return ntheta, npol
    return int(reference.theta.shape[0] - 1), 1


def _parallel_grid_from_eik_theta(theta):
    from .grids import build_parallel_grid
    from .types import ParallelGridSpec

    theta = np.asarray(theta, dtype=float)
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _coerce_geometry_reference_shape(name, values, shape):
    array = jnp.asarray(values, dtype=jnp.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return array


def _field_amplitude(field):
    return jnp.sqrt(jnp.mean(jnp.abs(field) ** 2))


def _cyclone_phi_normalization_amplitude(phi, setup, *, normalization_model: str):
    if normalization_model == "weighted":
        from .time_advance import mode_chain_amplitude

        return mode_chain_amplitude(
            phi,
            w_z=setup["geometry"].w_z,
            connectivity=setup["connectivity"],
        )
    if normalization_model == "gkw_unweighted":
        return jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)))
    raise ValueError("normalization_model must be 'weighted' or 'gkw_unweighted'")


def _gkw_rk4_step_reference(state, dt, rhs_fn):
    delta_1 = dt * rhs_fn(state)
    updated = state + delta_1 / 6.0
    stage = state + delta_1 / 2.0
    delta_2 = dt * rhs_fn(stage)
    updated = updated + delta_2 / 3.0
    stage = state + delta_2 / 2.0
    delta_3 = dt * rhs_fn(stage)
    updated = updated + delta_3 / 3.0
    stage = state + delta_3
    delta_4 = dt * rhs_fn(stage)
    return updated + delta_4 / 6.0


def _pack_gkw_phi_field(phi, *, field_offset: int):
    phi = jnp.asarray(phi)
    if phi.ndim != 3:
        raise ValueError("phi must have shape (n_z,n_kx,n_ky)")
    if field_offset < 0:
        raise ValueError("field_offset must be nonnegative")
    field_values = jnp.ravel(jnp.transpose(phi, (0, 2, 1)))
    prefix = jnp.zeros((field_offset,), dtype=phi.dtype)
    return jnp.concatenate([prefix, field_values])


def _unpack_gkw_phi_field(fdis, *, field_offset: int, n_z: int, n_kx: int, n_ky: int):
    fdis = jnp.asarray(fdis)
    if field_offset < 0:
        raise ValueError("field_offset must be nonnegative")
    if n_z < 1 or n_kx < 1 or n_ky < 1:
        raise ValueError("field dimensions must be positive")
    n_field = n_z * n_kx * n_ky
    field_values = fdis[field_offset : field_offset + n_field]
    if field_values.shape[0] != n_field:
        raise ValueError("packed field does not contain enough values")
    return jnp.transpose(jnp.reshape(field_values, (n_z, n_ky, n_kx)), (0, 2, 1))


def _gkw_parallel_phi_diagnostic(phi):
    phi = jnp.asarray(phi)
    if phi.ndim != 3:
        raise ValueError("phi must have shape (n_z,n_kx,n_ky)")
    _n_z, n_kx, n_ky = phi.shape
    return jnp.sum(jnp.abs(phi) ** 2, axis=(1, 2)) / (n_kx * n_ky)


def _gkw_ky_spectrum_diagnostic(phi):
    phi = jnp.asarray(phi)
    if phi.ndim != 3:
        raise ValueError("phi must have shape (n_z,n_kx,n_ky)")
    n_z, n_kx, _n_ky = phi.shape
    return jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1)) / (n_kx * n_z)


def _gkw_kx_spectrum_diagnostic(phi):
    phi = jnp.asarray(phi)
    if phi.ndim != 3:
        raise ValueError("phi must have shape (n_z,n_kx,n_ky)")
    n_z, _n_kx, n_ky = phi.shape
    return jnp.sum(jnp.abs(phi) ** 2, axis=(0, 2)) / (n_ky * n_z)


def _deterministic_complex_probe(shape, dtype):
    size = int(np.prod(shape))
    index = jnp.arange(size, dtype=jnp.float64).reshape(shape)
    probe = jnp.cos(index / 5.0) + 1j * jnp.sin(index / 7.0)
    return probe.astype(dtype)


def _gkw_triplets_from_dense_matrix(matrix: np.ndarray, *, threshold: float):
    matrix = np.asarray(matrix)
    rows, cols = np.nonzero(np.abs(matrix) > threshold)
    values = matrix[rows, cols]
    return rows.astype(np.int64), cols.astype(np.int64), values.astype(matrix.dtype)


def _gkw_compress_triplets(rows, cols, values, *, n_rows: int):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values)
    if rows.shape != cols.shape or rows.shape != values.shape:
        raise ValueError("rows, cols, and values must have matching shapes")
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if rows.size == 0:
        return {
            "rows": rows,
            "cols": cols,
            "values": values,
            "iac": np.ones(n_rows + 1, dtype=np.int64),
        }
    duplicate_rows = np.concatenate([rows, rows])
    duplicate_cols = np.concatenate([cols, cols])
    duplicate_values = np.concatenate([0.4 * values, 0.6 * values])
    order = np.lexsort((duplicate_cols, duplicate_rows))
    sorted_rows = duplicate_rows[order]
    sorted_cols = duplicate_cols[order]
    sorted_values = duplicate_values[order]

    out_rows = []
    out_cols = []
    out_values = []
    row = int(sorted_rows[0])
    col = int(sorted_cols[0])
    value = sorted_values[0]
    for next_row, next_col, next_value in zip(
        sorted_rows[1:],
        sorted_cols[1:],
        sorted_values[1:],
        strict=False,
    ):
        if int(next_row) == row and int(next_col) == col:
            value = value + next_value
        else:
            out_rows.append(row)
            out_cols.append(col)
            out_values.append(value)
            row = int(next_row)
            col = int(next_col)
            value = next_value
    out_rows.append(row)
    out_cols.append(col)
    out_values.append(value)
    out_rows = np.asarray(out_rows, dtype=np.int64)
    out_cols = np.asarray(out_cols, dtype=np.int64)
    out_values = np.asarray(out_values, dtype=values.dtype)
    iac = np.empty(n_rows + 1, dtype=np.int64)
    cursor = 0
    for row_index in range(n_rows):
        iac[row_index] = cursor
        while cursor < out_rows.size and out_rows[cursor] == row_index:
            cursor += 1
    iac[n_rows] = out_rows.size
    return {"rows": out_rows, "cols": out_cols, "values": out_values, "iac": iac}


def _gkw_apply_compressed_triplets(compressed, vector):
    vector = np.asarray(vector)
    out = np.zeros(
        (compressed["iac"].shape[0] - 1,), dtype=np.result_type(compressed["values"], vector)
    )
    for row in range(out.shape[0]):
        start = compressed["iac"][row]
        stop = compressed["iac"][row + 1]
        cols = compressed["cols"][start:stop]
        values = compressed["values"][start:stop]
        if cols.size:
            out[row] = np.sum(values * vector[cols])
    return out


def _gkw_complex_real_split_action(matrix, vector, *, threshold: float):
    matrix = np.asarray(matrix)
    vector = np.asarray(vector)
    real_mask = np.abs(np.imag(matrix)) <= threshold
    real_matrix = np.where(real_mask, np.real(matrix), 0.0)
    complex_matrix = np.where(real_mask, 0.0, matrix)
    action = real_matrix @ vector + complex_matrix @ vector
    n_real = int(np.count_nonzero((np.abs(matrix) > threshold) & real_mask))
    n_complex = int(np.count_nonzero((np.abs(matrix) > threshold) & ~real_mask))
    return action, n_real, n_complex


def _l2_norm(values):
    return jnp.sqrt(jnp.mean(jnp.abs(jnp.asarray(values)) ** 2))


def _trace_fitted_growth(times, amplitudes):
    times = jnp.asarray(times, dtype=jnp.float64)
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.float64)
    if times.shape[0] < 2:
        return jnp.asarray(0.0, dtype=jnp.float64)
    log_amplitude = jnp.log(jnp.maximum(amplitudes, jnp.asarray(1.0e-300, dtype=jnp.float64)))
    centered_time = times - jnp.mean(times)
    centered_log = log_amplitude - jnp.mean(log_amplitude)
    denominator = jnp.sum(centered_time**2)
    return jnp.sum(centered_time * centered_log) / denominator


def _post_window_trace_view(solver_trace: CycloneTrace, reference_times):
    solver_times = jnp.asarray(solver_trace.times, dtype=jnp.float64)
    reference_times = jnp.asarray(reference_times, dtype=jnp.float64)
    solver_growth = jnp.asarray(solver_trace.window_growth, dtype=jnp.float64)
    solver_amplitude = jnp.asarray(solver_trace.physical_amplitude, dtype=jnp.float64)
    if solver_times.shape == reference_times.shape:
        return solver_times, solver_growth, solver_amplitude
    if solver_times.shape[0] == reference_times.shape[0] + 1:
        return solver_times[1:], solver_growth[1:], solver_amplitude[1:]
    raise ValueError("solver trace must either match reference times or include one initial sample")


def _late_mean_from_samples(values, start_fraction: float):
    values = jnp.asarray(values, dtype=jnp.float64)
    start = max(0, min(int(values.shape[0] * start_fraction), values.shape[0] - 1))
    return jnp.mean(values[start:])


def _late_fit_from_samples(times, amplitudes, start_fraction: float):
    times = jnp.asarray(times, dtype=jnp.float64)
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.float64)
    start = max(0, min(int(times.shape[0] * start_fraction), times.shape[0] - 2))
    return _trace_fitted_growth(times[start:], amplitudes[start:])


def _build_cyclone_base_case_setup(
    target: BenchmarkTarget,
    *,
    n_z: int,
    n_vpar: int,
    n_mu: int,
    vpar_max: float,
    mu_max: float | None,
    nperiod: int,
    parallel_recurrence_rate: float,
    parallel_backend: str,
    parallel_boundary: str,
    parallel_derivative_model: str,
    velocity_backend: str,
    initial_profile: str,
    velocity_recurrence_rate: float = 0.0,
):
    from .geometry import build_s_alpha_geometry
    from .grids import build_fourier_grid, build_mode_connectivity, build_velocity_grid
    from .physics import AdiabaticElectronParams
    from .solver import build_linear_residual_precompute
    from .types import FourierGridSpec, GeometryScalarParams, SpeciesParams, VelocityGridSpec

    metadata = dict(target.metadata)
    if mu_max is None:
        mu_max = 0.5 * vpar_max**2
    if parallel_boundary not in ("periodic", "zero"):
        raise ValueError("parallel_boundary must be 'periodic' or 'zero'")
    ky = float(metadata.get("k_theta_rhos", 0.5))
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=n_vpar,
            n_mu=n_mu,
            vpar_max=vpar_max,
            mu_max=mu_max,
            backend=velocity_backend,
        )
    )
    parallel = _build_gkw_cell_centered_parallel_grid(
        n_z,
        nperiod=nperiod,
        derivative_backend=parallel_backend,
        periodic=parallel_boundary != "zero",
    )
    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(ky,)))
    connectivity = build_mode_connectivity(fourier)
    geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(
            q=float(metadata.get("q", 1.4)),
            shat=float(metadata.get("shat", 0.78)),
            eps=float(metadata.get("epsilon", 0.19)),
        ),
    )
    species = SpeciesParams(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        density_gradient=float(metadata.get("R_over_Ln", 2.2)),
        temperature_gradient=float(metadata.get("R_over_LT", 6.9)),
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=False,
        ),
        parallel_recurrence_rate=parallel_recurrence_rate,
        parallel_recurrence_velocity_model="rms",
        velocity_recurrence_rate=velocity_recurrence_rate,
        velocity_recurrence_velocity_model="rms",
        mode_connectivity=connectivity,
        parallel_derivative_model=parallel_derivative_model,
    )
    if initial_profile == "cosine2":
        profile = 1.0 + jnp.cos(2.0 * jnp.pi * parallel.z)
    elif initial_profile == "cosine":
        profile = jnp.cos(2.0 * jnp.pi * parallel.z)
    else:
        raise ValueError("initial_profile must be 'cosine2' or 'cosine'")
    state = jnp.ones((n_vpar, n_mu, 1, 1, 1), dtype=jnp.complex128)
    state = 1.0e-4 * state * profile[None, None, :, None, None]
    return {
        "velocity": velocity,
        "parallel": parallel,
        "fourier": fourier,
        "geometry": geometry,
        "connectivity": connectivity,
        "precompute": precompute,
        "state": state,
        "selected_ky_index": 0,
    }


def _fit_growth_from_log_amplitudes(times, log_amplitudes, *, start_fraction: float):
    times = jnp.asarray(times, dtype=jnp.float64)
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.float64)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if log_amplitudes.ndim != 2:
        raise ValueError("log_amplitudes must have shape (n_time,n_ky)")
    if times.shape[0] != log_amplitudes.shape[0]:
        raise ValueError("times and log_amplitudes length must match")
    if times.shape[0] < 2:
        raise ValueError("at least two amplitude samples are required")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1)")
    n_time = times.shape[0]
    start = max(0, min(int(n_time * start_fraction), n_time - 2))
    window_times = times[start:]
    window_logs = log_amplitudes[start:]
    centered_time = window_times - jnp.mean(window_times)
    centered_logs = window_logs - jnp.mean(window_logs, axis=0)
    denominator = jnp.sum(centered_time**2)
    return jnp.sum(centered_time[:, None] * centered_logs, axis=0) / denominator


def _cyclone_boundary_map_error(setup):
    stencil = setup["precompute"].rhs.gkw_parallel_stencil
    valid = np.asarray(stencil.valid_shift)
    n_shift, n_z, n_kx, n_ky = valid.shape
    expected = np.zeros_like(valid, dtype=bool)
    offsets = np.arange(-(n_shift // 2), n_shift // 2 + 1)
    for shift_index, offset in enumerate(offsets):
        for iz in range(n_z):
            expected[shift_index, iz, :, :] = 0 <= iz + offset < n_z
    return _max_abs_error(valid.astype(np.float64), expected.astype(np.float64))


def _cyclone_normalization_error(setup, metadata):
    n_z = int(setup["parallel"].z.shape[0])
    n_vpar = int(setup["velocity"].vpar.shape[0])
    n_mu = int(setup["velocity"].mu.shape[0])
    vpar_max = float(metadata.get("vpar_max", 3.0))
    nperiod = int(metadata.get("nperiod", 5))
    sgrmax = nperiod - 0.5
    dz = 2.0 * sgrmax / n_z
    expected_z = -sgrmax + 0.5 * dz + dz * jnp.arange(n_z, dtype=setup["parallel"].z.dtype)
    dv = 2.0 * vpar_max / n_vpar
    expected_vpar = (
        -vpar_max
        + 0.5 * dv
        + dv
        * jnp.arange(
            n_vpar,
            dtype=setup["velocity"].vpar.dtype,
        )
    )
    dvperp = vpar_max / n_mu
    vperp = dvperp * (jnp.arange(n_mu, dtype=setup["velocity"].mu.dtype) + 0.5)
    expected_mu = 0.5 * vperp**2
    expected_w_mu = 2.0 * jnp.pi * vperp * dvperp
    errors = jnp.asarray(
        [
            _max_abs_error(setup["parallel"].z, expected_z),
            _max_abs_error(setup["parallel"].w_z, jnp.full((n_z,), dz)),
            _max_abs_error(setup["velocity"].vpar, expected_vpar),
            _max_abs_error(setup["velocity"].w_vpar, jnp.full((n_vpar,), dv)),
            _max_abs_error(setup["velocity"].mu, expected_mu),
            _max_abs_error(setup["velocity"].w_mu, expected_w_mu),
            _max_abs_error(setup["fourier"].ky, jnp.asarray([metadata.get("k_theta_rhos", 0.5)])),
        ],
        dtype=jnp.float64,
    )
    return jnp.max(errors)


def _field_power(field, weights):
    return jnp.sum(jnp.asarray(weights) * jnp.abs(jnp.asarray(field)) ** 2)


def _selected_mode_rms_profile(values, ix: int, iy: int):
    selected = jnp.asarray(values)[..., :, ix, iy]
    if selected.ndim == 1:
        return jnp.abs(selected)
    return jnp.sqrt(jnp.mean(jnp.abs(selected) ** 2, axis=tuple(range(selected.ndim - 1))))


def _cyclone_source_term_arrays(setup, metadata):
    """Rebuild the CBC source formulas used by selected GKW ``linear_terms``."""

    rhs = setup["precompute"].rhs
    velocity = setup["velocity"]
    geometry = setup["geometry"]
    fourier = setup["fourier"]
    vpar = jnp.asarray(velocity.vpar)
    mu = jnp.asarray(velocity.mu)
    B = jnp.asarray(geometry.B)

    vpar_5 = vpar[:, None, None, None, None]
    mu_5 = mu[None, :, None, None, None]
    B_5 = B[None, None, :, None, None]
    kx = fourier.kx[None, None, None, :, None]
    ky = fourier.ky[None, None, None, None, :]
    D_x = geometry.D_x[None, None, :, None, None]
    D_y = geometry.D_y[None, None, :, None, None]

    drift_frequency = (vpar_5**2 + mu_5 * B_5) * (kx * D_x + ky * D_y)
    energy = vpar[:, None, None] ** 2 + 2.0 * mu[None, :, None] * B[None, None, :]
    maxwellian_value = jnp.exp(-energy) / jnp.pi**1.5
    drive_factor = float(metadata.get("R_over_Ln", 2.2)) + float(metadata.get("R_over_LT", 6.9)) * (
        energy - 1.5
    )
    bessel_j0 = rhs.flr_factors.bessel_j0[0]
    equilibrium_drive_coeff = (
        1j
        * geometry.E_y[None, None, :, None, None]
        * fourier.ky[None, None, None, None, :]
        * bessel_j0[None, :, :, :, :]
        * maxwellian_value[..., None, None]
        * drive_factor[..., None, None]
    )
    current_drive_coeff = (
        1j
        * rhs.E_y[None, None, :, None, None]
        * rhs.ky[None, None, None, None, :]
        * rhs.flr_factors.bessel_j0[0][None, :, :, :, :]
        * rhs.maxwellian[0][..., None, None]
        * rhs.drive_factor[0][..., None, None]
    )

    mirror_coeff = mu[:, None] * B[None, :] * geometry.G[None, :]
    parallel_coeff = vpar[:, None] * geometry.F[None, :]
    parallel_field_coeff = -parallel_coeff[:, None, :] * maxwellian_value
    current_field_coeff = (
        -rhs.charge_over_temperature[0]
        * rhs.parallel_streaming_coeff[0][:, None, :]
        * rhs.maxwellian[0]
    )
    drift_field_coeff = (
        -1j * drift_frequency * maxwellian_value[..., None, None] * bessel_j0[None, :, :, :, :]
    )
    current_drift_field_coeff = (
        -rhs.charge_over_temperature[0]
        * 1j
        * rhs.magnetic_drift_frequency[0]
        * rhs.maxwellian[0][..., None, None]
        * rhs.flr_factors.bessel_j0[0][None, :, :, :, :]
    )
    return {
        "drift_frequency": drift_frequency,
        "mirror_coeff": mirror_coeff,
        "equilibrium_drive_coeff": equilibrium_drive_coeff,
        "current_drive_coeff": current_drive_coeff,
        "parallel_field_coeff": parallel_field_coeff,
        "current_field_coeff": current_field_coeff,
        "drift_field_coeff": drift_field_coeff,
        "current_drift_field_coeff": current_drift_field_coeff,
    }


def _apply_first_axis_matrix(matrix, values):
    matrix = jnp.asarray(matrix)
    values = jnp.asarray(values)
    return jnp.tensordot(matrix, values, axes=((1,), (0,)))


def _gkw_fortran_term_vii_reference(phi, field_coefficient, bessel_j0, stencil):
    phi_np = np.asarray(phi)
    coefficient_np = np.asarray(field_coefficient)
    bessel_np = np.asarray(bessel_j0)
    if phi_np.ndim != 3 or phi_np.shape[1:] != (1, 1):
        raise ValueError("Term VII source audit expects a single selected mode")
    if coefficient_np.ndim != 3:
        raise ValueError("field_coefficient must have shape (n_vpar,n_mu,n_z)")
    if bessel_np.shape[1:] != phi_np.shape:
        raise ValueError("bessel_j0 must have shape (n_mu,n_z,1,1)")

    n_vpar, n_mu, n_z = coefficient_np.shape
    d1_pos = np.asarray(stencil.d1_pos[:, :, 0, 0], dtype=float).T
    d1_neg = np.asarray(stencil.d1_neg[:, :, 0, 0], dtype=float).T
    s_shift = np.asarray(stencil.s_shift[:, :, 0, 0], dtype=np.int32)
    valid_shift = np.asarray(stencil.valid_shift[:, :, 0, 0], dtype=bool)
    expected = np.zeros((n_vpar, n_mu, n_z, 1, 1), dtype=np.result_type(phi_np, 1j))
    for iv in range(n_vpar):
        for imu in range(n_mu):
            for iz in range(n_z):
                row_coeff = coefficient_np[iv, imu, iz]
                table = d1_pos if row_coeff < 0.0 else d1_neg
                for shift_index in range(table.shape[1]):
                    if not valid_shift[shift_index, iz]:
                        continue
                    source_z = int(s_shift[shift_index, iz])
                    expected[iv, imu, iz, 0, 0] += (
                        row_coeff
                        * table[iz, shift_index]
                        * bessel_np[imu, source_z, 0, 0]
                        * phi_np[source_z, 0, 0]
                    )
    return jnp.asarray(expected, dtype=jnp.asarray(phi).dtype)


def _gkw_fortran_igh_reference(state, setup, *, disp_par: float, disp_vp: float):
    state_np = np.asarray(state)
    if state_np.ndim != 5 or state_np.shape[-2:] != (1, 1):
        raise ValueError("GKW igh audit currently expects a single selected mode")
    n_vpar, n_mu, n_z, _, _ = state_np.shape
    if n_vpar < 5 or n_z < 5:
        raise ValueError("GKW igh audit requires at least five vpar and z points")

    velocity = setup["velocity"]
    geometry = setup["geometry"]
    vpar = np.asarray(velocity.vpar, dtype=float)
    mu = np.asarray(velocity.mu, dtype=float)
    B = np.asarray(geometry.B, dtype=float)
    F = np.asarray(geometry.F, dtype=float)
    G = np.asarray(geometry.G, dtype=float)
    dvp = float(vpar[1] - vpar[0])
    ds = float(setup["precompute"].rhs.gkw_parallel_stencil.spacing)
    vpgr_rms = float(np.sqrt(np.mean(vpar**2)))
    mugr_rms = float(np.sqrt(np.mean(mu**2)))

    total = np.zeros_like(state_np)
    hamiltonian = np.zeros_like(state_np)
    parallel_diffusion = np.zeros_like(state_np)
    velocity_diffusion = np.zeros_like(state_np)

    def vpar_at(iv):
        return vpar[0] + iv * dvp

    def hh(iz, imu, iv):
        iz_ref = min(max(iz, 0), n_z - 1)
        return 0.5 * vpar_at(iv) ** 2 + mu[imu] * B[iz_ref]

    def add(out, row_iv, row_imu, row_iz, col_iv, col_iz, value):
        if 0 <= col_iv < n_vpar and 0 <= col_iz < n_z:
            out[row_iv, row_imu, row_iz, 0, 0] += value * state_np[col_iv, row_imu, col_iz, 0, 0]

    for iv in range(n_vpar):
        for imu in range(n_mu):
            for iz in range(n_z):
                dum = F[iz] / (ds * dvp)
                dum2 = -F[iz] * vpar[iv]
                position = _gkw_open_boundary_position_class(iz, n_z)
                if position in (-2, 2):
                    if dum2 * position < 0.0:
                        _gkw_igh_zero_two(
                            hamiltonian,
                            row_iv=iv,
                            row_imu=imu,
                            row_iz=iz,
                            dum=dum,
                            position=position,
                            hh=hh,
                            add=add,
                        )
                    elif dum2 * position > 0.0:
                        _gkw_igh_jhg_interior(
                            hamiltonian,
                            row_iv=iv,
                            row_imu=imu,
                            row_iz=iz,
                            dum=dum,
                            hh=hh,
                            add=add,
                        )
                elif position in (-1, 1):
                    if dum2 * position < 0.0:
                        _gkw_igh_two(
                            hamiltonian,
                            row_iv=iv,
                            row_imu=imu,
                            row_iz=iz,
                            dum=dum,
                            position=position,
                            hh=hh,
                            add=add,
                        )
                    elif dum2 * position > 0.0:
                        _gkw_igh_jhg_interior(
                            hamiltonian,
                            row_iv=iv,
                            row_imu=imu,
                            row_iz=iz,
                            dum=dum,
                            hh=hh,
                            add=add,
                        )
                else:
                    _gkw_igh_jhg_interior(
                        hamiltonian,
                        row_iv=iv,
                        row_imu=imu,
                        row_iz=iz,
                        dum=dum,
                        hh=hh,
                        add=add,
                    )
                    disp_s_dum = F[iz] * vpgr_rms
                    disp_v_dum = mugr_rms * B[iz] * G[iz]
                    _gkw_igh_diffus(
                        parallel_diffusion,
                        row_iv=iv,
                        row_imu=imu,
                        row_iz=iz,
                        kdiff=disp_par,
                        dum=disp_s_dum,
                        dx=ds,
                        direction="s",
                        add=add,
                    )
                    _gkw_igh_diffus(
                        velocity_diffusion,
                        row_iv=iv,
                        row_imu=imu,
                        row_iz=iz,
                        kdiff=disp_vp,
                        dum=disp_v_dum,
                        dx=dvp,
                        direction="vpar",
                        add=add,
                    )

    total = hamiltonian + parallel_diffusion + velocity_diffusion
    dtype = jnp.asarray(state).dtype
    return (
        jnp.asarray(total, dtype=dtype),
        jnp.asarray(hamiltonian, dtype=dtype),
        jnp.asarray(parallel_diffusion, dtype=dtype),
        jnp.asarray(velocity_diffusion, dtype=dtype),
    )


def _gkw_igh_jhg_interior(out, *, row_iv, row_imu, row_iz, dum, hh, add):
    d1 = dum / 6.0
    d2 = -dum / 24.0
    iv = row_iv
    iz = row_iz
    imu = row_imu
    entries = (
        (iv, iz - 2, d2 * (hh(iz - 1, imu, iv + 1) - hh(iz - 1, imu, iv - 1))),
        (
            iv - 1,
            iz - 1,
            d1 * (hh(iz - 1, imu, iv) - hh(iz, imu, iv - 1))
            + d2
            * (
                hh(iz - 2, imu, iv)
                + hh(iz - 1, imu, iv + 1)
                - hh(iz, imu, iv - 2)
                - hh(iz + 1, imu, iv - 1)
            ),
        ),
        (
            iv,
            iz - 1,
            d1
            * (
                hh(iz - 1, imu, iv + 1)
                - hh(iz - 1, imu, iv - 1)
                - hh(iz, imu, iv - 1)
                + hh(iz, imu, iv + 1)
            ),
        ),
        (
            iv + 1,
            iz - 1,
            d1 * (hh(iz, imu, iv + 1) - hh(iz - 1, imu, iv))
            + d2
            * (
                hh(iz + 1, imu, iv + 1)
                - hh(iz - 1, imu, iv - 1)
                - hh(iz - 2, imu, iv)
                + hh(iz, imu, iv + 2)
            ),
        ),
        (iv - 2, iz, d2 * (hh(iz - 1, imu, iv - 1) - hh(iz + 1, imu, iv - 1))),
        (
            iv - 1,
            iz,
            d1
            * (
                hh(iz - 1, imu, iv - 1)
                + hh(iz - 1, imu, iv)
                - hh(iz + 1, imu, iv - 1)
                - hh(iz + 1, imu, iv)
            ),
        ),
        (
            iv + 1,
            iz,
            d1
            * (
                hh(iz + 1, imu, iv)
                + hh(iz + 1, imu, iv + 1)
                - hh(iz - 1, imu, iv)
                - hh(iz - 1, imu, iv + 1)
            ),
        ),
        (iv + 2, iz, d2 * (hh(iz + 1, imu, iv + 1) - hh(iz - 1, imu, iv + 1))),
        (
            iv - 1,
            iz + 1,
            d1 * (hh(iz, imu, iv - 1) - hh(iz + 1, imu, iv))
            + d2
            * (
                hh(iz - 1, imu, iv - 1)
                - hh(iz + 1, imu, iv + 1)
                - hh(iz + 2, imu, iv)
                + hh(iz, imu, iv - 2)
            ),
        ),
        (
            iv,
            iz + 1,
            d1
            * (
                hh(iz, imu, iv - 1)
                + hh(iz + 1, imu, iv - 1)
                - hh(iz, imu, iv + 1)
                - hh(iz + 1, imu, iv + 1)
            ),
        ),
        (
            iv + 1,
            iz + 1,
            d1 * (hh(iz + 1, imu, iv) - hh(iz, imu, iv + 1))
            + d2
            * (
                hh(iz + 1, imu, iv - 1)
                - hh(iz - 1, imu, iv + 1)
                + hh(iz + 2, imu, iv)
                - hh(iz, imu, iv + 2)
            ),
        ),
        (iv, iz + 2, d2 * (hh(iz + 1, imu, iv - 1) - hh(iz + 1, imu, iv + 1))),
    )
    for col_iv, col_iz, value in entries:
        add(out, row_iv, row_imu, row_iz, col_iv, col_iz, value)


def _gkw_igh_zero_two(out, *, row_iv, row_imu, row_iz, dum, position, hh, add):
    fac = 0.25
    iv = row_iv
    iz = row_iz
    imu = row_imu
    if position == 2:
        val = fac * dum * (3.0 * hh(iz, imu, iv) - 4.0 * hh(iz - 1, imu, iv) + hh(iz - 2, imu, iv))
        entries = ((iv - 1, iz, -val), (iv + 1, iz, val))
        val = fac * dum * (hh(iz, imu, iv + 1) - hh(iz, imu, iv - 1))
        entries += ((iv, iz, -3.0 * val), (iv, iz - 1, 4.0 * val), (iv, iz - 2, -val))
    elif position == -2:
        val = fac * dum * (-3.0 * hh(iz, imu, iv) + 4.0 * hh(iz + 1, imu, iv) - hh(iz + 2, imu, iv))
        entries = ((iv - 1, iz, -val), (iv + 1, iz, val))
        val = fac * dum * (hh(iz, imu, iv + 1) - hh(iz, imu, iv - 1))
        entries += ((iv, iz, 3.0 * val), (iv, iz + 1, -4.0 * val), (iv, iz + 2, val))
    else:
        raise ValueError("position must be -2 or 2")
    for col_iv, col_iz, value in entries:
        add(out, row_iv, row_imu, row_iz, col_iv, col_iz, value)


def _gkw_igh_two(out, *, row_iv, row_imu, row_iz, dum, position, hh, add):
    fac = 1.0 / 72.0
    iv = row_iv
    iz = row_iz
    imu = row_imu
    if position == -1:
        val = (
            fac
            * dum
            * (
                -2.0 * hh(iz - 1, imu, iv)
                - 3.0 * hh(iz, imu, iv)
                + 6.0 * hh(iz + 1, imu, iv)
                - hh(iz + 2, imu, iv)
            )
        )
        entries = ((iv - 2, iz, val), (iv - 1, iz, -8.0 * val))
        entries += ((iv + 1, iz, 8.0 * val), (iv + 2, iz, -val))
        val = (
            -fac
            * dum
            * (
                hh(iz, imu, iv - 2)
                - 8.0 * hh(iz, imu, iv - 1)
                + 8.0 * hh(iz, imu, iv + 1)
                - hh(iz, imu, iv + 2)
            )
        )
        entries += (
            (iv, iz - 1, -2.0 * val),
            (iv, iz, -3.0 * val),
            (iv, iz + 1, 6.0 * val),
            (iv, iz + 2, -val),
        )
    elif position == 1:
        val = (
            fac
            * dum
            * (
                hh(iz - 2, imu, iv)
                - 6.0 * hh(iz - 1, imu, iv)
                + 3.0 * hh(iz, imu, iv)
                + 2.0 * hh(iz + 1, imu, iv)
            )
        )
        entries = ((iv - 2, iz, val), (iv - 1, iz, -8.0 * val))
        entries += ((iv + 1, iz, 8.0 * val), (iv + 2, iz, -val))
        val = (
            -fac
            * dum
            * (
                hh(iz, imu, iv - 2)
                - 8.0 * hh(iz, imu, iv - 1)
                + 8.0 * hh(iz, imu, iv + 1)
                - hh(iz, imu, iv + 2)
            )
        )
        entries += (
            (iv, iz - 2, val),
            (iv, iz - 1, -6.0 * val),
            (iv, iz, 3.0 * val),
            (iv, iz + 1, 2.0 * val),
        )
    else:
        raise ValueError("position must be -1 or 1")
    for col_iv, col_iz, value in entries:
        add(out, row_iv, row_imu, row_iz, col_iv, col_iz, value)


def _gkw_igh_diffus(out, *, row_iv, row_imu, row_iz, kdiff, dum, dx, direction, add):
    if kdiff <= 0.0:
        return
    stencil = (1.0, -4.0, 6.0, -4.0, 1.0)
    coefficient = -float(kdiff) * abs(float(dum)) / (12.0 * float(dx))
    for offset, weight in zip(range(-2, 3), stencil, strict=True):
        col_iv = row_iv + offset if direction == "vpar" else row_iv
        col_iz = row_iz + offset if direction == "s" else row_iz
        add(out, row_iv, row_imu, row_iz, col_iv, col_iz, coefficient * weight)


def _gkw_fortran_term_i_reference(
    state, parallel_coeff, recurrence_coeff, stencil, vpar, recurrence_rate: float
):
    state_np = np.asarray(state)
    parallel_np = np.asarray(parallel_coeff, dtype=float)
    recurrence_np = np.asarray(recurrence_coeff, dtype=float)
    if state_np.ndim != 5 or state_np.shape[-2:] != (1, 1):
        raise ValueError("GKW Fortran Term I audit currently expects a single selected mode")
    if parallel_np.shape != state_np.shape[:1] + state_np.shape[2:3]:
        raise ValueError("parallel_coeff must have shape (n_vpar,n_z)")
    if recurrence_np.shape != parallel_np.shape:
        raise ValueError("recurrence_coeff must match parallel_coeff")

    n_vpar, n_mu, n_z, _, _ = state_np.shape
    offsets = np.arange(-4, 5, dtype=np.int32)
    source_coefficients = np.zeros((n_vpar, n_z, offsets.shape[0]), dtype=float)
    expected = np.zeros_like(state_np)
    spacing = float(stencil.spacing)
    for iv in range(n_vpar):
        for iz in range(n_z):
            dum = -parallel_np[iv, iz]
            recurrence = recurrence_np[iv, iz]
            for offset, coefficient in _gkw_fortran_term_i_coefficients(
                _gkw_open_boundary_position_class(iz, n_z),
                dum,
                recurrence,
                spacing,
            ):
                shift_index = int(offset + 4)
                source_coefficients[iv, iz, shift_index] = coefficient
                shifted_z = iz + offset
                if 0 <= shifted_z < n_z:
                    expected[iv, :, iz, 0, 0] += coefficient * state_np[iv, :, shifted_z, 0, 0]

    current_coefficients = _current_gkw_term_i_coefficients(
        parallel_np,
        recurrence_np,
        stencil,
    )
    expected_recurrence = _expected_gkw_idisp2_recurrence_speed(
        parallel_np,
        np.asarray(vpar, dtype=float),
        recurrence_rate,
    )
    return (
        jnp.asarray(expected, dtype=jnp.asarray(state).dtype),
        jnp.asarray(source_coefficients, dtype=jnp.float64),
        jnp.asarray(current_coefficients, dtype=jnp.float64),
        jnp.asarray(expected_recurrence, dtype=jnp.float64),
    )


def _gkw_fortran_term_i_coefficients(
    position_class: int, dum: float, recurrence: float, spacing: float
):
    ds = float(spacing)
    rec = abs(float(recurrence))
    if dum > 0.0:
        if position_class == -2:
            return ((0, -3.0 * abs(dum) / (2.0 * ds)), (1, 2.0 * dum / ds), (2, -dum / (2.0 * ds)))
        if position_class == -1:
            return (
                (0, -dum / (2.0 * ds)),
                (1, dum / ds),
                (2, -dum / (6.0 * ds)),
                (-1, -dum / (3.0 * ds)),
            )
        if position_class == 1:
            return (
                (-2, (dum - rec) / (12.0 * ds)),
                (-1, (-8.0 * dum + 4.0 * rec) / (12.0 * ds)),
                (0, -6.0 * rec / (12.0 * ds)),
                (1, (8.0 * dum + 4.0 * rec) / (12.0 * ds)),
            )
        if position_class == 2:
            return ((-1, (-dum + 2.0 * rec) / (2.0 * ds)), (0, -2.0 * rec / ds))
    elif dum < 0.0:
        if position_class == -2:
            return ((0, -2.0 * rec / ds), (1, (dum + 2.0 * rec) / (2.0 * ds)))
        if position_class == -1:
            return (
                (-1, (-8.0 * dum + 4.0 * rec) / (12.0 * ds)),
                (0, -6.0 * rec / (12.0 * ds)),
                (1, (8.0 * dum + 4.0 * rec) / (12.0 * ds)),
                (2, (-dum - rec) / (12.0 * ds)),
            )
        if position_class == 1:
            return (
                (0, dum / (2.0 * ds)),
                (1, dum / (3.0 * ds)),
                (-1, -dum / ds),
                (-2, dum / (6.0 * ds)),
            )
        if position_class == 2:
            return ((0, 3.0 * dum / (2.0 * ds)), (-1, -2.0 * dum / ds), (-2, dum / (2.0 * ds)))
    if position_class == 0:
        return (
            (-2, (dum - rec) / (12.0 * ds)),
            (-1, (-8.0 * dum + 4.0 * rec) / (12.0 * ds)),
            (0, -6.0 * rec / (12.0 * ds)),
            (1, (8.0 * dum + 4.0 * rec) / (12.0 * ds)),
            (2, (-dum - rec) / (12.0 * ds)),
        )
    return ()


def _current_gkw_term_i_coefficients(parallel_coeff, recurrence_coeff, stencil):
    sign = -np.asarray(parallel_coeff, dtype=float)
    recurrence = np.asarray(recurrence_coeff, dtype=float)
    d1_pos = np.asarray(stencil.d1_pos[:, :, 0, 0], dtype=float).T
    d1_neg = np.asarray(stencil.d1_neg[:, :, 0, 0], dtype=float).T
    d4_pos = np.asarray(stencil.d4_pos[:, :, 0, 0], dtype=float).T
    d4_neg = np.asarray(stencil.d4_neg[:, :, 0, 0], dtype=float).T
    selected_d1 = np.where(sign[:, :, None] > 0.0, d1_pos[None, :, :], d1_neg[None, :, :])
    selected_d4 = np.where(sign[:, :, None] > 0.0, d4_pos[None, :, :], d4_neg[None, :, :])
    return sign[:, :, None] * selected_d1 + recurrence[:, :, None] * selected_d4


def _expected_gkw_idisp2_recurrence_speed(parallel_coeff, vpar, recurrence_rate: float):
    if recurrence_rate == 0.0:
        return np.zeros_like(parallel_coeff)
    parallel = np.asarray(parallel_coeff, dtype=float)
    vpar = np.asarray(vpar, dtype=float)
    if vpar.shape != parallel.shape[:1]:
        raise ValueError("vpar must match the first axis of parallel_coeff")
    eps = 1.0e-300
    ffun = np.where(np.abs(vpar[:, None]) > eps, parallel / vpar[:, None], 0.0)
    vpgr_rms = np.sqrt(np.mean(vpar**2))
    return recurrence_rate * np.abs(ffun * vpgr_rms)


def _gkw_open_boundary_position_class(index: int, size: int) -> int:
    if index == 0:
        return -2
    if index == 1:
        return -1
    if index == size - 2:
        return 1
    if index == size - 1:
        return 2
    return 0


def _build_gkw_cell_centered_parallel_grid(
    n_z: int,
    nperiod: int = 1,
    *,
    derivative_backend: str = "fourier",
    periodic: bool = True,
):
    from .grids import build_finite_difference_operators, build_parallel_grid
    from .types import DerivativeBackend, ParallelGrid, ParallelGridSpec

    if nperiod < 1:
        raise ValueError("nperiod must be at least 1")
    sgrmax = nperiod - 0.5
    lower = -sgrmax + sgrmax / n_z
    length = 2.0 * sgrmax
    if derivative_backend == DerivativeBackend.FINITE_DIFFERENCE.value:
        spacing = length / n_z
        operators = build_finite_difference_operators(
            n_z,
            spacing,
            periodic=periodic,
        )
        identity = jnp.eye(n_z, dtype=operators.D1.dtype)
        z = lower + spacing * jnp.arange(n_z, dtype=operators.D1.dtype)
        return ParallelGrid(
            z=z,
            w_z=jnp.full((n_z,), spacing, dtype=operators.D1.dtype),
            D_z=operators.D1,
            modal_transform=identity,
            inverse_modal_transform=identity,
            backend=DerivativeBackend.FINITE_DIFFERENCE.value,
            topology="periodic" if periodic else "open",
        )
    if not periodic:
        raise ValueError("nonperiodic GKW cell-centered grids require finite_difference backend")
    if derivative_backend != DerivativeBackend.FOURIER.value:
        raise ValueError("derivative_backend must be 'fourier' or 'finite_difference'")
    return build_parallel_grid(
        ParallelGridSpec(
            n_z=n_z,
            z_min=lower,
            z_max=lower + length,
            topology="periodic",
        )
    )


def _max_abs_error(left, right):
    return jnp.max(jnp.abs(jnp.asarray(left) - jnp.asarray(right)))
