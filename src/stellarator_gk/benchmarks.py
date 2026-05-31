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
        object.__setattr__(self, "max_abs_error", jnp.asarray(self.max_abs_error, dtype=jnp.float64))
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
        object.__setattr__(self, "max_abs_error", jnp.asarray(self.max_abs_error, dtype=jnp.float64))
        object.__setattr__(self, "passed", jnp.asarray(self.passed, dtype=bool))
        object.__setattr__(self, "field_names", tuple(self.field_names))


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
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=1, kx_max=kx_rhos, ky_values=(0.0,))
    )
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
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=1, kx_max=kx_rhos, ky_values=(0.0,))
    )
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

    use_filter = any(
        rate > 0.0 for rate in (z_modal_damping, vpar_modal_damping, mu_modal_damping)
    )
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
        lambda state_value: integrate_fixed_step(
            state_value,
            dt,
            diagnostic_steps,
            linear_residual,
            precompute,
            filter_fn=filter_fn,
            store_history=False,
        ).state
    )
    solve_phi = jax.jit(lambda state_value: solve_adiabatic_electron_phi(state_value, precompute.field))
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
    parallel_recurrence_rate: float | None = None,
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
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
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

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
    velocity_backend = str(
        metadata.get("velocity_backend", "finite_difference")
        if velocity_backend is None
        else velocity_backend
    )
    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")

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
    )
    state = setup["state"]
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    times = []
    log_amplitudes = []
    advance_window = jax.jit(
        lambda state_value: integrate_fixed_step(
            state_value,
            dt,
            steps_per_window,
            linear_residual,
            setup["precompute"],
            store_history=False,
        ).state
    )
    solve_phi = jax.jit(lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field))

    def append_log_amplitude(time_value, state_value, accumulated_log):
        phi = solve_phi(state_value)
        amplitude = mode_chain_amplitude(
            phi,
            w_z=setup["geometry"].w_z,
            connectivity=setup["connectivity"],
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

    fitted_growth = _fit_growth_from_log_amplitudes(
        jnp.asarray(times, dtype=jnp.float64),
        jnp.stack(log_amplitudes),
        start_fraction=growth_window_fraction,
    )
    observed = fitted_growth[setup["selected_ky_index"]]
    return evaluate_benchmark_gate(
        observed,
        target,
        notes=(
            "production-control CBC gate with GKW cell-centered s grid, "
            f"n_z={n_z}, nperiod={nperiod}, n_vpar={n_vpar}, n_mu={n_mu}, "
            f"parallel_backend={parallel_backend}, velocity_backend={velocity_backend}, "
            f"parallel_boundary={parallel_boundary}, "
            f"parallel_derivative_model={parallel_derivative_model}, "
            f"steps_per_window={steps_per_window}, n_windows={n_windows}, "
            f"normalize_each_window={normalize_each_window}; "
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
    parallel_backend: str | None = None,
    parallel_boundary: str | None = None,
    parallel_derivative_model: str | None = None,
    velocity_backend: str | None = None,
    normalize_each_window: bool = True,
    target: BenchmarkTarget | None = None,
) -> CycloneTrace:
    """Record selected-mode CBC evolution at fixed diagnostic windows."""

    from .physics import solve_adiabatic_electron_phi
    from .solver import linear_residual
    from .time_advance import integrate_fixed_step, mode_chain_amplitude, normalize_by_ky_amplitude

    if steps_per_window < 1:
        raise ValueError("steps_per_window must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
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
        parallel_backend=parallel_backend,
        parallel_boundary=parallel_boundary,
        parallel_derivative_model=parallel_derivative_model,
        velocity_backend=velocity_backend,
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

    solve_phi = jax.jit(lambda state_value: solve_adiabatic_electron_phi(state_value, setup["precompute"].field))
    advance_window = jax.jit(
        lambda state_value: integrate_fixed_step(
            state_value,
            dt,
            steps_per_window,
            linear_residual,
            setup["precompute"],
            store_history=False,
        ).state
    )

    def append_snapshot(time_value, state_value, accumulated_log, previous_physical_value):
        phi = solve_phi(state_value)
        amplitude = mode_chain_amplitude(
            phi,
            w_z=setup["geometry"].w_z,
            connectivity=setup["connectivity"],
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
            "window growth, fitted growth, phi norm, state norm, and RHS norm"
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
            f"observed={observed.source}; reference={reference.source}; "
            "trace-level CBC comparison"
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
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(ky,))
    )
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
        mode_connectivity=connectivity,
        parallel_derivative_model=parallel_derivative_model,
    )
    profile = 1.0 + jnp.cos(2.0 * jnp.pi * parallel.z)
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
    expected_vpar = -vpar_max + 0.5 * dv + dv * jnp.arange(
        n_vpar,
        dtype=setup["velocity"].vpar.dtype,
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
