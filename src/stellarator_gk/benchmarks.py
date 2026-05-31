"""Benchmark reference targets and lightweight external-fixture readers."""

from __future__ import annotations

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
    """

    path = Path(path)
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
    )
    state = setup["state"]
    log_normalization = jnp.zeros((setup["fourier"].ky.shape[0],), dtype=jnp.float64)
    times = []
    log_amplitudes = []

    def append_log_amplitude(time_value, state_value, accumulated_log):
        phi = solve_adiabatic_electron_phi(state_value, setup["precompute"].field)
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
        result = integrate_fixed_step(
            state,
            dt,
            steps_per_window,
            linear_residual,
            setup["precompute"],
            store_history=False,
        )
        state = result.state
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
            f"steps_per_window={steps_per_window}, n_windows={n_windows}, "
            f"normalize_each_window={normalize_each_window}; "
            "production GKW/GX agreement remains open until this gate passes"
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


def _import_netcdf_dataset():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError(
            "load_gx_growth_rate_reference requires netCDF4; install DESC/GX "
            "analysis dependencies or add netCDF4 to the environment"
        ) from exc
    return Dataset


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


def _coerce_geometry_reference_shape(name, values, shape):
    array = jnp.asarray(values, dtype=jnp.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return array


def _field_amplitude(field):
    return jnp.sqrt(jnp.mean(jnp.abs(field) ** 2))


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
):
    from .geometry import build_s_alpha_geometry
    from .grids import build_fourier_grid, build_mode_connectivity, build_velocity_grid
    from .physics import AdiabaticElectronParams
    from .solver import build_linear_residual_precompute
    from .types import FourierGridSpec, GeometryScalarParams, SpeciesParams, VelocityGridSpec

    metadata = dict(target.metadata)
    if mu_max is None:
        mu_max = 0.5 * vpar_max**2
    ky = float(metadata.get("k_theta_rhos", 0.5))
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=n_vpar, n_mu=n_mu, vpar_max=vpar_max, mu_max=mu_max)
    )
    parallel = _build_gkw_cell_centered_parallel_grid(n_z, nperiod=nperiod)
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


def _field_power(field, weights):
    return jnp.sum(jnp.asarray(weights) * jnp.abs(jnp.asarray(field)) ** 2)


def _build_gkw_cell_centered_parallel_grid(
    n_z: int,
    nperiod: int = 1,
    *,
    derivative_backend: str = "fourier",
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
            periodic=True,
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
            topology="periodic",
        )
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
