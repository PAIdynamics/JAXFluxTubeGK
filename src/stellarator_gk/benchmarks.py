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
    ``theta, bmag, gradpar, gds2, gds21, gds22, gbdrift, gbdrift0, cvdrift, cvdrift0``.
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
        gbdrift=data[:, 6],
        gbdrift0=data[:, 7],
        cvdrift=data[:, 8],
        cvdrift0=data[:, 9],
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
    field_names = (
        "B/bmag",
        "F/gradpar",
        "g_xx/gds22",
        "g_xy/gds21",
        "g_yy/gds2",
        "D_x/gbdrift0+cvdrift0",
        "D_y/gbdrift+cvdrift",
        "G/-(gbdrift+cvdrift)",
        "kperp2",
    )
    field_errors = jnp.asarray(
        [
            _max_abs_error(geometry.B, bmag),
            _max_abs_error(geometry.F, gradpar),
            _max_abs_error(geometry.g_xx, gds22),
            _max_abs_error(geometry.g_xy, gds21),
            _max_abs_error(geometry.g_yy, gds2),
            _max_abs_error(geometry.D_x, gx_radial_drift),
            _max_abs_error(geometry.D_y, gx_binormal_drift),
            _max_abs_error(geometry.G, gx_mirror_proxy),
            _max_abs_error(
                k_perp_squared(geometry, fourier_grid),
                gx_eik_kperp2(reference, fourier_grid.kx, fourier_grid.ky),
            ),
        ],
        dtype=jnp.float64,
    )
    return GxEikGeometryParityReport(
        field_errors=field_errors,
        max_abs_error=jnp.max(field_errors),
        max_abs_kperp2_error=field_errors[-1],
        field_names=field_names,
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
    vpar_max: float = 4.0,
    mu_max: float = 8.0,
    dt: float = 0.01,
    n_steps: int = 100,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Evolve the present reduced zonal-flow setup and compare its residual.

    This is an executable validation gate for the current solver.  It is not a
    claim that the present reduced model passes the production RH benchmark.
    """

    from .geometry import build_boozer_parallel_grid, build_s_alpha_geometry
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
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=n_vpar, n_mu=n_mu, vpar_max=vpar_max, mu_max=mu_max)
    )
    parallel = build_boozer_parallel_grid(n_z=n_z, n_turns=1)
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
    )
    ix = 0
    maxwellian = precompute.rhs.maxwellian[0]
    state = jnp.zeros((n_vpar, n_mu, n_z, 3, 1), dtype=jnp.complex128)
    state = state.at[:, :, :, ix, 0].set(1.0e-4 * maxwellian)
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
        notes="reduced executable RH gate; production t=100 convergence remains open",
    )


def run_reduced_cyclone_base_case_gate(
    *,
    n_z: int = 16,
    n_vpar: int = 12,
    n_mu: int = 6,
    vpar_max: float = 4.0,
    mu_max: float = 8.0,
    dt: float = 0.003,
    n_steps: int = 100,
    growth_window_fraction: float = 0.5,
    target: BenchmarkTarget | None = None,
) -> BenchmarkGateResult:
    """Run the present reduced Cyclone setup and compare selected growth."""

    from .geometry import build_boozer_parallel_grid, build_s_alpha_geometry
    from .grids import build_fourier_grid, build_mode_connectivity, build_velocity_grid
    from .physics import AdiabaticElectronParams, solve_adiabatic_electron_phi
    from .solver import build_linear_residual_precompute, linear_residual
    from .time_advance import integrate_fixed_step, windowed_linear_growth_diagnostics
    from .types import FourierGridSpec, GeometryScalarParams, SpeciesParams, VelocityGridSpec

    target = target or cyclone_base_case_growth_target()
    metadata = dict(target.metadata)
    ky = float(metadata.get("k_theta_rhos", 0.5))
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=n_vpar, n_mu=n_mu, vpar_max=vpar_max, mu_max=mu_max)
    )
    parallel = build_boozer_parallel_grid(n_z=n_z, n_turns=1)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=2, kx_max=0.0, ky_values=(0.0, ky))
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
    )
    shape = (n_vpar, n_mu, n_z, 1, 2)
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    state = 1.0e-6 * (jnp.cos(index / 13.0) + 1j * jnp.sin(index / 17.0))
    result = integrate_fixed_step(
        state,
        dt,
        n_steps,
        linear_residual,
        precompute,
        store_history=True,
    )
    phi_history = jax.vmap(
        lambda snapshot: solve_adiabatic_electron_phi(snapshot, precompute.field)
    )(result.history)
    diagnostics = windowed_linear_growth_diagnostics(
        phi_history,
        result.times,
        start_fraction=growth_window_fraction,
        w_z=geometry.w_z,
        connectivity=connectivity,
    )
    return evaluate_benchmark_gate(
        diagnostics.growth_rate[1],
        target,
        notes=(
            "reduced executable CBC gate with late-window growth fit; "
            "production GKW/GX agreement remains open"
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


def _max_abs_error(left, right):
    return jnp.max(jnp.abs(jnp.asarray(left) - jnp.asarray(right)))
