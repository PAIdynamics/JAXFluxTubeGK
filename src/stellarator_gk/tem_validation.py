"""Kinetic-electron TEM preflight before quantitative external validation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import build_s_alpha_geometry
from .grids import (
    build_finite_difference_operators,
    build_fourier_grid,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
)
from .diagnostics import mode_amplitude
from .physics import kinetic_quasineutrality_residual, solve_kinetic_electron_phi
from .solver import build_linear_residual_precompute, linear_residual
from .time_advance import estimate_linear_cfl_dt, integrate_fixed_step
from .types import (
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGrid,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
)


TEM_PREFLIGHT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TemCaseSpec:
    """Static electrostatic heavy-electron TEM-favorable smoke parameters."""

    q: float = 1.4
    shat: float = 0.8
    eps: float = 0.18
    ky: float = 0.7
    density_gradient: float = 2.2
    ion_temperature_gradient: float = 0.0
    electron_temperature_gradient: float = 6.9
    electron_mass: float = 0.01
    n_z: int = 16
    n_vpar: int = 8
    n_mu: int = 4
    vpar_max: float = 3.0
    mu_max: float = 4.5
    field_periods: float = 3.0
    velocity_backend: str = "chebyshev"
    parallel_backend: str = "fourier"
    parallel_derivative_model: str = "matrix"
    initial_condition: str = "generic"
    parallel_recurrence_rate: float = 1.0
    velocity_recurrence_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.q <= 0.0 or self.eps <= 0.0 or self.ky <= 0.0:
            raise ValueError("q, eps, and ky must be positive")
        if not 0.0 < self.electron_mass < 1.0:
            raise ValueError("electron_mass must lie between zero and the ion mass")
        if min(self.n_z, self.n_vpar, self.n_mu) < 2:
            raise ValueError("TEM preflight grid sizes must be at least two")
        if self.vpar_max <= 0.0 or self.mu_max <= 0.0:
            raise ValueError("velocity bounds must be positive")
        if self.parallel_recurrence_rate < 0.0 or self.velocity_recurrence_rate < 0.0:
            raise ValueError("recurrence-control rates must be nonnegative")
        if self.parallel_backend not in ("fourier", "finite_difference"):
            raise ValueError("parallel_backend must be 'fourier' or 'finite_difference'")
        if self.parallel_derivative_model not in ("matrix", "gkw_igh"):
            raise ValueError("parallel_derivative_model must be 'matrix' or 'gkw_igh'")
        if self.parallel_derivative_model == "gkw_igh" and (
            self.parallel_backend != "finite_difference"
            or self.velocity_backend != "finite_difference"
        ):
            raise ValueError("gkw_igh requires finite-difference parallel and velocity grids")
        if self.initial_condition not in ("generic", "gyaradax_cosine2"):
            raise ValueError("unsupported TEM initial condition")


@dataclass(frozen=True)
class TemPhysicsPreflightReport:
    """Algebraic readiness evidence that deliberately stops short of TEM parity."""

    schema_version: int
    status: str
    passed: bool
    field_residual_max_abs: float
    rhs_max_abs: float
    estimated_cfl_dt: float
    electron_to_ion_streaming_ratio: float
    expected_streaming_ratio: float
    background_charge_density: float
    n_species: int
    field_model: str
    external_growth_frequency_validated: bool


@dataclass(frozen=True)
class TemLinearSmokeResult:
    """Reduced time-advanced TEM discriminator, not an external parity result."""

    schema_version: int
    status: str
    growth_rate: float
    frequency: float
    late_window_growth_delta: float
    final_time: float
    dt: float
    estimated_cfl_dt: float
    steps_per_window: int
    n_windows: int
    finite: bool
    electron_direction_frequency: bool
    externally_validated: bool


def tem_species(spec: TemCaseSpec | None = None) -> tuple[SpeciesParams, SpeciesParams]:
    """Return charge-neutral kinetic ions/electrons for the TEM preflight."""

    spec = spec or TemCaseSpec()
    return (
        SpeciesParams(
            charge=1.0,
            mass=1.0,
            density=1.0,
            temperature=1.0,
            density_gradient=spec.density_gradient,
            temperature_gradient=spec.ion_temperature_gradient,
        ),
        SpeciesParams(
            charge=-1.0,
            mass=spec.electron_mass,
            density=1.0,
            temperature=1.0,
            density_gradient=spec.density_gradient,
            temperature_gradient=spec.electron_temperature_gradient,
        ),
    )


def gyaradax_tem_case_spec() -> TemCaseSpec:
    """Return the exact local discretization profile for the pinned TEM producer."""

    return TemCaseSpec(
        n_z=32,
        n_vpar=32,
        n_mu=16,
        velocity_backend="finite_difference",
        parallel_backend="finite_difference",
        parallel_derivative_model="gkw_igh",
        initial_condition="gyaradax_cosine2",
    )


def run_tem_physics_preflight(
    spec: TemCaseSpec | None = None,
    *,
    field_tolerance: float = 1.0e-10,
    streaming_tolerance: float = 1.0e-10,
) -> TemPhysicsPreflightReport:
    """Exercise the coupled kinetic-electron path without claiming TEM parity."""

    spec = spec or TemCaseSpec()
    velocity, parallel, _fourier, _geometry, _species, precompute = _build_tem_system(spec)
    state = _initial_tem_state(precompute, parallel, spec)
    phi = solve_kinetic_electron_phi(state, precompute.field)
    field_residual = kinetic_quasineutrality_residual(phi, state, precompute.field)
    rhs = jax.jit(linear_residual)(state, precomputed=precompute)
    streaming = np.asarray(precompute.rhs.parallel_streaming_coeff)
    ion_streaming = float(np.max(np.abs(streaming[0])))
    electron_streaming = float(np.max(np.abs(streaming[1])))
    streaming_ratio = electron_streaming / ion_streaming
    expected_ratio = float(1.0 / np.sqrt(spec.electron_mass))
    field_error = float(jnp.max(jnp.abs(field_residual)))
    rhs_max = float(jnp.max(jnp.abs(rhs)))
    cfl = float(estimate_linear_cfl_dt(precompute))
    charge_density = float(sum(item.charge * item.density for item in _species))
    passed = bool(
        np.isfinite(rhs_max)
        and np.isfinite(cfl)
        and cfl > 0.0
        and field_error <= field_tolerance
        and abs(streaming_ratio - expected_ratio)
        <= streaming_tolerance * max(1.0, expected_ratio)
        and abs(charge_density) <= 1.0e-14
    )
    return TemPhysicsPreflightReport(
        schema_version=TEM_PREFLIGHT_SCHEMA_VERSION,
        status=(
            "kinetic_electron_algebra_preflight_passed_external_tem_parity_open"
            if passed
            else "kinetic_electron_algebra_preflight_failed"
        ),
        passed=passed,
        field_residual_max_abs=field_error,
        rhs_max_abs=rhs_max,
        estimated_cfl_dt=cfl,
        electron_to_ion_streaming_ratio=streaming_ratio,
        expected_streaming_ratio=expected_ratio,
        background_charge_density=charge_density,
        n_species=precompute.n_species,
        field_model=precompute.field_model,
        external_growth_frequency_validated=False,
    )


def run_reduced_tem_linear_smoke(
    spec: TemCaseSpec | None = None,
    *,
    dt: float | None = None,
    cfl_fraction: float = 0.5,
    steps_per_window: int = 20,
    n_windows: int = 12,
    late_fraction: float = 0.5,
) -> TemLinearSmokeResult:
    """Evolve a reduced kinetic-electron case and report branch diagnostics."""

    spec = spec or TemCaseSpec()
    if not 0.0 < cfl_fraction <= 1.0:
        raise ValueError("cfl_fraction must lie in (0, 1]")
    if steps_per_window < 1 or n_windows < 3:
        raise ValueError("steps_per_window must be positive and n_windows at least three")
    if not 0.0 <= late_fraction < 1.0:
        raise ValueError("late_fraction must lie in [0, 1)")
    _velocity, parallel, _fourier, _geometry, _species, precompute = _build_tem_system(spec)
    cfl = float(estimate_linear_cfl_dt(precompute))
    timestep = cfl_fraction * cfl if dt is None else float(dt)
    if timestep <= 0.0 or timestep > cfl:
        raise ValueError(f"dt must be positive and no larger than estimated CFL {cfl:.6g}")
    state = _initial_tem_state(precompute, parallel, spec)
    field = solve_kinetic_electron_phi(state, precompute.field)
    amplitude = float(mode_amplitude(field)[0, 0])
    state = state / amplitude
    log_normalization = np.log(amplitude)
    times = [0.0]
    log_amplitudes = [log_normalization]
    probes = [complex(np.asarray(field[parallel.z.shape[0] // 2, 0, 0]) / amplitude)]
    for window in range(n_windows):
        result = integrate_fixed_step(
            state,
            timestep,
            steps_per_window,
            linear_residual,
            precompute,
            store_history=False,
        )
        state = result.state
        field = solve_kinetic_electron_phi(state, precompute.field)
        amplitude = float(mode_amplitude(field)[0, 0])
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            break
        state = state / amplitude
        log_normalization += np.log(amplitude)
        time = (window + 1) * steps_per_window * timestep
        times.append(time)
        log_amplitudes.append(log_normalization)
        probes.append(complex(np.asarray(field[parallel.z.shape[0] // 2, 0, 0]) / amplitude))
    times_array = np.asarray(times)
    logs_array = np.asarray(log_amplitudes)
    probes_array = np.asarray(probes)
    start = min(max(int(len(times_array) * late_fraction), 0), len(times_array) - 2)
    growth = float(np.polyfit(times_array[start:], logs_array[start:], 1)[0])
    phases = np.unwrap(np.angle(probes_array))
    frequency = float(np.polyfit(times_array[start:], phases[start:], 1)[0])
    window_growth = np.diff(logs_array) / np.diff(times_array)
    late_delta = float(abs(window_growth[-1] - window_growth[-2]))
    finite = bool(
        len(times_array) == n_windows + 1
        and np.all(np.isfinite(logs_array))
        and np.isfinite(growth)
        and np.isfinite(frequency)
    )
    return TemLinearSmokeResult(
        schema_version=TEM_PREFLIGHT_SCHEMA_VERSION,
        status="reduced_tem_time_advance_not_external_validation",
        growth_rate=growth,
        frequency=frequency,
        late_window_growth_delta=late_delta,
        final_time=float(times_array[-1]),
        dt=timestep,
        estimated_cfl_dt=cfl,
        steps_per_window=steps_per_window,
        n_windows=n_windows,
        finite=finite,
        electron_direction_frequency=bool(frequency < 0.0),
        externally_validated=False,
    )


def _build_tem_system(spec: TemCaseSpec):
    species = tem_species(spec)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=spec.n_vpar,
            n_mu=spec.n_mu,
            vpar_max=spec.vpar_max,
            mu_max=spec.mu_max,
            backend=spec.velocity_backend,
        )
    )
    parallel = _build_tem_parallel_grid(spec)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(spec.ky,))
    )
    mode_connectivity = (
        build_mode_connectivity(fourier) if spec.parallel_derivative_model == "gkw_igh" else None
    )
    geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(q=spec.q, shat=spec.shat, eps=spec.eps),
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="kinetic",
        parallel_recurrence_rate=spec.parallel_recurrence_rate,
        velocity_recurrence_rate=spec.velocity_recurrence_rate,
        mode_connectivity=mode_connectivity,
        parallel_derivative_model=spec.parallel_derivative_model,
    )
    return velocity, parallel, fourier, geometry, species, precompute


def _build_tem_parallel_grid(spec: TemCaseSpec) -> ParallelGrid:
    half_span = 0.5 * spec.field_periods
    spacing = spec.field_periods / spec.n_z
    lower = -half_span + 0.5 * spacing
    if spec.parallel_backend == "finite_difference":
        operators = build_finite_difference_operators(spec.n_z, spacing, periodic=False)
        identity = jnp.eye(spec.n_z, dtype=operators.D1.dtype)
        return ParallelGrid(
            z=lower + spacing * jnp.arange(spec.n_z, dtype=operators.D1.dtype),
            w_z=jnp.full((spec.n_z,), spacing, dtype=operators.D1.dtype),
            D_z=operators.D1,
            modal_transform=identity,
            inverse_modal_transform=identity,
            backend="finite_difference",
            topology="open",
        )
    return build_parallel_grid(
        ParallelGridSpec(
            n_z=spec.n_z,
            z_min=lower,
            z_max=lower + spec.field_periods,
            topology="periodic",
        )
    )


def _initial_tem_state(precompute, parallel, spec: TemCaseSpec | None = None):
    spec = spec or TemCaseSpec()
    if spec.initial_condition == "gyaradax_cosine2":
        profile = 1.0e-3 * (jnp.cos(2.0 * jnp.pi * parallel.z) + 1.0)
        shape = (
            precompute.n_species,
            precompute.rhs.maxwellian.shape[1],
            precompute.rhs.maxwellian.shape[2],
            parallel.z.shape[0],
            1,
            1,
        )
        return jnp.broadcast_to(profile[None, None, None, :, None, None], shape).astype(
            jnp.complex128
        )
    maxwellian = jnp.asarray(precompute.rhs.maxwellian)
    half_span = 0.5 * float(jnp.sum(parallel.w_z))
    phase = jnp.exp(0.07j * 2.0 * jnp.pi * parallel.z)
    profile = jnp.cos(jnp.pi * parallel.z / half_span) ** 2
    species_phase = jnp.asarray([1.0 + 0.2j, -0.4 + 0.7j])
    state = (
        1.0e-3
        * species_phase[:, None, None, None, None, None]
        * maxwellian[..., None, None]
        * (profile * phase)[None, None, None, :, None, None]
    )
    return state


__all__ = [
    "TEM_PREFLIGHT_SCHEMA_VERSION",
    "TemCaseSpec",
    "TemPhysicsPreflightReport",
    "TemLinearSmokeResult",
    "gyaradax_tem_case_spec",
    "run_reduced_tem_linear_smoke",
    "run_tem_physics_preflight",
    "tem_species",
]
