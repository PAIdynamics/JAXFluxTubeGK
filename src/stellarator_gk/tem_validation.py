"""Kinetic-electron TEM preflight before quantitative external validation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import build_s_alpha_geometry
from .grids import build_fourier_grid, build_parallel_grid, build_velocity_grid
from .physics import kinetic_quasineutrality_residual, solve_kinetic_electron_phi
from .solver import build_linear_residual_precompute, linear_residual
from .time_advance import estimate_linear_cfl_dt
from .types import (
    FourierGridSpec,
    GeometryScalarParams,
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

    def __post_init__(self) -> None:
        if self.q <= 0.0 or self.eps <= 0.0 or self.ky <= 0.0:
            raise ValueError("q, eps, and ky must be positive")
        if not 0.0 < self.electron_mass < 1.0:
            raise ValueError("electron_mass must lie between zero and the ion mass")
        if min(self.n_z, self.n_vpar, self.n_mu) < 2:
            raise ValueError("TEM preflight grid sizes must be at least two")
        if self.vpar_max <= 0.0 or self.mu_max <= 0.0:
            raise ValueError("velocity bounds must be positive")


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


def run_tem_physics_preflight(
    spec: TemCaseSpec | None = None,
    *,
    field_tolerance: float = 1.0e-10,
    streaming_tolerance: float = 1.0e-10,
) -> TemPhysicsPreflightReport:
    """Exercise the coupled kinetic-electron path without claiming TEM parity."""

    spec = spec or TemCaseSpec()
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
    half_span = 0.5 * spec.field_periods
    parallel = build_parallel_grid(
        ParallelGridSpec(
            n_z=spec.n_z,
            z_min=-half_span,
            z_max=half_span,
            topology="periodic",
        )
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(spec.ky,))
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
    )
    maxwellian = jnp.asarray(precompute.rhs.maxwellian)
    phase = jnp.exp(0.07j * 2.0 * jnp.pi * parallel.z)
    profile = jnp.cos(jnp.pi * parallel.z / half_span) ** 2
    species_phase = jnp.asarray([1.0 + 0.2j, -0.4 + 0.7j])
    state = (
        1.0e-3
        * species_phase[:, None, None, None, None, None]
        * maxwellian[..., None, None]
        * (profile * phase)[None, None, None, :, None, None]
    )
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
    charge_density = float(sum(item.charge * item.density for item in species))
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


__all__ = [
    "TEM_PREFLIGHT_SCHEMA_VERSION",
    "TemCaseSpec",
    "TemPhysicsPreflightReport",
    "run_tem_physics_preflight",
    "tem_species",
]
