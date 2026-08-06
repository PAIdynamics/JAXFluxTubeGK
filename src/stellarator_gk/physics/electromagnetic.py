"""Algebraic electromagnetic field foundations for kinetic species."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.special import i1e

from ..geometry.analytic import k_perp_squared
from ..types import FourierGrid, SpeciesParams, VelocityGrid, _PyTreeDataclass
from .primitives import FLRFactors, bessel_j1_hat, normalized_energy, thermal_speed


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParallelAmperePrecompute(_PyTreeDataclass):
    """Weights and diagonal for the mixed-variable parallel Ampere solve."""

    source_weight: object
    denominator: object
    kperp_squared: object
    beta: object
    denominator_floor: float
    n_species: int
    representation: str = "mixed_g"

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "source_weight",
        "denominator",
        "kperp_squared",
        "beta",
        "denominator_floor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("n_species", "representation")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PerpendicularMagneticPrecompute(_PyTreeDataclass):
    """Coefficients for the coupled kinetic ``phi``/``B_parallel`` solve."""

    phi_weight: object
    bpar_weight: object
    denominator: object
    bpar_chi_factor: object
    beta: object
    denominator_floor: float
    n_species: int
    representation: str = "physical_f"

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "phi_weight",
        "bpar_weight",
        "denominator",
        "bpar_chi_factor",
        "beta",
        "denominator_floor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("n_species", "representation")


def build_parallel_ampere_precompute(
    velocity_grid: VelocityGrid,
    geometry,
    fourier_grid: FourierGrid,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    flr_factors: FLRFactors,
    *,
    beta,
    denominator_floor: float = 1.0e-14,
) -> ParallelAmperePrecompute:
    """Build GKW-normalized ``A_parallel`` weights for the evolved mixed variable.

    The diagonal uses the numerical velocity-grid moment, ensuring that the
    mixed-variable ``g -> f`` correction and Ampere response use the same
    finite-domain quadrature. This function establishes the field solve only;
    electromagnetic RHS evolution is a separate acceptance gate.
    """

    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be positive")
    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    beta = jnp.asarray(beta)
    charges = jnp.asarray([item.charge for item in species_tuple])
    densities = jnp.asarray([item.density for item in species_tuple])
    masses = jnp.asarray([item.mass for item in species_tuple])
    thermal_speeds = jnp.asarray([thermal_speed(item) for item in species_tuple])
    bessel = jnp.asarray(flr_factors.bessel_j0)
    if n_species == 1 and bessel.ndim == 4:
        bessel = bessel[None, ...]

    vpar = jnp.asarray(velocity_grid.vpar)[None, :, None, None, None, None]
    measure = (
        jnp.asarray(velocity_grid.w_vpar)[None, :, None, None, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, None, :, None, None, None]
        * jnp.asarray(geometry.B)[None, None, None, :, None, None]
    )
    source_weight = (
        beta
        * charges[:, None, None, None, None, None]
        * densities[:, None, None, None, None, None]
        * thermal_speeds[:, None, None, None, None, None]
        * vpar
        * bessel[:, None, ...]
        * measure
    )

    energy = normalized_energy(
        velocity_grid.vpar, velocity_grid.mu, geometry.B, species_tuple
    )
    numerical_maxwellian = (
        densities[:, None, None, None] * jnp.exp(-energy) / jnp.pi**1.5
    )
    gamma_numerical = jnp.sum(
        2.0
        * measure
        * bessel[:, None, ...] ** 2
        * vpar**2
        * numerical_maxwellian[..., None, None],
        axis=(1, 2),
    )
    skin = beta * jnp.sum(
        charges[:, None, None, None] ** 2
        * densities[:, None, None, None]
        * gamma_numerical
        / masses[:, None, None, None],
        axis=0,
    )
    kperp2 = k_perp_squared(geometry, fourier_grid)
    denominator = kperp2 + skin
    return ParallelAmperePrecompute(
        source_weight=source_weight,
        denominator=denominator,
        kperp_squared=kperp2,
        beta=beta,
        denominator_floor=float(denominator_floor),
        n_species=n_species,
    )


def build_perpendicular_magnetic_precompute(
    velocity_grid: VelocityGrid,
    geometry,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    flr_factors: FLRFactors,
    *,
    beta,
    denominator_floor: float = 1.0e-14,
) -> PerpendicularMagneticPrecompute:
    """Build the GKW kinetic coupled-field coefficients for ``phi`` and ``B_parallel``."""

    if denominator_floor <= 0.0:
        raise ValueError("denominator_floor must be positive")
    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    beta = jnp.asarray(beta)
    charges = jnp.asarray([item.charge for item in species_tuple])
    densities = jnp.asarray([item.density for item in species_tuple])
    temperatures = jnp.asarray([item.temperature for item in species_tuple])
    bessel = jnp.asarray(flr_factors.bessel_j0)
    argument = jnp.asarray(flr_factors.bessel_argument)
    gamma_zero = jnp.asarray(flr_factors.gamma0)
    gamma_one = i1e(jnp.asarray(flr_factors.polarization_argument))
    if n_species == 1 and bessel.ndim == 4:
        bessel = bessel[None, ...]
        argument = argument[None, ...]
        gamma_zero = gamma_zero[None, ...]
        gamma_one = gamma_one[None, ...]

    B = jnp.asarray(geometry.B)[None, :, None, None]
    kperp_zero = jnp.all(jnp.abs(argument) < 1.0e-5, axis=(0, 1))
    gamma_difference = jnp.where(kperp_zero[None, ...], 1.0, gamma_zero - gamma_one)
    gamma_zero_field = jnp.where(kperp_zero[None, ...], 1.0, gamma_zero)

    species_4 = (slice(None), None, None, None)
    charge_4 = charges[species_4]
    density_4 = densities[species_4]
    temperature_4 = temperatures[species_4]
    f_sp1 = jnp.sum(
        charge_4**2 * density_4 * (gamma_zero_field - 1.0) / temperature_4,
        axis=0,
    )
    f_sp2 = jnp.sum(charge_4 * beta * density_4 * gamma_difference / (2.0 * B), axis=0)
    b_sp1 = jnp.sum(charge_4 * density_4 * gamma_difference / B, axis=0)
    b_sp2 = jnp.sum(temperature_4 * density_4 * beta * gamma_difference / B**2, axis=0)
    denominator = f_sp1 * (1.0 + b_sp2) - f_sp2 * b_sp1

    measure = (
        jnp.asarray(velocity_grid.w_vpar)[None, :, None, None, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, None, :, None, None, None]
        * jnp.asarray(geometry.B)[None, None, None, :, None, None]
    )
    charge_6 = charges[:, None, None, None, None, None]
    density_6 = densities[:, None, None, None, None, None]
    temperature_6 = temperatures[:, None, None, None, None, None]
    mu_6 = jnp.asarray(velocity_grid.mu)[None, None, :, None, None, None]
    j1hat = jnp.where(kperp_zero[None, None, ...], 0.5, bessel_j1_hat(argument))
    i_sp1 = charge_6 * density_6 * measure * bessel[:, None, ...]
    i_sp2 = beta * temperature_6 * density_6 * measure * mu_6 * j1hat[:, None, ...]
    phi_weight = i_sp1 * (1.0 + b_sp2[None, None, None, ...]) - i_sp2 * b_sp1[
        None, None, None, ...
    ]
    bpar_weight = i_sp2 * f_sp1[None, None, None, ...] - i_sp1 * f_sp2[
        None, None, None, ...
    ]
    bpar_chi_factor = 2.0 * mu_6 * temperature_6 / charge_6 * j1hat[:, None, ...]
    return PerpendicularMagneticPrecompute(
        phi_weight=phi_weight,
        bpar_weight=bpar_weight,
        denominator=denominator,
        bpar_chi_factor=bpar_chi_factor,
        beta=beta,
        denominator_floor=float(denominator_floor),
        n_species=n_species,
    )


def solve_parallel_ampere(mixed_distribution, precompute: ParallelAmperePrecompute):
    """Solve normalized parallel Ampere's law from the evolved mixed variable."""

    distribution = _with_species_axis(mixed_distribution, precompute.n_species)
    numerator = jnp.sum(precompute.source_weight * distribution, axis=(0, 1, 2))
    denominator = _safe_denominator(precompute)
    return numerator / denominator


def parallel_ampere_residual(apar, mixed_distribution, precompute: ParallelAmperePrecompute):
    """Return ``numerator - denominator*A_parallel`` for field diagnostics."""

    distribution = _with_species_axis(mixed_distribution, precompute.n_species)
    numerator = jnp.sum(precompute.source_weight * distribution, axis=(0, 1, 2))
    return numerator - precompute.denominator * jnp.asarray(apar)


def solve_perpendicular_magnetic_fields(
    distribution, precompute: PerpendicularMagneticPrecompute
):
    """Solve the coupled kinetic quasineutrality/``B_parallel`` system."""

    distribution = _with_species_axis(distribution, precompute.n_species)
    denominator = _safe_field_denominator(precompute.denominator, precompute.denominator_floor)
    phi_numerator = jnp.sum(precompute.phi_weight * distribution, axis=(0, 1, 2))
    bpar_numerator = jnp.sum(precompute.bpar_weight * distribution, axis=(0, 1, 2))
    return -phi_numerator / denominator, -bpar_numerator / denominator


def _safe_denominator(precompute: ParallelAmperePrecompute):
    return _safe_field_denominator(precompute.denominator, precompute.denominator_floor)


def _safe_field_denominator(denominator, denominator_floor):
    denominator = jnp.asarray(denominator)
    floor = jnp.asarray(denominator_floor, dtype=denominator.dtype)
    return jnp.where(jnp.abs(denominator) < floor, floor, denominator)


def _with_species_axis(values, n_species: int):
    values = jnp.asarray(values)
    if values.ndim == 5 and n_species == 1:
        return values[None, ...]
    if values.ndim != 6 or values.shape[0] != n_species:
        raise ValueError("mixed distribution has incompatible species or phase-space shape")
    return values


__all__ = [
    "ParallelAmperePrecompute",
    "PerpendicularMagneticPrecompute",
    "build_parallel_ampere_precompute",
    "build_perpendicular_magnetic_precompute",
    "parallel_ampere_residual",
    "solve_parallel_ampere",
    "solve_perpendicular_magnetic_fields",
]
