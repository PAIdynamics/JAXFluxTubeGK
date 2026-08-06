"""Algebraic electromagnetic field foundations for kinetic species."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from ..geometry.analytic import k_perp_squared
from ..types import FourierGrid, SpeciesParams, VelocityGrid, _PyTreeDataclass
from .primitives import FLRFactors, normalized_energy, thermal_speed


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


def _safe_denominator(precompute: ParallelAmperePrecompute):
    denominator = jnp.asarray(precompute.denominator)
    floor = jnp.asarray(precompute.denominator_floor, dtype=denominator.dtype)
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
    "build_parallel_ampere_precompute",
    "parallel_ampere_residual",
    "solve_parallel_ampere",
]
