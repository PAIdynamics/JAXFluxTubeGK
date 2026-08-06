"""Conservative model collision operators for velocity collocation grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from ..types import SpeciesParams, VelocityGrid, _PyTreeDataclass
from .primitives import maxwellian, normalized_energy


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConservingBGKPrecompute(_PyTreeDataclass):
    """Projection data for a species-local, moment-conserving BGK model."""

    frequency: object
    invariants: object
    equilibrium_basis: object
    moment_inverse: object
    measure: object
    n_species: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "frequency",
        "invariants",
        "equilibrium_basis",
        "moment_inverse",
        "measure",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("n_species",)


def build_conserving_bgk_precompute(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    frequency,
) -> ConservingBGKPrecompute:
    """Build a linearized BGK projection conserving three discrete moments.

    The null space spans a Maxwellian multiplied by ``1``, ``v_parallel``,
    and normalized particle energy. Conservation is exact for the supplied
    collocation quadrature, independently at every parallel location and for
    every Fourier mode. This is a model self-collision operator; inter-species
    exchange and a Landau/Fokker--Planck operator require separate validation.
    """

    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    frequency_array = jnp.broadcast_to(jnp.asarray(frequency), (n_species,))
    if bool(jnp.any(frequency_array < 0.0)):
        raise ValueError("collision frequencies must be nonnegative")

    B = jnp.asarray(B)
    energy = jnp.asarray(normalized_energy(velocity_grid.vpar, velocity_grid.mu, B, species_tuple))
    vpar = jnp.broadcast_to(
        jnp.asarray(velocity_grid.vpar)[None, :, None, None], energy.shape
    )
    invariants = jnp.stack((jnp.ones_like(energy), vpar, energy), axis=1)
    equilibrium = jnp.asarray(maxwellian(velocity_grid.vpar, velocity_grid.mu, B, species_tuple))
    equilibrium_basis = invariants * equilibrium[:, None, ...]
    measure = (
        jnp.asarray(velocity_grid.w_vpar)[:, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, :, None]
        * B[None, None, :]
    )
    moment_matrix = jnp.einsum(
        "savmz,sbvmz,vmz->szab",
        invariants,
        equilibrium_basis,
        measure,
    )
    return ConservingBGKPrecompute(
        frequency=frequency_array,
        invariants=invariants,
        equilibrium_basis=equilibrium_basis,
        moment_inverse=jnp.linalg.inv(moment_matrix),
        measure=measure,
        n_species=n_species,
    )


def conserving_bgk_collision(distribution, precompute: ConservingBGKPrecompute):
    """Return ``-nu (f - P f)`` with an exactly conservative discrete projection."""

    distribution = jnp.asarray(distribution)
    original_ndim = distribution.ndim
    if original_ndim == 5 and precompute.n_species == 1:
        distribution = distribution[None, ...]
    if distribution.ndim != 6 or distribution.shape[0] != precompute.n_species:
        raise ValueError(
            "distribution must have shape (species,vpar,mu,z,kx,ky), "
            "with the species axis optional for one species"
        )
    moments = jnp.einsum(
        "savmz,vmz,svmzxy->szxya",
        precompute.invariants,
        precompute.measure,
        distribution,
    )
    coefficients = jnp.einsum("szba,szxya->szxyb", precompute.moment_inverse, moments)
    projection = jnp.einsum(
        "sbvmz,szxyb->svmzxy", precompute.equilibrium_basis, coefficients
    )
    result = -precompute.frequency[:, None, None, None, None, None] * (
        distribution - projection
    )
    return result[0] if original_ndim == 5 else result


def collision_moments(values, precompute: ConservingBGKPrecompute):
    """Return density, parallel-momentum, and energy moments for diagnostics."""

    values = jnp.asarray(values)
    if values.ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != precompute.n_species:
        raise ValueError("values have incompatible phase-space shape")
    return jnp.einsum(
        "savmz,vmz,svmzxy->sazxy",
        precompute.invariants,
        precompute.measure,
        values,
    )


__all__ = [
    "ConservingBGKPrecompute",
    "build_conserving_bgk_precompute",
    "collision_moments",
    "conserving_bgk_collision",
]
