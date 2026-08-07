"""Conservative model collision operators for velocity collocation grids."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, pi
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, gamma, gammainc, gammaincc

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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FokkerPlanckPrecompute(_PyTreeDataclass):
    """Nine-point GKW-style test-particle Fokker--Planck stencil."""

    stencil: object
    row_sum_bound: object
    conservation_invariants: object
    conservation_basis: object
    conservation_inverse: object
    measure: object
    xu_momentum_factor: object
    xu_energy_factor: object
    xu_vpar_weight: object
    xu_energy_weight: object
    pair_stencil: object
    pair_conservation_invariants: object
    pair_conservation_basis: object
    pair_conservation_inverse: object
    pair_reciprocal_inverse: object
    pair_conservation_measure: object
    n_species: int
    conserve_exchange: bool = False
    conservation_model: str = "none"
    pair_indices: tuple[tuple[int, int], ...] = ()

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "stencil",
        "row_sum_bound",
        "conservation_invariants",
        "conservation_basis",
        "conservation_inverse",
        "measure",
        "xu_momentum_factor",
        "xu_energy_factor",
        "xu_vpar_weight",
        "xu_energy_weight",
        "pair_stencil",
        "pair_conservation_invariants",
        "pair_conservation_basis",
        "pair_conservation_inverse",
        "pair_reciprocal_inverse",
        "pair_conservation_measure",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_species",
        "conserve_exchange",
        "conservation_model",
        "pair_indices",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StellaTestParticlePrimitives(_PyTreeDataclass):
    """Analytic velocity-space coefficients used by stella's Landau stencil."""

    speed: object
    maxwellian: object
    parallel_diffusion: object
    deflection: object
    mixed_diffusion: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "speed",
        "maxwellian",
        "parallel_diffusion",
        "deflection",
        "mixed_diffusion",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ()


def build_stella_test_particle_primitives(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    pair_frequency,
    *,
    deflection_scale=1.0,
    electron_parallel_scale=1.0,
    electron_deflection_scale=1.0,
    mixed_scale=1.0,
    electron_index: int | None = 1,
    ion_index: int | None = 0,
    electron_ion_mass_ratio_approximation: bool = False,
) -> StellaTestParticlePrimitives:
    """Construct stella's normalized ``nupa``, ``nuD``, ``nux``, and ``mw``.

    This is the analytic coefficient layer below stella's finite-difference
    test-particle matrix. Velocities are normalized to each target species'
    thermal speed, matching stella's equal-temperature local collision model.
    Arrays use ``(target, background, vpar, mu, z)`` ordering.
    """

    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    frequencies = jnp.asarray(pair_frequency)
    if frequencies.shape != (n_species, n_species):
        raise ValueError(
            f"pair_frequency has shape {frequencies.shape}, expected {(n_species, n_species)}"
        )
    if electron_index is not None and not 0 <= electron_index < n_species:
        raise ValueError("electron_index is outside the species axis")
    if ion_index is not None and not 0 <= ion_index < n_species:
        raise ValueError("ion_index is outside the species axis")

    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    magnetic_field = jnp.asarray(B)
    speed = jnp.sqrt(
        vpar[:, None, None] ** 2
        + 2.0 * mu[None, :, None] * magnetic_field[None, None, :]
    )
    masses = jnp.asarray([item.mass for item in species_tuple])
    mass_ratio = masses[:, None] / masses[None, :]
    normalized_speed = speed[None, None, ...] / jnp.sqrt(
        mass_ratio[:, :, None, None, None]
    )
    safe_speed = jnp.maximum(speed, jnp.sqrt(jnp.finfo(speed.dtype).tiny))
    safe_normalized = jnp.maximum(
        normalized_speed,
        jnp.sqrt(jnp.finfo(normalized_speed.dtype).tiny),
    )
    erf_value = erf(normalized_speed)
    chandrasekhar = (
        erf_value
        - 2.0 / jnp.sqrt(jnp.asarray(pi, dtype=speed.dtype))
        * normalized_speed
        * jnp.exp(-(normalized_speed**2))
    ) / (2.0 * safe_normalized**2)
    frequency = frequencies[:, :, None, None, None]
    parallel_diffusion = frequency * 2.0 * chandrasekhar / safe_speed[None, None, ...] ** 3
    deflection = (
        jnp.asarray(deflection_scale)
        * frequency
        * (erf_value - chandrasekhar)
        / safe_speed[None, None, ...] ** 3
    )

    electron_ion_pair = (
        electron_index is not None
        and ion_index is not None
        and electron_index != ion_index
    )
    if electron_ion_pair and electron_ion_mass_ratio_approximation:
        deflection = deflection.at[electron_index, ion_index].set(
            jnp.asarray(deflection_scale)
            * frequencies[electron_index, ion_index]
            / safe_speed**3
        )
    mixed_diffusion = jnp.asarray(mixed_scale) * (
        parallel_diffusion - jnp.asarray(deflection_scale) * deflection
    )
    if electron_ion_pair:
        mixed_diffusion = mixed_diffusion.at[electron_index, ion_index].set(
            jnp.asarray(mixed_scale)
            * (
                jnp.asarray(electron_parallel_scale)
                * parallel_diffusion[electron_index, ion_index]
                - jnp.asarray(electron_deflection_scale)
                * jnp.asarray(deflection_scale)
                * deflection[electron_index, ion_index]
            )
        )
        deflection = deflection.at[electron_index, ion_index].multiply(
            jnp.asarray(electron_deflection_scale)
        )
    maxwell = jnp.exp(
        -(vpar[:, None, None] ** 2)
        - 2.0 * mu[None, :, None] * magnetic_field[None, None, :]
    )
    maxwell = jnp.broadcast_to(maxwell, (n_species, *maxwell.shape))
    return StellaTestParticlePrimitives(
        speed=speed,
        maxwellian=maxwell,
        parallel_diffusion=parallel_diffusion,
        deflection=deflection,
        mixed_diffusion=mixed_diffusion,
    )


def build_stella_test_particle_gyro_diagonal(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    primitives: StellaTestParticlePrimitives,
    dt,
    *,
    gyro_scale=1.0,
    deflection_scale=1.0,
    electron_parallel_scale=1.0,
    electron_deflection_scale=1.0,
    electron_index: int | None = 1,
    ion_index: int | None = 0,
):
    """Return stella's contribution to ``I-dt*C_test`` per unit ``kperp2``.

    The result has ``(species, vpar, mu, z)`` layout and is purely diagonal in
    velocity and species. ``primitives`` must use the same electron-ion knob
    values; stella applies the corresponding factors once more in this term.
    """

    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    magnetic_field = jnp.asarray(B)
    parallel_scale = jnp.ones((n_species, n_species), dtype=magnetic_field.dtype)
    deflection_pair_scale = jnp.ones_like(parallel_scale)
    if (
        electron_index is not None
        and ion_index is not None
        and electron_index != ion_index
    ):
        parallel_scale = parallel_scale.at[electron_index, ion_index].set(
            electron_parallel_scale
        )
        deflection_pair_scale = deflection_pair_scale.at[
            electron_index, ion_index
        ].set(electron_deflection_scale)
    parallel = jnp.sum(
        primitives.parallel_diffusion
        * parallel_scale[:, :, None, None, None],
        axis=1,
    )
    deflection = jnp.sum(
        primitives.deflection
        * deflection_pair_scale[:, :, None, None, None],
        axis=1,
    )
    bmu = magnetic_field[None, None, :] * mu[None, :, None]
    velocity_squared = vpar[:, None, None] ** 2
    smz = jnp.asarray(
        [
            jnp.abs(jnp.sqrt(item.temperature * item.mass) / item.charge)
            for item in species_tuple
        ]
    )
    return (
        jnp.asarray(dt)
        * jnp.asarray(gyro_scale)
        * 0.5
        * (smz[:, None, None, None] / magnetic_field[None, None, None, :]) ** 2
        * (
            parallel * bmu[None, ...]
            + jnp.asarray(deflection_scale)
            * deflection
            * (velocity_squared + bmu)[None, ...]
        )
    )


def assemble_stella_test_particle_blocks(lower, diagonal, upper):
    """Assemble stella velocity blocks into the species-local implicit matrix.

    Each input has ``(target, background, vpar, row_mu, col_mu, z)`` layout
    and contains an already time-discretized contribution to ``-dt*C_test``.
    Background-species blocks are summed, the identity is added, and the
    returned dense matrices use ``(z, target, state, state)`` layout with
    ``state = vpar * n_mu + mu``.  Following stella's convention,
    ``lower[:, :, iv]`` couples row ``iv`` to column ``iv - 1`` and
    ``upper[:, :, iv]`` couples row ``iv`` to column ``iv + 1``.
    """

    lower_array = jnp.asarray(lower)
    diagonal_array = jnp.asarray(diagonal)
    upper_array = jnp.asarray(upper)
    if lower_array.shape != diagonal_array.shape or lower_array.shape != upper_array.shape:
        raise ValueError("lower, diagonal, and upper blocks must have identical shapes")
    if lower_array.ndim != 6:
        raise ValueError(
            "collision blocks must have (target, background, vpar, row_mu, col_mu, z) layout"
        )
    n_target, _, n_vpar, n_row_mu, n_col_mu, n_z = lower_array.shape
    if n_row_mu != n_col_mu:
        raise ValueError("collision velocity blocks must be square in mu")

    lower_sum = jnp.sum(lower_array, axis=1).transpose(0, 4, 1, 2, 3)
    diagonal_sum = jnp.sum(diagonal_array, axis=1).transpose(0, 4, 1, 2, 3)
    upper_sum = jnp.sum(upper_array, axis=1).transpose(0, 4, 1, 2, 3)
    matrix = jnp.zeros(
        (n_target, n_z, n_vpar, n_row_mu, n_vpar, n_col_mu),
        dtype=jnp.result_type(lower_array, diagonal_array, upper_array),
    )
    for iv in range(n_vpar):
        matrix = matrix.at[:, :, iv, :, iv, :].set(diagonal_sum[:, :, iv])
        if iv > 0:
            matrix = matrix.at[:, :, iv, :, iv - 1, :].set(lower_sum[:, :, iv])
        if iv + 1 < n_vpar:
            matrix = matrix.at[:, :, iv, :, iv + 1, :].set(upper_sum[:, :, iv])
    matrix = matrix.reshape(n_target, n_z, n_vpar * n_row_mu, n_vpar * n_col_mu)
    identity = jnp.eye(n_vpar * n_row_mu, dtype=matrix.dtype)
    return (matrix + identity[None, None, ...]).transpose(1, 0, 2, 3)


def build_stella_two_mu_diffusion_blocks(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    pair_frequency,
    primitives: StellaTestParticlePrimitives,
    dt,
    *,
    deflection_scale=1.0,
    electron_parallel_scale=1.0,
    electron_deflection_scale=1.0,
    electron_index: int | None = 1,
    ion_index: int | None = 0,
    electron_ion_mass_ratio_approximation: bool = False,
):
    """Construct stella's pure-mu boundary blocks for a two-node mu grid.

    This implements the default non-density-conserving ghost-cell formulas at
    both mu boundaries. The returned ``(lower, diagonal, upper)`` arrays use
    the block layout accepted by :func:`assemble_stella_test_particle_blocks`.
    Only ``diagonal`` is nonzero because the pure-mu operator is local in
    parallel velocity.
    """

    mu = jnp.asarray(velocity_grid.mu)
    if mu.shape != (2,):
        raise ValueError("the two-mu boundary constructor requires exactly two mu nodes")
    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    frequency = jnp.asarray(pair_frequency)
    if frequency.shape != (n_species, n_species):
        raise ValueError("pair_frequency must have (target, background) shape")
    magnetic_field = jnp.asarray(B)
    vpar = jnp.asarray(velocity_grid.vpar)
    dmu = mu[1] - mu[0]

    pair_parallel_scale = jnp.ones((n_species, n_species), dtype=magnetic_field.dtype)
    pair_deflection_scale = jnp.ones_like(pair_parallel_scale)
    electron_ion_pair = (
        electron_index is not None
        and ion_index is not None
        and electron_index != ion_index
    )
    if electron_ion_pair:
        pair_parallel_scale = pair_parallel_scale.at[electron_index, ion_index].set(
            electron_parallel_scale
        )
        pair_deflection_scale = pair_deflection_scale.at[
            electron_index, ion_index
        ].set(electron_deflection_scale)

    half_mu = 0.5 * (mu[0] + mu[1])
    half_speed = jnp.sqrt(
        vpar[:, None] ** 2 + 2.0 * half_mu * magnetic_field[None, :]
    )
    masses = jnp.asarray([item.mass for item in species_tuple])
    mass_ratio = masses[:, None] / masses[None, :]
    normalized = half_speed[None, None, ...] / jnp.sqrt(
        mass_ratio[:, :, None, None]
    )
    erf_value = erf(normalized)
    chandrasekhar = (
        erf_value
        - 2.0 / jnp.sqrt(jnp.asarray(pi, dtype=half_speed.dtype))
        * normalized
        * jnp.exp(-(normalized**2))
    ) / (2.0 * normalized**2)
    half_parallel = (
        frequency[:, :, None, None]
        * 2.0
        * chandrasekhar
        / half_speed[None, None, ...] ** 3
    )
    half_deflection = (
        jnp.asarray(deflection_scale)
        * frequency[:, :, None, None]
        * (erf_value - chandrasekhar)
        / half_speed[None, None, ...] ** 3
    )
    if electron_ion_pair and electron_ion_mass_ratio_approximation:
        half_deflection = half_deflection.at[electron_index, ion_index].set(
            jnp.asarray(deflection_scale)
            * frequency[electron_index, ion_index]
            / half_speed**3
        )
    if electron_ion_pair:
        half_deflection = half_deflection.at[electron_index, ion_index].multiply(
            electron_deflection_scale
        )

    node_maxwell = jnp.asarray(primitives.maxwellian)[:, None, ...]
    node_parallel = jnp.asarray(primitives.parallel_diffusion)
    node_deflection = jnp.asarray(primitives.deflection)
    node_velocity_squared = vpar[None, None, :, None, None] ** 2
    node_field = magnetic_field[None, None, None, None, :]
    node_mu = mu[None, None, None, :, None]
    node_gamma = 2.0 * (
        pair_parallel_scale[:, :, None, None, None]
        * node_parallel
        * node_mu**2
        + pair_deflection_scale[:, :, None, None, None]
        * jnp.asarray(deflection_scale)
        * node_deflection
        * node_velocity_squared
        / (2.0 * node_field)
        * node_mu
    ) * node_maxwell
    half_maxwell = jnp.exp(
        -(vpar[None, :, None] ** 2)
        - 2.0 * half_mu * magnetic_field[None, None, :]
    )
    half_gamma = 2.0 * (
        pair_parallel_scale[:, :, None, None] * half_parallel * half_mu**2
        + pair_deflection_scale[:, :, None, None]
        * jnp.asarray(deflection_scale)
        * half_deflection
        * vpar[None, None, :, None] ** 2
        / (2.0 * magnetic_field[None, None, None, :])
        * half_mu
    ) * half_maxwell[:, None, ...]

    gamma_lower = node_gamma[:, :, :, 0, :]
    gamma_upper = node_gamma[:, :, :, 1, :]
    maxwell_lower = node_maxwell[:, 0, :, 0, :]
    maxwell_upper = node_maxwell[:, 0, :, 1, :]
    lower_width = 0.5 * dmu + mu[0]
    upper_width = 1.5 * dmu
    diagonal = jnp.zeros(
        (n_species, n_species, vpar.size, 2, 2, magnetic_field.size),
        dtype=node_parallel.dtype,
    )
    diagonal = diagonal.at[:, :, :, 0, 0, :].set(
        -jnp.asarray(dt)
        * (
            -gamma_lower / dmu * dmu / (2.0 * mu[0])
            + (-half_gamma / dmu + gamma_lower / dmu) * mu[0] / (0.5 * dmu)
        )
        / maxwell_lower[:, None, ...]
        / lower_width
    )
    diagonal = diagonal.at[:, :, :, 0, 1, :].set(
        -jnp.asarray(dt)
        * (
            gamma_lower / dmu * dmu / (2.0 * mu[0])
            + (half_gamma / dmu - gamma_lower / dmu) * mu[0] / (0.5 * dmu)
        )
        / maxwell_upper[:, None, ...]
        / lower_width
    )
    diagonal = diagonal.at[:, :, :, 1, 1, :].set(
        -jnp.asarray(dt)
        * (
            (gamma_upper / dmu - half_gamma / dmu) * dmu / (0.5 * dmu)
            + (-gamma_upper / dmu) * dmu / 2.0 / dmu
        )
        / maxwell_upper[:, None, ...]
        / upper_width
    )
    diagonal = diagonal.at[:, :, :, 1, 0, :].set(
        -jnp.asarray(dt)
        * (
            (-gamma_upper / dmu + half_gamma / dmu) * dmu / (0.5 * dmu)
            + gamma_upper / dmu * dmu / 2.0 / dmu
        )
        / maxwell_lower[:, None, ...]
        / upper_width
    )
    zero = jnp.zeros_like(diagonal)
    return zero, diagonal, zero


def build_stella_two_mu_vpar_mixed_blocks(
    velocity_grid: VelocityGrid,
    primitives: StellaTestParticlePrimitives,
    dt,
):
    """Construct stella's vpar-path mixed blocks on a two-node mu grid."""

    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    if mu.shape != (2,):
        raise ValueError("the two-mu mixed constructor requires exactly two mu nodes")
    nux = jnp.asarray(primitives.mixed_diffusion)
    maxwell = jnp.asarray(primitives.maxwellian)
    n_target, n_background, n_vpar, _, n_z = nux.shape
    dvpar = vpar[1] - vpar[0]
    dmu = mu[1] - mu[0]
    lower = jnp.zeros((n_target, n_background, n_vpar, 2, 2, n_z), dtype=nux.dtype)
    diagonal = jnp.zeros_like(lower)
    upper = jnp.zeros_like(lower)

    def node_flux(iv, imu):
        return vpar[iv] * mu[imu] * nux[:, :, iv, imu, :] * maxwell[:, None, iv, imu, :]

    # Lower-vpar boundary: second-order ghost flux for the lower-mu row.
    averaged = 0.5 * (node_flux(0, 0) + node_flux(0, 1))
    next_averaged = 0.5 * (node_flux(1, 0) + node_flux(1, 1))
    factor = 0.5 * jnp.asarray(dt) / (dvpar * dmu)
    diagonal = diagonal.at[:, :, 0, 0, 0, :].set(factor * averaged / maxwell[:, None, 0, 0, :])
    diagonal = diagonal.at[:, :, 0, 0, 1, :].set(-factor * averaged / maxwell[:, None, 0, 1, :])
    upper = upper.at[:, :, 0, 0, 0, :].set(factor * next_averaged / maxwell[:, None, 1, 0, :])
    upper = upper.at[:, :, 0, 0, 1, :].set(-factor * next_averaged / maxwell[:, None, 1, 1, :])
    upper = upper.at[:, :, 0, 1, 0, :].set(
        jnp.asarray(dt) * node_flux(1, 1) / maxwell[:, None, 1, 0, :] / (2.0 * dvpar * dmu)
    )
    upper = upper.at[:, :, 0, 1, 1, :].set(
        -jnp.asarray(dt) * node_flux(1, 1) / maxwell[:, None, 1, 1, :] / (2.0 * dvpar * dmu)
    )

    for iv in range(1, n_vpar - 1):
        lower_flux = node_flux(iv - 1, 0)
        upper_flux = node_flux(iv + 1, 0)
        lower = lower.at[:, :, iv, 0, 0, :].set(
            -jnp.asarray(dt) * lower_flux / maxwell[:, None, iv - 1, 0, :] / (2.0 * dvpar * dmu)
        )
        lower = lower.at[:, :, iv, 0, 1, :].set(
            jnp.asarray(dt) * lower_flux / maxwell[:, None, iv - 1, 1, :] / (2.0 * dvpar * dmu)
        )
        upper = upper.at[:, :, iv, 0, 0, :].set(
            jnp.asarray(dt) * upper_flux / maxwell[:, None, iv + 1, 0, :] / (2.0 * dvpar * dmu)
        )
        upper = upper.at[:, :, iv, 0, 1, :].set(
            -jnp.asarray(dt) * upper_flux / maxwell[:, None, iv + 1, 1, :] / (2.0 * dvpar * dmu)
        )
        lower_flux = node_flux(iv - 1, 1)
        upper_flux = node_flux(iv + 1, 1)
        lower = lower.at[:, :, iv, 1, 0, :].set(
            -jnp.asarray(dt) * lower_flux / maxwell[:, None, iv - 1, 0, :] / (2.0 * dvpar * dmu)
        )
        lower = lower.at[:, :, iv, 1, 1, :].set(
            jnp.asarray(dt) * lower_flux / maxwell[:, None, iv - 1, 1, :] / (2.0 * dvpar * dmu)
        )
        upper = upper.at[:, :, iv, 1, 0, :].set(
            jnp.asarray(dt) * upper_flux / maxwell[:, None, iv + 1, 0, :] / (2.0 * dvpar * dmu)
        )
        upper = upper.at[:, :, iv, 1, 1, :].set(
            -jnp.asarray(dt) * upper_flux / maxwell[:, None, iv + 1, 1, :] / (2.0 * dvpar * dmu)
        )

    last = n_vpar - 1
    averaged = 0.5 * (node_flux(last, 0) + node_flux(last, 1))
    previous_averaged = 0.5 * (node_flux(last - 1, 0) + node_flux(last - 1, 1))
    diagonal = diagonal.at[:, :, last, 0, 0, :].set(-factor * averaged / maxwell[:, None, last, 0, :])
    diagonal = diagonal.at[:, :, last, 0, 1, :].set(factor * averaged / maxwell[:, None, last, 1, :])
    lower = lower.at[:, :, last, 0, 0, :].set(-factor * previous_averaged / maxwell[:, None, last - 1, 0, :])
    lower = lower.at[:, :, last, 0, 1, :].set(factor * previous_averaged / maxwell[:, None, last - 1, 1, :])
    lower = lower.at[:, :, last, 1, 0, :].set(
        -jnp.asarray(dt) * node_flux(last - 1, 1) / maxwell[:, None, last - 1, 0, :] / (2.0 * dvpar * dmu)
    )
    lower = lower.at[:, :, last, 1, 1, :].set(
        jnp.asarray(dt) * node_flux(last - 1, 1) / maxwell[:, None, last - 1, 1, :] / (2.0 * dvpar * dmu)
    )
    return lower, diagonal, upper


def build_stella_vpar_diffusion_blocks(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    pair_frequency,
    primitives: StellaTestParticlePrimitives,
    dt,
    *,
    deflection_scale=1.0,
    electron_parallel_scale=1.0,
    electron_deflection_scale=1.0,
    electron_index: int | None = 1,
    ion_index: int | None = 0,
    electron_ion_mass_ratio_approximation: bool = False,
):
    """Construct stella's pure parallel-velocity diffusion blocks."""

    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    frequency = jnp.asarray(pair_frequency)
    if frequency.shape != (n_species, n_species):
        raise ValueError("pair_frequency must have (target, background) shape")
    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    magnetic_field = jnp.asarray(B)
    dvpar = vpar[1] - vpar[0]
    half_vpar = 0.5 * (vpar[:-1] + vpar[1:])
    half_speed = jnp.sqrt(
        half_vpar[:, None, None] ** 2
        + 2.0 * mu[None, :, None] * magnetic_field[None, None, :]
    )
    masses = jnp.asarray([item.mass for item in species_tuple])
    mass_ratio = masses[:, None] / masses[None, :]
    normalized = half_speed[None, None, ...] / jnp.sqrt(
        mass_ratio[:, :, None, None, None]
    )
    erf_value = erf(normalized)
    chandrasekhar = (
        erf_value
        - 2.0 / jnp.sqrt(jnp.asarray(pi, dtype=half_speed.dtype))
        * normalized
        * jnp.exp(-(normalized**2))
    ) / (2.0 * normalized**2)
    half_parallel = (
        frequency[:, :, None, None, None]
        * 2.0
        * chandrasekhar
        / half_speed[None, None, ...] ** 3
    )
    half_deflection = (
        jnp.asarray(deflection_scale)
        * frequency[:, :, None, None, None]
        * (erf_value - chandrasekhar)
        / half_speed[None, None, ...] ** 3
    )
    electron_ion_pair = (
        electron_index is not None
        and ion_index is not None
        and electron_index != ion_index
    )
    if electron_ion_pair and electron_ion_mass_ratio_approximation:
        half_deflection = half_deflection.at[electron_index, ion_index].set(
            jnp.asarray(deflection_scale)
            * frequency[electron_index, ion_index]
            / half_speed**3
        )
    parallel_scale = jnp.ones((n_species, n_species), dtype=magnetic_field.dtype)
    deflection_pair_scale = jnp.ones_like(parallel_scale)
    if electron_ion_pair:
        parallel_scale = parallel_scale.at[electron_index, ion_index].set(
            electron_parallel_scale
        )
        half_deflection = half_deflection.at[electron_index, ion_index].multiply(
            electron_deflection_scale
        )
        deflection_pair_scale = deflection_pair_scale.at[
            electron_index, ion_index
        ].set(electron_deflection_scale)
    half_maxwell = jnp.exp(
        -(half_vpar[None, :, None, None] ** 2)
        - 2.0 * mu[None, None, :, None] * magnetic_field[None, None, None, :]
    )
    coefficient = (
        0.5
        * jnp.asarray(dt)
        * (
            parallel_scale[:, :, None, None, None]
            * half_parallel
            * half_vpar[None, None, :, None, None] ** 2
            + deflection_pair_scale[:, :, None, None, None]
            * jnp.asarray(deflection_scale)
            * 2.0
            * half_deflection
            * magnetic_field[None, None, None, None, :]
            * mu[None, None, None, :, None]
        )
        * half_maxwell[:, None, ...]
        / dvpar**2
    )
    maxwell = jnp.asarray(primitives.maxwellian)
    shape = (n_species, n_species, vpar.size, mu.size, mu.size, magnetic_field.size)
    lower = jnp.zeros(shape, dtype=coefficient.dtype)
    diagonal = jnp.zeros_like(lower)
    upper = jnp.zeros_like(lower)
    for imu in range(mu.size):
        upper = upper.at[:, :, :-1, imu, imu, :].set(
            -coefficient[:, :, :, imu, :] / maxwell[:, None, 1:, imu, :]
        )
        lower = lower.at[:, :, 1:, imu, imu, :].set(
            -coefficient[:, :, :, imu, :] / maxwell[:, None, :-1, imu, :]
        )
        diagonal = diagonal.at[:, :, 0, imu, imu, :].set(
            coefficient[:, :, 0, imu, :] / maxwell[:, None, 0, imu, :]
        )
        diagonal = diagonal.at[:, :, -1, imu, imu, :].set(
            coefficient[:, :, -1, imu, :] / maxwell[:, None, -1, imu, :]
        )
        diagonal = diagonal.at[:, :, 1:-1, imu, imu, :].set(
            (coefficient[:, :, :-1, imu, :] + coefficient[:, :, 1:, imu, :])
            / maxwell[:, None, 1:-1, imu, :]
        )
    return lower, diagonal, upper


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LaguerreLegendreCollisionPrecompute(_PyTreeDataclass):
    """Low-rank field-particle coefficients on a collocation grid.

    ``driver[a,b,c]`` contracts the distribution of background species ``b``
    into the scalar response for target ``a`` and component ``c``.
    ``response[a,b,c]`` maps that scalar back to the target velocity grid.
    The component labels and coefficient normalization are supplied by a
    separately validated builder or external parity workflow.
    """

    driver: object
    response: object
    row_sum_bound: object
    n_species: int
    component_labels: tuple[tuple[int, int, int], ...]

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "driver",
        "response",
        "row_sum_bound",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("n_species", "component_labels")


def _stella_associated_legendre(degree: int, order: int, x):
    """Return stella's Condon--Shortley associated Legendre polynomial."""

    absolute_order = abs(order)
    if absolute_order > degree:
        return jnp.zeros_like(x)
    diagonal = jnp.ones_like(x)
    for diagonal_degree in range(1, absolute_order + 1):
        diagonal = diagonal * (-(2 * diagonal_degree - 1)) * jnp.sqrt(jnp.maximum(1.0 - x * x, 0.0))
    if degree == absolute_order:
        result = diagonal
    else:
        previous = jnp.zeros_like(x)
        current = diagonal
        for recurrence_degree in range(absolute_order + 1, degree + 1):
            following = (
                (2 * recurrence_degree - 1) * x * current
                - (recurrence_degree - 1 + absolute_order) * previous
            ) / (recurrence_degree - absolute_order)
            previous, current = current, following
        result = current
    if order < 0:
        result = (
            (-1) ** absolute_order
            * factorial(degree - absolute_order)
            / factorial(degree + absolute_order)
            * result
        )
    return result


def stella_laguerre_legendre_delta0(
    speed,
    target_mass,
    background_mass,
    *,
    laguerre_degree: int,
    legendre_degree: int,
):
    """Evaluate stella's analytic lowest-order Landau response ``Delta_0``.

    The result is the collision-frequency-free response of target species
    ``a`` to ``x_b^l L_n^{l+1/2}(x_b^2) exp(-x_b^2)`` from background ``b``.
    ``speed`` is normalized to the target thermal speed.  The formula is
    differentiable in speed and the two species masses away from zero speed.
    """

    if laguerre_degree < 0 or legendre_degree < 0:
        raise ValueError("Laguerre and Legendre degrees must be nonnegative")
    xa = jnp.asarray(speed)
    mass_ratio = jnp.asarray(target_mass) / jnp.asarray(background_mass)
    xb = xa / jnp.sqrt(mass_ratio)
    # The native velocity quadrature has strictly positive total speed.  Keep
    # the expression defined at an exact origin for generic collocation grids.
    safe_xb = jnp.maximum(xb, jnp.sqrt(jnp.finfo(xb.dtype).tiny))
    result = jnp.zeros_like(xb)
    degree = legendre_degree
    for index in range(laguerre_degree + 1):
        coefficient = (
            (-1) ** index
            * gamma(laguerre_degree + degree + 1.5)
            / (
                gamma(laguerre_degree - index + 1.0)
                * gamma(degree + index + 1.5)
                * gamma(index + 1.0)
            )
        )
        argument = xb**2
        gamma_lower_1 = gamma(1.5 + degree + index) * gammainc(1.5 + degree + index, argument)
        gamma_lower_2 = gamma(2.5 + degree + index) * gammainc(2.5 + degree + index, argument)
        gamma_upper_1 = gamma(1.0 + index) * gammaincc(1.0 + index, argument)
        gamma_upper_2 = gamma(2.0 + index) * gammaincc(2.0 + index, argument)
        result = result + coefficient * (
            (2 * degree + 1.0) * xb ** (degree + 2 * index) * jnp.exp(-argument)
            - xb
            * (1.0 - mass_ratio)
            * (
                -(degree + 1.0) / safe_xb ** (degree + 2) * gamma_lower_1
                + degree * xb ** (degree - 1) * gamma_upper_1
            )
            - (gamma_lower_1 / safe_xb ** (degree + 1) + xb**degree * gamma_upper_1)
            + mass_ratio
            * xb**2
            * (
                (degree + 1.0)
                * (degree + 2.0)
                / (2 * degree + 3.0)
                * (gamma_lower_2 / safe_xb ** (degree + 3) + xb**degree * gamma_upper_1)
                - degree
                * (degree - 1.0)
                / (2 * degree - 1.0)
                * (gamma_lower_1 / safe_xb ** (degree + 1) + xb ** (degree - 2) * gamma_upper_2)
            )
        )
    prefactor = 4.0 * pi / pi**1.5 * jnp.exp(-(xa**2)) * mass_ratio / (2 * degree + 1.0)
    return jnp.nan_to_num(result * prefactor)


def _associated_laguerre(degree: int, alpha: float, argument):
    if degree == 0:
        return jnp.ones_like(argument)
    previous = jnp.ones_like(argument)
    current = 1.0 + alpha - argument
    for index in range(2, degree + 1):
        following = (
            (2 * index - 1 + alpha - argument) * current - (index - 1 + alpha) * previous
        ) / index
        previous, current = current, following
    return current


def build_stella_laguerre_legendre_delta(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    velocity_measure,
    *,
    component_labels: tuple[tuple[int, int, int], ...],
):
    """Construct stella's collision-frequency-free ``Delta_j`` responses.

    ``velocity_measure`` is the full ``integrate_vmu`` weight product with
    shape ``(vpar, mu, z)``. The returned array has shape
    ``(target, background, component, vpar, mu, z)``. This implements the
    recursive orthogonalization used by stella, including its mass-ratio
    self-adjointness branch, without depending on stella at runtime.
    """

    species_tuple = species if isinstance(species, tuple) else (species,)
    labels = tuple(tuple(int(value) for value in label) for label in component_labels)
    if any(len(label) != 3 for label in labels):
        raise ValueError("component labels must contain (l,m,j)")
    if any(degree < 0 or abs(order) > degree or laguerre < 0 for degree, order, laguerre in labels):
        raise ValueError("component labels require l >= |m| and j >= 0")
    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    magnetic_field = jnp.asarray(B)
    measure = jnp.asarray(velocity_measure)
    expected_measure = (vpar.size, mu.size, magnetic_field.size)
    if measure.shape != expected_measure:
        raise ValueError(f"velocity_measure has shape {measure.shape}, expected {expected_measure}")
    speed = jnp.sqrt(
        vpar[:, None, None] ** 2 + 2.0 * mu[None, :, None] * magnetic_field[None, None, :]
    )
    masses = jnp.stack([jnp.asarray(item.mass) for item in species_tuple])
    delta_cache = {}
    psi_cache = {}

    def velocity_polynomial(laguerre: int, legendre: int):
        return speed**legendre * _associated_laguerre(laguerre, legendre + 0.5, speed**2)

    def integrate(values):
        return jnp.sum(measure * values, axis=(0, 1))

    def delta(
        orthogonal_degree: int,
        input_degree: int,
        legendre: int,
        target: int,
        background: int,
    ):
        key = (orthogonal_degree, input_degree, legendre, target, background)
        if key not in delta_cache:
            if orthogonal_degree == 0:
                value = stella_laguerre_legendre_delta0(
                    speed,
                    masses[target],
                    masses[background],
                    laguerre_degree=input_degree,
                    legendre_degree=legendre,
                )
            else:
                value = delta(
                    orthogonal_degree - 1,
                    input_degree,
                    legendre,
                    target,
                    background,
                ) - psi(
                    orthogonal_degree - 1,
                    input_degree,
                    legendre,
                    target,
                    background,
                )[None, None, :] * delta(
                    orthogonal_degree - 1,
                    orthogonal_degree - 1,
                    legendre,
                    target,
                    background,
                )
            delta_cache[key] = value
        return delta_cache[key]

    def psi(
        orthogonal_degree: int,
        input_degree: int,
        legendre: int,
        target: int,
        background: int,
    ):
        key = (orthogonal_degree, input_degree, legendre, target, background)
        if key not in psi_cache:
            swapped = delta(
                orthogonal_degree,
                orthogonal_degree,
                legendre,
                background,
                target,
            )
            numerator = integrate(velocity_polynomial(input_degree, legendre) * swapped)
            direct = delta(
                orthogonal_degree,
                orthogonal_degree,
                legendre,
                target,
                background,
            )
            direct_denominator = integrate(
                velocity_polynomial(orthogonal_degree, legendre)
                * direct
                * (masses[background] / masses[target]) ** 3.5
            )
            swapped_denominator = integrate(
                velocity_polynomial(orthogonal_degree, legendre) * swapped
            )
            denominator = jnp.where(
                masses[background] / masses[target] < 1.0,
                direct_denominator,
                swapped_denominator,
            )
            psi_cache[key] = numerator / denominator
        return psi_cache[key]

    pairs = []
    for target in range(len(species_tuple)):
        backgrounds = []
        for background in range(len(species_tuple)):
            components = [
                delta(laguerre, laguerre, legendre, target, background)
                for legendre, _order, laguerre in labels
            ]
            backgrounds.append(jnp.stack(components))
        pairs.append(jnp.stack(backgrounds))
    return jnp.stack(pairs)


def build_stella_laguerre_legendre_response(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    pair_frequency,
    delta_j,
    gyroaverage,
    *,
    component_labels: tuple[tuple[int, int, int], ...],
):
    """Construct stella-normalized field-particle response coefficients.

    ``delta_j`` has shape ``(target, background, component, vpar, mu, z)``.
    ``gyroaverage`` has shape ``(target, |m|, mu, z, kx, ky)`` and contains
    stella's ``J_|m|`` factors.  The returned response has the low-rank
    coefficient shape ``(target, background, component, vpar, mu, z, kx, ky)``.

    This builder implements the independently testable response side of
    stella's Laguerre--Legendre field-particle operator.  Construction of
    ``delta_j`` from incomplete-gamma velocity integrals is a separate step.
    """

    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    labels = tuple(tuple(int(value) for value in label) for label in component_labels)
    if any(len(label) != 3 for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("component labels must be unique (l,m,j) triples")
    if any(degree < 0 or abs(order) > degree or laguerre < 0 for degree, order, laguerre in labels):
        raise ValueError("component labels require l >= |m| and j >= 0")

    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    magnetic_field = jnp.asarray(B)
    frequencies = jnp.asarray(pair_frequency)
    deltas = jnp.asarray(delta_j)
    gyroaverages = jnp.asarray(gyroaverage)
    expected_delta = (n_species, n_species, len(labels), vpar.size, mu.size, magnetic_field.size)
    if deltas.shape != expected_delta:
        raise ValueError(f"delta_j has shape {deltas.shape}, expected {expected_delta}")
    if frequencies.shape != (n_species, n_species):
        raise ValueError(
            f"pair_frequency has shape {frequencies.shape}, expected {(n_species, n_species)}"
        )
    if gyroaverages.ndim != 6 or gyroaverages.shape[:1] != (n_species,):
        raise ValueError("gyroaverage must have shape (target,|m|,mu,z,kx,ky)")
    if gyroaverages.shape[2:4] != (mu.size, magnetic_field.size):
        raise ValueError("gyroaverage mu/z axes do not match the velocity and field grids")
    maximum_order = max((abs(m) for _, m, _ in labels), default=0)
    if gyroaverages.shape[1] <= maximum_order:
        raise ValueError("gyroaverage does not contain every required |m| order")

    speed = jnp.sqrt(
        vpar[:, None, None] ** 2 + 2.0 * mu[None, :, None] * magnetic_field[None, None, :]
    )
    xi = jnp.where(speed > 0.0, vpar[:, None, None] / speed, 0.0)
    masses = jnp.asarray([item.mass for item in species_tuple])
    mass_factor = (masses[:, None] / masses[None, :]) ** -1.5
    responses = []
    for component, (degree, order, _laguerre) in enumerate(labels):
        clm = (
            (2 * degree + 1) * factorial(degree - order) / (4.0 * pi * factorial(degree + order))
        ) ** 0.5
        legendre = _stella_associated_legendre(degree, order, xi)
        sign = -1.0 if order < 0 and abs(order) % 2 else 1.0
        velocity_basis = sign * clm * legendre
        bessel = gyroaverages[:, abs(order), ...]
        response = (
            frequencies[:, :, None, None, None, None, None]
            * mass_factor[:, :, None, None, None, None, None]
            * deltas[:, :, component, :, :, :, None, None]
            * velocity_basis[None, None, :, :, :, None, None]
            * bessel[:, None, None, :, :, :, :]
        )
        responses.append(response)
    return jnp.stack(responses, axis=2)


def build_stella_laguerre_legendre_driver(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    velocity_measure,
    delta_j,
    gyroaverage,
    *,
    component_labels: tuple[tuple[int, int, int], ...],
):
    """Construct stella-normalized Laguerre--Legendre driver coefficients.

    This is the non-exact-conservation branch used by the pinned collision
    discriminator. ``delta_j`` follows the response builder's six-dimensional
    pair/component contract and ``gyroaverage`` follows its six-dimensional
    species/order contract. The returned driver includes the complete velocity
    measure and ``psijnorm`` normalization.
    """

    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    labels = tuple(tuple(int(value) for value in label) for label in component_labels)
    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    magnetic_field = jnp.asarray(B)
    measure = jnp.asarray(velocity_measure)
    deltas = jnp.asarray(delta_j)
    gyroaverages = jnp.asarray(gyroaverage)
    expected_delta = (
        n_species,
        n_species,
        len(labels),
        vpar.size,
        mu.size,
        magnetic_field.size,
    )
    if deltas.shape != expected_delta:
        raise ValueError(f"delta_j has shape {deltas.shape}, expected {expected_delta}")
    if measure.shape != expected_delta[3:]:
        raise ValueError(
            f"velocity_measure has shape {measure.shape}, expected {expected_delta[3:]}"
        )
    if gyroaverages.ndim != 6 or gyroaverages.shape[0] != n_species:
        raise ValueError("gyroaverage must have shape (species,|m|,mu,z,kx,ky)")
    if gyroaverages.shape[2:4] != (mu.size, magnetic_field.size):
        raise ValueError("gyroaverage mu/z axes do not match the velocity and field grids")
    if any(degree < 0 or abs(order) > degree or laguerre < 0 for degree, order, laguerre in labels):
        raise ValueError("component labels require l >= |m| and j >= 0")

    speed = jnp.sqrt(
        vpar[:, None, None] ** 2 + 2.0 * mu[None, :, None] * magnetic_field[None, None, :]
    )
    maxwellian = jnp.exp(-(speed**2))
    masses = jnp.stack([jnp.asarray(item.mass) for item in species_tuple])
    drivers = []
    for component, (degree, order, laguerre) in enumerate(labels):
        polynomial = speed**degree * _associated_laguerre(laguerre, degree + 0.5, speed**2)
        pair_norms = []
        for target in range(n_species):
            background_norms = []
            for background in range(n_species):
                if degree == 0 and laguerre == 0:
                    norm = jnp.ones_like(magnetic_field)
                else:
                    mass_ratio = masses[target] / masses[background]
                    direct = deltas[target, background, component]
                    swapped = deltas[background, target, component]
                    if degree == 0 and laguerre == 1:
                        direct_integrand = -(speed**2) * direct * mass_ratio**-3.5
                        swapped_integrand = -(speed**2) * swapped
                    else:
                        direct_integrand = polynomial * direct * mass_ratio**-3.5
                        swapped_integrand = polynomial * swapped
                    norm = jnp.where(
                        mass_ratio < 1.0,
                        jnp.sum(measure * swapped_integrand, axis=(0, 1)),
                        jnp.sum(measure * direct_integrand, axis=(0, 1)),
                    ) / (4.0 * pi)
                background_norms.append(norm)
            pair_norms.append(jnp.stack(background_norms))
        psijnorm = jnp.stack(pair_norms)
        clm = (-1) ** order * (
            (2 * degree + 1) * factorial(degree + order) / (4.0 * pi * factorial(degree - order))
        ) ** 0.5
        legendre = _stella_associated_legendre(degree, -order, vpar[:, None, None] / speed)
        sign = -1.0 if order < 0 and abs(order) % 2 else 1.0
        component_drivers = []
        for target in range(n_species):
            background_drivers = []
            for background in range(n_species):
                coefficient = (
                    sign
                    * clm
                    * measure[:, :, :, None, None]
                    / maxwellian[:, :, :, None, None]
                    * legendre[:, :, :, None, None]
                    * deltas[background, target, component, :, :, :, None, None]
                    * gyroaverages[background, abs(order), None, :, :, :, :]
                    / psijnorm[target, background, None, None, :, None, None]
                )
                background_drivers.append(coefficient)
            component_drivers.append(jnp.stack(background_drivers))
        drivers.append(jnp.stack(component_drivers))
    return jnp.stack(drivers, axis=2)


def build_stella_laguerre_legendre_collision_precompute(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    pair_frequency,
    velocity_measure,
    gyroaverage,
    *,
    component_labels: tuple[tuple[int, int, int], ...],
) -> LaguerreLegendreCollisionPrecompute:
    """Assemble locally constructed stella-normalized collision coefficients."""

    delta = build_stella_laguerre_legendre_delta(
        velocity_grid,
        B,
        species,
        velocity_measure,
        component_labels=component_labels,
    )
    driver = build_stella_laguerre_legendre_driver(
        velocity_grid,
        B,
        species,
        velocity_measure,
        delta,
        gyroaverage,
        component_labels=component_labels,
    )
    response = build_stella_laguerre_legendre_response(
        velocity_grid,
        B,
        species,
        pair_frequency,
        delta,
        gyroaverage,
        component_labels=component_labels,
    )
    return build_laguerre_legendre_collision_precompute(
        driver,
        response,
        component_labels=component_labels,
    )


def build_laguerre_legendre_collision_precompute(
    driver,
    response,
    *,
    component_labels: tuple[tuple[int, int, int], ...],
) -> LaguerreLegendreCollisionPrecompute:
    """Validate a low-rank field-particle coefficient contract.

    Coefficients have shape
    ``(target, background, component, vpar, mu, z, kx, ky)``.  The driver
    includes the velocity quadrature and every normalization needed by its
    scalar moment.  This function establishes application mechanics only; it
    does not certify the supplied coefficients as a Landau operator.
    """

    driver = jnp.asarray(driver)
    response = jnp.asarray(response)
    if driver.ndim != 8 or response.ndim != 8 or driver.shape != response.shape:
        raise ValueError(
            "driver and response must have matching shape "
            "(target,background,component,vpar,mu,z,kx,ky)"
        )
    if driver.shape[0] != driver.shape[1] or driver.shape[0] < 1:
        raise ValueError("field-particle coefficients require a square species pair grid")
    labels = tuple(tuple(int(value) for value in label) for label in component_labels)
    if any(len(label) != 3 for label in labels):
        raise ValueError("each component label must contain (l,m,j)")
    if len(labels) != driver.shape[2] or len(set(labels)) != len(labels):
        raise ValueError("component labels must uniquely match the coefficient axis")
    driver_norm = jnp.sum(jnp.abs(driver), axis=(3, 4))
    induced_rows = jnp.sum(
        jnp.abs(response) * driver_norm[:, :, :, None, None, ...],
        axis=(1, 2),
    )
    row_sum_bound = jnp.max(induced_rows, axis=(1, 2, 3, 4, 5))
    return LaguerreLegendreCollisionPrecompute(
        driver=driver,
        response=response,
        row_sum_bound=row_sum_bound,
        n_species=driver.shape[0],
        component_labels=labels,
    )


def laguerre_legendre_collision_components(
    distribution, precompute: LaguerreLegendreCollisionPrecompute
):
    """Return directed target/background field-particle contributions."""

    values = jnp.asarray(distribution)
    if values.ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    expected = (precompute.n_species, *precompute.driver.shape[3:])
    if values.shape != expected:
        raise ValueError(f"distribution has shape {values.shape}, expected {expected}")
    moments = jnp.einsum("abcvmzxy,bvmzxy->abczxy", precompute.driver, values)
    return laguerre_legendre_collision_components_from_moments(moments, precompute)


def laguerre_legendre_collision_components_from_moments(
    moments, precompute: LaguerreLegendreCollisionPrecompute
):
    """Map pair/component scalar moments back to directed phase-space actions."""

    moments = jnp.asarray(moments)
    expected = (*precompute.response.shape[:3], *precompute.response.shape[5:])
    if moments.shape != expected:
        raise ValueError(f"moments have shape {moments.shape}, expected {expected}")
    return jnp.einsum("abcvmzxy,abczxy->abvmzxy", precompute.response, moments)


def laguerre_legendre_collision(distribution, precompute: LaguerreLegendreCollisionPrecompute):
    """Apply a supplied Laguerre--Legendre low-rank collision contract."""

    values = jnp.asarray(distribution)
    original_ndim = values.ndim
    components = laguerre_legendre_collision_components(values, precompute)
    result = jnp.sum(components, axis=1)
    return result[0] if original_ndim == 5 else result


def implicit_laguerre_legendre_collision(
    distribution,
    test_particle_matrix,
    precompute: LaguerreLegendreCollisionPrecompute,
    dt,
):
    """Apply the coupled backward-Euler test/field-particle collision solve.

    ``test_particle_matrix`` is ``I - dt*C_tp`` with shape ``(state,state)``
    or ``(z,kx,ky,state,state)``, where ``state = species*vpar*mu``. The
    low-rank response system is solved through the Woodbury identity, matching
    stella's implicit ordering without forming a dense field-particle matrix.
    """

    values = jnp.asarray(distribution)
    original_ndim = values.ndim
    if original_ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    expected = (precompute.n_species, *precompute.driver.shape[3:])
    if values.shape != expected:
        raise ValueError(f"distribution has shape {values.shape}, expected {expected}")
    n_species, n_vpar, n_mu, n_z, n_kx, n_ky = values.shape
    state_size = n_species * n_vpar * n_mu
    pair_size = n_species * n_species * precompute.driver.shape[2]
    batch_shape = (n_z, n_kx, n_ky)
    matrix = jnp.asarray(test_particle_matrix)
    if matrix.shape == (state_size, state_size):
        matrix = jnp.broadcast_to(matrix, (*batch_shape, state_size, state_size))
    elif matrix.shape != (*batch_shape, state_size, state_size):
        raise ValueError(
            "test_particle_matrix must have shape (state,state) or "
            f"(z,kx,ky,state,state), got {matrix.shape}"
        )
    state = values.transpose(3, 4, 5, 0, 1, 2).reshape(*batch_shape, state_size)
    species_identity = jnp.eye(n_species, dtype=values.dtype)
    driver_matrix = jnp.einsum(
        "abcvmzxy,sb->zxyabcsvm",
        precompute.driver,
        species_identity,
    ).reshape(*batch_shape, pair_size, state_size)
    response_matrix = jnp.einsum(
        "abcvmzxy,sa->zxysvmabc",
        precompute.response,
        species_identity,
    ).reshape(*batch_shape, state_size, pair_size)
    inhomogeneous = jnp.linalg.solve(matrix, state[..., None])[..., 0]
    response = jnp.linalg.solve(matrix, jnp.asarray(dt) * response_matrix)
    response_system = jnp.eye(pair_size, dtype=values.dtype) - jnp.matmul(driver_matrix, response)
    moments = jnp.linalg.solve(
        response_system,
        jnp.matmul(driver_matrix, inhomogeneous[..., None]),
    )
    advanced = inhomogeneous + jnp.matmul(response, moments)[..., 0]
    collision = (advanced - state) / jnp.asarray(dt)
    result = collision.reshape(*batch_shape, n_species, n_vpar, n_mu).transpose(3, 4, 5, 0, 1, 2)
    return result[0] if original_ndim == 5 else result


def build_fokker_planck_test_particle_matrix(
    precompute: FokkerPlanckPrecompute,
    dt,
    *,
    n_kx: int = 1,
    n_ky: int = 1,
):
    """Materialize ``I-dt*C_tp`` from the validated differential stencil.

    Species are coupled only through the field-particle completion, so this
    matrix contains block-diagonal target-species test-particle stencils. The
    returned layout is ``(z,kx,ky,state,state)`` for direct use by
    :func:`implicit_laguerre_legendre_collision`.
    """

    if n_kx < 1 or n_ky < 1:
        raise ValueError("n_kx and n_ky must be positive")
    stencil = jnp.asarray(precompute.stencil)
    n_species, _neighbors, n_vpar, n_mu, n_z = stencil.shape
    state_size = n_species * n_vpar * n_mu

    def collision_at_z(flat_state, z_index):
        state = flat_state.reshape(n_species, n_vpar, n_mu)
        actions = []
        for species_index in range(n_species):
            distribution = state[species_index, :, :, None, None, None]
            species_stencil = stencil[species_index, :, :, :, z_index : z_index + 1]
            action = _apply_fokker_planck_stencil(distribution, species_stencil)
            actions.append(action[:, :, 0, 0, 0])
        return jnp.stack(actions).reshape(state_size)

    zero = jnp.zeros(state_size, dtype=stencil.dtype)
    matrices = []
    for z_index in range(n_z):
        collision_matrix = jax.jacfwd(collision_at_z, argnums=0)(zero, z_index)
        matrices.append(jnp.eye(state_size, dtype=stencil.dtype) - dt * collision_matrix)
    matrix = jnp.stack(matrices)
    return jnp.broadcast_to(
        matrix[:, None, None, :, :],
        (n_z, n_kx, n_ky, state_size, state_size),
    )


def build_fokker_planck_precompute(
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    frequency,
    *,
    pitch_angle: bool = True,
    energy_scattering: bool = True,
    friction: bool = True,
    mass_conserving_boundary: bool = True,
    conserve_exchange: bool = False,
    conservation_model: str | None = None,
) -> FokkerPlanckPrecompute:
    """Build a multispecies test-particle differential collision stencil.

    Each target species sums scattering from every supplied background species.
    The discretization follows GKW's nine-point ``(v_parallel, mu)`` stencil.
    ``xu_species_local`` reproduces GKW's local defect correction,
    ``global_exchange`` applies one algebraic multispecies projection, and
    ``pairwise_exchange`` retains ordered target/background stencils and
    projects each unordered collision pair independently.
    ``reciprocal_exchange`` instead uses partner-driven momentum and energy
    responses, matching the dataflow of a reciprocal field-particle term.  It
    remains a low-rank model rather than a coefficient-level Landau operator.
    """

    if conservation_model is None:
        conservation_model = "global_exchange" if conserve_exchange else "none"
    elif conserve_exchange:
        raise ValueError("use conservation_model or conserve_exchange, not both")
    if conservation_model not in (
        "none",
        "global_exchange",
        "xu_species_local",
        "pairwise_exchange",
        "reciprocal_exchange",
    ):
        raise ValueError(
            "conservation_model must be 'none', 'global_exchange', "
            "'xu_species_local', 'pairwise_exchange', or 'reciprocal_exchange'"
        )
    if velocity_grid.backend != "finite_difference":
        raise ValueError("Fokker-Planck collisions require the finite-difference velocity grid")
    species_tuple = species if isinstance(species, tuple) else (species,)
    n_species = len(species_tuple)
    frequencies = jnp.broadcast_to(jnp.asarray(frequency), (n_species,))
    if bool(jnp.any(frequencies < 0.0)):
        raise ValueError("collision frequencies must be nonnegative")
    vpar = jnp.asarray(velocity_grid.vpar)
    mu = jnp.asarray(velocity_grid.mu)
    B = jnp.asarray(B)
    if vpar.size < 2 or mu.size < 2:
        raise ValueError("Fokker-Planck collisions require at least two velocity nodes")
    dvpar = float(vpar[1] - vpar[0])
    vperp = jnp.sqrt(2.0 * mu)
    dvperp = float(2.0 * vperp[0])

    stencils = []
    pair_stencils = []
    for target_index, target in enumerate(species_tuple):
        target_stencil = jnp.zeros((9, vpar.size, mu.size, B.size), dtype=vpar.dtype)
        target_pair_stencils = []
        target_vth = jnp.sqrt(target.temperature / target.mass)
        for background in species_tuple:
            background_vth = jnp.sqrt(background.temperature / background.mass)
            gamma_prefactor = (
                frequencies[target_index]
                * target.charge**2
                * background.charge**2
                * background.density
                / target.temperature**2
            )
            pair_stencil = _build_fokker_planck_stencil(
                vpar,
                mu,
                B,
                dvpar,
                dvperp,
                gamma_prefactor,
                target_vth,
                target_vth / background_vth,
                target.mass / background.mass,
                pitch_angle,
                energy_scattering,
                friction,
                mass_conserving_boundary,
            )
            target_pair_stencils.append(pair_stencil)
            target_stencil = target_stencil + pair_stencil
        pair_stencils.append(jnp.stack(target_pair_stencils))
        stencils.append(target_stencil)
    stencil = jnp.stack(stencils)
    pair_stencil = jnp.stack(pair_stencils)
    invariants = basis = inverse = measure = None
    xu_momentum = xu_energy = xu_vpar_weight = xu_energy_weight = None
    pair_indices = ()
    pair_invariants = pair_basis = pair_inverse = pair_measure = None
    pair_reciprocal_inverse = None
    reciprocal_response_gains = None
    conservation_gain = jnp.asarray(1.0)
    if conservation_model == "global_exchange":
        invariants, basis, inverse, measure, conservation_gain = _build_fokker_planck_conservation(
            velocity_grid,
            B,
            species_tuple,
        )
    elif conservation_model == "xu_species_local":
        (
            xu_momentum,
            xu_energy,
            xu_vpar_weight,
            xu_energy_weight,
            conservation_gain,
        ) = _build_xu_conservation(velocity_grid, B, species_tuple)
    elif conservation_model in ("pairwise_exchange", "reciprocal_exchange"):
        invariants, basis, inverse, measure, _ = _build_fokker_planck_conservation(
            velocity_grid,
            B,
            species_tuple,
        )
        pair_indices = tuple(
            (first, second) for first in range(n_species) for second in range(first, n_species)
        )
        pair_data = [
            _build_fokker_planck_conservation(
                velocity_grid,
                B,
                (species_tuple[first],)
                if first == second
                else (species_tuple[first], species_tuple[second]),
            )
            for first, second in pair_indices
        ]
        pair_invariants = tuple(item[0] for item in pair_data)
        pair_basis = tuple(item[1] for item in pair_data)
        pair_inverse = tuple(item[2] for item in pair_data)
        pair_measure = tuple(item[3] for item in pair_data)
        if conservation_model == "reciprocal_exchange":
            reciprocal_inverses = []
            reciprocal_gains = []
            for item in pair_data:
                local_species = item[0].shape[1]
                target_inverses = []
                target_gains = []
                for target in range(local_species):
                    rows = jnp.asarray((target, local_species, local_species + 1))
                    local_invariants = item[0][rows, target]
                    local_basis = item[1][rows, target]
                    matrix = jnp.einsum(
                        "cvmz,dvmz,vmz->zcd",
                        local_invariants,
                        local_basis,
                        item[3],
                    )
                    target_inverse = jnp.linalg.inv(matrix)
                    target_inverses.append(target_inverse)
                    partner = target if local_species == 1 else 1 - target
                    driver_invariants = jnp.stack(
                        (
                            item[0][target, target],
                            item[0][local_species, partner],
                            item[0][local_species + 1, partner],
                        )
                    )
                    response = jnp.einsum("dvmz,zdc->cvmz", local_basis, target_inverse)
                    driver_norm = jnp.sum(
                        jnp.abs(driver_invariants) * item[3][None, ...],
                        axis=(1, 2),
                    )
                    density_gain = jnp.max(jnp.abs(response[0]) * driver_norm[0][None, None, :])
                    exchange_gain = jnp.max(
                        jnp.sum(
                            jnp.abs(response[1:]) * driver_norm[1:, None, None, :],
                            axis=0,
                        )
                    )
                    target_gains.append(jnp.stack((density_gain, exchange_gain)))
                reciprocal_inverses.append(jnp.stack(target_inverses))
                reciprocal_gains.append(jnp.stack(target_gains))
            pair_reciprocal_inverse = tuple(reciprocal_inverses)
            reciprocal_response_gains = tuple(reciprocal_gains)

    if conservation_model == "pairwise_exchange":
        pair_row_bounds = jnp.max(jnp.sum(jnp.abs(pair_stencil), axis=2), axis=(2, 3, 4))
        pair_gains = jnp.ones((n_species, n_species), dtype=vpar.dtype)
        for (first, second), item in zip(pair_indices, pair_data, strict=True):
            pair_gains = pair_gains.at[first, second].set(item[4])
            pair_gains = pair_gains.at[second, first].set(item[4])
        row_sum_bound = jnp.sum(pair_row_bounds * pair_gains, axis=1)
    elif conservation_model == "reciprocal_exchange":
        pair_row_bounds = jnp.max(jnp.sum(jnp.abs(pair_stencil), axis=2), axis=(2, 3, 4))
        row_sum_bound = jnp.zeros((n_species,), dtype=vpar.dtype)
        for (first, second), gains in zip(pair_indices, reciprocal_response_gains, strict=True):
            if first == second:
                row_sum_bound = row_sum_bound.at[first].add(
                    pair_row_bounds[first, first] * (1.0 + jnp.sum(gains[0]))
                )
            else:
                row_sum_bound = row_sum_bound.at[first].add(
                    pair_row_bounds[first, second] * (1.0 + gains[0, 0])
                    + pair_row_bounds[second, first] * gains[0, 1]
                )
                row_sum_bound = row_sum_bound.at[second].add(
                    pair_row_bounds[second, first] * (1.0 + gains[1, 0])
                    + pair_row_bounds[first, second] * gains[1, 1]
                )
    else:
        row_sum_bound = (
            jnp.max(jnp.sum(jnp.abs(stencil), axis=1), axis=(1, 2, 3)) * conservation_gain
        )
    return FokkerPlanckPrecompute(
        stencil=stencil,
        row_sum_bound=row_sum_bound,
        conservation_invariants=invariants,
        conservation_basis=basis,
        conservation_inverse=inverse,
        measure=measure,
        xu_momentum_factor=xu_momentum,
        xu_energy_factor=xu_energy,
        xu_vpar_weight=xu_vpar_weight,
        xu_energy_weight=xu_energy_weight,
        pair_stencil=pair_stencil,
        pair_conservation_invariants=pair_invariants,
        pair_conservation_basis=pair_basis,
        pair_conservation_inverse=pair_inverse,
        pair_reciprocal_inverse=pair_reciprocal_inverse,
        pair_conservation_measure=pair_measure,
        n_species=n_species,
        conserve_exchange=conservation_model
        in ("global_exchange", "pairwise_exchange", "reciprocal_exchange"),
        conservation_model=conservation_model,
        pair_indices=pair_indices,
    )


def fokker_planck_collision(distribution, precompute: FokkerPlanckPrecompute):
    """Apply the nine-point test-particle stencil to a distribution."""

    values = jnp.asarray(distribution)
    original_ndim = values.ndim
    if original_ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != precompute.n_species:
        raise ValueError("distribution has incompatible species or phase-space shape")
    if precompute.conservation_model == "pairwise_exchange":
        result = jnp.sum(fokker_planck_pairwise_components(values, precompute), axis=1)
    elif precompute.conservation_model == "reciprocal_exchange":
        result = jnp.sum(fokker_planck_reciprocal_components(values, precompute), axis=1)
    else:
        result = jax.vmap(_apply_fokker_planck_stencil)(values, precompute.stencil)
    if precompute.conservation_model == "global_exchange":
        result = _project_collision_constraints(
            result,
            precompute.conservation_invariants,
            precompute.conservation_basis,
            precompute.conservation_inverse,
            precompute.measure,
        )
    elif precompute.conservation_model == "xu_species_local":
        momentum_defect = jnp.einsum("svmz,svmzxy->szxy", precompute.xu_vpar_weight, result)
        energy_defect = jnp.einsum("svmz,svmzxy->szxy", precompute.xu_energy_weight, result)
        result = result - (
            momentum_defect[:, None, None, :, :, :] * precompute.xu_momentum_factor[..., None, None]
            + energy_defect[:, None, None, :, :, :] * precompute.xu_energy_factor[..., None, None]
        )
    return result[0] if original_ndim == 5 else result


def fokker_planck_pairwise_components(distribution, precompute: FokkerPlanckPrecompute):
    """Return directed contributions from every conservative collision pair.

    Entry ``[a, b]`` is the contribution to target species ``a`` from its
    interaction with background species ``b``.  Off-diagonal entries are
    coupled by a shared density/momentum/energy projection.
    """

    if precompute.conservation_model != "pairwise_exchange":
        raise ValueError("pairwise components require conservation_model='pairwise_exchange'")
    values = jnp.asarray(distribution)
    if values.ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != precompute.n_species:
        raise ValueError("distribution has incompatible species or phase-space shape")
    components = jnp.zeros(
        (precompute.n_species, precompute.n_species, *values.shape[1:]),
        dtype=values.dtype,
    )
    pair_data = zip(
        precompute.pair_indices,
        precompute.pair_conservation_invariants,
        precompute.pair_conservation_basis,
        precompute.pair_conservation_inverse,
        precompute.pair_conservation_measure,
        strict=True,
    )
    for (first, second), invariants, basis, inverse, measure in pair_data:
        if first == second:
            raw = _apply_fokker_planck_stencil(
                values[first], precompute.pair_stencil[first, second]
            )[None, ...]
        else:
            raw = jnp.stack(
                (
                    _apply_fokker_planck_stencil(
                        values[first], precompute.pair_stencil[first, second]
                    ),
                    _apply_fokker_planck_stencil(
                        values[second], precompute.pair_stencil[second, first]
                    ),
                )
            )
        corrected = _project_collision_constraints(raw, invariants, basis, inverse, measure)
        components = components.at[first, second].set(corrected[0])
        if first != second:
            components = components.at[second, first].set(corrected[1])
    return components


def fokker_planck_reciprocal_components(distribution, precompute: FokkerPlanckPrecompute):
    """Return pair actions whose field terms are driven by the collision partner.

    For an off-diagonal pair, each target keeps its own test-particle momentum
    and energy defect while receiving the equal-and-opposite low-rank response
    driven by the other species.  This mirrors stella's reciprocal dataflow but
    does not yet claim parity with its Laguerre--Legendre coefficients.
    """

    if precompute.conservation_model != "reciprocal_exchange":
        raise ValueError("reciprocal components require conservation_model='reciprocal_exchange'")
    values = jnp.asarray(distribution)
    if values.ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != precompute.n_species:
        raise ValueError("distribution has incompatible species or phase-space shape")
    components = jnp.zeros(
        (precompute.n_species, precompute.n_species, *values.shape[1:]),
        dtype=values.dtype,
    )
    pair_data = zip(
        precompute.pair_indices,
        precompute.pair_conservation_invariants,
        precompute.pair_conservation_basis,
        precompute.pair_conservation_measure,
        precompute.pair_reciprocal_inverse,
        strict=True,
    )
    for (first, second), invariants, basis, measure, inverses in pair_data:
        indices = (first,) if first == second else (first, second)
        raw = jnp.stack(
            tuple(
                _apply_fokker_planck_stencil(
                    values[target], precompute.pair_stencil[target, background]
                )
                for target, background in (
                    ((first, first),) if first == second else ((first, second), (second, first))
                )
            )
        )
        local_species = len(indices)
        local_moments = []
        local_bases = []
        for local_target in range(local_species):
            rows = jnp.asarray((local_target, local_species, local_species + 1))
            local_invariants = invariants[rows, local_target]
            local_basis = basis[rows, local_target]
            local_moments.append(
                jnp.einsum(
                    "cvmz,vmz,vmzxy->czxy",
                    local_invariants,
                    measure,
                    raw[local_target],
                )
            )
            local_bases.append(local_basis)
        for local_target, target in enumerate(indices):
            partner = local_target if local_species == 1 else 1 - local_target
            desired = jnp.stack(
                (
                    -local_moments[local_target][0],
                    -local_moments[partner][1],
                    -local_moments[partner][2],
                )
            )
            coefficients = jnp.einsum("zdc,czxy->zdxy", inverses[local_target], desired)
            correction = jnp.einsum("dvmz,zdxy->vmzxy", local_bases[local_target], coefficients)
            background = second if target == first else first
            components = components.at[target, background].set(raw[local_target] + correction)
    return components


def fokker_planck_conserved_moments(values, precompute: FokkerPlanckPrecompute):
    """Return per-species density plus total momentum/energy constraints."""

    if not precompute.conserve_exchange:
        raise ValueError("exchange conservation was not enabled in the precompute")
    values = jnp.asarray(values)
    if values.ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != precompute.n_species:
        raise ValueError("values have incompatible species or phase-space shape")
    return jnp.einsum(
        "csvmz,vmz,svmzxy->czxy",
        precompute.conservation_invariants,
        precompute.measure,
        values,
    )


def _project_collision_constraints(values, invariants, basis, inverse, measure):
    moments = jnp.einsum("csvmz,vmz,svmzxy->czxy", invariants, measure, values)
    coefficients = jnp.einsum("zdc,czxy->zdxy", inverse, moments)
    correction = jnp.einsum("dsvmz,zdxy->svmzxy", basis, coefficients)
    return values - correction


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
    vpar = jnp.broadcast_to(jnp.asarray(velocity_grid.vpar)[None, :, None, None], energy.shape)
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
    projection = jnp.einsum("sbvmz,szxyb->svmzxy", precompute.equilibrium_basis, coefficients)
    result = -precompute.frequency[:, None, None, None, None, None] * (distribution - projection)
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


def _build_fokker_planck_conservation(velocity_grid, B, species):
    n_species = len(species)
    maxwellians = jnp.asarray(maxwellian(velocity_grid.vpar, velocity_grid.mu, B, species))
    energy = jnp.asarray(normalized_energy(velocity_grid.vpar, velocity_grid.mu, B, species))
    vpar = jnp.asarray(velocity_grid.vpar)[None, :, None, None]
    momentum_scale = jnp.sqrt(jnp.asarray([item.mass * item.temperature for item in species]))[
        :, None, None, None
    ]
    physical_momentum = momentum_scale * vpar * jnp.ones_like(energy)
    physical_energy = (
        jnp.asarray([item.temperature for item in species])[:, None, None, None] * energy
    )
    measure = (
        jnp.asarray(velocity_grid.w_vpar)[:, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, :, None]
        * B[None, None, :]
    )
    density_invariants = (
        jnp.eye(n_species)[:, :, None, None, None] * jnp.ones_like(energy)[None, ...]
    )
    invariants = jnp.concatenate(
        (
            density_invariants,
            physical_momentum[None, ...],
            physical_energy[None, ...],
        ),
        axis=0,
    )
    density_basis = density_invariants * maxwellians[None, ...]
    density_norm = jnp.einsum("svmz,vmz->sz", maxwellians, measure)
    energy_mean = jnp.einsum(
        "svmz,svmz,vmz->sz", physical_energy, maxwellians, measure
    ) / jnp.maximum(density_norm, 1.0e-14)
    centered_energy = physical_energy - energy_mean[:, None, None, :]
    basis = jnp.concatenate(
        (
            density_basis,
            (physical_momentum * maxwellians)[None, ...],
            (centered_energy * maxwellians)[None, ...],
        ),
        axis=0,
    )
    moment_matrix = jnp.einsum("csvmz,dsvmz,vmz->zcd", invariants, basis, measure)
    inverse = jnp.linalg.inv(moment_matrix)
    basis_norm = jnp.max(jnp.sum(jnp.abs(basis), axis=0))
    inverse_norm = jnp.max(jnp.sum(jnp.abs(inverse), axis=2))
    invariant_norm = jnp.max(
        jnp.sum(jnp.abs(invariants) * measure[None, None, ...], axis=(1, 2, 3))
    )
    gain = 1.0 + basis_norm * inverse_norm * invariant_norm
    return invariants, basis, inverse, measure, gain


def _build_xu_conservation(velocity_grid, B, species):
    """Build the species-local Xu correction used by GKW/Gyaradax.

    This correction removes the parallel-momentum and energy defects of the
    test-particle stencil independently for each species.  It is therefore a
    useful reference-parity model, but it does not represent reciprocal
    inter-species field-particle exchange.
    """

    vpar = jnp.asarray(velocity_grid.vpar)[:, None, None]
    mu = jnp.asarray(velocity_grid.mu)[None, :, None]
    B = jnp.asarray(B)[None, None, :]
    speed_squared = vpar**2 + 2.0 * mu * B
    measure = (
        jnp.asarray(velocity_grid.w_vpar)[:, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, :, None]
        * B
    )
    vpar_weight = vpar * measure
    energy_weight = speed_squared * measure

    momentum_factors = []
    energy_factors = []
    for item in species:
        temperature = jnp.asarray(item.temperature)
        density = jnp.asarray(item.density)
        maxwellian_envelope = jnp.exp(-speed_squared / temperature) / (
            jnp.sqrt(temperature * jnp.pi) ** 3
        )
        equilibrium = density * maxwellian_envelope
        particle_norm = jnp.sum(equilibrium * measure, axis=(0, 1))
        energy_norm = jnp.sum(speed_squared * equilibrium * measure, axis=(0, 1))
        energy_mean = energy_norm / jnp.maximum(particle_norm, 1.0e-14)
        momentum_norm = jnp.sum(vpar**2 * equilibrium * measure, axis=(0, 1))
        centered_energy = speed_squared - energy_mean[None, None, :]
        centered_energy_norm = jnp.sum(
            speed_squared * centered_energy * equilibrium * measure,
            axis=(0, 1),
        )
        momentum_factors.append(
            vpar * equilibrium / jnp.maximum(momentum_norm, 1.0e-14)[None, None, :]
        )
        energy_factors.append(
            centered_energy
            * equilibrium
            / jnp.maximum(centered_energy_norm, 1.0e-14)[None, None, :]
        )

    momentum_factor = jnp.stack(momentum_factors)
    energy_factor = jnp.stack(energy_factors)
    weight_shape = momentum_factor.shape
    vpar_weights = jnp.broadcast_to(vpar_weight[None, ...], weight_shape)
    energy_weights = jnp.broadcast_to(energy_weight[None, ...], weight_shape)
    momentum_gain = jnp.max(jnp.abs(momentum_factor), axis=(1, 2)) * jnp.sum(
        jnp.abs(vpar_weights), axis=(1, 2)
    )
    energy_gain = jnp.max(jnp.abs(energy_factor), axis=(1, 2)) * jnp.sum(
        jnp.abs(energy_weights), axis=(1, 2)
    )
    gain = jnp.max(1.0 + momentum_gain + energy_gain)
    return momentum_factor, energy_factor, vpar_weights, energy_weights, gain


def _erf_derivative(value):
    return 2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(value**2))


def _pitch_diffusion(speed, prefactor, background_scale):
    safe_speed = jnp.maximum(speed, 1.0e-14)
    background_speed = jnp.maximum(safe_speed * background_scale, 1.0e-14)
    numerator = (2.0 - 1.0 / background_speed**2) * erf(background_speed) + _erf_derivative(
        background_speed
    ) / background_speed
    return prefactor * numerator / (4.0 * safe_speed)


def _energy_diffusion(speed, prefactor, background_scale):
    safe_speed = jnp.maximum(speed, 1.0e-14)
    background_speed = jnp.maximum(safe_speed * background_scale, 1.0e-14)
    numerator = (
        erf(background_speed) / background_speed**2
        - _erf_derivative(background_speed) / background_speed
    )
    return prefactor * numerator / (2.0 * safe_speed)


def _friction(speed, prefactor, background_scale, mass_ratio):
    safe_speed = jnp.maximum(speed, 1.0e-14)
    background_speed = jnp.maximum(safe_speed * background_scale, 1.0e-14)
    numerator = erf(background_speed) - _erf_derivative(background_speed) * background_speed
    return prefactor * mass_ratio * numerator / safe_speed**2


def _build_fokker_planck_stencil(
    vpar,
    mu,
    B,
    dvpar,
    dvperp,
    prefactor,
    target_vth,
    background_scale,
    mass_ratio,
    pitch_angle,
    energy_scattering,
    friction,
    mass_conserving_boundary,
):
    nv, nmu, nz = vpar.size, mu.size, B.size
    vp = vpar.reshape(nv, 1, 1)
    vperp = jnp.sqrt(jnp.maximum(2.0 * mu, 0.0)).reshape(1, nmu, 1)
    sqrt_B = jnp.sqrt(jnp.maximum(B.reshape(1, 1, nz), 1.0e-14))
    dvrp = sqrt_B * dvperp
    vperp_physical = vperp * sqrt_B

    def diffusion_terms(parallel, perpendicular):
        speed_squared = jnp.maximum(parallel**2 + perpendicular**2, 1.0e-14)
        speed = jnp.sqrt(speed_squared)
        pitch = jnp.where(
            pitch_angle,
            _pitch_diffusion(speed, prefactor, background_scale),
            0.0,
        )
        energy = jnp.where(
            energy_scattering,
            _energy_diffusion(speed, prefactor, background_scale),
            0.0,
        )
        drag = jnp.where(
            friction,
            _friction(speed, prefactor, background_scale, mass_ratio),
            0.0,
        )
        return speed_squared, pitch, energy, drag

    par_plus = vp + 0.5 * dvpar
    sq_a, pitch_a, energy_a, drag_a = diffusion_terms(par_plus, vperp_physical)
    a_pitch = vperp_physical**2 * pitch_a / (sq_a * dvpar**2)
    a_energy = par_plus**2 * energy_a / (sq_a * dvpar**2)
    a_drag = par_plus * drag_a / (jnp.sqrt(sq_a) * dvpar)
    top = (jnp.arange(nv).reshape(nv, 1, 1) < nv - 1) | (not mass_conserving_boundary)
    a_pitch, a_energy, a_drag = (
        jnp.where(top, value, 0.0) for value in (a_pitch, a_energy, a_drag)
    )

    par_minus = vp - 0.5 * dvpar
    sq_b, pitch_b, energy_b, drag_b = diffusion_terms(par_minus, vperp_physical)
    b_pitch = vperp_physical**2 * pitch_b / (sq_b * dvpar**2)
    b_energy = par_minus**2 * energy_b / (sq_b * dvpar**2)
    b_drag = par_minus * drag_b / (jnp.sqrt(sq_b) * dvpar)
    bottom = (jnp.arange(nv).reshape(nv, 1, 1) > 0) | (not mass_conserving_boundary)
    b_pitch, b_energy, b_drag = (
        jnp.where(bottom, value, 0.0) for value in (b_pitch, b_energy, b_drag)
    )

    safe_vperp = jnp.maximum(vperp_physical, 1.0e-14)
    perp_plus = vperp_physical + 0.5 * dvrp
    sq_c, pitch_c, energy_c, drag_c = diffusion_terms(vp, perp_plus)
    c_pitch = perp_plus * vp**2 * pitch_c / (sq_c * safe_vperp * dvrp**2)
    c_energy = perp_plus**3 * energy_c / (sq_c * safe_vperp * dvrp**2)
    c_drag = perp_plus**2 * drag_c / (jnp.sqrt(sq_c) * safe_vperp * dvrp)
    mu_top = (jnp.arange(nmu).reshape(1, nmu, 1) < nmu - 1) | (not mass_conserving_boundary)
    c_pitch, c_energy, c_drag = (
        jnp.where(mu_top, value, 0.0) for value in (c_pitch, c_energy, c_drag)
    )

    perp_minus = vperp_physical - 0.5 * dvrp
    sq_d, pitch_d, energy_d, drag_d = diffusion_terms(vp, perp_minus)
    d_pitch = perp_minus * vp**2 * pitch_d / (sq_d * safe_vperp * dvrp**2)
    d_energy = perp_minus**3 * energy_d / (sq_d * safe_vperp * dvrp**2)
    d_drag = perp_minus**2 * drag_d / (jnp.sqrt(sq_d) * safe_vperp * dvrp)

    cross_e = (-(perp_plus**2) * vp * pitch_c + perp_plus**2 * vp * energy_c) / (safe_vperp * sq_c)
    cross_e = jnp.where(mu_top, cross_e, 0.0)
    mu_bottom = (jnp.arange(nmu).reshape(1, nmu, 1) > 0) | (not mass_conserving_boundary)
    cross_f = (-(perp_minus**2) * vp * pitch_d + perp_minus**2 * vp * energy_d) / (
        safe_vperp * sq_d
    )
    cross_f = jnp.where(mu_bottom, cross_f, 0.0)
    cross_g = vperp_physical * par_plus * (-pitch_a + energy_a) / sq_a
    cross_g = jnp.where(top, cross_g, 0.0)
    cross_h = vperp_physical * par_minus * (-pitch_b + energy_b) / sq_b
    cross_h = jnp.where(bottom, cross_h, 0.0)

    sum_a, sum_b = a_pitch + a_energy, b_pitch + b_energy
    sum_c, sum_d = c_pitch + c_energy, d_pitch + d_energy
    cross_factor = 1.0 / (4.0 * dvpar * dvrp)
    e, f = cross_e * cross_factor, cross_f * cross_factor
    g, h = cross_g * cross_factor, cross_h * cross_factor
    delta = 0.5
    return target_vth * jnp.stack(
        (
            -(sum_a + sum_b + sum_c + sum_d) + delta * (a_drag - b_drag + c_drag - d_drag),
            sum_a + (1.0 - delta) * a_drag + e - f,
            sum_b - (1.0 - delta) * b_drag - e + f,
            sum_c + (1.0 - delta) * c_drag + g - h,
            sum_d - (1.0 - delta) * d_drag - g + h,
            e + g,
            -e - h,
            -f - g,
            f + h,
        )
    )


def _apply_fokker_planck_stencil(distribution, stencil):
    nv, nmu = distribution.shape[:2]
    iv, imu = jnp.arange(nv), jnp.arange(nmu)
    shifts = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
    result = jnp.zeros_like(distribution)
    for index, (v_shift, mu_shift) in enumerate(shifts):
        v_index = jnp.clip(iv + v_shift, 0, nv - 1)
        mu_index = jnp.clip(imu + mu_shift, 0, nmu - 1)
        valid = ((iv + v_shift >= 0) & (iv + v_shift < nv))[:, None] & (
            (imu + mu_shift >= 0) & (imu + mu_shift < nmu)
        )[None, :]
        shifted = jnp.take(distribution, v_index, axis=0)
        shifted = jnp.take(shifted, mu_index, axis=1)
        shifted = jnp.where(valid[:, :, None, None, None], shifted, 0.0)
        result = result + stencil[index, :, :, :, None, None] * shifted
    return result


__all__ = [
    "ConservingBGKPrecompute",
    "FokkerPlanckPrecompute",
    "LaguerreLegendreCollisionPrecompute",
    "StellaTestParticlePrimitives",
    "assemble_stella_test_particle_blocks",
    "build_stella_two_mu_diffusion_blocks",
    "build_stella_two_mu_vpar_mixed_blocks",
    "build_stella_vpar_diffusion_blocks",
    "build_conserving_bgk_precompute",
    "build_fokker_planck_precompute",
    "build_fokker_planck_test_particle_matrix",
    "build_laguerre_legendre_collision_precompute",
    "build_stella_laguerre_legendre_response",
    "build_stella_laguerre_legendre_delta",
    "build_stella_laguerre_legendre_driver",
    "build_stella_laguerre_legendre_collision_precompute",
    "build_stella_test_particle_primitives",
    "build_stella_test_particle_gyro_diagonal",
    "collision_moments",
    "conserving_bgk_collision",
    "fokker_planck_collision",
    "fokker_planck_conserved_moments",
    "fokker_planck_pairwise_components",
    "fokker_planck_reciprocal_components",
    "laguerre_legendre_collision",
    "laguerre_legendre_collision_components",
    "laguerre_legendre_collision_components_from_moments",
    "implicit_laguerre_legendre_collision",
    "stella_laguerre_legendre_delta0",
]
