"""Conservative model collision operators for velocity collocation grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

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
    n_species: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("stencil", "row_sum_bound")
    _static_fields: ClassVar[tuple[str, ...]] = ("n_species",)


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
) -> FokkerPlanckPrecompute:
    """Build a multispecies test-particle differential collision stencil.

    Each target species sums scattering from every supplied background species.
    The discretization follows GKW's nine-point ``(v_parallel, mu)`` stencil.
    This foundation does not include the reciprocal field-particle term needed
    to claim exact inter-species momentum and energy exchange.
    """

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
    for target_index, target in enumerate(species_tuple):
        target_stencil = jnp.zeros((9, vpar.size, mu.size, B.size), dtype=vpar.dtype)
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
            target_stencil = target_stencil + _build_fokker_planck_stencil(
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
        stencils.append(target_stencil)
    stencil = jnp.stack(stencils)
    return FokkerPlanckPrecompute(
        stencil=stencil,
        row_sum_bound=jnp.max(jnp.sum(jnp.abs(stencil), axis=1), axis=(1, 2, 3)),
        n_species=n_species,
    )


def fokker_planck_collision(distribution, precompute: FokkerPlanckPrecompute):
    """Apply the nine-point test-particle stencil to a distribution."""

    values = jnp.asarray(distribution)
    original_ndim = values.ndim
    if original_ndim == 5 and precompute.n_species == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != precompute.n_species:
        raise ValueError("distribution has incompatible species or phase-space shape")
    result = jax.vmap(_apply_fokker_planck_stencil)(values, precompute.stencil)
    return result[0] if original_ndim == 5 else result


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


def _erf_derivative(value):
    return 2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(value**2))


def _pitch_diffusion(speed, prefactor, background_scale):
    safe_speed = jnp.maximum(speed, 1.0e-14)
    background_speed = jnp.maximum(safe_speed * background_scale, 1.0e-14)
    numerator = (
        (2.0 - 1.0 / background_speed**2) * erf(background_speed)
        + _erf_derivative(background_speed) / background_speed
    )
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
    mu_top = (jnp.arange(nmu).reshape(1, nmu, 1) < nmu - 1) | (
        not mass_conserving_boundary
    )
    c_pitch, c_energy, c_drag = (
        jnp.where(mu_top, value, 0.0) for value in (c_pitch, c_energy, c_drag)
    )

    perp_minus = vperp_physical - 0.5 * dvrp
    sq_d, pitch_d, energy_d, drag_d = diffusion_terms(vp, perp_minus)
    d_pitch = perp_minus * vp**2 * pitch_d / (sq_d * safe_vperp * dvrp**2)
    d_energy = perp_minus**3 * energy_d / (sq_d * safe_vperp * dvrp**2)
    d_drag = perp_minus**2 * drag_d / (jnp.sqrt(sq_d) * safe_vperp * dvrp)

    cross_e = (-perp_plus**2 * vp * pitch_c + perp_plus**2 * vp * energy_c) / (
        safe_vperp * sq_c
    )
    cross_e = jnp.where(mu_top, cross_e, 0.0)
    mu_bottom = (jnp.arange(nmu).reshape(1, nmu, 1) > 0) | (
        not mass_conserving_boundary
    )
    cross_f = (-perp_minus**2 * vp * pitch_d + perp_minus**2 * vp * energy_d) / (
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
            -(sum_a + sum_b + sum_c + sum_d)
            + delta * (a_drag - b_drag + c_drag - d_drag),
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
        valid = (
            ((iv + v_shift >= 0) & (iv + v_shift < nv))[:, None]
            & ((imu + mu_shift >= 0) & (imu + mu_shift < nmu))[None, :]
        )
        shifted = jnp.take(distribution, v_index, axis=0)
        shifted = jnp.take(shifted, mu_index, axis=1)
        shifted = jnp.where(valid[:, :, None, None, None], shifted, 0.0)
        result = result + stencil[index, :, :, :, None, None] * shifted
    return result


__all__ = [
    "ConservingBGKPrecompute",
    "FokkerPlanckPrecompute",
    "build_conserving_bgk_precompute",
    "build_fokker_planck_precompute",
    "collision_moments",
    "conserving_bgk_collision",
    "fokker_planck_collision",
]
