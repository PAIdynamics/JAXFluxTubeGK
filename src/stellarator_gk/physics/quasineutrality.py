"""Electrostatic quasineutrality solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from ..types import FourierGrid, SpeciesParams, VelocityGrid, _PyTreeDataclass
from .primitives import FLRFactors


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdiabaticElectronParams(_PyTreeDataclass):
    """Adiabatic-electron response parameters for the electrostatic phi solve."""

    density: float
    temperature: float
    zonal_correction: bool = True
    denominator_floor: float = 1.0e-14

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("density", "temperature", "denominator_floor")
    _static_fields: ClassVar[tuple[str, ...]] = ("zonal_correction",)

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("adiabatic electron density must be positive")
        if self.temperature <= 0:
            raise ValueError("adiabatic electron temperature must be positive")
        if self.denominator_floor <= 0:
            raise ValueError("denominator_floor must be positive")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdiabaticQuasineutralityPrecompute(_PyTreeDataclass):
    """Precomputed arrays for the adiabatic-electron electrostatic solve."""

    phi_weight: object
    denominator: object
    ion_polarization: object
    electron_response: object
    w_z: object
    ixzero: int
    iyzero: int
    has_zonal: bool
    zonal_correction: bool
    denominator_floor: float
    n_species: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "phi_weight",
        "denominator",
        "ion_polarization",
        "electron_response",
        "w_z",
        "denominator_floor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "ixzero",
        "iyzero",
        "has_zonal",
        "zonal_correction",
        "n_species",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KineticQuasineutralityPrecompute(_PyTreeDataclass):
    """Precomputed arrays for a fully kinetic electrostatic solve."""

    phi_weight: object
    denominator: object
    denominator_floor: float
    ixzero: int
    iyzero: int
    regularize_constant_mode: bool
    n_species: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "phi_weight",
        "denominator",
        "denominator_floor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "ixzero",
        "iyzero",
        "regularize_constant_mode",
        "n_species",
    )


def default_adiabatic_electron_params(
    ion_species: SpeciesParams | tuple[SpeciesParams, ...],
    *,
    temperature: float = 1.0,
    zonal_correction: bool = True,
    denominator_floor: float = 1.0e-14,
) -> AdiabaticElectronParams:
    """Choose electron density from background charge neutrality."""

    species_tuple = _as_species_tuple(ion_species)
    density = sum(species.charge * species.density for species in species_tuple)
    if density <= 0:
        raise ValueError("background ion charge density must be positive for adiabatic electrons")
    return AdiabaticElectronParams(
        density=float(density),
        temperature=temperature,
        zonal_correction=zonal_correction,
        denominator_floor=denominator_floor,
    )


def _velocity_measure(velocity_grid: VelocityGrid, B, phase_space_measure):
    """Return the full ``(species,vpar,mu,z,kx,ky)`` integration measure."""

    B = jnp.asarray(B)
    if phase_space_measure is None:
        w_v = jnp.asarray(velocity_grid.w_vpar)[None, :, None, None, None, None]
        w_mu = jnp.asarray(velocity_grid.w_mu)[None, None, :, None, None, None]
        return w_v * w_mu * B[None, None, None, :, None, None]
    measure = jnp.asarray(phase_space_measure)
    expected = (B.shape[0], velocity_grid.vpar.shape[0], velocity_grid.mu.shape[0])
    if measure.shape != expected:
        raise ValueError(
            f"phase_space_measure has shape {measure.shape}; expected {expected} in (z,vpar,mu) order"
        )
    return jnp.transpose(measure, (1, 2, 0))[None, :, :, :, None, None]


def build_adiabatic_quasineutrality_precompute(
    velocity_grid: VelocityGrid,
    B,
    flr_factors: FLRFactors,
    ion_species: SpeciesParams | tuple[SpeciesParams, ...],
    electron_params: AdiabaticElectronParams | None = None,
    *,
    fourier_grid: FourierGrid | None = None,
    w_z=None,
    ixzero: int | None = None,
    iyzero: int | None = None,
    phase_space_measure=None,
) -> AdiabaticQuasineutralityPrecompute:
    """Precompute velocity weights and field denominator for adiabatic electrons."""

    species_tuple = _as_species_tuple(ion_species)
    n_species = len(species_tuple)
    electrons = electron_params or default_adiabatic_electron_params(species_tuple)
    B = jnp.asarray(B)
    w_z_array = _normalize_parallel_weights(B, w_z)
    bessel = _with_species_axis(flr_factors.bessel_j0, n_species, "bessel_j0", single_ndim=4)
    gamma = _with_species_axis(flr_factors.gamma0, n_species, "gamma0", single_ndim=3)

    measure = _velocity_measure(velocity_grid, B, phase_space_measure)
    density_charge = jnp.asarray([s.charge * s.density for s in species_tuple])
    density_charge = density_charge.reshape(n_species, 1, 1, 1, 1, 1)
    phi_weight = density_charge * measure * bessel[:, None, :, :, :, :]

    polarization_coeff = jnp.asarray(
        [s.charge**2 * s.density / s.temperature for s in species_tuple]
    )
    ion_polarization = jnp.sum(
        polarization_coeff.reshape(n_species, 1, 1, 1) * (gamma - 1.0),
        axis=0,
    )
    electron_response = jnp.asarray(electrons.density / electrons.temperature)
    denominator = ion_polarization - electron_response

    if fourier_grid is not None:
        ix = fourier_grid.ixzero if ixzero is None else ixzero
        iy = fourier_grid.iyzero if iyzero is None else iyzero
    else:
        ix = -1 if ixzero is None else ixzero
        iy = -1 if iyzero is None else iyzero

    return AdiabaticQuasineutralityPrecompute(
        phi_weight=phi_weight,
        denominator=denominator,
        ion_polarization=ion_polarization,
        electron_response=electron_response,
        w_z=w_z_array,
        ixzero=int(ix),
        iyzero=int(iy),
        has_zonal=bool(iy is not None and iy >= 0),
        zonal_correction=bool(electrons.zonal_correction),
        denominator_floor=float(electrons.denominator_floor),
        n_species=n_species,
    )


def build_kinetic_quasineutrality_precompute(
    velocity_grid: VelocityGrid,
    B,
    flr_factors: FLRFactors,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    *,
    fourier_grid: FourierGrid | None = None,
    ixzero: int | None = None,
    iyzero: int | None = None,
    denominator_floor: float = 1.0e-14,
    regularize_constant_mode: bool = True,
    phase_space_measure=None,
) -> KineticQuasineutralityPrecompute:
    """Precompute weights and denominator for a fully kinetic electrostatic solve."""

    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    species_tuple = _as_species_tuple(species)
    n_species = len(species_tuple)
    B = jnp.asarray(B)
    bessel = _with_species_axis(flr_factors.bessel_j0, n_species, "bessel_j0", single_ndim=4)
    gamma = _with_species_axis(flr_factors.gamma0, n_species, "gamma0", single_ndim=3)

    measure = _velocity_measure(velocity_grid, B, phase_space_measure)
    density_charge = jnp.asarray([s.charge * s.density for s in species_tuple])
    density_charge = density_charge.reshape(n_species, 1, 1, 1, 1, 1)
    phi_weight = density_charge * measure * bessel[:, None, :, :, :, :]

    polarization_coeff = jnp.asarray(
        [s.charge**2 * s.density / s.temperature for s in species_tuple]
    )
    denominator = jnp.sum(
        polarization_coeff.reshape(n_species, 1, 1, 1) * (gamma - 1.0),
        axis=0,
    )

    if fourier_grid is not None:
        ix = fourier_grid.ixzero if ixzero is None else ixzero
        iy = fourier_grid.iyzero if iyzero is None else iyzero
    else:
        ix = -1 if ixzero is None else ixzero
        iy = -1 if iyzero is None else iyzero

    return KineticQuasineutralityPrecompute(
        phi_weight=phi_weight,
        denominator=denominator,
        denominator_floor=float(denominator_floor),
        ixzero=int(ix),
        iyzero=int(iy),
        regularize_constant_mode=bool(regularize_constant_mode),
        n_species=n_species,
    )


def adiabatic_density_numerator(distribution, precompute: AdiabaticQuasineutralityPrecompute):
    """Return the ion gyrocenter density numerator in quasineutrality."""

    distribution = _distribution_with_species_axis(distribution, precompute.n_species)
    return jnp.sum(precompute.phi_weight * distribution, axis=(0, 1, 2))


def kinetic_density_numerator(distribution, precompute: KineticQuasineutralityPrecompute):
    """Return the fully kinetic gyrocenter density numerator."""

    distribution = _distribution_with_species_axis(distribution, precompute.n_species)
    return jnp.sum(precompute.phi_weight * distribution, axis=(0, 1, 2))


def solve_adiabatic_electron_phi(distribution, precompute: AdiabaticQuasineutralityPrecompute):
    """Solve adiabatic-electron electrostatic quasineutrality for ``phi(z,kx,ky)``."""

    numerator = adiabatic_density_numerator(distribution, precompute)
    return solve_adiabatic_electron_phi_from_density(numerator, precompute)


def solve_adiabatic_electron_phi_from_density(
    numerator,
    precompute: AdiabaticQuasineutralityPrecompute,
):
    """Solve the field equation using a precomputed density numerator."""

    numerator = jnp.asarray(numerator)
    denominator = _safe_denominator(precompute.denominator, precompute.denominator_floor)
    phi = -numerator / denominator
    if precompute.zonal_correction and precompute.has_zonal:
        phi = _apply_zonal_adiabatic_correction(numerator, denominator, phi, precompute)
    return phi


def solve_kinetic_electron_phi(distribution, precompute: KineticQuasineutralityPrecompute):
    """Solve the fully kinetic electrostatic quasineutrality equation."""

    numerator = kinetic_density_numerator(distribution, precompute)
    return solve_kinetic_electron_phi_from_density(numerator, precompute)


def solve_kinetic_electron_phi_from_density(
    numerator,
    precompute: KineticQuasineutralityPrecompute,
):
    """Solve the kinetic field equation from a precomputed density numerator."""

    denominator = _safe_denominator(precompute.denominator, precompute.denominator_floor)
    phi = -jnp.asarray(numerator) / denominator
    if precompute.regularize_constant_mode and precompute.ixzero >= 0 and precompute.iyzero >= 0:
        phi = phi.at[:, precompute.ixzero, precompute.iyzero].set(0.0)
    return phi


def adiabatic_quasineutrality_residual(
    phi,
    distribution,
    precompute: AdiabaticQuasineutralityPrecompute,
):
    """Return the quasineutrality residual for a solved or trial potential."""

    numerator = adiabatic_density_numerator(distribution, precompute)
    return adiabatic_quasineutrality_residual_from_density(phi, numerator, precompute)


def adiabatic_quasineutrality_residual_from_density(
    phi,
    numerator,
    precompute: AdiabaticQuasineutralityPrecompute,
):
    """Return ``N + D phi`` plus the optional zonal adiabatic correction."""

    phi = jnp.asarray(phi)
    numerator = jnp.asarray(numerator)
    denominator = _safe_denominator(precompute.denominator, precompute.denominator_floor)
    residual = numerator + denominator * phi
    if precompute.zonal_correction and precompute.has_zonal:
        iy = precompute.iyzero
        kx_mask = _nonzero_kx_mask(phi.shape[1], precompute.ixzero, phi.dtype)
        weighted_average = _weighted_z_average(phi[:, :, iy], precompute.w_z)
        correction = precompute.electron_response * weighted_average[None, :] * kx_mask[None, :]
        zonal_residual = residual[:, :, iy] + correction
        residual = residual.at[:, :, iy].set(zonal_residual)
    return residual


def kinetic_quasineutrality_residual(
    phi,
    distribution,
    precompute: KineticQuasineutralityPrecompute,
):
    """Return the fully kinetic quasineutrality residual."""

    numerator = kinetic_density_numerator(distribution, precompute)
    return kinetic_quasineutrality_residual_from_density(phi, numerator, precompute)


def kinetic_quasineutrality_residual_from_density(
    phi,
    numerator,
    precompute: KineticQuasineutralityPrecompute,
):
    """Return ``N + D phi`` for the fully kinetic field equation."""

    residual = jnp.asarray(numerator) + precompute.denominator * jnp.asarray(phi)
    if precompute.regularize_constant_mode and precompute.ixzero >= 0 and precompute.iyzero >= 0:
        residual = residual.at[:, precompute.ixzero, precompute.iyzero].set(0.0)
    return residual


def _apply_zonal_adiabatic_correction(numerator, denominator, phi, precompute):
    iy = precompute.iyzero
    zonal_numerator = numerator[:, :, iy]
    zonal_denominator = denominator[:, :, iy]
    w_z = precompute.w_z
    total_weight = jnp.sum(w_z)
    weighted_n_over_d = jnp.sum(w_z[:, None] * zonal_numerator / zonal_denominator, axis=0)
    weighted_inv_d = jnp.sum(w_z[:, None] / zonal_denominator, axis=0)
    correction_average = (
        precompute.electron_response
        * weighted_n_over_d
        / (total_weight + precompute.electron_response * weighted_inv_d)
    )
    corrected_zonal = phi[:, :, iy] + correction_average[None, :] / zonal_denominator
    kx_mask = _nonzero_kx_mask(phi.shape[1], precompute.ixzero, phi.dtype)
    corrected_zonal = jnp.where(kx_mask[None, :] > 0, corrected_zonal, phi[:, :, iy])
    return phi.at[:, :, iy].set(corrected_zonal)


def _as_species_tuple(species: SpeciesParams | tuple[SpeciesParams, ...]):
    return species if isinstance(species, tuple) else (species,)


def _with_species_axis(array, n_species: int, name: str, *, single_ndim: int):
    array = jnp.asarray(array)
    if n_species == 1 and array.ndim == single_ndim:
        return array[None, ...]
    if array.ndim == single_ndim + 1 and array.shape[0] == n_species:
        return array
    raise ValueError(
        f"{name} must have a single-species shape with {single_ndim} dimensions or "
        f"a leading species axis with {single_ndim + 1} dimensions; got {array.shape}"
    )


def _distribution_with_species_axis(distribution, n_species: int):
    distribution = jnp.asarray(distribution)
    if n_species == 1 and distribution.ndim == 5:
        return distribution[None, ...]
    if distribution.ndim == 6 and distribution.shape[0] == n_species:
        return distribution
    raise ValueError(
        "distribution must have shape (n_vpar,n_mu,n_z,n_kx,n_ky) "
        "or (n_species,n_vpar,n_mu,n_z,n_kx,n_ky)"
    )


def _normalize_parallel_weights(B, w_z):
    if w_z is None:
        return jnp.ones_like(B) / B.shape[0]
    return jnp.asarray(w_z)


def _safe_denominator(denominator, floor):
    denominator = jnp.asarray(denominator)
    floor = jnp.asarray(floor, dtype=denominator.real.dtype)
    sign = jnp.where(jnp.real(denominator) < 0.0, -1.0, 1.0)
    replacement = sign.astype(denominator.dtype) * floor.astype(denominator.dtype)
    return jnp.where(jnp.abs(denominator) < floor, replacement, denominator)


def _weighted_z_average(values, w_z):
    return jnp.sum(w_z[:, None] * values, axis=0) / jnp.sum(w_z)


def _nonzero_kx_mask(n_kx: int, ixzero: int, dtype):
    mask = jnp.ones((n_kx,), dtype=dtype)
    if ixzero >= 0:
        mask = mask.at[ixzero].set(0.0)
    return mask
