"""Linear electrostatic gyrokinetic RHS terms.

The functions here implement the term-level contract from ``main.tex`` using
matrix-free JAX array operations.  All topology and grid choices are assumed to
have been handled before this layer; the RHS sees only collocation derivative
matrices, Fourier mode arrays, and precomputed geometry/species coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from ..types import FourierGrid, ParallelGrid, SpeciesParams, VelocityGrid, _PyTreeDataclass
from .primitives import (
    FLRFactors,
    magnetic_drift_frequency as build_magnetic_drift_frequency,
    maxwellian as build_maxwellian,
    mirror_force_coefficient,
    normalized_energy,
    parallel_streaming_coefficient,
    species_flr_factors,
    thermodynamic_drive_factor,
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearRHSPrecompute(_PyTreeDataclass):
    """Precomputed coefficients for the linear electrostatic RHS."""

    D_z: object
    D_vpar: object
    ky: object
    E_y: object
    flr_factors: FLRFactors
    maxwellian: object
    drive_factor: object
    parallel_streaming_coeff: object
    mirror_force_coeff: object
    magnetic_drift_frequency: object
    charge_over_temperature: object
    perpendicular_damping: object
    parallel_recurrence_operator: object
    parallel_recurrence_coeff: object
    n_species: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "D_z",
        "D_vpar",
        "ky",
        "E_y",
        "flr_factors",
        "maxwellian",
        "drive_factor",
        "parallel_streaming_coeff",
        "mirror_force_coeff",
        "magnetic_drift_frequency",
        "charge_over_temperature",
        "perpendicular_damping",
        "parallel_recurrence_operator",
        "parallel_recurrence_coeff",
    )
    _static_fields: ClassVar[tuple[str, ...]] = ("n_species",)

    def __post_init__(self):
        if self.n_species < 1:
            raise ValueError("n_species must be at least 1")


def build_linear_rhs_precompute(
    velocity_grid: VelocityGrid,
    parallel_grid: ParallelGrid,
    fourier_grid: FourierGrid,
    geometry,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    *,
    flr_factors: FLRFactors | None = None,
    perpendicular_damping=None,
    parallel_recurrence_rate: float = 0.0,
    parallel_recurrence_velocity_model: str = "rms",
) -> LinearRHSPrecompute:
    """Precompute geometry/species coefficients used by the linear RHS terms."""

    species_tuple = _as_species_tuple(species)
    n_species = len(species_tuple)
    kperp2 = _k_perp_squared(geometry, fourier_grid)
    flr = flr_factors or species_flr_factors(species, velocity_grid.mu, geometry.B, kperp2)
    flr = _with_species_axis_flr(flr, n_species)

    energy = normalized_energy(velocity_grid.vpar, velocity_grid.mu, geometry.B, species)
    fmax = _with_species_axis(
        build_maxwellian(velocity_grid.vpar, velocity_grid.mu, geometry.B, species),
        n_species,
        "maxwellian",
        single_ndim=3,
    )
    drive = _with_species_axis(
        thermodynamic_drive_factor(energy, species),
        n_species,
        "drive_factor",
        single_ndim=3,
    )
    parallel = _with_species_axis(
        parallel_streaming_coefficient(velocity_grid.vpar, geometry.F, species),
        n_species,
        "parallel_streaming_coeff",
        single_ndim=2,
    )
    mirror = _with_species_axis(
        mirror_force_coefficient(velocity_grid.mu, geometry.B, geometry.G, species),
        n_species,
        "mirror_force_coeff",
        single_ndim=2,
    )
    drift = _with_species_axis(
        build_magnetic_drift_frequency(
            velocity_grid.vpar,
            velocity_grid.mu,
            geometry.B,
            geometry.D_x,
            geometry.D_y,
            fourier_grid.kx,
            fourier_grid.ky,
            species,
        ),
        n_species,
        "magnetic_drift_frequency",
        single_ndim=5,
    )
    charge_over_temperature = jnp.asarray(
        [item.charge / item.temperature for item in species_tuple],
        dtype=jnp.asarray(geometry.B).dtype,
    )

    damping = _normalize_perpendicular_damping(
        perpendicular_damping,
        fourier_grid.kx.shape[0],
        fourier_grid.ky.shape[0],
        dtype=jnp.asarray(geometry.B).dtype,
    )
    recurrence_operator = _gkw_parallel_recurrence_operator(
        parallel_grid,
        dtype=jnp.asarray(geometry.B).dtype,
    )
    recurrence_coeff = _parallel_recurrence_coefficient(
        velocity_grid.vpar,
        parallel,
        parallel_recurrence_rate,
        parallel_recurrence_velocity_model,
    )

    return LinearRHSPrecompute(
        D_z=parallel_grid.D_z,
        D_vpar=velocity_grid.D_vpar,
        ky=fourier_grid.ky,
        E_y=geometry.E_y,
        flr_factors=flr,
        maxwellian=fmax,
        drive_factor=drive,
        parallel_streaming_coeff=parallel,
        mirror_force_coeff=mirror,
        magnetic_drift_frequency=drift,
        charge_over_temperature=charge_over_temperature,
        perpendicular_damping=damping,
        parallel_recurrence_operator=recurrence_operator,
        parallel_recurrence_coeff=recurrence_coeff,
        n_species=n_species,
    )


def parallel_streaming(distribution, D_z, parallel_coefficient):
    """Return ``-a_parallel partial_z f``."""

    coefficient = jnp.asarray(parallel_coefficient)
    n_species = _coefficient_species_count(coefficient, single_ndim=2, name="parallel_coefficient")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    coefficient_s = _with_species_axis(
        coefficient,
        n_species,
        "parallel_coefficient",
        single_ndim=2,
    )
    dz_distribution = _parallel_derivative(distribution_s, D_z)
    result = -coefficient_s[:, :, None, :, None, None] * dz_distribution
    return _restore_distribution_shape(result, original_ndim)


def magnetic_drift_advection(distribution, drift_frequency):
    """Return ``-i omega_d f``."""

    drift = jnp.asarray(drift_frequency)
    n_species = _coefficient_species_count(drift, single_ndim=5, name="drift_frequency")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    drift_s = _with_species_axis(drift, n_species, "drift_frequency", single_ndim=5)
    result = -1j * drift_s * distribution_s
    return _restore_distribution_shape(result, original_ndim)


def mirror_force(distribution, D_vpar, mirror_coefficient):
    """Return ``a_mu partial_vparallel f``."""

    coefficient = jnp.asarray(mirror_coefficient)
    n_species = _coefficient_species_count(coefficient, single_ndim=2, name="mirror_coefficient")
    original_ndim = jnp.asarray(distribution).ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    coefficient_s = _with_species_axis(
        coefficient,
        n_species,
        "mirror_coefficient",
        single_ndim=2,
    )
    dv_distribution = _vpar_derivative(distribution_s, D_vpar)
    result = coefficient_s[:, None, :, :, None, None] * dv_distribution
    return _restore_distribution_shape(result, original_ndim)


def equilibrium_drive(phi, precompute: LinearRHSPrecompute):
    """Return ``i ky E_y J0 phi F_M Xi``."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    coefficient = (
        1j
        * precompute.E_y[None, None, None, :, None, None]
        * precompute.ky[None, None, None, None, None, :]
    )
    result = (
        coefficient
        * gyro_phi[:, None, :, :, :, :]
        * precompute.maxwellian[..., None, None]
        * precompute.drive_factor[..., None, None]
    )
    return _drop_single_species(result, precompute.n_species)


def parallel_field_drive(phi, D_z, precompute: LinearRHSPrecompute):
    """Return ``-(Z/T) a_parallel F_M partial_z(J0 phi)``."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    dz_gyro_phi = _parallel_derivative(gyro_phi, D_z)
    result = (
        -precompute.charge_over_temperature[:, None, None, None, None, None]
        * precompute.parallel_streaming_coeff[:, :, None, :, None, None]
        * precompute.maxwellian[..., None, None]
        * dz_gyro_phi[:, None, :, :, :, :]
    )
    return _drop_single_species(result, precompute.n_species)


def drift_field_drive(phi, precompute: LinearRHSPrecompute):
    """Return ``-(Z/T) i omega_d F_M J0 phi``."""

    gyro_phi = _gyroaveraged_potential(phi, precompute)
    result = (
        -precompute.charge_over_temperature[:, None, None, None, None, None]
        * 1j
        * precompute.magnetic_drift_frequency
        * precompute.maxwellian[..., None, None]
        * gyro_phi[:, None, :, :, :, :]
    )
    return _drop_single_species(result, precompute.n_species)


def dissipation(distribution, damping_rate=None):
    """Return optional linear perpendicular damping, zero by default."""

    distribution = jnp.asarray(distribution)
    if damping_rate is None:
        return jnp.zeros_like(distribution)
    damping = jnp.asarray(damping_rate)
    if damping.ndim == 2:
        damping = damping.reshape((1,) * (distribution.ndim - 2) + damping.shape)
    return -damping * distribution


def parallel_recurrence_control(distribution, operator, coefficient):
    """Return the GKW-scaled parallel fourth-order recurrence-control term."""

    distribution = jnp.asarray(distribution)
    coefficient = jnp.asarray(coefficient)
    if coefficient.ndim == 2:
        n_species = 1
        coefficient_s = coefficient[None, ...]
    elif coefficient.ndim == 3:
        n_species = int(coefficient.shape[0])
        coefficient_s = coefficient
    else:
        raise ValueError(
            "parallel_recurrence_coeff must have shape (n_vpar,n_z) "
            "or (n_species,n_vpar,n_z)"
        )
    original_ndim = distribution.ndim
    distribution_s = _distribution_with_species_axis(distribution, n_species)
    d4_distribution = _parallel_derivative(distribution_s, operator)
    result = coefficient_s[:, :, None, :, None, None] * d4_distribution
    return _restore_distribution_shape(result, original_ndim)


def linear_residual_from_phi(distribution, phi, precompute: LinearRHSPrecompute):
    """Assemble the linear RHS for a supplied electrostatic potential."""

    return (
        parallel_streaming(distribution, precompute.D_z, precompute.parallel_streaming_coeff)
        + magnetic_drift_advection(distribution, precompute.magnetic_drift_frequency)
        + mirror_force(distribution, precompute.D_vpar, precompute.mirror_force_coeff)
        + equilibrium_drive(phi, precompute)
        + parallel_field_drive(phi, precompute.D_z, precompute)
        + drift_field_drive(phi, precompute)
        + dissipation(distribution, precompute.perpendicular_damping)
        + parallel_recurrence_control(
            distribution,
            precompute.parallel_recurrence_operator,
            precompute.parallel_recurrence_coeff,
        )
    )


def _gyroaveraged_potential(phi, precompute: LinearRHSPrecompute):
    phi = jnp.asarray(phi)
    if phi.ndim != 3:
        raise ValueError("phi must have shape (n_z,n_kx,n_ky)")
    return precompute.flr_factors.bessel_j0 * phi[None, None, :, :, :]


def _parallel_derivative(values, D_z):
    axis = jnp.asarray(values).ndim - 3
    return _apply_matrix_along_axis(D_z, values, axis)


def _vpar_derivative(values, D_vpar):
    return _apply_matrix_along_axis(D_vpar, values, axis=1)


def _apply_matrix_along_axis(matrix, values, axis: int):
    values = jnp.asarray(values)
    matrix = jnp.asarray(matrix)
    moved = jnp.moveaxis(values, axis, 0)
    differentiated = jnp.tensordot(matrix, moved, axes=((1,), (0,)))
    return jnp.moveaxis(differentiated, 0, axis)


def _k_perp_squared(geometry, fourier_grid: FourierGrid):
    kx = fourier_grid.kx[None, :, None]
    ky = fourier_grid.ky[None, None, :]
    return (
        geometry.g_xx[:, None, None] * kx**2
        + 2.0 * geometry.g_xy[:, None, None] * kx * ky
        + geometry.g_yy[:, None, None] * ky**2
    )


def _with_species_axis_flr(flr: FLRFactors, n_species: int) -> FLRFactors:
    return FLRFactors(
        bessel_argument=_with_species_axis(
            flr.bessel_argument,
            n_species,
            "flr.bessel_argument",
            single_ndim=4,
        ),
        bessel_j0=_with_species_axis(
            flr.bessel_j0,
            n_species,
            "flr.bessel_j0",
            single_ndim=4,
        ),
        polarization_argument=_with_species_axis(
            flr.polarization_argument,
            n_species,
            "flr.polarization_argument",
            single_ndim=3,
        ),
        gamma0=_with_species_axis(
            flr.gamma0,
            n_species,
            "flr.gamma0",
            single_ndim=3,
        ),
    )


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


def _restore_distribution_shape(distribution, original_ndim: int):
    if original_ndim == 5:
        return distribution[0]
    if original_ndim == 6:
        return distribution
    raise ValueError(
        "distribution must have shape (n_vpar,n_mu,n_z,n_kx,n_ky) "
        "or (n_species,n_vpar,n_mu,n_z,n_kx,n_ky)"
    )


def _drop_single_species(values, n_species: int):
    if n_species == 1:
        return values[0]
    return values


def _coefficient_species_count(array, *, single_ndim: int, name: str) -> int:
    if array.ndim == single_ndim:
        return 1
    if array.ndim == single_ndim + 1:
        return int(array.shape[0])
    raise ValueError(
        f"{name} must have {single_ndim} dimensions or a leading species axis; got {array.shape}"
    )


def _normalize_perpendicular_damping(damping, n_kx: int, n_ky: int, *, dtype):
    if damping is None:
        return jnp.zeros((n_kx, n_ky), dtype=dtype)
    damping = jnp.asarray(damping, dtype=dtype)
    if damping.ndim == 0:
        return jnp.full((n_kx, n_ky), damping, dtype=dtype)
    if damping.shape != (n_kx, n_ky):
        raise ValueError("perpendicular_damping must be scalar or have shape (n_kx,n_ky)")
    return damping


def _gkw_parallel_recurrence_operator(parallel_grid: ParallelGrid, *, dtype):
    """Return the negative-semidefinite GKW-scaled fourth derivative operator.

    GKW's ``disp_par`` stencil adds
    ``[-1, 4, -6, 4, -1] / (12 * ds)`` to the parallel streaming stencil.  For
    the spectral target backend, the equivalent low-wavenumber scaling is
    ``-(ds**3 / 12) * d^4/dz^4``; this keeps recurrence control inside the
    residual while preserving the benchmark discretization's resolution
    scaling.
    """

    d_z = jnp.asarray(parallel_grid.D_z, dtype=dtype)
    d4 = d_z @ d_z @ d_z @ d_z
    spacing = jnp.sum(jnp.asarray(parallel_grid.w_z, dtype=dtype)) / d_z.shape[0]
    return -(spacing**3 / 12.0) * d4


def _parallel_recurrence_coefficient(vpar, parallel_coeff, rate: float, velocity_model: str):
    if rate < 0.0:
        raise ValueError("parallel_recurrence_rate must be nonnegative")
    if velocity_model not in ("local", "rms"):
        raise ValueError("parallel_recurrence_velocity_model must be 'local' or 'rms'")
    parallel_coeff = jnp.asarray(parallel_coeff)
    if velocity_model == "local":
        speed = jnp.abs(parallel_coeff)
    else:
        vpar = jnp.asarray(vpar, dtype=parallel_coeff.dtype)
        rms = jnp.sqrt(jnp.mean(vpar**2))
        eps = jnp.asarray(1.0e-300, dtype=parallel_coeff.dtype)
        if parallel_coeff.ndim == 2:
            v_abs = jnp.abs(vpar)[:, None]
            safe_v_abs = jnp.where(v_abs > eps, v_abs, 1.0)
            scale = jnp.where(v_abs > eps, jnp.abs(parallel_coeff) / safe_v_abs, 0.0)
            speed = rms * jnp.max(scale, axis=0, keepdims=True)
        elif parallel_coeff.ndim == 3:
            v_abs = jnp.abs(vpar)[None, :, None]
            safe_v_abs = jnp.where(v_abs > eps, v_abs, 1.0)
            scale = jnp.where(v_abs > eps, jnp.abs(parallel_coeff) / safe_v_abs, 0.0)
            speed = rms * jnp.max(scale, axis=1, keepdims=True)
        else:
            raise ValueError("parallel_coeff must have shape (n_vpar,n_z) or (n_species,n_vpar,n_z)")
        speed = jnp.broadcast_to(speed, parallel_coeff.shape)
    return jnp.asarray(rate, dtype=parallel_coeff.dtype) * speed


def _as_species_tuple(species: SpeciesParams | tuple[SpeciesParams, ...]):
    return species if isinstance(species, tuple) else (species,)
