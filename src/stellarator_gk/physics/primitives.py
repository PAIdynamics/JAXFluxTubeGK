"""Differentiable physics factors shared by future RHS implementations.

The functions in this module intentionally operate on plain arrays and
``SpeciesParams`` objects rather than on a specific velocity-space backend.  The
first solver uses Chebyshev collocation nodes, while a later Hermite-Laguerre
backend can reuse the same species, FLR, drift, and drive conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.special import i0e

from ..types import SpeciesParams, _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FLRFactors(_PyTreeDataclass):
    """Finite-Larmor-radius factors for one species or a species stack."""

    bessel_argument: object
    bessel_j0: object
    polarization_argument: object
    gamma0: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "bessel_argument",
        "bessel_j0",
        "polarization_argument",
        "gamma0",
    )


@jax.custom_jvp
def bessel_j0(x):
    """Evaluate ``J_0(x)`` with a JAX-differentiable Cephes-style approximation."""

    x = jnp.asarray(x)
    ax = jnp.abs(x)
    y = x * x
    tiny = 1.0 - y / 4.0 + y * y / 64.0
    small = _bessel_j0_small(jnp.where(ax < 8.0, x, 0.0))
    large = _bessel_j0_large(jnp.where(ax >= 8.0, ax, 8.0))
    return jnp.where(ax < 1.0e-5, tiny, jnp.where(ax < 8.0, small, large))


@bessel_j0.defjvp
def _bessel_j0_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    return bessel_j0(x), -_bessel_j1(x) * x_dot


def gamma0(b):
    """Evaluate the gyrokinetic polarization factor ``Gamma_0(b) = I_0(b) exp(-b)``."""

    return i0e(jnp.asarray(b))


@jax.custom_jvp
def _nonnegative_sqrt(x):
    """Square root with a finite zero-mode tangent for static nonnegative factors."""

    return jnp.sqrt(jnp.maximum(jnp.asarray(x), 0.0))


@_nonnegative_sqrt.defjvp
def _nonnegative_sqrt_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    y = _nonnegative_sqrt(x)
    safe_y = jnp.where(x > 0.0, y, jnp.ones_like(y))
    tangent = jnp.where(x > 0.0, 0.5 * x_dot / safe_y, jnp.zeros_like(x_dot))
    return y, tangent


def thermal_speed(species: SpeciesParams):
    """Return the normalized thermal speed ``sqrt(T_s / m_s)``."""

    return jnp.sqrt(jnp.asarray(species.temperature) / jnp.asarray(species.mass))


def normalized_energy(vpar, mu, B, species: SpeciesParams | tuple[SpeciesParams, ...]):
    """Return ``(v_parallel^2 + 2 mu B) / T_s`` on ``(..., v, mu, z)`` grids."""

    if isinstance(species, tuple):
        return jnp.stack([normalized_energy(vpar, mu, B, item) for item in species])
    vpar_grid, mu_grid, B_grid = _velocity_3d(vpar, mu, B)
    return (vpar_grid**2 + 2.0 * mu_grid * B_grid) / jnp.asarray(species.temperature)


def maxwellian(vpar, mu, B, species: SpeciesParams | tuple[SpeciesParams, ...]):
    """Return the normalized Maxwellian ``F_M(v_parallel, mu, z)``."""

    if isinstance(species, tuple):
        return jnp.stack([maxwellian(vpar, mu, B, item) for item in species])
    energy = normalized_energy(vpar, mu, B, species)
    prefactor = jnp.asarray(species.density) / (jnp.pi * jnp.asarray(species.temperature)) ** 1.5
    return prefactor * jnp.exp(-energy)


def thermodynamic_drive_factor(energy, species: SpeciesParams | tuple[SpeciesParams, ...]):
    """Return ``Xi_s = omega_n + omega_T (energy - 3/2)``."""

    energy = jnp.asarray(energy)
    if isinstance(species, tuple):
        density_gradient = _species_array(species, "density_gradient", energy.ndim)
        temperature_gradient = _species_array(species, "temperature_gradient", energy.ndim)
    else:
        density_gradient = jnp.asarray(species.density_gradient)
        temperature_gradient = jnp.asarray(species.temperature_gradient)
    return density_gradient + temperature_gradient * (energy - 1.5)


def equilibrium_gradient_drive_coefficient(maxwellian_value, drive_factor, E_y, ky):
    """Return the real coefficient ``ky E_y F_M Xi`` for Term V."""

    maxwellian_value = jnp.asarray(maxwellian_value)
    drive_factor = jnp.asarray(drive_factor)
    E_y = jnp.asarray(E_y)
    ky = jnp.asarray(ky)
    prefix_ndim = maxwellian_value.ndim - 1
    E_y_b = jnp.reshape(E_y, (1,) * prefix_ndim + (E_y.shape[0], 1))
    ky_b = jnp.reshape(ky, (1,) * (prefix_ndim + 1) + (ky.shape[0],))
    return maxwellian_value[..., :, None] * drive_factor[..., :, None] * E_y_b * ky_b


def polarization_argument(B, kperp_squared, species: SpeciesParams | tuple[SpeciesParams, ...]):
    """Return ``b_s`` for ``Gamma_0`` on ``(z, kx, ky)`` grids."""

    if isinstance(species, tuple):
        return jnp.stack([polarization_argument(B, kperp_squared, item) for item in species])
    charge = _nonzero_charge(species)
    B_b = jnp.asarray(B)[:, None, None]
    kperp2 = jnp.maximum(jnp.asarray(kperp_squared), 0.0)
    rho_factor = jnp.asarray(species.mass) * thermal_speed(species) / charge
    return 0.5 * (rho_factor**2) * kperp2 / B_b**2


def species_flr_factors(
    species: SpeciesParams | tuple[SpeciesParams, ...],
    mu,
    B,
    kperp_squared,
) -> FLRFactors:
    """Return Bessel and polarization factors for the supplied species."""

    if isinstance(species, tuple):
        factors = [species_flr_factors(item, mu, B, kperp_squared) for item in species]
        return FLRFactors(
            bessel_argument=jnp.stack([item.bessel_argument for item in factors]),
            bessel_j0=jnp.stack([item.bessel_j0 for item in factors]),
            polarization_argument=jnp.stack([item.polarization_argument for item in factors]),
            gamma0=jnp.stack([item.gamma0 for item in factors]),
        )

    charge_abs = jnp.abs(_nonzero_charge(species))
    mu_b = jnp.asarray(mu)[:, None, None, None]
    B_b = jnp.asarray(B)[None, :, None, None]
    kperp = _nonnegative_sqrt(jnp.asarray(kperp_squared))[None, :, :, :]
    rho_factor = jnp.asarray(species.mass) * thermal_speed(species) / charge_abs
    argument = rho_factor * kperp * _nonnegative_sqrt(2.0 * mu_b / B_b)
    b_arg = polarization_argument(B, kperp_squared, species)
    return FLRFactors(
        bessel_argument=argument,
        bessel_j0=bessel_j0(argument),
        polarization_argument=b_arg,
        gamma0=gamma0(b_arg),
    )


def magnetic_drift_frequency(
    vpar,
    mu,
    B,
    D_x,
    D_y,
    kx,
    ky,
    species: SpeciesParams | tuple[SpeciesParams, ...],
):
    """Return ``omega_d = (v_parallel^2 + mu B) (kx D_x + ky D_y) / Z_s``."""

    if isinstance(species, tuple):
        return jnp.stack(
            [magnetic_drift_frequency(vpar, mu, B, D_x, D_y, kx, ky, item) for item in species]
        )
    charge = _nonzero_charge(species)
    vpar_b, mu_b, B_b = _velocity_5d(vpar, mu, B)
    kdotD = _kdot_geometry(D_x, D_y, kx, ky)
    return (vpar_b**2 + mu_b * B_b) * kdotD / charge


def mirror_force_coefficient(mu, B, G, species: SpeciesParams | tuple[SpeciesParams, ...]):
    """Return the mirror-force velocity-space coefficient ``v_th mu B G``."""

    if isinstance(species, tuple):
        return jnp.stack([mirror_force_coefficient(mu, B, G, item) for item in species])
    mu_b = jnp.asarray(mu)[:, None]
    B_b = jnp.asarray(B)[None, :]
    G_b = jnp.asarray(G)[None, :]
    return thermal_speed(species) * mu_b * B_b * G_b


def parallel_streaming_coefficient(vpar, F, species: SpeciesParams | tuple[SpeciesParams, ...]):
    """Return the parallel-streaming coefficient ``v_th F v_parallel``."""

    if isinstance(species, tuple):
        return jnp.stack([parallel_streaming_coefficient(vpar, F, item) for item in species])
    return thermal_speed(species) * jnp.asarray(vpar)[:, None] * jnp.asarray(F)[None, :]


def _bessel_j0_small(x):
    y = x * x
    numerator = 57568490574.0 + y * (
        -13362590354.0
        + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456)))
    )
    denominator = 57568490411.0 + y * (
        1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))
    )
    return numerator / denominator


def _bessel_j0_large(ax):
    z = 8.0 / ax
    y = z * z
    phase = ax - 0.785398164
    amplitude = 1.0 + y * (
        -0.1098628627e-2
        + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
    )
    correction = -0.1562499995e-1 + y * (
        0.1430488765e-3
        + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7))
    )
    return jnp.sqrt(0.636619772 / ax) * (
        jnp.cos(phase) * amplitude - z * jnp.sin(phase) * correction
    )


def _bessel_j1(x):
    x = jnp.asarray(x)
    ax = jnp.abs(x)
    y = x * x
    tiny = x * (0.5 - y / 16.0 + y * y / 384.0)
    small = _bessel_j1_small(jnp.where(ax < 8.0, x, 0.0))
    large = _bessel_j1_large(jnp.where(ax >= 8.0, ax, 8.0))
    large = jnp.copysign(large, x)
    return jnp.where(ax < 1.0e-5, tiny, jnp.where(ax < 8.0, small, large))


def _bessel_j1_small(x):
    y = x * x
    numerator = x * (
        72362614232.0
        + y
        * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606))))
    )
    denominator = 144725228442.0 + y * (
        2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))
    )
    return numerator / denominator


def _bessel_j1_large(ax):
    z = 8.0 / ax
    y = z * z
    phase = ax - 2.356194491
    amplitude = 1.0 + y * (
        0.183105e-2
        + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6))
    )
    correction = 0.04687499995 + y * (
        -0.2002690873e-3
        + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
    )
    return jnp.sqrt(0.636619772 / ax) * (
        jnp.cos(phase) * amplitude - z * jnp.sin(phase) * correction
    )


def _velocity_3d(vpar, mu, B):
    return (
        jnp.asarray(vpar)[:, None, None],
        jnp.asarray(mu)[None, :, None],
        jnp.asarray(B)[None, None, :],
    )


def _velocity_5d(vpar, mu, B):
    return (
        jnp.asarray(vpar)[:, None, None, None, None],
        jnp.asarray(mu)[None, :, None, None, None],
        jnp.asarray(B)[None, None, :, None, None],
    )


def _kdot_geometry(D_x, D_y, kx, ky):
    D_x_b = jnp.asarray(D_x)[None, None, :, None, None]
    D_y_b = jnp.asarray(D_y)[None, None, :, None, None]
    kx_b = jnp.asarray(kx)[None, None, None, :, None]
    ky_b = jnp.asarray(ky)[None, None, None, None, :]
    return kx_b * D_x_b + ky_b * D_y_b


def _species_array(species: tuple[SpeciesParams, ...], name: str, ndim: int):
    values = jnp.asarray([getattr(item, name) for item in species])
    return jnp.reshape(values, (values.shape[0],) + (1,) * (ndim - 1))


def _nonzero_charge(species: SpeciesParams):
    return jnp.asarray(species.charge)
