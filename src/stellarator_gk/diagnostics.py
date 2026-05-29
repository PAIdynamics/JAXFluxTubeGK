"""Diagnostic reductions and quasilinear ingredients."""

from __future__ import annotations

import jax.numpy as jnp

from .types import VelocityGrid


def velocity_space_integral(values, velocity_grid: VelocityGrid, B=None):
    """Integrate over ``v_parallel`` and ``mu`` using the velocity-grid weights."""

    values = jnp.asarray(values)
    weights = velocity_grid.w_vpar[:, None] * velocity_grid.w_mu[None, :]
    weighted = weights.reshape(weights.shape + (1,) * (values.ndim - 2)) * values
    result = jnp.sum(weighted, axis=(0, 1))
    if B is not None:
        B = jnp.asarray(B)
        result = B.reshape((B.shape[0],) + (1,) * (result.ndim - 1)) * result
    return result


def mode_amplitude(field, w_z=None):
    """Return RMS mode amplitude over the parallel coordinate."""

    field = jnp.asarray(field)
    weights = _z_weights(field, w_z)
    mean_square = jnp.sum(weights[:, None, None] * jnp.abs(field) ** 2, axis=0)
    return jnp.sqrt(mean_square / jnp.sum(weights))


def kxky_spectrum(field, w_z=None, parseval=None):
    """Return the parallel-averaged ``|field|^2`` spectrum on ``(kx, ky)``."""

    field = jnp.asarray(field)
    weights = _z_weights(field, w_z)
    spectrum = jnp.sum(weights[:, None, None] * jnp.abs(field) ** 2, axis=0) / jnp.sum(weights)
    if parseval is not None:
        spectrum = spectrum * jnp.asarray(parseval)[None, :]
    return spectrum


def ky_spectrum(field, w_z=None, parseval=None):
    """Return a ``ky`` spectrum by summing the ``kx`` spectrum."""

    return jnp.sum(kxky_spectrum(field, w_z=w_z, parseval=parseval), axis=0)


def radial_flux_spectrum(phi, response, ky, w_z=None, parseval=None, sign: float = 1.0):
    """Return ``sign * ky * Im(conj(phi) response)`` averaged over ``z``.

    This is a reusable quasilinear ingredient.  The caller chooses ``response``
    to represent particle, heat, or another gyroaveraged moment response.
    """

    phi = jnp.asarray(phi)
    response = jnp.asarray(response)
    weights = _z_weights(phi, w_z)
    ky = jnp.asarray(ky)
    flux_z = sign * ky[None, None, :] * jnp.imag(jnp.conj(phi) * response)
    spectrum = jnp.sum(weights[:, None, None] * flux_z, axis=0) / jnp.sum(weights)
    if parseval is not None:
        spectrum = spectrum * jnp.asarray(parseval)[None, :]
    return spectrum


def total_radial_flux(phi, response, ky, w_z=None, parseval=None, sign: float = 1.0):
    """Return the total radial flux ingredient summed over ``kx`` and ``ky``."""

    return jnp.sum(
        radial_flux_spectrum(phi, response, ky, w_z=w_z, parseval=parseval, sign=sign)
    )


def _z_weights(field, w_z):
    if w_z is None:
        return jnp.ones((field.shape[0],), dtype=field.real.dtype)
    return jnp.asarray(w_z)
