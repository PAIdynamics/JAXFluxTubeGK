"""Diagnostic reductions and quasilinear ingredients."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .types import SpeciesParams, VelocityGrid


@dataclass(frozen=True)
class SaturatedFluxStatistics:
    """Time-window statistics for a candidate saturated radial-flux signal."""

    mean: object
    standard_deviation: object
    standard_error: object
    relative_window_drift: object
    n_samples: int
    n_blocks: int = 1
    block_duration: float = 0.0


def correlated_flux_statistics(
    times,
    flux,
    *,
    start_fraction: float = 0.5,
    block_duration: float = 5.0,
) -> SaturatedFluxStatistics:
    """Estimate flux statistics using physical-time blocks.

    Adaptive trajectories are sampled once per accepted step, so an ordinary
    sample mean overweights intervals with smaller timesteps. Equal-duration
    block means remove that bias and provide a conservative standard error for
    autocorrelated turbulence traces.
    """

    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if times.ndim != 1 or flux.shape != times.shape or times.size < 3:
        raise ValueError("flux statistics require matching one-dimensional traces")
    if not np.isfinite(times).all() or not np.isfinite(flux).all():
        raise ValueError("flux statistics require finite traces")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("flux-statistics times must be strictly increasing")
    if not 0.0 <= start_fraction < 1.0 or block_duration <= 0.0:
        raise ValueError("start fraction and block duration are invalid")
    start = min(int(times.size * start_fraction), times.size - 2)
    window_times = times[start:]
    window_flux = flux[start:]
    duration = float(window_times[-1] - window_times[0])
    if duration <= 0.0:
        raise ValueError("flux-statistics window must have positive duration")
    n_blocks = max(1, int(np.floor(duration / block_duration + 1.0e-12)))
    edges = np.linspace(window_times[0], window_times[-1], n_blocks + 1)
    block_means = np.asarray(
        [
            _interval_mean(window_times, window_flux, left, right)
            for left, right in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    mean = float(np.mean(block_means))
    variance = _interval_mean(window_times, (window_flux - mean) ** 2, edges[0], edges[-1])
    standard_deviation = float(np.sqrt(max(variance, 0.0)))
    standard_error = (
        float(np.std(block_means, ddof=1) / np.sqrt(n_blocks))
        if n_blocks > 1
        else standard_deviation / np.sqrt(window_flux.size)
    )
    if n_blocks > 1:
        centers = 0.5 * (edges[:-1] + edges[1:])
        centered_time = centers - np.mean(centers)
        slope = float(centered_time @ (block_means - mean) / (centered_time @ centered_time))
    else:
        centered_time = window_times - np.mean(window_times)
        slope = float(
            centered_time @ (window_flux - np.mean(window_flux)) / (centered_time @ centered_time)
        )
    relative_drift = slope * duration / max(abs(mean), 1.0e-14)
    return SaturatedFluxStatistics(
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        relative_window_drift=relative_drift,
        n_samples=int(window_flux.size),
        n_blocks=n_blocks,
        block_duration=duration / n_blocks,
    )


def _interval_mean(times: np.ndarray, values: np.ndarray, left: float, right: float) -> float:
    """Return the trapezoidal time-average of values over [left, right], interpolating at the endpoints."""
    interior = (times > left) & (times < right)
    sample_times = np.concatenate(([left], times[interior], [right]))
    sample_values = np.concatenate(
        ([np.interp(left, times, values)], values[interior], [np.interp(right, times, values)])
    )
    return float(np.trapezoid(sample_values, sample_times) / (right - left))


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

    return jnp.sum(radial_flux_spectrum(phi, response, ky, w_z=w_z, parseval=parseval, sign=sign))


def gyrokinetic_heat_response(
    distribution,
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    bessel_j0,
):
    """Return each species' gyroaveraged non-advective heat response.

    The response is the discrete velocity integral of
    ``J0 * T_s * (E_s - 3/2) * f_s``.  It can be passed directly to
    :func:`radial_flux_spectrum` with the electrostatic potential.
    """

    values = jnp.asarray(distribution)
    original_ndim = values.ndim
    species_tuple = species if isinstance(species, tuple) else (species,)
    if original_ndim == 5 and len(species_tuple) == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != len(species_tuple):
        raise ValueError("distribution has incompatible species or phase-space shape")
    gyroaverage = jnp.asarray(bessel_j0)
    if gyroaverage.ndim == 4 and len(species_tuple) == 1:
        gyroaverage = gyroaverage[None, ...]
    expected_gyro_shape = (
        len(species_tuple),
        values.shape[2],
        values.shape[3],
        values.shape[4],
        values.shape[5],
    )
    if gyroaverage.shape != expected_gyro_shape:
        raise ValueError("bessel_j0 has incompatible species, mu, z, or Fourier shape")

    vpar = jnp.asarray(velocity_grid.vpar)[None, :, None, None]
    mu = jnp.asarray(velocity_grid.mu)[None, None, :, None]
    B = jnp.asarray(B)[None, None, None, :]
    temperature = jnp.asarray([item.temperature for item in species_tuple])[:, None, None, None]
    normalized_energy = (vpar**2 + 2.0 * mu * B) / temperature
    heat_weight = temperature * (normalized_energy - 1.5)
    measure = (
        jnp.asarray(velocity_grid.w_vpar)[:, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, :, None]
        * B[0]
    )
    response = jnp.einsum(
        "vmz,svmzxy,smzxy,svmz->szxy",
        measure,
        values,
        gyroaverage,
        heat_weight,
    )
    return response[0] if original_ndim == 5 else response


def gyrokinetic_energy_response(
    distribution,
    velocity_grid: VelocityGrid,
    B,
    species: SpeciesParams | tuple[SpeciesParams, ...],
    bessel_j0,
):
    """Return a gyroaveraged total-energy response for heat-flux diagnostics.

    Unlike :func:`gyrokinetic_heat_response`, this moment does not subtract
    ``3/2 T_s`` times the particle response. The two fluxes agree only when
    the radial particle flux vanishes.
    """

    values = jnp.asarray(distribution)
    original_ndim = values.ndim
    species_tuple = species if isinstance(species, tuple) else (species,)
    if original_ndim == 5 and len(species_tuple) == 1:
        values = values[None, ...]
    if values.ndim != 6 or values.shape[0] != len(species_tuple):
        raise ValueError("distribution has incompatible species or phase-space shape")
    gyroaverage = jnp.asarray(bessel_j0)
    if gyroaverage.ndim == 4 and len(species_tuple) == 1:
        gyroaverage = gyroaverage[None, ...]
    expected_gyro_shape = (
        len(species_tuple),
        values.shape[2],
        values.shape[3],
        values.shape[4],
        values.shape[5],
    )
    if gyroaverage.shape != expected_gyro_shape:
        raise ValueError("bessel_j0 has incompatible species, mu, z, or Fourier shape")

    vpar = jnp.asarray(velocity_grid.vpar)[:, None, None]
    mu = jnp.asarray(velocity_grid.mu)[None, :, None]
    B = jnp.asarray(B)[None, None, :]
    energy = vpar**2 + 2.0 * mu * B
    measure = (
        jnp.asarray(velocity_grid.w_vpar)[:, None, None]
        * jnp.asarray(velocity_grid.w_mu)[None, :, None]
        * B
    )
    response = jnp.einsum(
        "vmz,svmzxy,smzxy,vmz->szxy",
        measure,
        values,
        gyroaverage,
        energy,
    )
    return response[0] if original_ndim == 5 else response


def saturated_radial_flux_statistics(
    phi_history,
    response_history,
    times,
    ky,
    *,
    start_fraction: float = 0.5,
    w_z=None,
    parseval=None,
    sign: float = 1.0,
) -> SaturatedFluxStatistics:
    """Summarize radial flux over a caller-selected candidate saturated window."""

    phi_history = jnp.asarray(phi_history)
    response_history = jnp.asarray(response_history)
    times = jnp.asarray(times)
    if phi_history.shape != response_history.shape or phi_history.ndim != 4:
        raise ValueError("field histories must share shape (time,z,kx,ky)")
    if times.shape != (phi_history.shape[0],):
        raise ValueError("times must have one value per history sample")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1)")
    start = min(int(phi_history.shape[0] * start_fraction), phi_history.shape[0] - 1)
    if phi_history.shape[0] - start < 2:
        raise ValueError("saturated window must contain at least two samples")
    weights = _z_weights(phi_history[0], w_z)
    flux_z = (
        sign
        * jnp.asarray(ky)[None, None, None, :]
        * jnp.imag(jnp.conj(phi_history) * response_history)
    )
    flux = jnp.sum(weights[None, :, None, None] * flux_z, axis=1) / jnp.sum(weights)
    if parseval is not None:
        flux = flux * jnp.asarray(parseval)[None, None, :]
    flux = jnp.sum(flux, axis=(1, 2))[start:]
    window_times = times[start:]
    mean = jnp.mean(flux)
    standard_deviation = jnp.std(flux, ddof=1)
    standard_error = standard_deviation / jnp.sqrt(flux.shape[0])
    centered_time = window_times - jnp.mean(window_times)
    slope = jnp.sum(centered_time * (flux - mean)) / jnp.sum(centered_time**2)
    relative_drift = (
        slope
        * (window_times[-1] - window_times[0])
        / jnp.maximum(jnp.abs(mean), jnp.asarray(1.0e-14, dtype=mean.dtype))
    )
    return SaturatedFluxStatistics(
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        relative_window_drift=relative_drift,
        n_samples=int(flux.shape[0]),
    )


def _z_weights(field, w_z):
    """Return w_z as an array, or uniform unit weights matching field's dtype if w_z is None."""
    if w_z is None:
        return jnp.ones((field.shape[0],), dtype=field.real.dtype)
    return jnp.asarray(w_z)
