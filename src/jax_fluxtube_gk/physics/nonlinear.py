"""Dealiased pseudo-spectral nonlinear ExB dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from ..types import FourierGrid, _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ExBPseudospectralPrecompute(_PyTreeDataclass):
    """Static maps from the retained half spectrum to a padded full plane."""

    retained_x: object
    conjugate_x: object
    retained_y: object
    conjugate_y: object
    padded_kx: object
    padded_ky: object
    n_kx: int
    n_ky: int
    n_x_padded: int
    n_y_padded: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "retained_x",
        "conjugate_x",
        "retained_y",
        "conjugate_y",
        "padded_kx",
        "padded_ky",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_kx",
        "n_ky",
        "n_x_padded",
        "n_y_padded",
    )


def build_exb_pseudospectral_precompute(
    fourier_grid: FourierGrid,
) -> ExBPseudospectralPrecompute:
    """Build Hermitian reconstruction and 3/2-padding maps.

    Nonlinear runs require centered, uniformly spaced ``kx`` modes and a
    uniformly spaced nonnegative ``ky`` half spectrum beginning at zero.
    """

    kx = np.asarray(fourier_grid.kx, dtype=float)
    ky = np.asarray(fourier_grid.ky, dtype=float)
    if kx.size < 3 or ky.size < 2:
        raise ValueError("nonlinear ExB dynamics require at least 3 kx and 2 ky modes")
    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])
    if dkx <= 0.0 or dky <= 0.0:
        raise ValueError("Fourier modes must be strictly increasing")
    if not np.allclose(np.diff(kx), dkx) or not np.allclose(np.diff(ky), dky):
        raise ValueError("nonlinear ExB dynamics require uniform Fourier spacing")
    if not np.isclose(ky[0], 0.0) or not np.isclose(kx[fourier_grid.ixzero], 0.0):
        raise ValueError("nonlinear ExB dynamics require retained kx=0 and ky=0 modes")
    kx_modes = np.rint(kx / dkx).astype(int)
    ky_modes = np.rint(ky / dky).astype(int)
    if not np.allclose(kx, kx_modes * dkx) or not np.allclose(ky, ky_modes * dky):
        raise ValueError("Fourier modes must be integer multiples of their fundamental spacing")

    n_x_padded = _odd_ceiling(1.5 * kx.size)
    n_y_padded = _odd_ceiling(1.5 * (2 * ky.size - 1))
    retained_x = np.mod(kx_modes, n_x_padded)
    conjugate_x = np.mod(-kx_modes, n_x_padded)
    retained_y = np.mod(ky_modes, n_y_padded)
    conjugate_y = np.mod(-ky_modes[1:], n_y_padded)
    padded_kx = 2.0 * np.pi * np.fft.fftfreq(n_x_padded, d=2.0 * np.pi / (dkx * n_x_padded))
    padded_ky = 2.0 * np.pi * np.fft.fftfreq(n_y_padded, d=2.0 * np.pi / (dky * n_y_padded))
    dtype = jnp.asarray(fourier_grid.kx).dtype
    return ExBPseudospectralPrecompute(
        retained_x=jnp.asarray(retained_x, dtype=jnp.int32),
        conjugate_x=jnp.asarray(conjugate_x, dtype=jnp.int32),
        retained_y=jnp.asarray(retained_y, dtype=jnp.int32),
        conjugate_y=jnp.asarray(conjugate_y, dtype=jnp.int32),
        padded_kx=jnp.asarray(padded_kx, dtype=dtype),
        padded_ky=jnp.asarray(padded_ky, dtype=dtype),
        n_kx=kx.size,
        n_ky=ky.size,
        n_x_padded=n_x_padded,
        n_y_padded=n_y_padded,
    )


def exb_pseudospectral_bracket(a, b, precompute: ExBPseudospectralPrecompute):
    """Return ``{a,b}=d_x a d_y b-d_y a d_x b`` on retained modes."""

    a_full = _expand_half_spectrum(a, precompute)
    b_full = _expand_half_spectrum(b, precompute)
    scale = precompute.n_x_padded * precompute.n_y_padded
    kx = precompute.padded_kx.reshape(
        (1,) * (a_full.ndim - 2) + (precompute.n_x_padded, 1)
    )
    ky = precompute.padded_ky.reshape(
        (1,) * (a_full.ndim - 2) + (1, precompute.n_y_padded)
    )
    ax = jnp.fft.ifft2(1j * kx * a_full, axes=(-2, -1)) * scale
    ay = jnp.fft.ifft2(1j * ky * a_full, axes=(-2, -1)) * scale
    bx = jnp.fft.ifft2(1j * kx * b_full, axes=(-2, -1)) * scale
    by = jnp.fft.ifft2(1j * ky * b_full, axes=(-2, -1)) * scale
    bracket_full = jnp.fft.fft2(ax * by - ay * bx, axes=(-2, -1)) / scale
    return bracket_full[..., precompute.retained_x[:, None], precompute.retained_y[None, :]]


def nonlinear_exb_term(distribution, phi, rhs_precompute, spectral_precompute):
    """Return gyroaveraged nonlinear Term III, ``-{J0 phi,f}``."""

    distribution = jnp.asarray(distribution)
    original_ndim = distribution.ndim
    if original_ndim == 5:
        distribution = distribution[None, ...]
    gyro_phi = (
        jnp.asarray(rhs_precompute.flr_factors.bessel_j0)
        * jnp.asarray(phi)[None, None, ...]
    )
    bracket = exb_pseudospectral_bracket(
        gyro_phi[:, None, ...], distribution, spectral_precompute
    )
    result = -bracket
    return result[0] if original_ndim == 5 else result


def estimate_nonlinear_exb_dt(
    gyro_phi,
    precompute: ExBPseudospectralPrecompute,
    *,
    safety: float = 0.8,
    floor: float = 1.0e-14,
):
    """Return an advective CFL estimate from padded real-space ExB speeds."""

    if safety <= 0.0:
        raise ValueError("safety must be positive")
    full = _expand_half_spectrum(gyro_phi, precompute)
    scale = precompute.n_x_padded * precompute.n_y_padded
    kx = precompute.padded_kx.reshape(
        (1,) * (full.ndim - 2) + (precompute.n_x_padded, 1)
    )
    ky = precompute.padded_ky.reshape(
        (1,) * (full.ndim - 2) + (1, precompute.n_y_padded)
    )
    velocity_y = jnp.fft.ifft2(1j * kx * full, axes=(-2, -1)) * scale
    velocity_x = -jnp.fft.ifft2(1j * ky * full, axes=(-2, -1)) * scale
    dx = 2.0 * jnp.pi / (
        jnp.abs(precompute.padded_kx[1]) * precompute.n_x_padded
    )
    dy = 2.0 * jnp.pi / (
        jnp.abs(precompute.padded_ky[1]) * precompute.n_y_padded
    )
    rate = jnp.max(jnp.abs(velocity_x) / dx + jnp.abs(velocity_y) / dy)
    return jnp.asarray(safety) / jnp.maximum(rate, jnp.asarray(floor))


def _expand_half_spectrum(values, precompute: ExBPseudospectralPrecompute):
    values = jnp.asarray(values)
    if values.shape[-2:] != (precompute.n_kx, precompute.n_ky):
        raise ValueError("last two array dimensions must match the retained Fourier grid")
    full = jnp.zeros(
        values.shape[:-2] + (precompute.n_x_padded, precompute.n_y_padded),
        dtype=values.dtype,
    )
    full = full.at[..., precompute.retained_x[:, None], precompute.retained_y[None, :]].set(
        values
    )
    return full.at[
        ..., precompute.conjugate_x[:, None], precompute.conjugate_y[None, :]
    ].set(jnp.conj(values[..., :, 1:]))


def _odd_ceiling(value: float) -> int:
    result = int(np.ceil(value))
    return result if result % 2 else result + 1


__all__ = [
    "ExBPseudospectralPrecompute",
    "build_exb_pseudospectral_precompute",
    "estimate_nonlinear_exb_dt",
    "exb_pseudospectral_bracket",
    "nonlinear_exb_term",
]
