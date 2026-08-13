"""Hermite-Laguerre velocity-space backend utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import numpy as np
from scipy import special

import jax
import jax.numpy as jnp

from ..types import _PyTreeDataclass
from .primitives import gamma0


class VelocityBasisKind(StrEnum):
    """Velocity-space backend identifiers."""

    CHEBYSHEV_COLLOCATION = "chebyshev_collocation"
    HERMITE_LAGUERRE = "hermite_laguerre"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VelocityBasisSpec(_PyTreeDataclass):
    """Static configuration for a velocity-space basis backend."""

    kind: str = VelocityBasisKind.HERMITE_LAGUERRE.value
    n_hermite: int = 8
    n_laguerre: int = 4
    n_hermite_grid: int | None = None
    n_laguerre_grid: int | None = None
    dtype: str = "float64"

    _static_fields: ClassVar[tuple[str, ...]] = (
        "kind",
        "n_hermite",
        "n_laguerre",
        "n_hermite_grid",
        "n_laguerre_grid",
        "dtype",
    )

    def __post_init__(self):
        """Normalize ``kind`` to its canonical value and validate mode/grid counts."""

        kind = VelocityBasisKind(self.kind)
        object.__setattr__(self, "kind", kind.value)
        if self.n_hermite < 1:
            raise ValueError("n_hermite must be at least 1")
        if self.n_laguerre < 1:
            raise ValueError("n_laguerre must be at least 1")
        if self.n_hermite_grid is not None and self.n_hermite_grid < self.n_hermite:
            raise ValueError("n_hermite_grid must be at least n_hermite")
        if self.n_laguerre_grid is not None and self.n_laguerre_grid < self.n_laguerre:
            raise ValueError("n_laguerre_grid must be at least n_laguerre")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HermiteLaguerreBasis(_PyTreeDataclass):
    """Gauss-grid transforms and modal coupling matrices for spectral moments."""

    hermite_nodes: object
    hermite_weights: object
    laguerre_nodes: object
    laguerre_weights: object
    hermite_to_grid: object
    hermite_to_spectral: object
    laguerre_to_grid: object
    laguerre_to_spectral: object
    hermite_derivative: object
    hermite_v: object
    hermite_v2: object
    laguerre_x: object
    n_hermite: int
    n_laguerre: int
    n_hermite_grid: int
    n_laguerre_grid: int
    dtype: str = "float64"

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "hermite_nodes",
        "hermite_weights",
        "laguerre_nodes",
        "laguerre_weights",
        "hermite_to_grid",
        "hermite_to_spectral",
        "laguerre_to_grid",
        "laguerre_to_spectral",
        "hermite_derivative",
        "hermite_v",
        "hermite_v2",
        "laguerre_x",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "n_hermite",
        "n_laguerre",
        "n_hermite_grid",
        "n_laguerre_grid",
        "dtype",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MomentRHSParams(_PyTreeDataclass):
    """Controls for the reduced Hermite-Laguerre linear RHS.

    This is a compact discriminator model for s-alpha, adiabatic-electron ITG
    mode-structure checks. It keeps Hermite/Laguerre source projections, linked
    parallel chains, and linked ``k_z`` hypercollision explicit. It is an
    experimental solver-owned model, not a production velocity backend.
    """

    density_gradient: float = 0.8
    temperature_gradient: float = 2.49
    tau: float = 1.0
    streaming_scale: float = 1.0
    drive_scale: float = 1.0
    drift_scale: float = 0.18
    magnetic_shear: float = 0.8
    nu_hyper_m: float = 1.0
    vt: float = 1.0
    gradpar_abs: float = 1.0
    denominator_floor: float = 1.0e-12
    include_curvature_drift: bool = True
    include_hypercollision: bool = True
    p_hyper_m: int | None = None
    n_links: int = 1
    z_periods: float = 1.0
    dealias: bool = False

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "density_gradient",
        "temperature_gradient",
        "tau",
        "streaming_scale",
        "drive_scale",
        "drift_scale",
        "magnetic_shear",
        "nu_hyper_m",
        "vt",
        "gradpar_abs",
        "denominator_floor",
    )
    _static_fields: ClassVar[tuple[str, ...]] = (
        "include_curvature_drift",
        "include_hypercollision",
        "p_hyper_m",
        "n_links",
        "z_periods",
        "dealias",
    )

    def __post_init__(self):
        """Validate that the physical and numerical parameters are in admissible ranges."""

        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.vt < 0.0:
            raise ValueError("vt must be non-negative")
        if self.gradpar_abs < 0.0:
            raise ValueError("gradpar_abs must be non-negative")
        if self.denominator_floor <= 0.0:
            raise ValueError("denominator_floor must be positive")
        if self.n_links < 1:
            raise ValueError("n_links must be at least 1")
        if self.z_periods <= 0.0:
            raise ValueError("z_periods must be positive")
        if self.p_hyper_m is not None and self.p_hyper_m < 1:
            raise ValueError("p_hyper_m must be positive when supplied")


def build_velocity_basis(spec: VelocityBasisSpec) -> HermiteLaguerreBasis:
    """Build the configured velocity basis.

    The Chebyshev collocation backend is still built by
    :func:`jax_fluxtube_gk.build_velocity_grid`; this dispatcher reserves a
    common entry point for future backend selection.
    """

    if spec.kind != VelocityBasisKind.HERMITE_LAGUERRE.value:
        raise ValueError("build_velocity_basis currently builds only hermite_laguerre bases")
    return build_hermite_laguerre_basis(spec)


def build_hermite_laguerre_basis(spec: VelocityBasisSpec) -> HermiteLaguerreBasis:
    """Build probabilists-Hermite and signed-Laguerre transforms."""

    n_h_grid = spec.n_hermite if spec.n_hermite_grid is None else spec.n_hermite_grid
    n_l_grid = spec.n_laguerre if spec.n_laguerre_grid is None else spec.n_laguerre_grid
    real_dtype = jnp.dtype(spec.dtype)

    hermite_nodes, hermite_weights = np.polynomial.hermite_e.hermegauss(n_h_grid)
    hermite_weights = hermite_weights / np.sqrt(2.0 * np.pi)
    laguerre_nodes, laguerre_weights = special.roots_laguerre(n_l_grid)

    hermite_grid = np.asarray(normalized_hermite_functions(spec.n_hermite, hermite_nodes))
    laguerre_grid = np.asarray(signed_laguerre_polynomials(spec.n_laguerre, laguerre_nodes))

    hermite_to_grid = hermite_grid.T
    hermite_to_spectral = hermite_grid * hermite_weights[None, :]
    laguerre_to_grid = laguerre_grid.T
    laguerre_to_spectral = laguerre_grid * laguerre_weights[None, :]

    hermite_v = _hermite_v_matrix(spec.n_hermite)
    hermite_derivative = _hermite_derivative_matrix(spec.n_hermite)
    laguerre_x = _laguerre_x_matrix(spec.n_laguerre)

    return HermiteLaguerreBasis(
        hermite_nodes=jnp.asarray(hermite_nodes, dtype=real_dtype),
        hermite_weights=jnp.asarray(hermite_weights, dtype=real_dtype),
        laguerre_nodes=jnp.asarray(laguerre_nodes, dtype=real_dtype),
        laguerre_weights=jnp.asarray(laguerre_weights, dtype=real_dtype),
        hermite_to_grid=jnp.asarray(hermite_to_grid, dtype=real_dtype),
        hermite_to_spectral=jnp.asarray(hermite_to_spectral, dtype=real_dtype),
        laguerre_to_grid=jnp.asarray(laguerre_to_grid, dtype=real_dtype),
        laguerre_to_spectral=jnp.asarray(laguerre_to_spectral, dtype=real_dtype),
        hermite_derivative=jnp.asarray(hermite_derivative, dtype=real_dtype),
        hermite_v=jnp.asarray(hermite_v, dtype=real_dtype),
        hermite_v2=jnp.asarray(hermite_v @ hermite_v, dtype=real_dtype),
        laguerre_x=jnp.asarray(laguerre_x, dtype=real_dtype),
        n_hermite=spec.n_hermite,
        n_laguerre=spec.n_laguerre,
        n_hermite_grid=n_h_grid,
        n_laguerre_grid=n_l_grid,
        dtype=spec.dtype,
    )


def normalized_hermite_functions(n: int, x):
    """Return ``He_m(x) / sqrt(m!)`` for ``m=0..n-1``."""

    if n < 1:
        raise ValueError("n must be at least 1")
    x = jnp.asarray(x)
    values = [jnp.ones_like(x)]
    if n > 1:
        values.append(x)
    for m in range(1, n - 1):
        values.append(x * values[m] - m * values[m - 1])
    scales = jnp.sqrt(jnp.asarray([special.factorial(m, exact=False) for m in range(n)], dtype=x.dtype))
    return jnp.stack(values[:n], axis=0) / scales.reshape((n,) + (1,) * x.ndim)


def signed_laguerre_polynomials(n: int, x):
    """Return ``psi_l(x)=(-1)^l L_l(x)`` for ``l=0..n-1``."""

    if n < 1:
        raise ValueError("n must be at least 1")
    x = jnp.asarray(x)
    values = [jnp.ones_like(x)]
    if n > 1:
        values.append(1.0 - x)
    for ell in range(1, n - 1):
        next_value = ((2 * ell + 1 - x) * values[ell] - ell * values[ell - 1]) / (ell + 1)
        values.append(next_value)
    signs = jnp.asarray([(-1.0) ** ell for ell in range(n)], dtype=x.dtype)
    return jnp.stack(values[:n], axis=0) * signs.reshape((n,) + (1,) * x.ndim)


def spectral_to_velocity_grid(coefficients, basis: HermiteLaguerreBasis):
    """Transform coefficients ``(laguerre, hermite, ...)`` to Gauss-grid values."""

    return jnp.einsum(
        "jl,vm,lm...->jv...",
        basis.laguerre_to_grid,
        basis.hermite_to_grid,
        jnp.asarray(coefficients),
    )


def velocity_grid_to_spectral(values, basis: HermiteLaguerreBasis):
    """Project Gauss-grid values ``(laguerre_grid, hermite_grid, ...)`` to moments."""

    return jnp.einsum(
        "lj,mv,jv...->lm...",
        basis.laguerre_to_spectral,
        basis.hermite_to_spectral,
        jnp.asarray(values),
    )


def gyroaverage_laguerre_coefficients(b, n_laguerre: int):
    """Return coefficients ``J_l = exp(-b/2) (-b/2)^l / l!``."""

    if n_laguerre < 1:
        raise ValueError("n_laguerre must be at least 1")
    b = jnp.asarray(b)
    values = [jnp.exp(-0.5 * b)]
    for ell in range(1, n_laguerre):
        values.append(values[-1] * (-0.5 * b) / ell)
    return jnp.stack(values, axis=0)


def truncated_gamma0_from_laguerre(b, n_laguerre: int):
    """Approximate ``Gamma_0`` by an energetically consistent truncated sum."""

    coefficients = gyroaverage_laguerre_coefficients(b, n_laguerre)
    return jnp.sum(coefficients**2, axis=0)


def density_moment(coefficients, gyroaverage_coefficients=None):
    """Return ``sum_l J_l H_{l,0}``, or ``H_{0,0}`` if no gyroaverage is supplied."""

    coefficients = jnp.asarray(coefficients)
    if gyroaverage_coefficients is None:
        return coefficients[0, 0]
    return jnp.einsum("l...,l...->...", gyroaverage_coefficients, coefficients[:, 0])


def parallel_flow_moment(coefficients, gyroaverage_coefficients=None):
    """Return ``sum_l J_l H_{l,1}``, or ``H_{0,1}`` without gyroaveraging."""

    coefficients = jnp.asarray(coefficients)
    if coefficients.shape[1] < 2:
        return jnp.zeros_like(coefficients[0, 0])
    if gyroaverage_coefficients is None:
        return coefficients[0, 1]
    return jnp.einsum("l...,l...->...", gyroaverage_coefficients, coefficients[:, 1])


def parallel_temperature_moment(coefficients, gyroaverage_coefficients=None):
    """Return ``sqrt(2) sum_l J_l H_{l,2}``, or the ungyroaveraged low moment."""

    coefficients = jnp.asarray(coefficients)
    if coefficients.shape[1] < 3:
        return jnp.zeros_like(coefficients[0, 0])
    if gyroaverage_coefficients is None:
        return jnp.sqrt(jnp.asarray(2.0, dtype=coefficients.dtype)) * coefficients[0, 2]
    return jnp.sqrt(jnp.asarray(2.0, dtype=coefficients.dtype)) * jnp.einsum(
        "l...,l...->...", gyroaverage_coefficients, coefficients[:, 2]
    )


def parallel_heat_flux_moment(coefficients, gyroaverage_coefficients=None):
    """Return ``sqrt(6) sum_l J_l H_{l,3}``, or the ungyroaveraged low moment."""

    coefficients = jnp.asarray(coefficients)
    if coefficients.shape[1] < 4:
        return jnp.zeros_like(coefficients[0, 0])
    factor = jnp.sqrt(jnp.asarray(6.0, dtype=coefficients.dtype))
    if gyroaverage_coefficients is None:
        return factor * coefficients[0, 3]
    return factor * jnp.einsum("l...,l...->...", gyroaverage_coefficients, coefficients[:, 3])


def perpendicular_temperature_moment(coefficients, gyroaverage_coefficients=None):
    """Return the perpendicular-temperature moment from Laguerre coefficients."""

    coefficients = jnp.asarray(coefficients)
    return _perpendicular_laguerre_weighted_moment(coefficients[:, 0], gyroaverage_coefficients)


def perpendicular_heat_flux_moment(coefficients, gyroaverage_coefficients=None):
    """Return a low-order perpendicular heat-flux-like moment.

    Without gyroaveraging this is ``int v_parallel (mu B - 1) h = H_{1,1}``.
    With gyroaveraging, the same Laguerre weights used for the perpendicular
    temperature moment are applied to the ``m=1`` Hermite moment.
    """

    coefficients = jnp.asarray(coefficients)
    if coefficients.shape[1] < 2:
        return jnp.zeros_like(coefficients[0, 0])
    return _perpendicular_laguerre_weighted_moment(coefficients[:, 1], gyroaverage_coefficients)


def _perpendicular_laguerre_weighted_moment(laguerre_slice, gyroaverage_coefficients=None):
    """Return the Laguerre mu-weighted contraction ``sum_l (l J_{l-1}+2l J_l+(l+1) J_{l+1}) G_l``, or ``G_1`` without gyroaveraging.

    Shared by :func:`perpendicular_temperature_moment` and
    :func:`perpendicular_heat_flux_moment` to apply the same mu-weighting to
    different Hermite slices of the Laguerre coefficients.
    """

    coefficients = jnp.asarray(laguerre_slice)
    n_laguerre = coefficients.shape[0]
    if gyroaverage_coefficients is None:
        return coefficients[1] if n_laguerre > 1 else jnp.zeros_like(coefficients[0])

    ell = jnp.arange(n_laguerre, dtype=coefficients.real.dtype)
    jm1 = _shift_laguerre_coefficients(gyroaverage_coefficients, 1)
    jp1 = _shift_laguerre_coefficients(gyroaverage_coefficients, -1)
    ell_shape = (-1,) + (1,) * (coefficients.ndim - 1)
    weights = ell.reshape(ell_shape) * jm1
    weights = weights + 2.0 * ell.reshape(ell_shape) * gyroaverage_coefficients
    weights = weights + (ell + 1.0).reshape(ell_shape) * jp1
    return jnp.einsum("l...,l...->...", weights, coefficients)


def heat_flux_moments(coefficients, gyroaverage_coefficients=None):
    """Return ``(parallel_heat_flux, perpendicular_heat_flux)`` diagnostics."""

    return (
        parallel_heat_flux_moment(coefficients, gyroaverage_coefficients),
        perpendicular_heat_flux_moment(coefficients, gyroaverage_coefficients),
    )


def fluid_moments(coefficients, gyroaverage_coefficients=None) -> dict[str, object]:
    """Return a compact set of low-order moment diagnostics."""

    qpar, qperp = heat_flux_moments(coefficients, gyroaverage_coefficients)
    return {
        "density": density_moment(coefficients, gyroaverage_coefficients),
        "parallel_flow": parallel_flow_moment(coefficients, gyroaverage_coefficients),
        "parallel_temperature": parallel_temperature_moment(coefficients, gyroaverage_coefficients),
        "perpendicular_temperature": perpendicular_temperature_moment(
            coefficients, gyroaverage_coefficients
        ),
        "parallel_heat_flux": qpar,
        "perpendicular_heat_flux": qperp,
    }


def free_energy_spectrum(coefficients, axis: int | tuple[int, ...] | None = None):
    """Return ``|G_lm|^2`` or a summed spectrum over selected axes."""

    spectrum = jnp.abs(jnp.asarray(coefficients)) ** 2
    if axis is None:
        return spectrum
    return jnp.sum(spectrum, axis=axis)


def hypercollision_damping_rates(
    n_laguerre: int,
    n_hermite: int,
    *,
    hermite_nu: float = 0.0,
    hermite_power: float | None = None,
    laguerre_nu: float = 0.0,
    laguerre_power: float | None = None,
    conserve_low_m: bool = True,
    dtype: str = "float64",
):
    """Return modal damping rates for a simple closure/hypercollision hook."""

    if n_laguerre < 1 or n_hermite < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")
    real_dtype = jnp.dtype(dtype)
    ell = jnp.arange(n_laguerre, dtype=real_dtype)[:, None]
    m = jnp.arange(n_hermite, dtype=real_dtype)[None, :]
    h_power = n_hermite / 2.0 if hermite_power is None else hermite_power
    l_power = max(n_laguerre / 2.0, 1.0) if laguerre_power is None else laguerre_power
    rates = hermite_nu * m**h_power + laguerre_nu * ell**l_power
    if conserve_low_m:
        rates = jnp.where(m <= 2.0, laguerre_nu * ell**l_power, rates)
        rates = rates.at[0, :3].set(0.0)
    return rates


def apply_hypercollision(coefficients, damping_rates):
    """Return the RHS contribution ``-nu_lm G_lm``."""

    coefficients = jnp.asarray(coefficients)
    rates = jnp.asarray(damping_rates)
    return -rates.reshape(rates.shape + (1,) * (coefficients.ndim - rates.ndim)) * coefficients


def linked_kz_wavenumbers(
    n_z: int,
    *,
    n_links: int = 1,
    z_periods: float = 1.0,
    dealias: bool = False,
    dtype: str = "float64",
):
    """Return linked-chain ``k_z`` values in FFT ordering.

    The chain has length ``n_z * n_links`` and stores modes as
    ``0, 1, ..., floor(N/2), floor(N/2)-N, ..., -1`` divided by
    ``z_periods * n_links``. The optional mask applies the two-thirds rule.
    """

    if n_z < 2:
        raise ValueError("n_z must be at least 2")
    if n_links < 1:
        raise ValueError("n_links must be at least 1")
    if z_periods <= 0:
        raise ValueError("z_periods must be positive")

    real_dtype = jnp.dtype(dtype)
    n_total = int(n_z) * int(n_links)
    indices = jnp.arange(n_total, dtype=real_dtype)
    signed = jnp.where(indices < (n_total // 2 + 1), indices, indices - n_total)
    kz = signed / (jnp.asarray(z_periods, dtype=real_dtype) * n_links)
    if dealias:
        cutoff = (n_total - 1.0) / 3.0
        active = (indices <= cutoff) | (indices >= n_total - cutoff)
        kz = jnp.where(active, kz, jnp.zeros_like(kz))
    return kz


def apply_linked_abs_kz(
    values,
    *,
    n_links: int = 1,
    z_periods: float = 1.0,
    axis: int = -1,
    dealias: bool = False,
):
    """Apply the linked-chain ``|k_z|`` pseudo-differential operator.

    The selected axis is treated as the full linked chain. JAX's inverse FFT
    supplies the matching ``1/N`` normalization.
    """

    values = jnp.asarray(values)
    axis = _normalize_axis(axis, values.ndim)
    if n_links < 1:
        raise ValueError("n_links must be at least 1")
    n_total = values.shape[axis]
    if n_total % n_links != 0:
        raise ValueError("the selected axis length must be divisible by n_links")

    n_z = n_total // n_links
    real_dtype = jnp.asarray(values.real).dtype
    kz = linked_kz_wavenumbers(
        n_z,
        n_links=n_links,
        z_periods=z_periods,
        dealias=dealias,
        dtype=str(real_dtype),
    )
    multiplier_shape = [1] * values.ndim
    multiplier_shape[axis] = n_total
    multiplier = jnp.abs(kz).reshape(multiplier_shape)
    return jnp.fft.ifft(multiplier * jnp.fft.fft(values, axis=axis), axis=axis)


def apply_linked_grad_z(
    values,
    *,
    n_links: int = 1,
    z_periods: float = 1.0,
    axis: int = -1,
    dealias: bool = False,
):
    """Apply the linked-chain first parallel derivative.

    The wavenumber ordering and normalization match
    :func:`linked_kz_wavenumbers` and the ``|k_z|`` helper.
    """

    values = jnp.asarray(values)
    axis = _normalize_axis(axis, values.ndim)
    if n_links < 1:
        raise ValueError("n_links must be at least 1")
    n_total = values.shape[axis]
    if n_total % n_links != 0:
        raise ValueError("the selected axis length must be divisible by n_links")

    n_z = n_total // n_links
    real_dtype = jnp.asarray(values.real).dtype
    kz = linked_kz_wavenumbers(
        n_z,
        n_links=n_links,
        z_periods=z_periods,
        dealias=dealias,
        dtype=str(real_dtype),
    )
    multiplier_shape = [1] * values.ndim
    multiplier_shape[axis] = n_total
    multiplier = (1j * kz).reshape(multiplier_shape)
    return jnp.fft.ifft(multiplier * jnp.fft.fft(values, axis=axis), axis=axis)


def moment_adiabatic_phi(coefficients, b, params: MomentRHSParams | None = None):
    """Solve the reduced adiabatic-electron field equation in moment variables.

    ``coefficients`` must use the spectral layout
    ``(laguerre, hermite, ky, z)``.  The numerator is the gyroaveraged density
    ``sum_l J_l(b) G_{l0}``; the denominator is the truncated
    adiabatic-electron polarization ``tau + 1 - Gamma_0``.
    """

    params = MomentRHSParams() if params is None else params
    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim != 4:
        raise ValueError("coefficients must have shape (n_laguerre,n_hermite,n_ky,n_z)")
    b = _broadcast_b_to_ky_z(b, coefficients.shape[2], coefficients.shape[3])
    gyro = gyroaverage_laguerre_coefficients(b, coefficients.shape[0])
    density = density_moment(coefficients, gyro)
    gamma = truncated_gamma0_from_laguerre(b, coefficients.shape[0])
    denominator = _safe_signed_denominator(
        jnp.asarray(params.tau, dtype=density.real.dtype) + 1.0 - gamma,
        params.denominator_floor,
    )
    return density / denominator


def moment_energy_multiply(coefficients, basis: HermiteLaguerreBasis):
    """Return ``(v_parallel^2/2 + mu B) G`` in Hermite-Laguerre moments."""

    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim < 2:
        raise ValueError("coefficients must include laguerre and hermite axes")
    v2_part = 0.5 * jnp.einsum("mn,ln...->lm...", basis.hermite_v2, coefficients)
    mu_part = jnp.einsum("ln,nm...->lm...", basis.laguerre_x, coefficients)
    return v2_part + mu_part


def s_alpha_curvature_drift_profile(z, params: MomentRHSParams | None = None):
    """Return a compact s-alpha curvature profile for the moment discriminator."""

    params = MomentRHSParams() if params is None else params
    z = jnp.asarray(z)
    theta = 2.0 * jnp.pi * z
    return jnp.cos(theta) + jnp.asarray(params.magnetic_shear, dtype=z.dtype) * theta * jnp.sin(
        theta
    )


def moment_itg_drive_source(
    phi,
    b,
    ky,
    basis: HermiteLaguerreBasis,
    params: MomentRHSParams | None = None,
):
    """Project the adiabatic-electron ITG density/temperature drive to moments."""

    params = MomentRHSParams() if params is None else params
    phi = jnp.asarray(phi)
    if phi.ndim != 2:
        raise ValueError("phi must have shape (n_ky,n_z)")
    ky = _validate_ky(ky, phi.shape[0], phi.real.dtype)
    b = _broadcast_b_to_ky_z(b, phi.shape[0], phi.shape[1])
    gyro = gyroaverage_laguerre_coefficients(b, basis.n_laguerre)
    source = jnp.zeros(
        (basis.n_laguerre, basis.n_hermite, phi.shape[0], phi.shape[1]),
        dtype=jnp.result_type(phi, 1j),
    )

    density_projection = jnp.asarray(params.density_gradient, dtype=phi.real.dtype) * gyro
    perpendicular_temperature = jnp.einsum(
        "ln,nkz->lkz",
        basis.laguerre_x - jnp.eye(basis.n_laguerre, dtype=basis.laguerre_x.dtype),
        gyro,
    )
    source = source.at[:, 0].add(
        (density_projection + params.temperature_gradient * perpendicular_temperature) * phi
    )
    if basis.n_hermite > 2:
        parallel_temperature = (jnp.sqrt(jnp.asarray(2.0, dtype=phi.real.dtype)) / 2.0) * gyro
        source = source.at[:, 2].add(
            params.temperature_gradient * parallel_temperature * phi
        )

    ky_factor = ky.reshape(1, 1, phi.shape[0], 1)
    return 1j * jnp.asarray(params.drive_scale, dtype=phi.real.dtype) * ky_factor * source


def moment_linear_rhs(
    coefficients,
    basis: HermiteLaguerreBasis,
    ky,
    z,
    b,
    params: MomentRHSParams | None = None,
    *,
    phi=None,
    drift_profile=None,
):
    """Return the reduced Hermite-Laguerre linear RHS.

    The RHS is intended for the multi-``ky`` physics discriminator: it combines
    linked-chain Hermite streaming, an ITG field drive, a simple s-alpha
    curvature-drift surrogate, and a linked ``k_z`` hypercollision. The state
    layout is ``(laguerre, hermite, ky, z)``.
    """

    params = MomentRHSParams() if params is None else params
    coefficients = jnp.asarray(coefficients)
    if coefficients.shape[:2] != (basis.n_laguerre, basis.n_hermite):
        raise ValueError("coefficient leading axes must match the Hermite-Laguerre basis")
    if coefficients.ndim != 4:
        raise ValueError("coefficients must have shape (n_laguerre,n_hermite,n_ky,n_z)")
    ky = _validate_ky(ky, coefficients.shape[2], coefficients.real.dtype)
    z = jnp.asarray(z, dtype=coefficients.real.dtype)
    if z.shape != (coefficients.shape[3],):
        raise ValueError("z must have shape (n_z,)")
    b = _broadcast_b_to_ky_z(b, coefficients.shape[2], coefficients.shape[3])
    phi = moment_adiabatic_phi(coefficients, b, params) if phi is None else jnp.asarray(phi)
    if phi.shape != (coefficients.shape[2], coefficients.shape[3]):
        raise ValueError("phi must have shape (n_ky,n_z)")

    d_dz = apply_linked_grad_z(
        coefficients,
        n_links=params.n_links,
        z_periods=params.z_periods,
        axis=-1,
        dealias=params.dealias,
    )
    rhs = -jnp.asarray(params.streaming_scale, dtype=coefficients.real.dtype) * jnp.einsum(
        "mn,lnkz->lmkz",
        basis.hermite_v,
        d_dz,
    )
    rhs = rhs + moment_itg_drive_source(phi, b, ky, basis, params)

    if params.include_curvature_drift:
        drift = (
            s_alpha_curvature_drift_profile(z, params)
            if drift_profile is None
            else jnp.asarray(drift_profile, dtype=coefficients.real.dtype)
        )
        if drift.shape == (coefficients.shape[3],):
            drift = jnp.broadcast_to(drift[None, :], (coefficients.shape[2], coefficients.shape[3]))
        elif drift.shape != (coefficients.shape[2], coefficients.shape[3]):
            raise ValueError("drift_profile must have shape (n_z,) or (n_ky,n_z)")
        energy = moment_energy_multiply(coefficients, basis)
        drift_factor = (
            -1j
            * jnp.asarray(params.drift_scale, dtype=coefficients.real.dtype)
            * ky.reshape(1, 1, coefficients.shape[2], 1)
            * drift.reshape(1, 1, coefficients.shape[2], coefficients.shape[3])
        )
        rhs = rhs + drift_factor * energy

    if params.include_hypercollision:
        rhs = rhs + apply_kz_hypercollision(
            coefficients,
            nu_hyper_m=params.nu_hyper_m,
            p_hyper_m=params.p_hyper_m,
            vt=params.vt,
            gradpar_abs=params.gradpar_abs,
            n_links=params.n_links,
            z_periods=params.z_periods,
            hermite_axis=1,
            parallel_axis=-1,
            dealias=params.dealias,
        )
    return rhs


def kz_hypercollision_prefactor(
    n_hermite: int,
    *,
    nu_hyper_m: float = 1.0,
    p_hyper_m: int | None = None,
    vt: float = 1.0,
    gradpar_abs: float = 1.0,
    dtype: str = "float64",
):
    """Return the linked-``k_z`` hypercollision prefactor."""

    if n_hermite < 1:
        raise ValueError("n_hermite must be at least 1")
    if vt < 0:
        raise ValueError("vt must be non-negative")
    if gradpar_abs < 0:
        raise ValueError("gradpar_abs must be non-negative")

    real_dtype = jnp.dtype(dtype)
    p = max(1, min(20, n_hermite // 2)) if p_hyper_m is None else int(p_hyper_m)
    if p < 1:
        raise ValueError("p_hyper_m must be positive")
    if n_hermite <= 3:
        return jnp.asarray(0.0, dtype=real_dtype)

    m_max = jnp.asarray(n_hermite - 1, dtype=real_dtype)
    p_value = jnp.asarray(p, dtype=real_dtype)
    return (
        jnp.asarray(nu_hyper_m, dtype=real_dtype)
        * (p_value + 0.5)
        / (m_max ** (p_value + 0.5))
        * jnp.asarray(2.3 * vt * gradpar_abs, dtype=real_dtype)
    )


def kz_hypercollision_hermite_rates(
    n_hermite: int,
    *,
    nu_hyper_m: float = 1.0,
    p_hyper_m: int | None = None,
    vt: float = 1.0,
    gradpar_abs: float = 1.0,
    dtype: str = "float64",
):
    """Return Hermite damping rates for the linked ``k_z`` hypercollision."""

    if n_hermite < 1:
        raise ValueError("n_hermite must be at least 1")
    real_dtype = jnp.dtype(dtype)
    p = max(1, min(20, n_hermite // 2)) if p_hyper_m is None else int(p_hyper_m)
    prefactor = kz_hypercollision_prefactor(
        n_hermite,
        nu_hyper_m=nu_hyper_m,
        p_hyper_m=p,
        vt=vt,
        gradpar_abs=gradpar_abs,
        dtype=dtype,
    )
    m = jnp.arange(n_hermite, dtype=real_dtype)
    rates = prefactor * m**jnp.asarray(p, dtype=real_dtype)
    return jnp.where(m > 2.0, rates, jnp.zeros_like(rates))


def apply_kz_hypercollision(
    coefficients,
    *,
    nu_hyper_m: float = 1.0,
    p_hyper_m: int | None = None,
    vt: float = 1.0,
    gradpar_abs: float = 1.0,
    n_links: int = 1,
    z_periods: float = 1.0,
    hermite_axis: int = 1,
    parallel_axis: int = -1,
    dealias: bool = False,
):
    """Return the linked ``k_z`` hypercollision RHS contribution.

    ``coefficients`` may use any axis order, but the default is the existing
    moment convention ``(laguerre, hermite, ..., z)``.  The operator first forms
    ``-nu_m G`` for Hermite modes ``m>2`` and then applies the linked
    pseudo-differential ``|k_z|`` operator along the parallel chain.
    """

    coefficients = jnp.asarray(coefficients)
    hermite_axis = _normalize_axis(hermite_axis, coefficients.ndim)
    parallel_axis = _normalize_axis(parallel_axis, coefficients.ndim)
    if hermite_axis == parallel_axis:
        raise ValueError("hermite_axis and parallel_axis must be different")

    n_hermite = coefficients.shape[hermite_axis]
    rates = kz_hypercollision_hermite_rates(
        n_hermite,
        nu_hyper_m=nu_hyper_m,
        p_hyper_m=p_hyper_m,
        vt=vt,
        gradpar_abs=gradpar_abs,
        dtype=str(jnp.asarray(coefficients.real).dtype),
    )
    rate_shape = [1] * coefficients.ndim
    rate_shape[hermite_axis] = n_hermite
    damped = -rates.reshape(rate_shape) * coefficients
    return apply_linked_abs_kz(
        damped,
        n_links=n_links,
        z_periods=z_periods,
        axis=parallel_axis,
        dealias=dealias,
    )


def _hermite_derivative_matrix(n_hermite: int):
    """Return the matrix representation of ``d/dv`` in the normalized Hermite basis."""

    matrix = np.zeros((n_hermite, n_hermite), dtype=float)
    for m in range(1, n_hermite):
        matrix[m - 1, m] = np.sqrt(m)
    return matrix


def _hermite_v_matrix(n_hermite: int):
    """Return the tridiagonal matrix representation of multiplication by ``v`` in the normalized Hermite basis."""

    matrix = np.zeros((n_hermite, n_hermite), dtype=float)
    for m in range(n_hermite):
        if m + 1 < n_hermite:
            matrix[m, m + 1] = np.sqrt(m + 1)
        if m - 1 >= 0:
            matrix[m, m - 1] = np.sqrt(m)
    return matrix


def _laguerre_x_matrix(n_laguerre: int):
    """Return the tridiagonal matrix representation of multiplication by ``x`` (mu) in the signed Laguerre basis."""

    matrix = np.zeros((n_laguerre, n_laguerre), dtype=float)
    for ell in range(n_laguerre):
        matrix[ell, ell] = 2 * ell + 1
        if ell + 1 < n_laguerre:
            matrix[ell, ell + 1] = ell + 1
        if ell - 1 >= 0:
            matrix[ell, ell - 1] = ell
    return matrix


def _shift_laguerre_coefficients(coefficients, offset: int):
    """Shift ``coefficients`` along the leading Laguerre axis by ``offset`` modes, zero-padding the vacated entries."""

    zeros = jnp.zeros_like(coefficients)
    if offset > 0:
        return jnp.concatenate([zeros[:offset], coefficients[:-offset]], axis=0)
    if offset < 0:
        return jnp.concatenate([coefficients[-offset:], zeros[offset:]], axis=0)
    return coefficients


def _normalize_axis(axis: int, ndim: int) -> int:
    """Return ``axis`` resolved to a non-negative index within ``[0, ndim)``, raising if out of bounds."""

    if ndim < 1:
        raise ValueError("expected at least one array dimension")
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError("axis out of bounds")
    return axis


def _broadcast_b_to_ky_z(b, n_ky: int, n_z: int):
    """Broadcast the gyroaverage argument ``b`` to shape ``(n_ky, n_z)`` from a scalar or a compatible partial shape."""

    b = jnp.asarray(b)
    if b.ndim == 0:
        return jnp.broadcast_to(b, (n_ky, n_z))
    if b.shape == (n_ky,):
        return jnp.broadcast_to(b[:, None], (n_ky, n_z))
    if b.shape == (n_z,):
        return jnp.broadcast_to(b[None, :], (n_ky, n_z))
    if b.shape == (n_ky, 1):
        return jnp.broadcast_to(b, (n_ky, n_z))
    if b.shape == (1, n_z):
        return jnp.broadcast_to(b, (n_ky, n_z))
    if b.shape != (n_ky, n_z):
        raise ValueError("b must be scalar or have shape (n_ky,), (n_z,), or (n_ky,n_z)")
    return b


def _validate_ky(ky, n_ky: int, dtype):
    """Cast ``ky`` to ``dtype`` and verify it has shape ``(n_ky,)``."""

    ky = jnp.asarray(ky, dtype=dtype)
    if ky.shape != (n_ky,):
        raise ValueError("ky must have shape (n_ky,)")
    return ky


def _safe_signed_denominator(denominator, floor):
    """Return ``denominator`` with its magnitude floored, preserving sign, to avoid division blow-up near zero."""

    denominator = jnp.asarray(denominator)
    floor = jnp.asarray(floor, dtype=denominator.dtype)
    sign = jnp.where(denominator.real < 0.0, -1.0, 1.0)
    return jnp.where(jnp.abs(denominator) < floor, sign * floor, denominator)


def gamma0_limit_error(b, n_laguerre: int):
    """Convenience diagnostic comparing truncated Laguerre polarization to ``Gamma_0``."""

    return truncated_gamma0_from_laguerre(b, n_laguerre) - gamma0(b)
