"""Grid, spectral-operator, and topology builders for Phase 2."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .types import (
    DerivativeBackend,
    FiniteDifferenceOperators,
    FourierGrid,
    FourierGridSpec,
    ModeConnectivity,
    ParallelGrid,
    ParallelGridSpec,
    VelocityGrid,
    VelocityGridSpec,
)


def build_velocity_grid(spec: VelocityGridSpec) -> VelocityGrid:
    """Build velocity nodes, weights, derivatives, and transforms."""

    if spec.backend == DerivativeBackend.CHEBYSHEV.value:
        vpar, w_vpar, d_vpar, t_vpar, ti_vpar = _chebyshev_collocation(
            spec.n_vpar, -spec.vpar_max, spec.vpar_max, spec.dtype
        )
        mu, w_mu, d_mu, t_mu, ti_mu = _chebyshev_collocation(
            spec.n_mu, 0.0, spec.mu_max, spec.dtype
        )
    elif spec.backend == DerivativeBackend.FINITE_DIFFERENCE.value:
        vpar, w_vpar, d_vpar, t_vpar, ti_vpar = _gkw_velocity_collocation(
            spec.n_vpar, spec.vpar_max, spec.dtype
        )
        mu, w_mu, d_mu, t_mu, ti_mu = _gkw_mu_collocation(
            spec.n_mu, spec.mu_max, spec.dtype
        )
    else:
        raise ValueError(f"unsupported velocity backend {spec.backend!r}")
    return VelocityGrid(
        vpar=vpar,
        mu=mu,
        w_vpar=w_vpar,
        w_mu=w_mu,
        D_vpar=d_vpar,
        D_mu=d_mu,
        vpar_modal_transform=t_vpar,
        vpar_inverse_modal_transform=ti_vpar,
        mu_modal_transform=t_mu,
        mu_inverse_modal_transform=ti_mu,
        backend=spec.backend,
    )


def build_parallel_grid(spec: ParallelGridSpec) -> ParallelGrid:
    """Build the parallel spectral grid for periodic or open field-line chains."""

    if spec.backend == DerivativeBackend.FOURIER.value:
        z, w_z, d_z, transform, inverse = _fourier_collocation(
            spec.n_z, spec.z_min, spec.z_max, spec.dtype
        )
    elif spec.backend == DerivativeBackend.CHEBYSHEV.value:
        z, w_z, d_z, transform, inverse = _chebyshev_collocation(
            spec.n_z, spec.z_min, spec.z_max, spec.dtype
        )
    else:
        raise ValueError(f"unsupported parallel backend {spec.backend!r}")
    return ParallelGrid(
        z=z,
        w_z=w_z,
        D_z=d_z,
        modal_transform=transform,
        inverse_modal_transform=inverse,
        backend=spec.backend,
        topology=spec.topology,
    )


def build_fourier_grid(spec: FourierGridSpec) -> FourierGrid:
    """Build centered radial and nonnegative binormal Fourier mode grids."""

    dtype = jnp.dtype(spec.dtype)
    ky = _build_ky_values(spec).astype(float)
    half = (spec.n_kx - 1) // 2
    if half == 0:
        kx = np.array([0.0])
    elif spec.use_gkw_shear_spacing and spec.n_ky > 1:
        spacing = abs(float(spec.q) * float(spec.shat) * ky[1] / (float(spec.eps) * spec.ikxspace))
        kx = np.arange(-half, half + 1, dtype=float) * spacing
    else:
        kx = np.linspace(-spec.kx_max, spec.kx_max, spec.n_kx)
    parseval = np.where(np.isclose(ky, 0.0), 1.0, 2.0)
    zero_ky_indices = np.where(np.isclose(ky, 0.0))[0]
    return FourierGrid(
        kx=jnp.asarray(kx, dtype=dtype),
        ky=jnp.asarray(ky, dtype=dtype),
        parseval=jnp.asarray(parseval, dtype=dtype),
        ixzero=int(np.argmin(np.abs(kx))),
        iyzero=int(zero_ky_indices[0]) if zero_ky_indices.size else -1,
        ikxspace=spec.ikxspace,
    )


def build_mode_connectivity(
    fourier_grid: FourierGrid, ikxspace: int | None = None, max_shift: int = 4
) -> ModeConnectivity:
    """Build static GKW-style mode labels and parallel-boundary kx connectivity."""

    if max_shift < 0:
        raise ValueError("max_shift must be nonnegative")
    spacing = fourier_grid.ikxspace if ikxspace is None else ikxspace
    if spacing < 1:
        raise ValueError("ikxspace must be at least 1")

    kx = np.asarray(fourier_grid.kx)
    ky = np.asarray(fourier_grid.ky)
    nkx = int(kx.shape[0])
    nky = int(ky.shape[0])
    ixzero = int(fourier_grid.ixzero)
    iyzero = int(fourier_grid.iyzero)

    mode_label = _build_mode_label(nkx, nky, iyzero, spacing)
    ixplus, ixminus = _build_ix_connectivity(mode_label, iyzero)
    kx_shift, valid_shift = _build_kx_shift_maps(ixplus, ixminus, iyzero, max_shift)

    return ModeConnectivity(
        mode_label=jnp.asarray(mode_label, dtype=jnp.int32),
        ixplus=jnp.asarray(ixplus, dtype=jnp.int32),
        ixminus=jnp.asarray(ixminus, dtype=jnp.int32),
        kx_shift=jnp.asarray(kx_shift, dtype=jnp.int32),
        valid_shift=jnp.asarray(valid_shift, dtype=bool),
        ixzero=ixzero,
        iyzero=iyzero,
        max_shift=max_shift,
    )


def build_finite_difference_operators(
    n: int, spacing: float, periodic: bool = False, dtype: str = "float64"
) -> FiniteDifferenceOperators:
    """Build fourth-order finite-difference fallback derivative matrices."""

    if n < 1:
        raise ValueError("n must be positive")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    d1 = np.zeros((n, n), dtype=float)
    d4 = np.zeros((n, n), dtype=float)
    d1_stencil = {(-2): 1.0 / (12.0 * spacing), (-1): -8.0 / (12.0 * spacing),
                  1: 8.0 / (12.0 * spacing), 2: -1.0 / (12.0 * spacing)}
    d4_stencil = {
        -2: 1.0 / spacing**4,
        -1: -4.0 / spacing**4,
        0: 6.0 / spacing**4,
        1: -4.0 / spacing**4,
        2: 1.0 / spacing**4,
    }
    _fill_stencil_matrix(d1, d1_stencil, periodic)
    _fill_stencil_matrix(d4, d4_stencil, periodic)
    return FiniteDifferenceOperators(
        D1=jnp.asarray(d1, dtype=jnp.dtype(dtype)),
        D4=jnp.asarray(d4, dtype=jnp.dtype(dtype)),
        n=n,
        spacing=float(spacing),
        periodic=bool(periodic),
    )


def _chebyshev_collocation(n: int, lower: float, upper: float, dtype: str):
    reference_nodes = -np.cos(np.pi * np.arange(n, dtype=float) / (n - 1))
    nodes = 0.5 * (upper + lower) + 0.5 * (upper - lower) * reference_nodes
    weights = 0.5 * (upper - lower) * _clenshaw_curtis_weights(n)
    derivative = _barycentric_derivative_matrix(nodes)
    inverse_modal = np.polynomial.chebyshev.chebvander(reference_nodes, n - 1)
    modal = np.linalg.inv(inverse_modal)
    jax_dtype = jnp.dtype(dtype)
    return (
        jnp.asarray(nodes, dtype=jax_dtype),
        jnp.asarray(weights, dtype=jax_dtype),
        jnp.asarray(derivative, dtype=jax_dtype),
        jnp.asarray(modal, dtype=jax_dtype),
        jnp.asarray(inverse_modal, dtype=jax_dtype),
    )


def _gkw_velocity_collocation(n: int, vpar_max: float, dtype: str):
    spacing = 2.0 * float(vpar_max) / n
    nodes = -float(vpar_max) + spacing * (np.arange(n, dtype=float) + 0.5)
    weights = np.full(n, spacing, dtype=float)
    operators = build_finite_difference_operators(n, spacing, periodic=False, dtype=dtype)
    identity = np.eye(n, dtype=float)
    jax_dtype = jnp.dtype(dtype)
    return (
        jnp.asarray(nodes, dtype=jax_dtype),
        jnp.asarray(weights, dtype=jax_dtype),
        operators.D1,
        jnp.asarray(identity, dtype=jax_dtype),
        jnp.asarray(identity, dtype=jax_dtype),
    )


def _gkw_mu_collocation(n: int, mu_max: float, dtype: str):
    vperp_max = np.sqrt(2.0 * float(mu_max))
    spacing = vperp_max / n
    vperp = spacing * (np.arange(n, dtype=float) + 0.5)
    nodes = 0.5 * vperp**2
    weights = 2.0 * np.pi * vperp * spacing
    derivative = _barycentric_derivative_matrix(nodes)
    identity = np.eye(n, dtype=float)
    jax_dtype = jnp.dtype(dtype)
    return (
        jnp.asarray(nodes, dtype=jax_dtype),
        jnp.asarray(weights, dtype=jax_dtype),
        jnp.asarray(derivative, dtype=jax_dtype),
        jnp.asarray(identity, dtype=jax_dtype),
        jnp.asarray(identity, dtype=jax_dtype),
    )


def _fourier_collocation(n: int, lower: float, upper: float, dtype: str):
    length = upper - lower
    nodes = lower + length * np.arange(n, dtype=float) / n
    weights = np.full(n, length / n, dtype=float)
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    identity = np.eye(n, dtype=complex)
    forward = np.fft.fft(identity, axis=0)
    inverse = np.fft.ifft(identity, axis=0)
    derivative = np.fft.ifft(1j * wavenumbers[:, None] * forward, axis=0)
    derivative = np.real_if_close(derivative, tol=1000).real
    real_dtype = jnp.dtype(dtype)
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64
    return (
        jnp.asarray(nodes, dtype=real_dtype),
        jnp.asarray(weights, dtype=real_dtype),
        jnp.asarray(derivative, dtype=real_dtype),
        jnp.asarray(forward, dtype=complex_dtype),
        jnp.asarray(inverse, dtype=complex_dtype),
    )


def _clenshaw_curtis_weights(n: int) -> np.ndarray:
    if n == 1:
        return np.array([2.0])
    order = n - 1
    theta = np.pi * np.arange(n) / order
    weights = np.zeros(n, dtype=float)
    interior = np.arange(1, order)
    values = np.ones(order - 1, dtype=float)
    if order % 2 == 0:
        weights[0] = 1.0 / (order**2 - 1.0)
        weights[-1] = weights[0]
        for k in range(1, order // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k**2 - 1.0)
        values -= np.cos(order * theta[interior]) / (order**2 - 1.0)
    else:
        weights[0] = 1.0 / order**2
        weights[-1] = weights[0]
        for k in range(1, (order + 1) // 2):
            values -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k**2 - 1.0)
    weights[interior] = 2.0 * values / order
    return weights


def _barycentric_derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    n = nodes.size
    diff = nodes[:, None] - nodes[None, :]
    weights = np.ones(n, dtype=float)
    for i in range(n):
        weights[i] = 1.0 / np.prod(nodes[i] - np.delete(nodes, i))
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = weights[j] / (weights[i] * diff[i, j])
        matrix[i, i] = -np.sum(matrix[i])
    return matrix


def _build_ky_values(spec: FourierGridSpec) -> np.ndarray:
    if spec.ky_values is not None:
        return np.asarray(spec.ky_values, dtype=float)
    if spec.n_ky == 1:
        return np.asarray([float(spec.ky_max)], dtype=float)
    return np.linspace(0.0, float(spec.ky_max), spec.n_ky)


def _build_mode_label(nkx: int, nky: int, iyzero: int, ikxspace: int) -> np.ndarray:
    mode_label = np.zeros((nkx, nky), dtype=np.int32)
    label = 1
    for iy in range(nky):
        if iy == iyzero:
            for ix in range(nkx):
                mode_label[ix, iy] = label
                label += 1
            continue
        for offset in range(ikxspace):
            chain = np.arange(offset, nkx, ikxspace)
            if chain.size == 0:
                continue
            mode_label[chain, iy] = label
            label += 1
    return mode_label


def _build_ix_connectivity(mode_label: np.ndarray, iyzero: int) -> tuple[np.ndarray, np.ndarray]:
    nkx, nky = mode_label.shape
    ixplus = -np.ones((nkx, nky), dtype=np.int32)
    ixminus = -np.ones((nkx, nky), dtype=np.int32)
    for iy in range(nky):
        if iy == iyzero:
            indices = np.arange(nkx, dtype=np.int32)
            ixplus[:, iy] = indices
            ixminus[:, iy] = indices
            continue
        for label in np.unique(mode_label[:, iy]):
            chain = np.where(mode_label[:, iy] == label)[0].astype(np.int32)
            chain.sort()
            if chain.size <= 1:
                continue
            ixplus[chain[:-1], iy] = chain[1:]
            ixminus[chain[1:], iy] = chain[:-1]
    return ixplus, ixminus


def _build_kx_shift_maps(
    ixplus: np.ndarray, ixminus: np.ndarray, iyzero: int, max_shift: int
) -> tuple[np.ndarray, np.ndarray]:
    nkx, nky = ixplus.shape
    offsets = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
    kx_shift = -np.ones((offsets.size, nkx, nky), dtype=np.int32)
    valid = np.zeros((offsets.size, nkx, nky), dtype=np.bool_)
    for offset_index, offset in enumerate(offsets):
        for ix in range(nkx):
            for iy in range(nky):
                target = ix
                ok = True
                if iy == iyzero:
                    kx_shift[offset_index, ix, iy] = ix
                    valid[offset_index, ix, iy] = True
                    continue
                for _ in range(abs(int(offset))):
                    next_target = ixplus[target, iy] if offset > 0 else ixminus[target, iy]
                    if next_target < 0:
                        ok = False
                        break
                    target = int(next_target)
                if ok:
                    kx_shift[offset_index, ix, iy] = target
                    valid[offset_index, ix, iy] = True
    return kx_shift, valid


def _fill_stencil_matrix(matrix: np.ndarray, stencil: dict[int, float], periodic: bool) -> None:
    n = matrix.shape[0]
    for row in range(n):
        for offset, value in stencil.items():
            col = row + offset
            if periodic:
                matrix[row, col % n] += value
            elif 0 <= col < n:
                matrix[row, col] += value
