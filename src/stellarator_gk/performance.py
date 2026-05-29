"""CPU performance and memory-footprint helpers.

These utilities keep profiling and sizing outside traced physics kernels.  They
are intentionally lightweight: benchmark helpers are smoke-test tools, while
memory estimates make target-grid costs visible before allocating large arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import numpy as np

from .solver import linear_residual


@dataclass(frozen=True)
class LinearMemoryEstimate:
    """Approximate memory footprint for a linear electrostatic run."""

    state_shape: tuple[int, ...]
    field_shape: tuple[int, ...]
    state_bytes: int
    field_bytes: int
    coefficient_bytes: int
    history_bytes: int
    total_bytes: int
    n_steps: int
    store_history: bool
    complex_dtype: str
    real_dtype: str


@dataclass(frozen=True)
class LinearResidualBenchmark:
    """Reduced-grid JIT timing smoke-test result."""

    compile_seconds: float
    mean_execute_seconds: float
    best_execute_seconds: float
    repeats: int
    state_bytes: int
    coefficient_bytes: int


def pytree_nbytes(tree) -> int:
    """Return the total byte count of array-like PyTree leaves."""

    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "dtype") and hasattr(leaf, "size"):
            total += int(leaf.size) * np.dtype(leaf.dtype).itemsize
        elif np.isscalar(leaf):
            total += np.asarray(leaf).nbytes
    return int(total)


def estimate_linear_memory_from_dimensions(
    *,
    n_vpar: int,
    n_mu: int,
    n_z: int,
    n_kx: int,
    n_ky: int,
    n_species: int = 1,
    complex_dtype: str = "complex128",
    real_dtype: str | None = None,
    n_steps: int = 0,
    store_history: bool = True,
) -> LinearMemoryEstimate:
    """Estimate state, field, history, and coefficient storage from dimensions."""

    _validate_dimensions(n_species, n_vpar, n_mu, n_z, n_kx, n_ky, n_steps)
    complex_itemsize, real_itemsize, real_dtype_name = _dtype_sizes(complex_dtype, real_dtype)
    state_shape = _state_shape(n_species, n_vpar, n_mu, n_z, n_kx, n_ky)
    field_shape = (n_z, n_kx, n_ky)
    state_bytes = _num_elements(state_shape) * complex_itemsize
    field_bytes = _num_elements(field_shape) * complex_itemsize
    coefficient_bytes = _estimated_coefficient_bytes(
        n_species,
        n_vpar,
        n_mu,
        n_z,
        n_kx,
        n_ky,
        real_itemsize,
    )
    history_bytes = _history_state_count(n_steps, store_history) * state_bytes
    return LinearMemoryEstimate(
        state_shape=state_shape,
        field_shape=field_shape,
        state_bytes=int(state_bytes),
        field_bytes=int(field_bytes),
        coefficient_bytes=int(coefficient_bytes),
        history_bytes=int(history_bytes),
        total_bytes=int(state_bytes + field_bytes + coefficient_bytes + history_bytes),
        n_steps=int(n_steps),
        store_history=bool(store_history),
        complex_dtype=np.dtype(complex_dtype).name,
        real_dtype=real_dtype_name,
    )


def estimate_linear_memory_from_precompute(
    precomputed,
    distribution_shape: tuple[int, ...],
    *,
    complex_dtype: str | None = None,
    n_steps: int = 0,
    store_history: bool = True,
) -> LinearMemoryEstimate:
    """Estimate memory from an assembled Phase 7 precompute and state shape."""

    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    if len(distribution_shape) not in (5, 6):
        raise ValueError("distribution_shape must have 5 or 6 dimensions")
    rhs = precomputed.rhs if hasattr(precomputed, "rhs") else precomputed
    n_z = int(rhs.D_z.shape[0])
    n_kx, n_ky = (int(value) for value in rhs.perpendicular_damping.shape)
    inferred_dtype = getattr(rhs.magnetic_drift_frequency, "dtype", np.dtype("complex128"))
    complex_dtype = np.dtype(complex_dtype or _complex_dtype_for(inferred_dtype)).name
    complex_itemsize, real_itemsize, real_dtype_name = _dtype_sizes(complex_dtype, None)
    state_shape = tuple(int(value) for value in distribution_shape)
    field_shape = (n_z, n_kx, n_ky)
    state_bytes = _num_elements(state_shape) * complex_itemsize
    field_bytes = _num_elements(field_shape) * complex_itemsize
    coefficient_bytes = pytree_nbytes(precomputed)
    history_bytes = _history_state_count(n_steps, store_history) * state_bytes
    return LinearMemoryEstimate(
        state_shape=state_shape,
        field_shape=field_shape,
        state_bytes=int(state_bytes),
        field_bytes=int(field_bytes),
        coefficient_bytes=int(coefficient_bytes),
        history_bytes=int(history_bytes),
        total_bytes=int(state_bytes + field_bytes + coefficient_bytes + history_bytes),
        n_steps=int(n_steps),
        store_history=bool(store_history),
        complex_dtype=complex_dtype,
        real_dtype=real_dtype_name,
    )


def benchmark_linear_residual(
    distribution,
    precomputed,
    *,
    repeats: int = 3,
    block_until_ready: bool = True,
) -> LinearResidualBenchmark:
    """Compile and time a reduced-grid self-consistent residual evaluation."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    compiled = jax.jit(lambda state: linear_residual(state, precomputed=precomputed))
    start = perf_counter()
    result = compiled(distribution)
    if block_until_ready:
        result.block_until_ready()
    compile_seconds = perf_counter() - start

    execute_times = []
    for _ in range(repeats):
        start = perf_counter()
        result = compiled(distribution)
        if block_until_ready:
            result.block_until_ready()
        execute_times.append(perf_counter() - start)

    return LinearResidualBenchmark(
        compile_seconds=float(compile_seconds),
        mean_execute_seconds=float(np.mean(execute_times)),
        best_execute_seconds=float(np.min(execute_times)),
        repeats=int(repeats),
        state_bytes=int(np.asarray(distribution).size * np.asarray(distribution).dtype.itemsize),
        coefficient_bytes=pytree_nbytes(precomputed),
    )


def format_bytes(nbytes: int) -> str:
    """Format a byte count with binary units."""

    if nbytes < 0:
        raise ValueError("nbytes must be nonnegative")
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _validate_dimensions(n_species, n_vpar, n_mu, n_z, n_kx, n_ky, n_steps):
    for name, value in (
        ("n_species", n_species),
        ("n_vpar", n_vpar),
        ("n_mu", n_mu),
        ("n_z", n_z),
        ("n_kx", n_kx),
        ("n_ky", n_ky),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")


def _dtype_sizes(complex_dtype, real_dtype):
    complex_dtype = np.dtype(complex_dtype)
    if real_dtype is None:
        real_dtype = np.float64 if complex_dtype.itemsize == 16 else np.float32
    real_dtype = np.dtype(real_dtype)
    return complex_dtype.itemsize, real_dtype.itemsize, real_dtype.name


def _complex_dtype_for(dtype) -> str:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return dtype.name
    return "complex128" if dtype.itemsize >= 8 else "complex64"


def _state_shape(n_species, n_vpar, n_mu, n_z, n_kx, n_ky):
    shape = (n_vpar, n_mu, n_z, n_kx, n_ky)
    if n_species == 1:
        return shape
    return (n_species,) + shape


def _num_elements(shape):
    return int(np.prod(shape, dtype=np.int64))


def _history_state_count(n_steps: int, store_history: bool):
    return n_steps + 1 if store_history else 2


def _estimated_coefficient_bytes(n_species, n_vpar, n_mu, n_z, n_kx, n_ky, real_itemsize):
    operator_terms = n_z * n_z + n_vpar * n_vpar + n_ky + n_z + n_species + n_kx * n_ky
    flr_terms = 2 * n_species * n_mu * n_z * n_kx * n_ky
    flr_terms += 2 * n_species * n_z * n_kx * n_ky
    velocity_terms = 2 * n_species * n_vpar * n_mu * n_z
    streaming_terms = n_species * (n_vpar + n_mu) * n_z
    drift_terms = n_species * n_vpar * n_mu * n_z * n_kx * n_ky
    total_real_values = (
        operator_terms
        + flr_terms
        + velocity_terms
        + streaming_terms
        + drift_terms
    )
    return int(total_real_values * real_itemsize)
