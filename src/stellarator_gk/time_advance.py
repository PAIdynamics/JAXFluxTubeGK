"""Fixed-step time advancement and linear growth diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from .types import ModeConnectivity, _PyTreeDataclass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TimeAdvanceResult(_PyTreeDataclass):
    """Result of a fixed-step time integration."""

    state: object
    history: object
    times: object
    dt: object
    n_steps: int

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("state", "history", "times", "dt")
    _static_fields: ClassVar[tuple[str, ...]] = ("n_steps",)

    def __post_init__(self):
        if self.n_steps < 0:
            raise ValueError("n_steps must be nonnegative")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearGrowthDiagnostics(_PyTreeDataclass):
    """Per-``ky`` linear growth, frequency, and mode-structure diagnostics."""

    amplitude_start: object
    amplitude_end: object
    growth_rate: object
    frequency: object
    mode_structure: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = (
        "amplitude_start",
        "amplitude_end",
        "growth_rate",
        "frequency",
        "mode_structure",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KyNormalizationResult(_PyTreeDataclass):
    """State and accumulated factors after per-``ky`` amplitude normalization."""

    state: object
    scale: object
    log_normalization: object

    _dynamic_fields: ClassVar[tuple[str, ...]] = ("state", "scale", "log_normalization")


def rk4_step(state, dt, rhs_fn, *rhs_args, filter_fn=None):
    """Advance one explicit fourth-order Runge--Kutta step."""

    state = jnp.asarray(state)
    dt = jnp.asarray(dt)
    k1 = rhs_fn(state, *rhs_args)
    k2 = rhs_fn(state + 0.5 * dt * k1, *rhs_args)
    k3 = rhs_fn(state + 0.5 * dt * k2, *rhs_args)
    k4 = rhs_fn(state + dt * k3, *rhs_args)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _apply_filter(next_state, filter_fn)


def integrate_fixed_step(state, dt, n_steps: int, rhs_fn, *rhs_args, filter_fn=None):
    """Advance ``state`` for ``n_steps`` fixed RK4 steps using ``jax.lax.scan``."""

    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    state = jnp.asarray(state)
    dt = jnp.asarray(dt)

    def body(carry, _index):
        next_state = rk4_step(carry, dt, rhs_fn, *rhs_args, filter_fn=filter_fn)
        return next_state, next_state

    final_state, states = jax.lax.scan(body, state, jnp.arange(n_steps))
    history = jnp.concatenate([state[None, ...], states], axis=0)
    time_dtype = jnp.result_type(dt, jnp.float64)
    times = dt * jnp.arange(n_steps + 1, dtype=time_dtype)
    return TimeAdvanceResult(
        state=final_state,
        history=history,
        times=times,
        dt=dt,
        n_steps=int(n_steps),
    )


def mode_chain_amplitude(field, w_z=None, connectivity: ModeConnectivity | None = None):
    """Return per-``ky`` potential amplitude on the connected ``kx`` chain."""

    field = jnp.asarray(field)
    if field.ndim != 3:
        raise ValueError("field must have shape (n_z,n_kx,n_ky)")
    weights = _z_weights(field, w_z)
    mask = _mode_chain_mask(field.shape[1], field.shape[2], connectivity, field.real.dtype)
    power = jnp.sum(weights[:, None, None] * mask[None, :, :] * jnp.abs(field) ** 2, axis=(0, 1))
    return jnp.sqrt(jnp.maximum(power, 0.0))


def growth_rate(amplitude_start, amplitude_end, t_start, t_end, amplitude_floor: float = 1.0e-300):
    """Return ``(log A_end - log A_start) / (t_end - t_start)``."""

    amplitude_start = jnp.asarray(amplitude_start)
    amplitude_end = jnp.asarray(amplitude_end)
    duration = jnp.asarray(t_end) - jnp.asarray(t_start)
    floor = jnp.asarray(amplitude_floor, dtype=amplitude_end.real.dtype)
    return (jnp.log(jnp.maximum(amplitude_end, floor)) - jnp.log(jnp.maximum(amplitude_start, floor))) / duration


def real_frequency(
    field_start,
    field_end,
    t_start,
    t_end,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
):
    """Return the per-``ky`` phase-rotation frequency from two mode structures."""

    field_start = jnp.asarray(field_start)
    field_end = jnp.asarray(field_end)
    if field_start.shape != field_end.shape:
        raise ValueError("field_start and field_end must have the same shape")
    if field_start.ndim != 3:
        raise ValueError("fields must have shape (n_z,n_kx,n_ky)")
    weights = _z_weights(field_start, w_z)
    mask = _mode_chain_mask(
        field_start.shape[1],
        field_start.shape[2],
        connectivity,
        field_start.real.dtype,
    )
    overlap = jnp.sum(
        weights[:, None, None] * mask[None, :, :] * jnp.conj(field_start) * field_end,
        axis=(0, 1),
    )
    duration = jnp.asarray(t_end) - jnp.asarray(t_start)
    return -jnp.angle(overlap) / duration


def linear_growth_diagnostics(
    field_start,
    field_end,
    t_start,
    t_end,
    *,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
    amplitude_floor: float = 1.0e-300,
) -> LinearGrowthDiagnostics:
    """Compute per-``ky`` amplitude, growth rate, frequency, and mode structure."""

    end = jnp.asarray(field_end)
    amplitude_start = mode_chain_amplitude(field_start, w_z=w_z, connectivity=connectivity)
    amplitude_end = mode_chain_amplitude(end, w_z=w_z, connectivity=connectivity)
    scale = jnp.maximum(amplitude_end, jnp.asarray(amplitude_floor, dtype=amplitude_end.dtype))
    mode_structure = end / scale[None, None, :]
    return LinearGrowthDiagnostics(
        amplitude_start=amplitude_start,
        amplitude_end=amplitude_end,
        growth_rate=growth_rate(
            amplitude_start,
            amplitude_end,
            t_start,
            t_end,
            amplitude_floor=amplitude_floor,
        ),
        frequency=real_frequency(
            field_start,
            end,
            t_start,
            t_end,
            w_z=w_z,
            connectivity=connectivity,
        ),
        mode_structure=mode_structure,
    )


def normalize_by_ky_amplitude(
    state,
    amplitude,
    *,
    log_normalization=None,
    amplitude_floor: float = 1.0e-300,
) -> KyNormalizationResult:
    """Normalize a phase-space state by per-``ky`` amplitudes and accumulate logs."""

    state = jnp.asarray(state)
    amplitude = jnp.asarray(amplitude)
    scale = jnp.maximum(amplitude, jnp.asarray(amplitude_floor, dtype=amplitude.dtype))
    normalized = state / scale.reshape((1,) * (state.ndim - 1) + (scale.shape[0],))
    log_scale = jnp.log(scale)
    if log_normalization is not None:
        log_scale = jnp.asarray(log_normalization) + log_scale
    return KyNormalizationResult(
        state=normalized,
        scale=scale,
        log_normalization=log_scale,
    )


def estimate_linear_cfl_dt(
    precompute,
    *,
    safety: float = 0.8,
    rk4_radius: float = 2.4,
    floor: float = 1.0e-14,
):
    """Return a conservative fixed-step estimate from RHS coefficient row sums."""

    rhs = precompute.rhs if hasattr(precompute, "rhs") else precompute
    dz_radius = jnp.max(jnp.sum(jnp.abs(rhs.D_z), axis=1))
    dv_radius = jnp.max(jnp.sum(jnp.abs(rhs.D_vpar), axis=1))
    parallel_radius = jnp.max(jnp.abs(rhs.parallel_streaming_coeff)) * dz_radius
    mirror_radius = jnp.max(jnp.abs(rhs.mirror_force_coeff)) * dv_radius
    drift_radius = jnp.max(jnp.abs(rhs.magnetic_drift_frequency))
    damping_radius = jnp.max(jnp.abs(rhs.perpendicular_damping))
    radius = parallel_radius + mirror_radius + drift_radius + damping_radius
    return jnp.asarray(safety * rk4_radius) / jnp.maximum(radius, jnp.asarray(floor))


def _apply_filter(state, filter_fn):
    if filter_fn is None:
        return state
    return filter_fn(state)


def _z_weights(field, w_z):
    if w_z is None:
        return jnp.ones((field.shape[0],), dtype=field.real.dtype)
    return jnp.asarray(w_z, dtype=field.real.dtype)


def _mode_chain_mask(n_kx: int, n_ky: int, connectivity: ModeConnectivity | None, dtype):
    if connectivity is None:
        return jnp.ones((n_kx, n_ky), dtype=dtype)
    labels = jnp.asarray(connectivity.mode_label)
    targets = labels[connectivity.ixzero, :]
    return (labels == targets[None, :]).astype(dtype)
