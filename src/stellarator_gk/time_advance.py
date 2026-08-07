"""Fixed-step time advancement and linear growth diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .types import DerivativeBackend, ModeConnectivity, _PyTreeDataclass


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


@dataclass(frozen=True)
class AdaptiveTimeAdvanceResult:
    """Result of host-controlled adaptive integration with accepted step sizes."""

    state: object
    history: object
    times: object
    dt_history: object
    n_steps: int


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


def integrate_fixed_step(
    state,
    dt,
    n_steps: int,
    rhs_fn,
    *rhs_args,
    filter_fn=None,
    store_history: bool = True,
):
    """Advance ``state`` for ``n_steps`` fixed RK4 steps.

    The default path stores every RK4 snapshot with ``jax.lax.scan``.  Passing
    ``store_history=False`` uses ``jax.lax.fori_loop`` and stores only the
    initial/final endpoints in ``history`` for memory-sensitive objective runs.
    """

    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    state = jnp.asarray(state)
    dt = jnp.asarray(dt)

    if not store_history:
        return _integrate_fixed_step_endpoints(
            state, dt, n_steps, rhs_fn, *rhs_args, filter_fn=filter_fn
        )

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


def integrate_adaptive(
    state,
    final_time: float,
    rhs_fn,
    timestep_fn,
    *rhs_args,
    max_steps: int = 100_000,
    filter_fn=None,
    store_history: bool = True,
    compile_step: bool = False,
):
    """Advance with RK4 using a state-dependent accepted timestep.

    Timestep selection and termination are intentionally host controlled; each
    accepted RK4 step remains JAX-compatible. Use fixed-step integration when
    differentiating through the complete trajectory because adaptive accept
    decisions are nonsmooth.
    """

    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state = jnp.asarray(state)
    initial_state = state
    time = 0.0
    states = [state] if store_history else []
    times = [time]
    steps = []
    if compile_step:
        evaluate_timestep = jax.jit(lambda value, *args: timestep_fn(value, *args))
        advance = jax.jit(
            lambda value, dt, *args: rk4_step(
                value,
                dt,
                rhs_fn,
                *args,
                filter_fn=filter_fn,
            )
        )
    else:

        def evaluate_timestep(value, *args):
            return timestep_fn(value, *args)

        def advance(value, dt, *args):
            return rk4_step(
                value,
                dt,
                rhs_fn,
                *args,
                filter_fn=filter_fn,
            )

    while time < final_time and len(steps) < max_steps:
        proposed = float(evaluate_timestep(state, *rhs_args))
        if not np.isfinite(proposed) or proposed <= 0.0:
            raise ValueError("timestep_fn must return a finite positive value")
        dt = min(proposed, final_time - time)
        state = advance(state, dt, *rhs_args)
        time += dt
        steps.append(dt)
        times.append(time)
        if store_history:
            states.append(state)
    if time < final_time:
        raise RuntimeError(f"adaptive integration exceeded max_steps={max_steps}")
    history = jnp.stack(states) if store_history else jnp.stack((initial_state, state))
    return AdaptiveTimeAdvanceResult(
        state=state,
        history=history,
        times=jnp.asarray(times),
        dt_history=jnp.asarray(steps),
        n_steps=len(steps),
    )


def semi_lagrangian_mirror_step(
    state,
    dt,
    vpar,
    mirror_coefficient,
    *,
    interpolation: str = "linear",
):
    """Advance ``dg/dt = a(mu,z) dg/dvpar`` by characteristic tracing.

    The distribution must use ``(vpar,mu,z,kx,ky)`` ordering and the parallel
    velocity nodes must be uniformly spaced. Values traced beyond the velocity
    domain use a zero-incoming boundary condition.
    """

    state = jnp.asarray(state)
    vpar = jnp.asarray(vpar)
    coefficient = jnp.asarray(mirror_coefficient)
    if coefficient.ndim == 3 and coefficient.shape[0] == 1:
        coefficient = coefficient[0]
    if state.ndim != 5:
        raise ValueError("state must have shape (vpar,mu,z,kx,ky)")
    if coefficient.shape != state.shape[1:3]:
        raise ValueError("mirror_coefficient must have shape (mu,z)")
    if vpar.ndim != 1 or vpar.shape[0] != state.shape[0] or vpar.shape[0] < 2:
        raise ValueError("vpar must be one-dimensional and match the state")
    if interpolation not in ("linear", "cubic", "stella_cubic"):
        raise ValueError("interpolation must be 'linear', 'cubic', or 'stella_cubic'")

    spacing = vpar[1] - vpar[0]
    displacement = jnp.asarray(dt, dtype=vpar.dtype) * coefficient[None, :, :] / spacing
    source_index = jnp.arange(vpar.shape[0], dtype=vpar.dtype)[:, None, None] + displacement
    lower = jnp.floor(source_index).astype(jnp.int32)
    fraction = source_index - lower
    upper = lower + 1
    mu_index = jnp.arange(state.shape[1])[None, :, None]
    z_index = jnp.arange(state.shape[2])[None, None, :]

    def values_at(index):
        values = state[jnp.clip(index, 0, state.shape[0] - 1), mu_index, z_index, :, :]
        valid = (index >= 0) & (index < state.shape[0])
        return jnp.where(valid[..., None, None], values, 0.0)

    fraction = fraction[..., None, None]
    if interpolation == "linear":
        return (1.0 - fraction) * values_at(lower) + fraction * values_at(upper)

    if interpolation == "stella_cubic":
        # stella traces characteristics with an integer shift truncated toward
        # zero, then uses a direction-aware four-point Lagrange stencil.  Its
        # outgoing point is deliberately linear when the integer shift is zero;
        # incoming points use zero ghost cells.  Keeping this as an explicit
        # provider convention avoids silently changing the generic cubic path.
        shift = jnp.trunc(displacement).astype(jnp.int32)
        direction = jnp.where(displacement >= 0.0, 1, -1).astype(jnp.int32)
        location = jnp.abs(displacement - shift)
        target = jnp.arange(vpar.shape[0], dtype=jnp.int32)[:, None, None]
        base = target + shift
        location_5d = location[..., None, None]
        w0 = -location_5d * (location_5d - 2.0) * (location_5d - 1.0) / 6.0
        w1 = 3.0 * (location_5d - 2.0) * (location_5d - 1.0) * (location_5d + 1.0) / 6.0
        w2 = -3.0 * location_5d * (location_5d - 2.0) * (location_5d + 1.0) / 6.0
        w3 = location_5d * (location_5d - 1.0) * (location_5d + 1.0) / 6.0
        cubic = (
            w0 * values_at(base - direction)
            + w1 * values_at(base)
            + w2 * values_at(base + direction)
            + w3 * values_at(base + 2 * direction)
        )
        linear = (1.0 - location_5d) * values_at(base) + location_5d * values_at(base + direction)
        outgoing = (shift == 0) & (target == jnp.where(direction > 0, 0, vpar.shape[0] - 1))
        return jnp.where(outgoing[..., None, None], linear, cubic)

    left = values_at(lower - 1)
    center_left = values_at(lower)
    center_right = values_at(lower + 1)
    right = values_at(lower + 2)
    weight_left = -fraction * (fraction - 1.0) * (fraction - 2.0) / 6.0
    weight_center_left = (fraction + 1.0) * (fraction - 1.0) * (fraction - 2.0) / 2.0
    weight_center_right = -((fraction + 1.0) * fraction * (fraction - 2.0) / 2.0)
    weight_right = (fraction + 1.0) * fraction * (fraction - 1.0) / 6.0
    return (
        weight_left * left
        + weight_center_left * center_left
        + weight_center_right * center_right
        + weight_right * right
    )


def integrate_fixed_step_split_mirror(
    state,
    dt,
    n_steps: int,
    rhs_fn,
    vpar,
    mirror_coefficient,
    *rhs_args,
    mirror_interpolation: str = "linear",
    parallel_streaming_propagator=None,
    parallel_response_step_fn=None,
    collision_step_fn=None,
    parallel_response_splitting: str = "strang",
    explicit_scheme: str = "rk4",
    filter_fn=None,
    store_history: bool = True,
):
    """Advance with Strang-split semi-Lagrangian mirror characteristics."""

    if n_steps < 0:
        raise ValueError("n_steps must be nonnegative")
    state = jnp.asarray(state)
    dt = jnp.asarray(dt)
    if parallel_streaming_propagator is not None and parallel_response_step_fn is not None:
        raise ValueError("choose either a streaming propagator or a coupled response step")
    if parallel_response_splitting not in ("strang", "after", "stella_after"):
        raise ValueError("parallel_response_splitting must be 'strang', 'after', or 'stella_after'")
    if explicit_scheme not in ("rk3", "rk4"):
        raise ValueError("explicit_scheme must be 'rk3' or 'rk4'")

    def parallel_step(value):
        if parallel_response_step_fn is not None:
            return parallel_response_step_fn(value)
        if parallel_streaming_propagator is not None:
            return implicit_parallel_streaming_step(value, parallel_streaming_propagator)
        return value

    def collision_step(value):
        return value if collision_step_fn is None else collision_step_fn(value)

    def step(value):
        if parallel_response_splitting == "stella_after":
            value = _ssp_rk3_step(value, dt, rhs_fn, *rhs_args, filter_fn=filter_fn)
            value = semi_lagrangian_mirror_step(
                value,
                dt,
                vpar,
                mirror_coefficient,
                interpolation=mirror_interpolation,
            )
            return collision_step(parallel_step(value))
        value = semi_lagrangian_mirror_step(
            value,
            0.5 * dt,
            vpar,
            mirror_coefficient,
            interpolation=mirror_interpolation,
        )
        if parallel_response_splitting == "strang":
            value = parallel_step(value)
        value = (
            _ssp_rk3_step(value, dt, rhs_fn, *rhs_args, filter_fn=filter_fn)
            if explicit_scheme == "rk3"
            else rk4_step(value, dt, rhs_fn, *rhs_args, filter_fn=filter_fn)
        )
        value = parallel_step(value)
        value = collision_step(value)
        return semi_lagrangian_mirror_step(
            value,
            0.5 * dt,
            vpar,
            mirror_coefficient,
            interpolation=mirror_interpolation,
        )

    if store_history:

        def body(carry, _index):
            next_state = step(carry)
            return next_state, next_state

        final_state, states = jax.lax.scan(body, state, jnp.arange(n_steps))
        history = jnp.concatenate([state[None, ...], states], axis=0)
        times = dt * jnp.arange(n_steps + 1, dtype=jnp.result_type(dt, jnp.float64))
    else:
        final_state = jax.lax.fori_loop(0, n_steps, lambda _index, value: step(value), state)
        history = jnp.stack([state, final_state], axis=0)
        times = jnp.asarray([0.0, dt * n_steps], dtype=jnp.result_type(dt, jnp.float64))
    return TimeAdvanceResult(
        state=final_state,
        history=history,
        times=times,
        dt=dt,
        n_steps=int(n_steps),
    )


def _ssp_rk3_step(state, dt, rhs_fn, *rhs_args, filter_fn=None):
    """Advance one explicit step with stella's three-stage SSP RK3 formula."""

    first_rhs = rhs_fn(state, *rhs_args)
    first = state + dt * first_rhs
    second_rhs = rhs_fn(first, *rhs_args)
    second = first + dt * second_rhs
    third_rhs = rhs_fn(second, *rhs_args)
    result = state / 3.0 + 0.5 * first + (second + dt * third_rhs) / 6.0
    return result if filter_fn is None else filter_fn(result)


def build_implicit_parallel_streaming_propagator(D_z, parallel_coefficient, dt):
    """Build batched implicit-midpoint propagators for ``-c(v,z) d/dz``.

    ``dt`` is the duration of one application. The returned array has shape
    ``(n_vpar,n_z,n_z)`` and remains differentiable with respect to the
    derivative matrix and streaming coefficient.
    """

    derivative = jnp.asarray(D_z)
    coefficient = jnp.asarray(parallel_coefficient)
    if coefficient.ndim == 3 and coefficient.shape[0] == 1:
        coefficient = coefficient[0]
    if derivative.ndim != 2 or derivative.shape[0] != derivative.shape[1]:
        raise ValueError("D_z must be square")
    if coefficient.ndim != 2 or coefficient.shape[1] != derivative.shape[0]:
        raise ValueError("parallel_coefficient must have shape (vpar,z)")

    operator = -coefficient[:, :, None] * derivative[None, :, :]
    identity = jnp.eye(derivative.shape[0], dtype=jnp.result_type(derivative, coefficient))
    half_dt = 0.5 * jnp.asarray(dt, dtype=identity.dtype)
    left = identity[None, :, :] - half_dt * operator
    right = identity[None, :, :] + half_dt * operator
    return jnp.linalg.solve(left, right)


def implicit_parallel_streaming_step(state, propagator):
    """Apply per-vpar parallel propagators to a single-species distribution."""

    state = jnp.asarray(state)
    propagator = jnp.asarray(propagator)
    if state.ndim != 5:
        raise ValueError("state must have shape (vpar,mu,z,kx,ky)")
    if propagator.shape != (state.shape[0], state.shape[2], state.shape[2]):
        raise ValueError("propagator must have shape (vpar,z,z)")
    return jnp.einsum("vij,vmjxy->vmixy", propagator, state)


def _integrate_fixed_step_endpoints(state, dt, n_steps: int, rhs_fn, *rhs_args, filter_fn=None):
    def body(_index, carry):
        return rk4_step(carry, dt, rhs_fn, *rhs_args, filter_fn=filter_fn)

    final_state = jax.lax.fori_loop(0, n_steps, body, state)
    time_dtype = jnp.result_type(dt, jnp.float64)
    times = jnp.asarray([0.0, dt * n_steps], dtype=time_dtype)
    return TimeAdvanceResult(
        state=final_state,
        history=jnp.stack([state, final_state], axis=0),
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
    return (
        jnp.log(jnp.maximum(amplitude_end, floor)) - jnp.log(jnp.maximum(amplitude_start, floor))
    ) / duration


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


def windowed_linear_growth_diagnostics(
    field_history,
    times,
    *,
    start_fraction: float = 0.5,
    start_index: int | None = None,
    end_index: int | None = None,
    w_z=None,
    connectivity: ModeConnectivity | None = None,
    amplitude_floor: float = 1.0e-300,
) -> LinearGrowthDiagnostics:
    """Fit per-``ky`` growth rates from a time window of field snapshots.

    The endpoint diagnostic is exact for a single exponential but is noisy for
    benchmark runs with transient phase mixing.  This helper fits
    ``log(amplitude)`` against time over a late window, matching the averaging
    convention used by GKW/GX-style growth-rate benchmarks.
    """

    history = jnp.asarray(field_history)
    times = jnp.asarray(times)
    if history.ndim != 4:
        raise ValueError("field_history must have shape (n_time,n_z,n_kx,n_ky)")
    if times.ndim != 1 or times.shape[0] != history.shape[0]:
        raise ValueError("times must be one-dimensional and match field_history length")
    if history.shape[0] < 2:
        raise ValueError("at least two field snapshots are required")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1)")

    n_time = history.shape[0]
    start = int(n_time * start_fraction) if start_index is None else int(start_index)
    end = n_time if end_index is None else int(end_index)
    start = max(0, min(start, n_time - 2))
    end = max(start + 2, min(end, n_time))
    window_history = history[start:end]
    window_times = times[start:end]

    amplitudes = jax.vmap(
        lambda field: mode_chain_amplitude(field, w_z=w_z, connectivity=connectivity)
    )(window_history)
    floor = jnp.asarray(amplitude_floor, dtype=amplitudes.dtype)
    log_amplitude = jnp.log(jnp.maximum(amplitudes, floor))
    centered_time = window_times - jnp.mean(window_times)
    centered_log = log_amplitude - jnp.mean(log_amplitude, axis=0)
    denominator = jnp.sum(centered_time**2)
    fitted_growth = jnp.sum(centered_time[:, None] * centered_log, axis=0) / denominator

    endpoint = linear_growth_diagnostics(
        window_history[0],
        window_history[-1],
        window_times[0],
        window_times[-1],
        w_z=w_z,
        connectivity=connectivity,
        amplitude_floor=amplitude_floor,
    )
    return LinearGrowthDiagnostics(
        amplitude_start=endpoint.amplitude_start,
        amplitude_end=endpoint.amplitude_end,
        growth_rate=fitted_growth,
        frequency=endpoint.frequency,
        mode_structure=endpoint.mode_structure,
    )


def build_modal_damping_filter(
    *,
    dt,
    velocity_grid=None,
    parallel_grid=None,
    vpar_rate: float = 0.0,
    mu_rate: float = 0.0,
    z_rate: float = 0.0,
    order: int = 4,
):
    """Return a post-step spectral modal damping filter for phase-space states.

    The rates are continuous-time rates.  One RK4 step applies
    ``exp(-dt * rate * (r/rmax)**order)`` in each enabled modal basis.  This is
    intended for benchmark-controlled recurrence damping, with all rates zero
    by default in solver paths.
    """

    if order <= 0:
        raise ValueError("order must be positive")
    if vpar_rate < 0.0 or mu_rate < 0.0 or z_rate < 0.0:
        raise ValueError("modal damping rates must be nonnegative")
    if vpar_rate and velocity_grid is None:
        raise ValueError("velocity_grid is required when vpar_rate is nonzero")
    if mu_rate and velocity_grid is None:
        raise ValueError("velocity_grid is required when mu_rate is nonzero")
    if z_rate and parallel_grid is None:
        raise ValueError("parallel_grid is required when z_rate is nonzero")

    dt = jnp.asarray(dt)
    vpar_factor = (
        _modal_damping_factor(
            velocity_grid.vpar.shape[0],
            dt,
            vpar_rate,
            order,
            DerivativeBackend.CHEBYSHEV.value,
        )
        if vpar_rate
        else None
    )
    mu_factor = (
        _modal_damping_factor(
            velocity_grid.mu.shape[0],
            dt,
            mu_rate,
            order,
            DerivativeBackend.CHEBYSHEV.value,
        )
        if mu_rate
        else None
    )
    z_factor = (
        _modal_damping_factor(
            parallel_grid.z.shape[0],
            dt,
            z_rate,
            order,
            parallel_grid.backend,
        )
        if z_rate
        else None
    )

    def filter_fn(state):
        out = jnp.asarray(state)
        vpar_axis, mu_axis, z_axis = _phase_space_axes(out.ndim)
        if vpar_factor is not None:
            out = _apply_modal_factor(
                out,
                vpar_axis,
                velocity_grid.vpar_modal_transform,
                velocity_grid.vpar_inverse_modal_transform,
                vpar_factor,
            )
        if mu_factor is not None:
            out = _apply_modal_factor(
                out,
                mu_axis,
                velocity_grid.mu_modal_transform,
                velocity_grid.mu_inverse_modal_transform,
                mu_factor,
            )
        if z_factor is not None:
            out = _apply_modal_factor(
                out,
                z_axis,
                parallel_grid.modal_transform,
                parallel_grid.inverse_modal_transform,
                z_factor,
            )
        return out

    return filter_fn


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
    """Return a conservative fixed-step estimate from coupled-operator row sums.

    When given a kinetic-electron ``LinearResidualPrecompute``, the estimate
    includes the quasineutrality-mediated field response as well as the
    distribution-side advection, drift, damping, and recurrence terms. Passing
    a bare RHS precompute, or the nonlocal zonal adiabatic solve, retains the
    distribution-only estimate.
    """

    rhs = precompute.rhs if hasattr(precompute, "rhs") else precompute
    dz_radius = jnp.max(jnp.sum(jnp.abs(rhs.D_z), axis=1))
    dv_radius = jnp.max(jnp.sum(jnp.abs(rhs.D_vpar), axis=1))
    parallel_radius = jnp.max(jnp.abs(rhs.parallel_streaming_coeff)) * dz_radius
    mirror_radius = jnp.max(jnp.abs(rhs.mirror_force_coeff)) * dv_radius
    drift_radius = jnp.max(jnp.abs(rhs.magnetic_drift_frequency))
    damping_radius = jnp.max(jnp.abs(rhs.perpendicular_damping))
    parallel_recurrence_radius = jnp.asarray(0.0)
    velocity_recurrence_radius = jnp.asarray(0.0)
    field_response_radius = jnp.asarray(0.0)
    collision_radius = jnp.asarray(0.0)
    distribution_gain = jnp.asarray(1.0)
    derivative_model = getattr(rhs, "parallel_derivative_model", "matrix")
    if derivative_model == "matrix" and hasattr(rhs, "parallel_recurrence_operator"):
        parallel_recurrence_radius = jnp.max(jnp.abs(rhs.parallel_recurrence_coeff)) * jnp.max(
            jnp.sum(jnp.abs(rhs.parallel_recurrence_operator), axis=1)
        )
    if derivative_model != "gkw_igh" and hasattr(rhs, "velocity_recurrence_operator"):
        velocity_recurrence_radius = jnp.max(jnp.abs(rhs.velocity_recurrence_coeff)) * jnp.max(
            jnp.sum(jnp.abs(rhs.velocity_recurrence_operator), axis=1)
        )
    if getattr(precompute, "field_model", None) == "kinetic":
        field_response_radius = _electrostatic_field_response_radius(precompute)
    if getattr(precompute, "field_model", None) == "electromagnetic":
        distribution_gain, field_response_radius = _electromagnetic_field_response_radius(
            precompute
        )
    if getattr(precompute, "collisions", None) is not None:
        if hasattr(precompute.collisions, "frequency"):
            # I-P is bounded by two in the weighted projection norm. The factor is
            # deliberately conservative for the infinity-norm estimate used here.
            collision_radius = 2.0 * jnp.max(jnp.abs(precompute.collisions.frequency))
        else:
            collision_radius = jnp.max(precompute.collisions.row_sum_bound)
    distribution_radius = (
        parallel_radius
        + mirror_radius
        + drift_radius
        + damping_radius
        + parallel_recurrence_radius
        + velocity_recurrence_radius
        + collision_radius
    )
    radius = distribution_gain * distribution_radius + field_response_radius
    return jnp.asarray(safety * rk4_radius) / jnp.maximum(radius, jnp.asarray(floor))


def _electrostatic_field_response_radius(precompute):
    """Bound the infinity norm of the algebraic-phi contribution to the RHS."""

    rhs = precompute.rhs
    field = precompute.field
    denominator = jnp.asarray(field.denominator)
    floor = jnp.asarray(field.denominator_floor, dtype=denominator.dtype)
    safe_denominator = jnp.where(
        jnp.abs(denominator) < floor,
        jnp.where(denominator < 0.0, -floor, floor),
        denominator,
    )
    # phi(z,kx,ky) is local in Fourier space and is a weighted velocity-space
    # reduction. This is its exact infinity-norm row sum before optional zonal
    # corrections, which are irrelevant to the non-zonal TEM path.
    phi_radius = jnp.sum(jnp.abs(field.phi_weight), axis=(0, 1, 2)) / jnp.abs(safe_denominator)

    bessel = jnp.asarray(rhs.flr_factors.bessel_j0)
    maxwellian = jnp.asarray(rhs.maxwellian)[..., None, None]
    drive = jnp.asarray(rhs.drive_factor)[..., None, None]
    ky = jnp.asarray(rhs.ky)[None, None, None, None, None, :]
    ey = jnp.asarray(rhs.E_y)[None, None, None, :, None, None]
    gyro_radius = jnp.abs(bessel[:, None, ...])

    equilibrium = jnp.max(jnp.abs(ey * ky * maxwellian * drive) * gyro_radius * phi_radius)
    drift = jnp.max(
        jnp.abs(
            rhs.charge_over_temperature[:, None, None, None, None, None]
            * rhs.magnetic_drift_frequency
            * maxwellian
        )
        * gyro_radius
        * phi_radius
    )

    dz_radius = jnp.max(jnp.sum(jnp.abs(rhs.D_z), axis=1))
    parallel = (
        jnp.max(
            jnp.abs(
                rhs.charge_over_temperature[:, None, None, None, None, None]
                * rhs.parallel_streaming_coeff[:, :, None, :, None, None]
                * maxwellian
            )
            * gyro_radius
        )
        * dz_radius
        * jnp.max(phi_radius)
    )
    return equilibrium + drift + parallel


def _electromagnetic_field_response_radius(precompute):
    """Bound mixed-state feedback from ``A_parallel``, ``phi``, and ``B_parallel``."""

    rhs = precompute.rhs
    fields = precompute.field
    ampere = fields.ampere
    perpendicular = fields.perpendicular

    ampere_denominator = jnp.asarray(ampere.denominator)
    ampere_floor = jnp.asarray(ampere.denominator_floor, dtype=ampere_denominator.dtype)
    safe_ampere = jnp.maximum(jnp.abs(ampere_denominator), ampere_floor)
    apar_radius = jnp.sum(jnp.abs(ampere.source_weight), axis=(0, 1, 2)) / safe_ampere
    physical_gain = jnp.max(
        1.0 + jnp.abs(ampere.g_to_f_factor) * apar_radius[None, None, None, ...]
    )

    perpendicular_denominator = jnp.asarray(perpendicular.denominator)
    perpendicular_floor = jnp.asarray(
        perpendicular.denominator_floor, dtype=perpendicular_denominator.dtype
    )
    safe_perpendicular = jnp.maximum(jnp.abs(perpendicular_denominator), perpendicular_floor)
    constant_mode = jnp.abs(ampere.kperp_squared) < perpendicular_floor
    phi_radius = (
        physical_gain
        * jnp.sum(jnp.abs(perpendicular.phi_weight), axis=(0, 1, 2))
        / safe_perpendicular
    )
    bpar_radius = (
        physical_gain
        * jnp.sum(jnp.abs(perpendicular.bpar_weight), axis=(0, 1, 2))
        / safe_perpendicular
    )
    phi_radius = jnp.where(constant_mode, 0.0, phi_radius)
    bpar_radius = jnp.where(constant_mode, 0.0, bpar_radius)

    bessel = jnp.asarray(rhs.flr_factors.bessel_j0)[:, None, ...]
    gyro_phi_radius = jnp.abs(bessel) * phi_radius[None, None, None, ...]
    gyro_bpar_radius = jnp.abs(perpendicular.bpar_chi_factor) * bpar_radius[None, None, None, ...]
    gyro_chi_radius = (
        gyro_phi_radius
        + jnp.abs(ampere.apar_chi_factor) * apar_radius[None, None, None, ...]
        + gyro_bpar_radius
    )

    maxwellian = jnp.asarray(rhs.maxwellian)[..., None, None]
    drive = jnp.asarray(rhs.drive_factor)[..., None, None]
    ky = jnp.asarray(rhs.ky)[None, None, None, None, None, :]
    ey = jnp.asarray(rhs.E_y)[None, None, None, :, None, None]
    equilibrium = jnp.max(jnp.abs(ey * ky * maxwellian * drive) * gyro_chi_radius)

    field_coefficient = jnp.abs(
        rhs.charge_over_temperature[:, None, None, None, None, None]
        * rhs.parallel_streaming_coeff[:, :, None, :, None, None]
        * maxwellian
    )
    dz_radius = jnp.max(jnp.sum(jnp.abs(rhs.D_z), axis=1))
    parallel = (
        jnp.max(field_coefficient)
        * dz_radius
        * (jnp.max(gyro_phi_radius) + jnp.max(gyro_bpar_radius))
    )

    drift_coefficient = jnp.abs(
        rhs.charge_over_temperature[:, None, None, None, None, None]
        * rhs.magnetic_drift_frequency
        * maxwellian
    )
    drift = jnp.max(drift_coefficient * (gyro_phi_radius + gyro_bpar_radius))
    return physical_gain, equilibrium + parallel + drift


def _apply_filter(state, filter_fn):
    if filter_fn is None:
        return state
    return filter_fn(state)


def _apply_modal_factor(values, axis: int, transform, inverse_transform, factor):
    coefficients = _apply_matrix_along_axis(transform, values, axis)
    shape = [1] * coefficients.ndim
    shape[axis] = factor.shape[0]
    coefficients = coefficients * factor.reshape(shape)
    return _apply_matrix_along_axis(inverse_transform, coefficients, axis)


def _apply_matrix_along_axis(matrix, values, axis: int):
    values = jnp.asarray(values)
    matrix = jnp.asarray(matrix)
    moved = jnp.moveaxis(values, axis, 0)
    transformed = jnp.tensordot(matrix, moved, axes=((1,), (0,)))
    return jnp.moveaxis(transformed, 0, axis)


def _modal_damping_factor(n: int, dt, rate: float, order: int, backend: str):
    relative = _relative_modal_index(n, backend)
    return jnp.exp(-jnp.asarray(dt) * jnp.asarray(rate) * relative**order)


def _relative_modal_index(n: int, backend: str):
    if n < 1:
        raise ValueError("modal dimension must be positive")
    indices = jnp.arange(n, dtype=jnp.float64)
    if backend == DerivativeBackend.FOURIER.value:
        denominator = jnp.maximum(n // 2, 1)
        return jnp.minimum(indices, n - indices) / denominator
    if backend == DerivativeBackend.CHEBYSHEV.value:
        return indices / jnp.maximum(n - 1, 1)
    raise ValueError(f"unsupported modal damping backend {backend!r}")


def _phase_space_axes(ndim: int):
    if ndim == 5:
        return 0, 1, 2
    if ndim == 6:
        return 1, 2, 3
    raise ValueError(
        "modal damping filter expects a 5D state "
        "(n_vpar,n_mu,n_z,n_kx,n_ky) or a 6D state with a leading species axis"
    )


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
