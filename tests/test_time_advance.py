from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    FourierGridSpec,
    ParallelGridSpec,
    VelocityGridSpec,
    build_fourier_grid,
    build_implicit_parallel_streaming_propagator,
    build_mode_connectivity,
    build_modal_damping_filter,
    build_parallel_grid,
    build_velocity_grid,
    estimate_linear_cfl_dt,
    integrate_adaptive,
    integrate_fixed_step,
    integrate_fixed_step_split_mirror,
    implicit_parallel_streaming_step,
    linear_growth_diagnostics,
    mode_chain_amplitude,
    normalize_by_ky_amplitude,
    real_frequency,
    rk4_step,
    semi_lagrangian_mirror_step,
    windowed_linear_growth_diagnostics,
)


def test_rk4_zero_input_invariance_and_history_times():
    def rhs(state):
        return (0.3 - 0.2j) * state

    state = jnp.zeros((3,), dtype=jnp.complex128)
    result = integrate_fixed_step(state, 0.1, 5, rhs)

    np.testing.assert_allclose(rk4_step(state, 0.1, rhs), 0.0, atol=0.0)
    np.testing.assert_allclose(result.state, 0.0, atol=0.0)
    np.testing.assert_allclose(result.history, 0.0, atol=0.0)
    np.testing.assert_allclose(result.times, jnp.linspace(0.0, 0.5, 6))
    assert result.history.shape == (6, 3)


def test_adaptive_integrator_reaches_final_time_and_records_accepted_steps():
    def rhs(state, rate):
        return rate * state

    def timestep(state, _rate):
        return 0.07 if state < 1.15 else 0.03

    result = integrate_adaptive(jnp.asarray(1.0), 0.2, rhs, timestep, 1.0)

    np.testing.assert_allclose(result.times[-1], 0.2, atol=0.0)
    np.testing.assert_allclose(jnp.sum(result.dt_history), 0.2, atol=2.0e-15)
    np.testing.assert_allclose(result.state, np.exp(0.2), rtol=3.0e-7)
    assert result.history.shape[0] == result.n_steps + 1
    assert jnp.min(result.dt_history) < jnp.max(result.dt_history)


def test_compiled_adaptive_step_matches_uncompiled_path():
    def rhs(state, rate):
        return rate * state

    def timestep(_state, _rate):
        return jnp.asarray(0.037)

    arguments = (jnp.asarray(1.2), 0.2, rhs, timestep, jnp.asarray(-0.4))
    direct = integrate_adaptive(*arguments)
    compiled = integrate_adaptive(*arguments, compile_step=True)

    np.testing.assert_allclose(compiled.state, direct.state, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(compiled.history, direct.history, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(compiled.times, direct.times, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(compiled.dt_history, direct.dt_history, rtol=0.0, atol=0.0)


def test_adaptive_observer_retains_compact_strided_diagnostics_without_state_history():
    result = integrate_adaptive(
        jnp.asarray([1.0, 2.0]),
        0.2,
        lambda state: state,
        lambda _state: jnp.asarray(0.05),
        store_history=False,
        compile_step=True,
        observation_fn=lambda state: jnp.asarray([jnp.sum(state), jnp.linalg.norm(state)]),
        observation_stride=3,
    )

    assert result.history.shape == (2, 2)
    assert result.observations.shape == (3, 2)
    np.testing.assert_allclose(result.observation_times, [0.0, 0.15, 0.2], atol=2.0e-15)
    np.testing.assert_allclose(result.observations[0], [3.0, np.sqrt(5.0)])
    np.testing.assert_allclose(result.observations[-1, 0], 3.0 * np.exp(0.2), rtol=3e-7)


def test_adaptive_observer_rejects_invalid_stride():
    with np.testing.assert_raises_regex(ValueError, "observation_stride"):
        integrate_adaptive(
            jnp.asarray(1.0),
            0.1,
            lambda state: state,
            lambda _state: 0.1,
            observation_stride=0,
        )


def test_semi_lagrangian_mirror_step_translates_linear_profile():
    vpar = jnp.linspace(-2.0, 2.0, 9)
    state = vpar[:, None, None, None, None].astype(jnp.complex128)
    coefficient = jnp.asarray([[0.4]])

    advanced = semi_lagrangian_mirror_step(state, 0.25, vpar, coefficient)

    np.testing.assert_allclose(advanced[1:-1, 0, 0, 0, 0], vpar[1:-1] + 0.1)

    with_species_axis = semi_lagrangian_mirror_step(
        state,
        0.25,
        vpar,
        coefficient[None, ...],
    )
    np.testing.assert_allclose(with_species_axis, advanced)


def test_cubic_semi_lagrangian_mirror_step_translates_cubic_profile():
    vpar = jnp.linspace(-2.0, 2.0, 17)
    profile = vpar**3 - 0.4 * vpar**2 + 0.2 * vpar - 0.7
    state = profile[:, None, None, None, None].astype(jnp.complex128)
    coefficient = jnp.asarray([[0.3]])

    advanced = semi_lagrangian_mirror_step(
        state,
        0.2,
        vpar,
        coefficient,
        interpolation="cubic",
    )
    shifted = vpar + 0.06
    expected = shifted**3 - 0.4 * shifted**2 + 0.2 * shifted - 0.7

    np.testing.assert_allclose(advanced[2:-2, 0, 0, 0, 0], expected[2:-2], atol=2e-13)


def test_stella_cubic_uses_linear_outgoing_point_and_zero_ghosts():
    vpar = jnp.linspace(-2.0, 2.0, 9)
    state = (vpar**3)[:, None, None, None, None].astype(jnp.complex128)
    coefficient = jnp.asarray([[0.4]])

    advanced = semi_lagrangian_mirror_step(
        state, 0.25, vpar, coefficient, interpolation="stella_cubic"
    )

    fraction = 0.2
    np.testing.assert_allclose(
        advanced[0, 0, 0, 0, 0],
        (1.0 - fraction) * state[0, 0, 0, 0, 0] + fraction * state[1, 0, 0, 0, 0],
    )
    shifted = vpar + 0.1
    np.testing.assert_allclose(advanced[1:-2, 0, 0, 0, 0], shifted[1:-2] ** 3, atol=2e-13)


def test_semi_lagrangian_mirror_step_rejects_unknown_interpolation():
    with np.testing.assert_raises_regex(ValueError, "interpolation"):
        semi_lagrangian_mirror_step(
            jnp.zeros((4, 1, 1, 1, 1)),
            0.1,
            jnp.linspace(-1.0, 1.0, 4),
            jnp.zeros((1, 1)),
            interpolation="spline",
        )


def test_split_mirror_integrator_matches_characteristic_for_zero_rhs():
    vpar = jnp.linspace(-2.0, 2.0, 9)
    state = vpar[:, None, None, None, None].astype(jnp.complex128)
    coefficient = jnp.asarray([[0.2]])

    result = integrate_fixed_step_split_mirror(
        state,
        0.1,
        4,
        lambda value: jnp.zeros_like(value),
        vpar,
        coefficient,
        store_history=False,
    )

    np.testing.assert_allclose(
        result.state[1:5, 0, 0, 0, 0],
        vpar[1:5] + 0.08,
        atol=1.0e-6,
    )


def test_split_integrator_can_apply_parallel_response_once_after_explicit_stage():
    state = jnp.ones((2, 1, 1, 1, 1), dtype=jnp.complex128)
    result = integrate_fixed_step_split_mirror(
        state,
        0.1,
        1,
        lambda value: jnp.ones_like(value),
        jnp.asarray([-1.0, 1.0]),
        jnp.zeros((1, 1)),
        parallel_response_step_fn=lambda value: 2.0 * value,
        parallel_response_splitting="after",
        store_history=False,
    )
    np.testing.assert_allclose(result.state, 2.2)


def test_stella_split_uses_ssp_rk3_then_full_mirror_then_response():
    state = jnp.ones((5, 1, 1, 1, 1), dtype=jnp.complex128)
    calls = []
    collision_calls = []

    def response(value):
        calls.append(value)
        return 2.0 * value

    def collision(value):
        collision_calls.append(value)
        return 3.0 * value

    result = integrate_fixed_step_split_mirror(
        state,
        0.1,
        1,
        lambda value: value,
        jnp.linspace(-2.0, 2.0, 5),
        jnp.zeros((1, 1)),
        mirror_interpolation="stella_cubic",
        parallel_response_step_fn=response,
        collision_step_fn=collision,
        parallel_response_splitting="stella_after",
        explicit_scheme="rk3",
        store_history=False,
    )

    expected_rk3 = 1.0 + 0.1 + 0.1**2 / 2.0 + 0.1**3 / 6.0
    np.testing.assert_allclose(result.state, 6.0 * expected_rk3)
    assert len(calls) == 1
    assert len(collision_calls) == 1


def test_implicit_parallel_streaming_matches_midpoint_amplification():
    derivative = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    coefficient = jnp.asarray([[0.7, 0.7], [-0.4, -0.4]])
    dt = 0.08
    propagator = build_implicit_parallel_streaming_propagator(
        derivative,
        coefficient,
        dt,
    )
    state = jnp.arange(8, dtype=jnp.float64).reshape(2, 1, 2, 2, 1).astype(jnp.complex128)

    observed = implicit_parallel_streaming_step(state, propagator)
    expected = jnp.stack([propagator[index] @ state[index, 0, :, :, 0] for index in range(2)])

    np.testing.assert_allclose(observed[:, 0, :, :, 0], expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        jnp.linalg.det(propagator[0]),
        1.0,
        rtol=2e-13,
        atol=2e-13,
    )


def test_rk4_fixed_step_has_fourth_order_scalar_convergence():
    rate = 0.25 - 0.9j
    state0 = 1.2 + 0.4j
    final_time = 0.8

    def rhs(state, coefficient):
        return coefficient * state

    def solve(n_steps):
        dt = final_time / n_steps
        return integrate_fixed_step(state0, dt, n_steps, rhs, rate).state

    exact = state0 * jnp.exp(rate * final_time)
    error_coarse = jnp.abs(solve(20) - exact)
    error_fine = jnp.abs(solve(40) - exact)

    assert error_fine < error_coarse / 12.0
    assert error_fine < 2.0e-8


def test_fixed_step_can_store_only_endpoints_for_memory_sensitive_runs():
    rate = 0.2 - 0.3j
    state0 = jnp.asarray([1.0 + 0.1j, -0.2 + 0.4j])
    dt = 0.04
    n_steps = 6

    def rhs(state, coefficient):
        return coefficient * state

    full = integrate_fixed_step(state0, dt, n_steps, rhs, rate)
    endpoints = integrate_fixed_step(state0, dt, n_steps, rhs, rate, store_history=False)

    np.testing.assert_allclose(endpoints.state, full.state, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(endpoints.history[0], state0, rtol=0, atol=0)
    np.testing.assert_allclose(endpoints.history[-1], full.state, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(endpoints.times, jnp.asarray([0.0, dt * n_steps]))
    assert endpoints.history.shape == (2,) + state0.shape


def test_mode_chain_growth_frequency_and_normalization():
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=5, n_ky=2, kx_max=1.0, ky_values=(0.0, 0.4), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    weights = jnp.asarray([0.5, 1.0, 1.5, 2.0])
    start = jnp.zeros((4, 5, 2), dtype=jnp.complex128)
    start = start.at[:, fourier.ixzero, fourier.iyzero].set(1.0 + 0.0j)
    start = start.at[:, [0, 2, 4], 1].set(1.0 + 0.5j)
    start = start.at[:, [1, 3], 1].set(100.0 + 0.0j)

    gamma = jnp.asarray([0.2, -0.1])
    omega = jnp.asarray([0.7, -1.2])
    duration = 0.6
    factors = jnp.exp((gamma - 1j * omega) * duration)
    end = start * factors[None, None, :]

    amplitude = mode_chain_amplitude(start, w_z=weights, connectivity=connectivity)
    diagnostics = linear_growth_diagnostics(
        start,
        end,
        0.0,
        duration,
        w_z=weights,
        connectivity=connectivity,
    )

    expected_amplitude = jnp.asarray(
        [
            jnp.sqrt(jnp.sum(weights)),
            jnp.sqrt(jnp.sum(weights) * 3.0 * 1.25),
        ]
    )
    np.testing.assert_allclose(amplitude, expected_amplitude, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(diagnostics.growth_rate, gamma, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(diagnostics.frequency, omega, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        real_frequency(start, end, 0.0, duration, w_z=weights, connectivity=connectivity),
        omega,
        rtol=2e-13,
        atol=2e-13,
    )

    normalized = normalize_by_ky_amplitude(
        jnp.ones((2, 3, 4, 5, 2)),
        diagnostics.amplitude_end,
        log_normalization=jnp.asarray([0.1, -0.2]),
    )
    np.testing.assert_allclose(normalized.scale, diagnostics.amplitude_end)
    np.testing.assert_allclose(
        normalized.log_normalization,
        jnp.asarray([0.1, -0.2]) + jnp.log(diagnostics.amplitude_end),
    )
    np.testing.assert_allclose(normalized.state[..., 0], 1.0 / diagnostics.amplitude_end[0])
    np.testing.assert_allclose(normalized.state[..., 1], 1.0 / diagnostics.amplitude_end[1])


def test_windowed_growth_diagnostics_fits_late_time_amplitudes():
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=5, n_ky=2, kx_max=1.0, ky_values=(0.0, 0.4), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    weights = jnp.asarray([0.5, 1.0, 1.5])
    base = jnp.zeros((3, 5, 2), dtype=jnp.complex128)
    base = base.at[:, fourier.ixzero, fourier.iyzero].set(1.0 + 0.0j)
    base = base.at[:, [0, 2, 4], 1].set(0.7 - 0.2j)
    base = base.at[:, [1, 3], 1].set(50.0 + 0.0j)
    times = jnp.linspace(0.0, 1.2, 7)
    gamma = jnp.asarray([0.15, -0.08])
    omega = jnp.asarray([0.4, -0.9])
    factors = jnp.exp((gamma - 1j * omega)[None, :] * times[:, None])
    history = base[None, :, :, :] * factors[:, None, None, :]

    diagnostics = windowed_linear_growth_diagnostics(
        history,
        times,
        start_index=2,
        w_z=weights,
        connectivity=connectivity,
    )

    np.testing.assert_allclose(diagnostics.growth_rate, gamma, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(diagnostics.frequency, omega, rtol=2e-13, atol=2e-13)
    assert diagnostics.mode_structure.shape == base.shape


def test_windowed_growth_diagnostics_validates_inputs():
    history = jnp.ones((1, 2, 1, 1), dtype=jnp.complex128)
    times = jnp.asarray([0.0])

    try:
        windowed_linear_growth_diagnostics(history, times)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("short history should fail")


def test_modal_damping_filter_preserves_low_modes_and_damps_high_modes():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=5, n_mu=4, vpar_max=1.0, mu_max=1.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=6, z_min=0.0, z_max=1.0, topology="periodic")
    )
    filter_fn = build_modal_damping_filter(
        dt=0.2,
        velocity_grid=velocity,
        parallel_grid=parallel,
        vpar_rate=0.7,
        z_rate=0.5,
    )
    v_coefficients = jnp.zeros((5, 4, 6, 1, 1), dtype=jnp.complex128)
    v_coefficients = v_coefficients.at[0, :, :, :, :].set(2.0)
    v_coefficients = v_coefficients.at[-1, :, :, :, :].set(1.0)
    nodal_state = jnp.tensordot(
        velocity.vpar_inverse_modal_transform,
        v_coefficients,
        axes=((1,), (0,)),
    )

    filtered = filter_fn(nodal_state)
    filtered_coefficients = jnp.tensordot(
        velocity.vpar_modal_transform,
        filtered,
        axes=((1,), (0,)),
    )

    np.testing.assert_allclose(filtered_coefficients[0], 2.0, rtol=2e-13, atol=2e-13)
    assert jnp.max(jnp.abs(filtered_coefficients[-1])) < 1.0


def test_fixed_step_scan_is_jittable_and_differentiable():
    state0 = 0.7
    dt = 0.05
    n_steps = 8

    def rhs(state, rate):
        return rate * state

    @jax.jit
    def final_state(rate):
        return integrate_fixed_step(state0, dt, n_steps, rhs, rate).state

    def objective(rate):
        state = final_state(rate)
        return state**2

    rate = 0.4
    step = 1.0e-5
    grad_value = jax.grad(objective)(rate)
    finite_difference = (objective(rate + step) - objective(rate - step)) / (2.0 * step)

    np.testing.assert_allclose(final_state(rate), state0 * jnp.exp(rate * dt * n_steps), rtol=1e-8)
    np.testing.assert_allclose(grad_value, finite_difference, rtol=3e-5, atol=2e-7)


def test_linear_cfl_estimate_uses_rhs_coefficient_row_sums():
    rhs = SimpleNamespace(
        D_z=jnp.asarray([[1.0, -1.0], [2.0, -2.0]]),
        D_vpar=jnp.asarray([[0.5, -0.5], [1.5, -1.5]]),
        parallel_streaming_coeff=jnp.asarray([[[3.0, -1.0]]]),
        mirror_force_coeff=jnp.asarray([[[2.0, -4.0]]]),
        magnetic_drift_frequency=jnp.ones((1, 1, 1, 2, 1, 1)) * 0.7,
        perpendicular_damping=jnp.asarray([[0.3]]),
    )

    estimate = estimate_linear_cfl_dt(rhs, safety=1.0, rk4_radius=2.4)
    expected_radius = 3.0 * 4.0 + 4.0 * 3.0 + 0.7 + 0.3

    np.testing.assert_allclose(estimate, 2.4 / expected_radius)


def test_linear_cfl_estimate_includes_explicit_recurrence_operators():
    rhs = SimpleNamespace(
        D_z=jnp.asarray([[1.0, -1.0], [2.0, -2.0]]),
        D_vpar=jnp.asarray([[0.5, -0.5], [1.5, -1.5]]),
        parallel_streaming_coeff=jnp.asarray([[[3.0, -1.0]]]),
        mirror_force_coeff=jnp.asarray([[[2.0, -4.0]]]),
        magnetic_drift_frequency=jnp.ones((1, 1, 1, 2, 1, 1)) * 0.7,
        perpendicular_damping=jnp.asarray([[0.3]]),
        parallel_recurrence_operator=jnp.asarray([[2.0, -2.0], [-3.0, 3.0]]),
        parallel_recurrence_coeff=jnp.asarray([[[0.5, 0.25]]]),
        velocity_recurrence_operator=jnp.asarray([[1.0, -1.0], [-4.0, 4.0]]),
        velocity_recurrence_coeff=jnp.asarray([[[0.25, 0.1]]]),
        parallel_derivative_model="matrix",
    )

    estimate = estimate_linear_cfl_dt(rhs, safety=1.0, rk4_radius=2.4)
    expected_radius = 3.0 * 4.0 + 4.0 * 3.0 + 0.7 + 0.3 + 0.5 * 6.0 + 0.25 * 8.0

    np.testing.assert_allclose(estimate, 2.4 / expected_radius)


def test_linear_cfl_estimate_includes_quasineutrality_field_response():
    rhs = SimpleNamespace(
        D_z=jnp.zeros((1, 1)),
        D_vpar=jnp.zeros((1, 1)),
        parallel_streaming_coeff=jnp.zeros((1, 1, 1)),
        mirror_force_coeff=jnp.zeros((1, 1, 1)),
        magnetic_drift_frequency=jnp.zeros((1, 1, 1, 1, 1, 1)),
        perpendicular_damping=jnp.zeros((1, 1)),
        ky=jnp.asarray([3.0]),
        E_y=jnp.asarray([2.0]),
        maxwellian=jnp.asarray([[[[5.0]]]]),
        drive_factor=jnp.asarray([[[[7.0]]]]),
        charge_over_temperature=jnp.asarray([1.0]),
        flr_factors=SimpleNamespace(bessel_j0=jnp.asarray([[[[[11.0]]]]])),
    )
    field = SimpleNamespace(
        phi_weight=jnp.asarray([[[[[[13.0]]]]]]),
        denominator=jnp.asarray([[[-17.0]]]),
        denominator_floor=1.0e-14,
    )

    estimate = estimate_linear_cfl_dt(
        SimpleNamespace(rhs=rhs, field=field, field_model="kinetic"),
        safety=1.0,
        rk4_radius=2.4,
    )
    expected_radius = 2.0 * 3.0 * 5.0 * 7.0 * 11.0 * 13.0 / 17.0

    np.testing.assert_allclose(estimate, 2.4 / expected_radius)
