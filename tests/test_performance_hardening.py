from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    benchmark_linear_residual,
    build_circular_geometry,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_parallel_grid,
    build_velocity_grid,
    estimate_linear_memory_from_dimensions,
    estimate_linear_memory_from_precompute,
    format_bytes,
    integrate_fixed_step,
    jitted_linear_residual,
    linear_residual,
    pytree_nbytes,
)


def _ion(**updates):
    base = dict(
        charge=1.0,
        mass=2.0,
        density=0.9,
        temperature=1.2,
        density_gradient=0.7,
        temperature_gradient=1.8,
    )
    base.update(updates)
    return SpeciesParams(**base)


def _small_problem(species=None):
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.4, mu_max=1.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=4, z_min=-0.5 + 0.5 / 4, z_max=0.5 + 0.5 / 4)
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.45, ky_values=(0.0, 0.35))
    )
    geometry = build_circular_geometry(
        parallel,
        GeometryScalarParams(q=1.25, shat=0.45, eps=0.16),
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        _ion() if species is None else species,
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=False,
        ),
    )
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    state = 0.01 * (jnp.cos(index / 5.0) + 1j * jnp.sin(index / 7.0))
    return velocity, parallel, fourier, geometry, precompute, state


def test_memory_estimates_scale_and_capture_history_savings():
    full_history = estimate_linear_memory_from_dimensions(
        n_vpar=16,
        n_mu=8,
        n_z=32,
        n_kx=5,
        n_ky=8,
        n_steps=20,
        store_history=True,
    )
    endpoints = estimate_linear_memory_from_dimensions(
        n_vpar=16,
        n_mu=8,
        n_z=32,
        n_kx=5,
        n_ky=8,
        n_steps=20,
        store_history=False,
    )
    target = estimate_linear_memory_from_dimensions(
        n_vpar=48,
        n_mu=16,
        n_z=96,
        n_kx=9,
        n_ky=16,
        n_steps=100,
        store_history=False,
    )

    assert full_history.state_shape == (16, 8, 32, 5, 8)
    assert full_history.field_shape == (32, 5, 8)
    assert full_history.history_bytes == 21 * full_history.state_bytes
    assert endpoints.history_bytes == 2 * endpoints.state_bytes
    assert endpoints.total_bytes < full_history.total_bytes
    assert target.total_bytes > endpoints.total_bytes
    assert format_bytes(1024) == "1.00 KiB"


def test_precompute_memory_estimate_and_jitted_residual_match_eager_path():
    *_unused, precompute, state = _small_problem()

    estimate = estimate_linear_memory_from_precompute(
        precompute,
        state.shape,
        n_steps=5,
        store_history=False,
    )
    eager = linear_residual(state, precomputed=precompute)
    compiled = jitted_linear_residual(state, precompute)

    assert estimate.coefficient_bytes == pytree_nbytes(precompute)
    assert estimate.history_bytes == 2 * estimate.state_bytes
    assert estimate.total_bytes > estimate.coefficient_bytes
    np.testing.assert_allclose(compiled, eager, rtol=2e-12, atol=2e-12)


def test_reduced_linear_residual_benchmark_smoke():
    *_unused, precompute, state = _small_problem()

    benchmark = benchmark_linear_residual(state, precompute, repeats=1)

    assert benchmark.repeats == 1
    assert benchmark.state_bytes == state.size * state.dtype.itemsize
    assert benchmark.coefficient_bytes == pytree_nbytes(precompute)
    assert np.isfinite(benchmark.compile_seconds)
    assert np.isfinite(benchmark.mean_execute_seconds)
    assert benchmark.best_execute_seconds < 5.0


def test_jitted_no_history_objective_gradient_is_finite_and_stable():
    velocity, parallel, fourier, geometry, _precompute, state = _small_problem()
    base_species = _ion()
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)

    def objective(temperature_gradient):
        species = replace(base_species, temperature_gradient=temperature_gradient)
        precompute = build_linear_residual_precompute(
            velocity,
            parallel,
            fourier,
            geometry,
            species,
            electron_params=electrons,
        )
        result = integrate_fixed_step(
            state,
            0.01,
            2,
            lambda distribution, pc: linear_residual(distribution, precomputed=pc),
            precompute,
            store_history=False,
        )
        return jnp.real(jnp.vdot(result.state, result.state))

    value, grad_value = jax.jit(jax.value_and_grad(objective))(1.8)
    step = 1.0e-5
    finite_difference = (objective(1.8 + step) - objective(1.8 - step)) / (2.0 * step)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad_value)
    np.testing.assert_allclose(grad_value, finite_difference, rtol=5e-4, atol=5e-6)
