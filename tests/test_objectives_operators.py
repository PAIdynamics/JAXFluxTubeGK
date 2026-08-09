from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_circular_geometry,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    dense_eigensystem,
    dense_matrix_from_action,
    initial_value_growth_objectives,
    k_perp_squared,
    kperp2_weighted_average,
    linear_growth_objectives,
    linear_operator_action,
    mode_chain_mask,
    mode_structure_penalty,
    project_to_ky,
    project_to_mode_chain,
    weighted_quasilinear_proxy,
)


def _ion(**updates):
    base = dict(
        charge=1.0,
        mass=2.0,
        density=0.8,
        temperature=1.4,
        density_gradient=1.0,
        temperature_gradient=2.0,
    )
    base.update(updates)
    return SpeciesParams(**base)


def _cell_centered_parallel_grid(n_z: int = 6):
    z_min = -0.5 + 0.5 / n_z
    return build_parallel_grid(
        ParallelGridSpec(n_z=n_z, z_min=z_min, z_max=z_min + 1.0, topology="periodic")
    )


def test_mode_projection_helpers_and_dense_eigensystem():
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=5, n_ky=2, kx_max=1.0, ky_values=(0.0, 0.4), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    state = jnp.ones((3, 5, 2), dtype=jnp.complex128)

    chain_mask = mode_chain_mask(5, 2, connectivity)
    projected_ky = project_to_ky(state, 1)
    projected_chain = project_to_mode_chain(state, connectivity, ky_index=1)

    np.testing.assert_allclose(projected_ky[..., 0], 0.0, atol=0.0)
    np.testing.assert_allclose(projected_ky[..., 1], 1.0, atol=0.0)
    np.testing.assert_allclose(projected_chain[..., 0], 0.0, atol=0.0)
    np.testing.assert_allclose(
        projected_chain[..., 1],
        jnp.broadcast_to(chain_mask[None, :, 1], projected_chain[..., 1].shape),
    )

    matrix_reference = jnp.asarray([[1.0 + 0.0j, 0.2j], [0.0 + 0.0j, 3.0 + 0.0j]])
    matrix = dense_matrix_from_action(lambda vector: matrix_reference @ vector, jnp.zeros(2))
    eigenvalues, _eigenvectors = dense_eigensystem(matrix)

    np.testing.assert_allclose(matrix, matrix_reference, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.sort_complex(np.asarray(eigenvalues)), np.asarray([1.0, 3.0]))


def test_linear_operator_action_restricts_one_mode_chain():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.5, mu_max=1.0))
    parallel = _cell_centered_parallel_grid(4)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.5, ky_values=(0.0, 0.4), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    geometry = build_circular_geometry(parallel, GeometryScalarParams(q=1.2, shat=0.4, eps=0.18))
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        _ion(),
        electron_params=AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False),
    )
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    state = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape) / 100.0

    action = linear_operator_action(state, precompute, ky_index=1, connectivity=connectivity)
    mask = mode_chain_mask(fourier.kx.shape[0], fourier.ky.shape[0], connectivity)
    restricted_mask = jnp.zeros_like(mask).at[:, 1].set(mask[:, 1])

    assert action.shape == state.shape
    np.testing.assert_allclose(action * (1.0 - restricted_mask), 0.0, atol=1e-13)
    jitted = jax.jit(lambda values: linear_operator_action(values, precompute, ky_index=1))(state)
    assert jitted.shape == state.shape


def test_growth_objectives_shapes_values_and_penalties():
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=5, n_ky=2, kx_max=1.0, ky_values=(0.0, 0.4), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    weights = jnp.asarray([0.7, 1.1, 1.3])
    start = jnp.ones((3, 5, 2), dtype=jnp.complex128)
    start = start.at[:, [1, 3], 1].set(20.0 + 0.0j)
    gamma = jnp.asarray([0.2, -0.1])
    omega = jnp.asarray([0.5, -0.7])
    duration = 0.4
    end = start * jnp.exp((gamma - 1j * omega) * duration)[None, None, :]
    kperp2 = jnp.ones_like(start.real) * jnp.asarray([2.0, 4.0])[None, None, :]

    values = linear_growth_objectives(
        start,
        end,
        0.0,
        duration,
        kperp2,
        selected_ky=1,
        w_z=weights,
        connectivity=connectivity,
        active_mask=jnp.asarray([True, True]),
    )
    target = values.mode_structure

    assert values.growth_rate.shape == (2,)
    assert values.frequency.shape == (2,)
    assert values.kperp2_average.shape == (2,)
    np.testing.assert_allclose(values.growth_rate, gamma, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(values.frequency, omega, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(values.selected_growth_rate, gamma[1])
    np.testing.assert_allclose(values.max_growth_rate, gamma[0])
    np.testing.assert_allclose(values.kperp2_average, jnp.asarray([2.0, 4.0]))
    np.testing.assert_allclose(
        values.quasilinear_proxy,
        weighted_quasilinear_proxy(gamma, jnp.asarray([2.0, 4.0])),
    )
    np.testing.assert_allclose(
        mode_structure_penalty(target, target, w_z=weights, connectivity=connectivity),
        0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        kperp2_weighted_average(kperp2, end, w_z=weights, connectivity=connectivity),
        jnp.asarray([2.0, 4.0]),
    )


def test_mode_structure_penalty_removes_independent_ky_phases():
    target = jnp.asarray(
        [
            [[1.0 + 0.5j, 0.2 - 0.4j]],
            [[-0.3 + 0.7j, 1.2 + 0.1j]],
        ]
    )
    phases = jnp.exp(1j * jnp.asarray([0.7, -1.1]))
    observed = target * phases[None, None, :]

    aligned = mode_structure_penalty(observed, target, phase_align=True)
    direct = mode_structure_penalty(observed, target, phase_align=False)

    np.testing.assert_allclose(aligned, 0.0, atol=2.0e-15)
    assert direct > 0.1


def test_initial_value_growth_objective_gradients_on_reduced_grid():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.6, mu_max=1.2))
    parallel = _cell_centered_parallel_grid(5)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.5, ky_values=(0.0, 0.35), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    base_species = _ion()
    base_geometry = GeometryScalarParams(q=1.25, shat=0.5, eps=0.18)
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    initial_state = 0.01 * (jnp.cos(index / 7.0) + 1j * jnp.sin(index / 11.0))

    def objective(density_gradient, temperature_gradient, q, shat, geometry_scale):
        species = replace(
            base_species,
            density_gradient=density_gradient,
            temperature_gradient=temperature_gradient,
        )
        geometry = build_circular_geometry(
            parallel,
            replace(base_geometry, q=q, shat=shat),
        )
        geometry = replace(
            geometry,
            B=geometry.B * (1.0 + 0.01 * geometry_scale),
            D_y=geometry.D_y * (1.0 - 0.02 * geometry_scale),
            g_yy=geometry.g_yy * (1.0 + 0.03 * geometry_scale),
        )
        precompute = build_linear_residual_precompute(
            velocity,
            parallel,
            fourier,
            geometry,
            species,
            electron_params=electrons,
        )
        values = initial_value_growth_objectives(
            initial_state,
            precompute,
            0.01,
            2,
            k_perp_squared(geometry, fourier),
            selected_ky=1,
            w_z=geometry.w_z,
            connectivity=connectivity,
            softplus_temperature=0.1,
        )
        return values.selected_growth_rate + 0.01 * values.quasilinear_proxy

    args = (1.0, 2.0, 1.25, 0.5, 0.3)
    grads = jax.grad(objective, argnums=(0, 1, 2, 3, 4))(*args)
    for grad_value in grads:
        assert jnp.isfinite(grad_value)

    step = 1.0e-5
    for arg_index in (1, 2, 4):
        plus = list(args)
        minus = list(args)
        plus[arg_index] += step
        minus[arg_index] -= step
        finite_difference = (objective(*plus) - objective(*minus)) / (2.0 * step)
        np.testing.assert_allclose(
            grads[arg_index],
            finite_difference,
            rtol=2e-4,
            atol=2e-5,
        )
