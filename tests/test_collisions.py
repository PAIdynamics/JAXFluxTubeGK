from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_conserving_bgk_precompute,
    build_fokker_planck_precompute,
    build_laguerre_legendre_collision_precompute,
    build_stella_laguerre_legendre_response,
    build_stella_laguerre_legendre_delta,
    build_stella_laguerre_legendre_driver,
    build_stella_laguerre_legendre_collision_precompute,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    collision_moments,
    conserving_bgk_collision,
    estimate_linear_cfl_dt,
    fokker_planck_collision,
    fokker_planck_conserved_moments,
    fokker_planck_pairwise_components,
    fokker_planck_reciprocal_components,
    laguerre_legendre_collision,
    laguerre_legendre_collision_components,
    laguerre_legendre_collision_components_from_moments,
    implicit_laguerre_legendre_collision,
    linear_residual,
    stella_laguerre_legendre_delta0,
)


def _collision_setup(n_species=1):
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=8, n_mu=6, vpar_max=3.5, mu_max=5.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=7, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    geometry = build_s_alpha_geometry(parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18))
    ion = SpeciesParams(1.0, 1.0, 1.0, 1.0, 2.2, 0.0)
    electron = SpeciesParams(-1.0, 0.01, 1.0, 1.0, 2.2, 6.9)
    species = ion if n_species == 1 else (ion, electron)
    return velocity, parallel, geometry, species


def test_conserving_bgk_preserves_discrete_density_momentum_and_energy():
    velocity, parallel, geometry, species = _collision_setup(n_species=2)
    precompute = build_conserving_bgk_precompute(
        velocity, geometry.B, species, frequency=(0.2, 0.7)
    )
    shape = (2, 8, 6, parallel.z.shape[0], 2, 3)
    state = jax.random.normal(jax.random.key(4), shape) + 1j * jax.random.normal(
        jax.random.key(5), shape
    )

    collision = jax.jit(conserving_bgk_collision)(state, precompute)
    moments = collision_moments(collision, precompute)

    np.testing.assert_allclose(moments, 0.0, atol=2.0e-11, rtol=0.0)


def test_conserving_bgk_null_space_and_frequency_scaling():
    velocity, parallel, geometry, species = _collision_setup()
    precompute = build_conserving_bgk_precompute(velocity, geometry.B, species, frequency=0.4)
    coefficients = jnp.ones((1, parallel.z.shape[0], 2, 1, 3))
    state = jnp.einsum("sbvmz,szxyb->svmzxy", precompute.equilibrium_basis, coefficients)[0]

    np.testing.assert_allclose(conserving_bgk_collision(state, precompute), 0.0, atol=2.0e-12)

    probe = jnp.sin(velocity.vpar)[:, None, None, None, None] * jnp.ones_like(state)
    low = build_conserving_bgk_precompute(velocity, geometry.B, species, frequency=0.2)
    high = build_conserving_bgk_precompute(velocity, geometry.B, species, frequency=0.6)
    np.testing.assert_allclose(
        conserving_bgk_collision(probe, high),
        3.0 * conserving_bgk_collision(probe, low),
        atol=2.0e-12,
    )


def test_collision_operator_is_integrated_in_residual_and_cfl():
    velocity, parallel, geometry, species = _collision_setup()
    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,)))
    collisionless = build_linear_residual_precompute(velocity, parallel, fourier, geometry, species)
    collisional = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        collision_frequency=0.5,
    )
    shape = (8, 6, parallel.z.shape[0], 1, 1)
    state = jax.random.normal(jax.random.key(8), shape).astype(jnp.complex128)

    difference = linear_residual(state, precomputed=collisional) - linear_residual(
        state, precomputed=collisionless
    )
    expected = conserving_bgk_collision(state, collisional.collisions)

    np.testing.assert_allclose(difference, expected, atol=2.0e-11, rtol=2.0e-11)
    assert float(estimate_linear_cfl_dt(collisional)) < float(estimate_linear_cfl_dt(collisionless))


def test_collision_frequency_remains_differentiable():
    velocity, _parallel, geometry, species = _collision_setup()
    state = jnp.sin(velocity.vpar)[:, None, None, None, None] * jnp.ones(
        (8, 6, geometry.B.shape[0], 1, 1)
    )

    def objective(frequency):
        precompute = build_conserving_bgk_precompute(
            velocity, geometry.B, species, frequency=frequency
        )
        value = conserving_bgk_collision(state, precompute)
        return jnp.real(jnp.vdot(value, value))

    value = objective(0.3)
    derivative = jax.grad(objective)(0.3)
    np.testing.assert_allclose(derivative, 2.0 * value / 0.3, rtol=2.0e-10)


def test_fokker_planck_foundation_is_finite_jittable_and_differentiable():
    velocity, parallel, geometry, species = _collision_setup()
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    state = jax.random.normal(
        jax.random.key(31),
        (8, 6, parallel.z.shape[0], 1, 1),
    ).astype(jnp.complex128)

    def objective(frequency):
        precompute = build_fokker_planck_precompute(velocity, geometry.B, species, frequency)
        collision = jax.jit(fokker_planck_collision)(state, precompute)
        return jnp.real(jnp.vdot(collision, collision))

    value = objective(0.2)
    derivative = jax.grad(objective)(0.2)
    assert np.isfinite(value)
    np.testing.assert_allclose(derivative, 2.0 * value / 0.2, rtol=2.0e-9)


def test_fokker_planck_rejects_nonuniform_velocity_backend():
    velocity, _parallel, geometry, species = _collision_setup()
    with np.testing.assert_raises_regex(ValueError, "finite-difference"):
        build_fokker_planck_precompute(velocity, geometry.B, species, 0.1)


def test_fokker_planck_operator_is_integrated_in_residual_and_cfl():
    _velocity, parallel, geometry, species = _collision_setup()
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,)))
    collisionless = build_linear_residual_precompute(velocity, parallel, fourier, geometry, species)
    collisional = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        collision_frequency=0.2,
        collision_model="fokker_planck",
    )
    state = jax.random.normal(
        jax.random.key(52),
        (8, 6, parallel.z.shape[0], 1, 1),
    ).astype(jnp.complex128)
    difference = linear_residual(state, precomputed=collisional) - linear_residual(
        state, precomputed=collisionless
    )

    np.testing.assert_allclose(
        difference,
        fokker_planck_collision(state, collisional.collisions),
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert float(estimate_linear_cfl_dt(collisional)) < float(estimate_linear_cfl_dt(collisionless))


def test_linear_precompute_rejects_unknown_collision_model():
    velocity, parallel, geometry, species = _collision_setup()
    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,)))
    with np.testing.assert_raises_regex(ValueError, "collision_model"):
        build_linear_residual_precompute(
            velocity,
            parallel,
            fourier,
            geometry,
            species,
            collision_frequency=0.2,
            collision_model="unknown",
        )


def test_fokker_planck_field_particle_completion_conserves_exchange_moments():
    _velocity, parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    state = jax.random.normal(
        jax.random.key(61),
        (2, 8, 6, parallel.z.shape[0], 1, 1),
    ) + 1j * jax.random.normal(
        jax.random.key(62),
        (2, 8, 6, parallel.z.shape[0], 1, 1),
    )
    raw = build_fokker_planck_precompute(velocity, geometry.B, species, frequency=(0.2, 0.7))
    conserving = build_fokker_planck_precompute(
        velocity,
        geometry.B,
        species,
        frequency=(0.2, 0.7),
        conserve_exchange=True,
    )
    raw_collision = fokker_planck_collision(state, raw)
    collision = jax.jit(fokker_planck_collision)(state, conserving)
    moments = fokker_planck_conserved_moments(collision, conserving)

    assert float(jnp.max(jnp.abs(raw_collision - collision))) > 1.0e-6
    np.testing.assert_allclose(moments, 0.0, atol=3.0e-10, rtol=0.0)
    assert float(jnp.max(conserving.row_sum_bound)) > float(jnp.max(raw.row_sum_bound))


def test_solver_builds_exchange_conserving_fokker_planck_precompute():
    _velocity, parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,)))
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="kinetic",
        collision_frequency=(0.2, 0.7),
        collision_model="fokker_planck",
        collision_conserve_exchange=True,
    )

    assert precompute.collisions.conserve_exchange
    assert np.isfinite(float(estimate_linear_cfl_dt(precompute)))


def test_xu_species_local_completion_removes_each_species_defect():
    _velocity, parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    state = jax.random.normal(jax.random.key(71), (2, 8, 6, parallel.z.size, 1, 1))
    precompute = build_fokker_planck_precompute(
        velocity,
        geometry.B,
        species,
        frequency=(0.2, 0.7),
        conservation_model="xu_species_local",
    )
    collision = jax.jit(fokker_planck_collision)(state, precompute)
    momentum = jnp.einsum("svmz,svmzxy->szxy", precompute.xu_vpar_weight, collision)
    energy = jnp.einsum("svmz,svmzxy->szxy", precompute.xu_energy_weight, collision)

    np.testing.assert_allclose(momentum, 0.0, atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(energy, 0.0, atol=2.0e-10, rtol=0.0)
    assert np.isfinite(float(jnp.max(precompute.row_sum_bound)))

    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,)))
    solver_precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="kinetic",
        collision_frequency=(0.2, 0.7),
        collision_model="fokker_planck",
        collision_conservation_model="xu_species_local",
    )
    assert solver_precompute.collisions.conservation_model == "xu_species_local"


def test_pairwise_exchange_conserves_each_pair_and_couples_species():
    _velocity, parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    state = jax.random.normal(
        jax.random.key(81), (2, 8, 6, parallel.z.size, 1, 1)
    ) + 1j * jax.random.normal(jax.random.key(82), (2, 8, 6, parallel.z.size, 1, 1))
    precompute = build_fokker_planck_precompute(
        velocity,
        geometry.B,
        species,
        frequency=(0.2, 0.7),
        conservation_model="pairwise_exchange",
    )
    components = jax.jit(fokker_planck_pairwise_components)(state, precompute)
    collision = jax.jit(fokker_planck_collision)(state, precompute)

    np.testing.assert_allclose(collision, jnp.sum(components, axis=1), atol=2e-12)
    np.testing.assert_allclose(
        fokker_planck_conserved_moments(collision, precompute),
        0.0,
        atol=4e-10,
        rtol=0.0,
    )
    for pair_index, (first, second) in enumerate(precompute.pair_indices):
        pair_values = (
            components[first, second][None, ...]
            if first == second
            else jnp.stack((components[first, second], components[second, first]))
        )
        pair_moments = jnp.einsum(
            "csvmz,vmz,svmzxy->czxy",
            precompute.pair_conservation_invariants[pair_index],
            precompute.pair_conservation_measure[pair_index],
            pair_values,
        )
        np.testing.assert_allclose(pair_moments, 0.0, atol=4e-10, rtol=0.0)

    changed_state = state.at[1].multiply(1.1)
    changed_components = fokker_planck_pairwise_components(changed_state, precompute)
    assert float(jnp.max(jnp.abs(changed_components[0, 1] - components[0, 1]))) > 1e-8
    assert np.isfinite(float(jnp.max(precompute.row_sum_bound)))


def test_reciprocal_exchange_conserves_pairs_and_uses_partner_response():
    _velocity, parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    state = jax.random.normal(
        jax.random.key(91), (2, 8, 6, parallel.z.size, 1, 1)
    ) + 1j * jax.random.normal(jax.random.key(92), (2, 8, 6, parallel.z.size, 1, 1))
    precompute = build_fokker_planck_precompute(
        velocity,
        geometry.B,
        species,
        frequency=(0.2, 0.7),
        conservation_model="reciprocal_exchange",
    )
    components = jax.jit(fokker_planck_reciprocal_components)(state, precompute)
    collision = jax.jit(fokker_planck_collision)(state, precompute)

    np.testing.assert_allclose(collision, jnp.sum(components, axis=1), atol=2e-12)
    np.testing.assert_allclose(
        fokker_planck_conserved_moments(collision, precompute),
        0.0,
        atol=4e-10,
        rtol=0.0,
    )
    for pair_index, (first, second) in enumerate(precompute.pair_indices):
        pair_values = (
            components[first, second][None, ...]
            if first == second
            else jnp.stack((components[first, second], components[second, first]))
        )
        pair_moments = jnp.einsum(
            "csvmz,vmz,svmzxy->czxy",
            precompute.pair_conservation_invariants[pair_index],
            precompute.pair_conservation_measure[pair_index],
            pair_values,
        )
        np.testing.assert_allclose(pair_moments, 0.0, atol=4e-10, rtol=0.0)

    electron_changed = state.at[1].multiply(1.1)
    changed_components = fokker_planck_reciprocal_components(electron_changed, precompute)
    assert float(jnp.max(jnp.abs(changed_components[0, 1] - components[0, 1]))) > 1e-8

    def objective(values):
        result = fokker_planck_collision(values, precompute)
        return jnp.real(jnp.vdot(result, result))

    gradient = jax.jit(jax.grad(objective))(state)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_reciprocal_exchange_row_sum_bound_bounds_dense_operator():
    _velocity, _parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    precompute = build_fokker_planck_precompute(
        velocity,
        geometry.B[:1],
        species,
        frequency=(0.2, 0.7),
        conservation_model="reciprocal_exchange",
    )
    shape = (2, 8, 6, 1, 1, 1)

    def flattened_operator(values):
        return fokker_planck_collision(values.reshape(shape), precompute).reshape(-1)

    matrix = jax.jacfwd(flattened_operator)(jnp.zeros(np.prod(shape)))
    exact_bounds = jnp.max(jnp.sum(jnp.abs(matrix), axis=1).reshape(2, -1), axis=1)
    assert bool(jnp.all(exact_bounds <= precompute.row_sum_bound * (1.0 + 2e-12)))


def test_solver_accepts_reciprocal_exchange_collision_model():
    _velocity, parallel, geometry, species = _collision_setup(n_species=2)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=6,
            vpar_max=3.5,
            mu_max=5.0,
            backend="finite_difference",
        )
    )
    fourier = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,)))
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="kinetic",
        collision_frequency=(0.2, 0.7),
        collision_model="fokker_planck",
        collision_conservation_model="reciprocal_exchange",
    )

    assert precompute.collisions.conservation_model == "reciprocal_exchange"
    assert np.isfinite(float(estimate_linear_cfl_dt(precompute)))


def test_laguerre_legendre_low_rank_contract_is_jittable_and_differentiable():
    coefficient_shape = (2, 2, 3, 2, 2, 2, 1, 1)
    driver = jax.random.normal(jax.random.key(101), coefficient_shape) / 7.0
    response = jax.random.normal(jax.random.key(102), coefficient_shape) / 5.0
    precompute = build_laguerre_legendre_collision_precompute(
        driver,
        response,
        component_labels=((0, 0, 1), (1, -1, 0), (1, 1, 0)),
    )
    state = jax.random.normal(jax.random.key(103), (2, 2, 2, 2, 1, 1))
    components = jax.jit(laguerre_legendre_collision_components)(state, precompute)
    collision = jax.jit(laguerre_legendre_collision)(state, precompute)

    moments = jnp.einsum("abcvmzxy,bvmzxy->abczxy", driver, state)
    expected_components = jnp.einsum("abcvmzxy,abczxy->abvmzxy", response, moments)
    from_moments = jax.jit(laguerre_legendre_collision_components_from_moments)(moments, precompute)
    np.testing.assert_allclose(components, expected_components, atol=2e-13)
    np.testing.assert_allclose(from_moments, expected_components, atol=2e-13)
    np.testing.assert_allclose(collision, jnp.sum(components, axis=1), atol=2e-13)

    def objective(values):
        result = laguerre_legendre_collision(values, precompute)
        return jnp.vdot(result, result)

    gradient = jax.jit(jax.grad(objective))(state)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_laguerre_legendre_row_sum_bound_bounds_dense_operator():
    coefficient_shape = (2, 2, 2, 2, 2, 1, 1, 1)
    driver = jax.random.normal(jax.random.key(111), coefficient_shape)
    response = jax.random.normal(jax.random.key(112), coefficient_shape)
    precompute = build_laguerre_legendre_collision_precompute(
        driver,
        response,
        component_labels=((0, 0, 1), (1, 0, 0)),
    )
    state_shape = (2, 2, 2, 1, 1, 1)

    def flattened_operator(values):
        return laguerre_legendre_collision(values.reshape(state_shape), precompute).reshape(-1)

    matrix = jax.jacfwd(flattened_operator)(jnp.zeros(np.prod(state_shape)))
    exact_bounds = jnp.max(jnp.sum(jnp.abs(matrix), axis=1).reshape(2, -1), axis=1)
    assert bool(jnp.all(exact_bounds <= precompute.row_sum_bound * (1.0 + 2e-12)))


def test_implicit_laguerre_legendre_solve_matches_dense_backward_euler():
    shape = (1, 1, 2, 2, 2, 1, 1, 1)
    driver = jax.random.normal(jax.random.key(121), shape) / 20.0
    response = jax.random.normal(jax.random.key(122), shape) / 15.0
    precompute = build_laguerre_legendre_collision_precompute(
        driver,
        response,
        component_labels=((0, 0, 1), (1, 0, 0)),
    )
    state = jax.random.normal(jax.random.key(123), (1, 2, 2, 1, 1, 1))
    test_particle = jnp.asarray(
        (
            (1.2, -0.03, 0.0, 0.0),
            (-0.02, 1.1, -0.01, 0.0),
            (0.0, -0.04, 1.15, -0.02),
            (0.0, 0.0, -0.03, 1.08),
        )
    )
    dt = 0.07

    collision = jax.jit(implicit_laguerre_legendre_collision)(state, test_particle, precompute, dt)
    driver_matrix = driver.reshape(2, 4)
    response_matrix = response.reshape(2, 4).T
    dense_matrix = test_particle - dt * response_matrix @ driver_matrix
    expected_advanced = jnp.linalg.solve(dense_matrix, state.reshape(4))
    expected = (expected_advanced - state.reshape(4)) / dt

    np.testing.assert_allclose(collision.reshape(4), expected, rtol=2e-13, atol=2e-13)

    def objective(values):
        result = implicit_laguerre_legendre_collision(values, test_particle, precompute, dt)
        return jnp.vdot(result, result)

    gradient = jax.jit(jax.grad(objective))(state)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_laguerre_legendre_contract_rejects_invalid_component_labels():
    coefficients = jnp.ones((1, 1, 2, 2, 2, 1, 1, 1))
    with pytest.raises(ValueError, match="uniquely match"):
        build_laguerre_legendre_collision_precompute(
            coefficients,
            coefficients,
            component_labels=((0, 0, 1), (0, 0, 1)),
        )


def test_stella_laguerre_legendre_response_matches_normalized_basis_formula():
    velocity, parallel, geometry, species = _collision_setup(n_species=2)
    labels = ((0, 0, 0), (1, -1, 0), (1, 0, 0), (1, 1, 0))
    pair_frequency = jnp.asarray(((0.2, 0.3), (0.4, 0.5)))
    delta = jnp.linspace(
        0.5,
        1.5,
        2 * 2 * len(labels) * 8 * 6 * parallel.z.size,
    ).reshape(2, 2, len(labels), 8, 6, parallel.z.size)
    gyroaverage = jnp.linspace(
        0.7,
        1.1,
        2 * 2 * 6 * parallel.z.size * 2 * 3,
    ).reshape(2, 2, 6, parallel.z.size, 2, 3)

    response = build_stella_laguerre_legendre_response(
        velocity,
        geometry.B,
        species,
        pair_frequency,
        delta,
        gyroaverage,
        component_labels=labels,
    )

    vpar = np.asarray(velocity.vpar)[:, None, None]
    mu = np.asarray(velocity.mu)[None, :, None]
    magnetic_field = np.asarray(geometry.B)[None, None, :]
    speed = np.sqrt(vpar**2 + 2.0 * mu * magnetic_field)
    xi = np.divide(vpar, speed, out=np.zeros_like(speed), where=speed > 0.0)
    perpendicular = np.sqrt(np.maximum(1.0 - xi**2, 0.0))
    polynomials = (np.ones_like(xi), 0.5 * perpendicular, xi, -perpendicular)
    normalizations = (
        np.sqrt(1.0 / (4.0 * np.pi)),
        np.sqrt(3.0 / (2.0 * np.pi)),
        np.sqrt(3.0 / (4.0 * np.pi)),
        np.sqrt(3.0 / (8.0 * np.pi)),
    )
    masses = np.asarray([item.mass for item in species])
    mass_factor = (masses[:, None] / masses[None, :]) ** -1.5
    expected_components = []
    for component, (_l, m, _j) in enumerate(labels):
        negative_m_sign = -1.0 if m < 0 else 1.0
        expected_components.append(
            np.asarray(pair_frequency)[:, :, None, None, None, None, None]
            * mass_factor[:, :, None, None, None, None, None]
            * np.asarray(delta)[:, :, component, :, :, :, None, None]
            * negative_m_sign
            * normalizations[component]
            * polynomials[component][None, None, :, :, :, None, None]
            * np.asarray(gyroaverage)[:, None, abs(m), None, :, :, :, :]
        )
    expected = np.stack(expected_components, axis=2)

    assert response.shape == (2, 2, 4, 8, 6, parallel.z.size, 2, 3)
    np.testing.assert_allclose(response, expected, rtol=2e-13, atol=2e-13)


def test_stella_laguerre_legendre_response_is_jittable_and_differentiable():
    velocity, parallel, geometry, species = _collision_setup()
    labels = ((0, 0, 0), (1, 0, 0))
    frequency = jnp.asarray(((0.3,),))
    delta = jnp.ones((1, 1, 2, 8, 6, parallel.z.size))
    gyroaverage = jnp.ones((1, 1, 6, parallel.z.size, 1, 1))

    def objective(magnetic_field):
        response = build_stella_laguerre_legendre_response(
            velocity,
            magnetic_field,
            species,
            frequency,
            delta,
            gyroaverage,
            component_labels=labels,
        )
        return jnp.sum(response**2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(geometry.B)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_stella_laguerre_legendre_response_validates_coefficient_axes():
    velocity, parallel, geometry, species = _collision_setup()
    with pytest.raises(ValueError, match="gyroaverage"):
        build_stella_laguerre_legendre_response(
            velocity,
            geometry.B,
            species,
            jnp.ones((1, 1)),
            jnp.ones((1, 1, 1, 8, 6, parallel.z.size)),
            jnp.ones((1, 1, 6, parallel.z.size)),
            component_labels=((0, 0, 0),),
        )


def test_stella_delta0_is_jittable_and_differentiable():
    speed = jnp.linspace(0.2, 4.0, 32)

    def objective(mass_ratio):
        values = stella_laguerre_legendre_delta0(
            speed,
            mass_ratio,
            1.0,
            laguerre_degree=1,
            legendre_degree=1,
        )
        return jnp.vdot(values, values)

    value, derivative = jax.jit(jax.value_and_grad(objective))(2.5)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(derivative))


def test_stella_delta0_rejects_negative_polynomial_degree():
    with pytest.raises(ValueError, match="must be nonnegative"):
        stella_laguerre_legendre_delta0(
            1.0,
            1.0,
            1.0,
            laguerre_degree=-1,
            legendre_degree=0,
        )


def test_stella_recursive_delta_reduces_to_analytic_delta0():
    velocity, parallel, geometry, species = _collision_setup(n_species=2)
    labels = ((0, 0, 0), (1, -1, 0), (1, 0, 0), (1, 1, 0))
    measure = (
        velocity.w_vpar[:, None, None]
        * velocity.w_mu[None, :, None]
        * jnp.ones((1, 1, parallel.z.size))
    )
    recursive = build_stella_laguerre_legendre_delta(
        velocity,
        geometry.B,
        species,
        measure,
        component_labels=labels,
    )
    speed = jnp.sqrt(
        velocity.vpar[:, None, None] ** 2
        + 2.0 * velocity.mu[None, :, None] * geometry.B[None, None, :]
    )
    expected = stella_laguerre_legendre_delta0(
        speed,
        species[0].mass,
        species[1].mass,
        laguerre_degree=0,
        legendre_degree=1,
    )

    np.testing.assert_allclose(recursive[0, 1, 1], expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(recursive[0, 1, 2], expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(recursive[0, 1, 3], expected, rtol=2e-13, atol=2e-13)


def test_stella_recursive_delta1_is_jittable_and_differentiable():
    velocity, parallel, geometry, species = _collision_setup(n_species=2)
    measure = jnp.ones((8, 6, parallel.z.size)) / (8 * 6)

    def objective(magnetic_field):
        delta = build_stella_laguerre_legendre_delta(
            velocity,
            magnetic_field,
            species,
            measure,
            component_labels=((0, 0, 1),),
        )
        return jnp.vdot(delta, delta)

    value, gradient = jax.jit(jax.value_and_grad(objective))(geometry.B)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_stella_normalized_driver_is_jittable_and_differentiable():
    velocity, parallel, geometry, species = _collision_setup(n_species=2)
    labels = ((1, 0, 0),)
    measure = jnp.ones((8, 6, parallel.z.size)) / (8 * 6)
    gyroaverage = jnp.ones((2, 1, 6, parallel.z.size, 1, 1))

    def objective(magnetic_field):
        delta = build_stella_laguerre_legendre_delta(
            velocity,
            magnetic_field,
            species,
            measure,
            component_labels=labels,
        )
        driver = build_stella_laguerre_legendre_driver(
            velocity,
            magnetic_field,
            species,
            measure,
            delta,
            gyroaverage,
            component_labels=labels,
        )
        return jnp.vdot(driver, driver)

    value, gradient = jax.jit(jax.value_and_grad(objective))(geometry.B)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_stella_collision_precompute_assembles_local_driver_and_response():
    velocity, parallel, geometry, species = _collision_setup(n_species=2)
    measure = jnp.ones((8, 6, parallel.z.size)) / (8 * 6)
    gyroaverage = jnp.ones((2, 1, 6, parallel.z.size, 1, 1))
    precompute = build_stella_laguerre_legendre_collision_precompute(
        velocity,
        geometry.B,
        species,
        jnp.asarray(((0.2, 0.3), (0.4, 0.5))),
        measure,
        gyroaverage,
        component_labels=((1, 0, 0),),
    )
    state = jax.random.normal(jax.random.key(141), (2, 8, 6, parallel.z.size, 1, 1))
    result = jax.jit(laguerre_legendre_collision)(state, precompute)

    assert result.shape == state.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    assert bool(jnp.all(jnp.isfinite(precompute.row_sum_bound)))


@pytest.mark.external
def test_fokker_planck_stencil_and_action_match_pinned_gyaradax(gyaradax_root):
    import sys

    if str(gyaradax_root) not in sys.path:
        sys.path.insert(0, str(gyaradax_root))
    from gyaradax.collisions import (
        collision_rhs,
        conservation_correction,
        precompute_collisions,
    )
    from gyaradax.geometry import compute_geometry
    from gyaradax.params import GKParams

    reference_geometry = compute_geometry(
        q=1.57,
        shat=1.07,
        eps=0.177,
        ns=6,
        nkx=1,
        nky=1,
        nvpar=8,
        nmu=4,
        nperiod=1,
        krhomax=0.5,
    )
    reference_params = GKParams(
        collisions=True,
        coll_freq=0.1,
        disp_vp=0.0,
        mas=1.0,
        tmp=1.0,
        de=1.0,
        signz=1.0,
        vthrat=1.0,
        adiabatic_electrons=True,
        dvp=float(reference_geometry["dvp"]),
        sgr_dist=float(reference_geometry["sgr_dist"]),
    )
    reference_stencil = precompute_collisions(reference_geometry, reference_params)["coll_stencil"]
    reference_mu = np.asarray(reference_geometry["mugr"])
    dvperp = 2.0 * np.sqrt(2.0 * reference_mu[0])
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=4,
            vpar_max=3.0,
            mu_max=0.5 * (4 * dvperp) ** 2,
            backend="finite_difference",
        )
    )
    species = SpeciesParams(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    observed = build_fokker_planck_precompute(
        velocity,
        reference_geometry["bn"],
        species,
        frequency=0.1,
    )
    np.testing.assert_allclose(observed.stencil[0], reference_stencil, rtol=2e-12, atol=2e-12)

    state = jax.random.normal(jax.random.key(44), (8, 4, 6, 1, 1)) + 1j * jax.random.normal(
        jax.random.key(45), (8, 4, 6, 1, 1)
    )
    np.testing.assert_allclose(
        fokker_planck_collision(state, observed),
        collision_rhs(state, reference_stencil),
        rtol=2e-12,
        atol=2e-12,
    )

    conserving_reference_params = replace(
        reference_params,
        coll_mom_conservation=True,
        coll_ene_conservation=True,
    )
    conserving_reference = precompute_collisions(reference_geometry, conserving_reference_params)
    conserving_observed = build_fokker_planck_precompute(
        velocity,
        reference_geometry["bn"],
        species,
        frequency=0.1,
        conservation_model="xu_species_local",
    )
    np.testing.assert_allclose(
        conserving_observed.xu_momentum_factor[0],
        conserving_reference["coll_mom_factor"],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        conserving_observed.xu_energy_factor[0],
        conserving_reference["coll_ene_factor"],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        conserving_observed.xu_vpar_weight[0],
        conserving_reference["coll_vpar_weight"],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        conserving_observed.xu_energy_weight[0],
        conserving_reference["coll_vsq_weight"],
        rtol=2e-12,
        atol=2e-12,
    )
    reference_rhs = collision_rhs(state, conserving_reference["coll_stencil"])
    reference_rhs = reference_rhs + conservation_correction(
        reference_rhs,
        conserving_reference["coll_mom_factor"],
        conserving_reference["coll_ene_factor"],
        conserving_reference["coll_vpar_weight"],
        conserving_reference["coll_vsq_weight"],
    )
    np.testing.assert_allclose(
        fokker_planck_collision(state, conserving_observed),
        reference_rhs,
        rtol=2e-12,
        atol=2e-12,
    )
