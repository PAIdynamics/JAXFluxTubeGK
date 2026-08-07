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
    linear_residual,
)


def _collision_setup(n_species=1):
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=8, n_mu=6, vpar_max=3.5, mu_max=5.0)
    )
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=7, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    geometry = build_s_alpha_geometry(
        parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18)
    )
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
    state = jnp.einsum(
        "sbvmz,szxyb->svmzxy", precompute.equilibrium_basis, coefficients
    )[0]

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
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,))
    )
    collisionless = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species
    )
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
    assert float(estimate_linear_cfl_dt(collisional)) < float(
        estimate_linear_cfl_dt(collisionless)
    )


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
        precompute = build_fokker_planck_precompute(
            velocity, geometry.B, species, frequency
        )
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
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,))
    )
    collisionless = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species
    )
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
    assert float(estimate_linear_cfl_dt(collisional)) < float(
        estimate_linear_cfl_dt(collisionless)
    )


def test_linear_precompute_rejects_unknown_collision_model():
    velocity, parallel, geometry, species = _collision_setup()
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,))
    )
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
    state = (
        jax.random.normal(
            jax.random.key(61),
            (2, 8, 6, parallel.z.shape[0], 1, 1),
        )
        + 1j
        * jax.random.normal(
            jax.random.key(62),
            (2, 8, 6, parallel.z.shape[0], 1, 1),
        )
    )
    raw = build_fokker_planck_precompute(
        velocity, geometry.B, species, frequency=(0.2, 0.7)
    )
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
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,))
    )
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
    state = jax.random.normal(
        jax.random.key(71), (2, 8, 6, parallel.z.size, 1, 1)
    )
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

    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,))
    )
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
    state = (
        jax.random.normal(
            jax.random.key(81), (2, 8, 6, parallel.z.size, 1, 1)
        )
        + 1j
        * jax.random.normal(
            jax.random.key(82), (2, 8, 6, parallel.z.size, 1, 1)
        )
    )
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
    reference_stencil = precompute_collisions(reference_geometry, reference_params)[
        "coll_stencil"
    ]
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

    state = (
        jax.random.normal(jax.random.key(44), (8, 4, 6, 1, 1))
        + 1j * jax.random.normal(jax.random.key(45), (8, 4, 6, 1, 1))
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
    conserving_reference = precompute_collisions(
        reference_geometry, conserving_reference_params
    )
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
