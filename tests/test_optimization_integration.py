from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    AdiabaticElectronParams,
    DesignObjectiveSpec,
    FourierGridSpec,
    OptimizationKnobs,
    ParallelGridSpec,
    SingleSurfaceOptimizationConfig,
    SpeciesParams,
    VelocityGridSpec,
    build_fourier_grid,
    build_desc_geometry_from_arrays,
    build_mode_connectivity,
    build_optimization_geometry,
    build_optimization_species,
    build_parallel_grid,
    build_velocity_grid,
    design_objective,
    scan_single_surface_objective,
    single_surface_objective,
    toy_gradient_descent_step,
)


def _grids():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.5, mu_max=1.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=5, z_min=-0.5 + 0.5 / 5, z_max=0.5 + 0.5 / 5)
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.5, ky_values=(0.0, 0.35), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    initial_state = 0.01 * (jnp.cos(index / 6.0) + 1j * jnp.sin(index / 8.0))
    return velocity, parallel, fourier, connectivity, initial_state


def _knobs():
    return OptimizationKnobs(
        density=0.9,
        temperature=1.2,
        density_gradient=0.8,
        temperature_gradient=2.1,
        q=1.25,
        shat=0.45,
        eps=0.17,
        rho=0.45,
        alpha=0.2,
        beta=0.03,
        pressure_gradient=0.04,
        equilibrium_coefficients=(0.04, -0.015),
    )


def _desc_geometry(parallel, amplitude=0.06):
    z = parallel.z
    ones = jnp.ones_like(z)
    return build_desc_geometry_from_arrays(
        parallel,
        theta=0.15 + 0.6 * z,
        phi=z,
        alpha=0.15 * ones,
        rho=0.45,
        B=1.0 + amplitude * jnp.cos(2.0 * jnp.pi * z),
        b_dot_grad_z=1.0 + 0.02 * jnp.sin(2.0 * jnp.pi * z),
        grad_psi_sq=1.1 + 0.03 * jnp.cos(2.0 * jnp.pi * z),
        grad_alpha_sq=1.4 + 0.02 * jnp.sin(2.0 * jnp.pi * z),
        grad_psi_dot_grad_alpha=0.04 * jnp.sin(2.0 * jnp.pi * z),
        B_cross_gradB_dot_grad_psi=0.02 * jnp.cos(2.0 * jnp.pi * z),
        B_cross_gradB_dot_grad_alpha=0.03 * jnp.sin(2.0 * jnp.pi * z),
        b_cross_kappa_dot_grad_psi=0.04 * ones,
        b_cross_kappa_dot_grad_alpha=0.05 * jnp.cos(2.0 * jnp.pi * z),
    )


def _config(**updates):
    base = SingleSurfaceOptimizationConfig(
        geometry_model="circular",
        dt=0.01,
        n_steps=2,
        selected_ky=1,
        objective_kind="growth_plus_quasilinear",
        quasilinear_weight=0.02,
        softplus_temperature=0.1,
        store_history=False,
    )
    return replace(base, **updates)


def test_optimization_knobs_build_species_and_geometry():
    _velocity, parallel, _fourier, _connectivity, _state = _grids()
    base_species = SpeciesParams(
        charge=1.0,
        mass=2.0,
        density=1.0,
        temperature=1.0,
        density_gradient=0.0,
        temperature_gradient=0.0,
    )
    knobs = _knobs()

    species = build_optimization_species(knobs, base_species)
    geometry = build_optimization_geometry(parallel, knobs, _config())

    np.testing.assert_allclose(species.charge, 1.0)
    np.testing.assert_allclose(species.mass, 2.0)
    np.testing.assert_allclose(species.density, knobs.density)
    np.testing.assert_allclose(species.temperature_gradient, knobs.temperature_gradient)
    assert geometry.B.shape == parallel.z.shape
    assert jnp.all(jnp.isfinite(geometry.B))
    assert jnp.all(jnp.isfinite(geometry.g_yy))


def test_desc_geometry_model_requires_supplied_geometry():
    _velocity, parallel, _fourier, _connectivity, _state = _grids()
    with pytest.raises(ValueError, match="imported geometry models require"):
        build_optimization_geometry(parallel, _knobs(), _config(geometry_model="desc"))


def test_single_surface_objective_is_jittable_and_differentiable():
    velocity, parallel, fourier, connectivity, initial_state = _grids()
    knobs = _knobs()
    config = _config(objective_kind="selected_growth", quasilinear_weight=0.0)
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)

    def objective(local_knobs):
        return single_surface_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
        ).scalar_objective

    result = single_surface_objective(
        knobs,
        velocity,
        parallel,
        fourier,
        initial_state,
        electron_params=electrons,
        connectivity=connectivity,
        config=config,
    )
    value, gradient = jax.jit(jax.value_and_grad(objective))(knobs)

    assert jnp.isfinite(result.scalar_objective)
    assert result.values.growth_rate.shape == (fourier.ky.shape[0],)
    np.testing.assert_allclose(value, result.scalar_objective, rtol=2e-12, atol=2e-12)
    assert jnp.isfinite(gradient.temperature_gradient)
    assert jnp.isfinite(gradient.q)
    assert gradient.equilibrium_coefficients.shape == knobs.equilibrium_coefficients.shape

    step = 1.0e-5
    finite_difference = (
        objective(replace(knobs, temperature_gradient=knobs.temperature_gradient + step))
        - objective(replace(knobs, temperature_gradient=knobs.temperature_gradient - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        gradient.temperature_gradient,
        finite_difference,
        rtol=5e-4,
        atol=5e-6,
    )


def test_single_surface_objective_accepts_desc_geometry_arrays():
    velocity, parallel, fourier, connectivity, initial_state = _grids()
    knobs = _knobs()
    config = _config(
        geometry_model="desc",
        objective_kind="selected_growth",
        quasilinear_weight=0.0,
    )
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    geometry = _desc_geometry(parallel)

    def objective(amplitude):
        local_geometry = _desc_geometry(parallel, amplitude=amplitude)
        return single_surface_objective(
            knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
            geometry=local_geometry,
        ).scalar_objective

    result = single_surface_objective(
        knobs,
        velocity,
        parallel,
        fourier,
        initial_state,
        electron_params=electrons,
        connectivity=connectivity,
        config=config,
        geometry=geometry,
    )
    value, gradient = jax.jit(jax.value_and_grad(objective))(0.06)
    step = 1.0e-5
    finite_difference = (objective(0.06 + step) - objective(0.06 - step)) / (2.0 * step)

    assert result.geometry.source == "desc"
    assert jnp.isfinite(result.scalar_objective)
    assert result.values.growth_rate.shape == (fourier.ky.shape[0],)
    np.testing.assert_allclose(value, result.scalar_objective, rtol=2e-12, atol=2e-12)
    assert jnp.isfinite(gradient)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-4, atol=1e-6)

    bad_geometry = replace(geometry, B=geometry.B[:-1])
    with pytest.raises(ValueError, match="supplied geometry.B"):
        single_surface_objective(
            knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
            geometry=bad_geometry,
        )


def test_design_objective_composes_named_physical_terms_and_gradients():
    velocity, parallel, fourier, connectivity, initial_state = _grids()
    knobs = _knobs()
    config = _config(objective_kind="selected_growth", quasilinear_weight=0.0)
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    baseline = single_surface_objective(
        knobs,
        velocity,
        parallel,
        fourier,
        initial_state,
        electron_params=electrons,
        connectivity=connectivity,
        config=config,
    )
    target = baseline.values.mode_structure * jnp.exp(0.83j)
    spec = DesignObjectiveSpec(
        selected_ky=1,
        growth_weight=0.7,
        frequency_weight=0.2,
        mode_structure_weight=0.3,
        quasilinear_weight=0.01,
        frequency_target=0.04,
    )

    def objective(temperature_gradient):
        local_knobs = replace(knobs, temperature_gradient=temperature_gradient)
        return design_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            spec,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
            target_mode_structure=target,
        ).scalar_objective

    result = design_objective(
        knobs,
        velocity,
        parallel,
        fourier,
        initial_state,
        spec,
        electron_params=electrons,
        connectivity=connectivity,
        config=config,
        target_mode_structure=target,
    )
    expected = (
        spec.growth_weight * result.growth_objective
        + spec.frequency_weight * result.frequency_penalty
        + spec.mode_structure_weight * result.mode_structure_penalty
        + spec.quasilinear_weight * result.quasilinear_proxy
    )
    value, gradient = jax.jit(jax.value_and_grad(objective))(knobs.temperature_gradient)
    # The reduced grid includes a zonal k_perp=0 contribution, so the
    # quasilinear component is O(1e9); use a scale-resolving difference step.
    step = 3.0e-2
    finite_difference = (
        objective(knobs.temperature_gradient + step)
        - objective(knobs.temperature_gradient - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(result.scalar_objective, expected, rtol=2.0e-13)
    np.testing.assert_allclose(result.mode_structure_penalty, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(value, result.scalar_objective, rtol=2.0e-12)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-3, atol=1.0e-6)


def test_design_objective_rejects_ambiguous_frequency_branch():
    with pytest.raises(ValueError, match="frequency objective requires selected_ky"):
        DesignObjectiveSpec(
            selected_ky=None,
            growth_aggregation="max",
            frequency_weight=1.0,
        )


def test_scan_single_surface_objective_accepts_supplied_desc_geometry():
    velocity, parallel, fourier, connectivity, initial_state = _grids()
    scan = scan_single_surface_objective(
        _knobs(),
        velocity,
        parallel,
        fourier,
        initial_state,
        rho_values=jnp.asarray([0.45]),
        alpha_values=jnp.asarray([0.0, 0.4]),
        ky_indices=(1,),
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=False,
        ),
        connectivity=connectivity,
        config=_config(geometry_model="desc", objective_kind="selected_growth"),
        geometry=_desc_geometry(parallel),
    )

    assert scan.objectives.shape == (1, 2, 1)
    assert jnp.all(jnp.isfinite(scan.objectives))


def test_scan_single_surface_objective_over_rho_alpha_and_ky():
    velocity, parallel, fourier, connectivity, initial_state = _grids()
    scan = scan_single_surface_objective(
        _knobs(),
        velocity,
        parallel,
        fourier,
        initial_state,
        rho_values=jnp.asarray([0.35, 0.55]),
        alpha_values=jnp.asarray([0.0, 0.4, 0.8]),
        ky_indices=(0, 1),
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=False,
        ),
        connectivity=connectivity,
        config=_config(objective_kind="selected_growth"),
    )

    assert scan.objectives.shape == (2, 3, 2)
    assert scan.ky_indices.dtype == jnp.int32
    assert jnp.all(jnp.isfinite(scan.objectives))
    assert jnp.max(jnp.abs(scan.objectives[:, 0, :] - scan.objectives[:, 1, :])) > 0.0


def test_toy_gradient_descent_step_updates_differentiable_knobs():
    velocity, parallel, fourier, connectivity, initial_state = _grids()
    knobs = _knobs()
    config = _config(objective_kind="quasilinear_proxy")

    def objective(local_knobs):
        return single_surface_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            electron_params=AdiabaticElectronParams(
                density=1.0,
                temperature=1.0,
                zonal_correction=False,
            ),
            connectivity=connectivity,
            config=config,
        ).scalar_objective

    step = toy_gradient_descent_step(objective, knobs, learning_rate=1.0e-13)

    assert jnp.isfinite(step.value)
    assert jnp.isfinite(step.gradient.temperature_gradient)
    assert jnp.isfinite(step.gradient.q)
    assert step.updated_knobs.equilibrium_coefficients.shape == knobs.equilibrium_coefficients.shape
    assert not np.isclose(np.asarray(step.updated_knobs.density), np.asarray(knobs.density))
