from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    AdiabaticElectronParams,
    BenchmarkTarget,
    FourierGridSpec,
    OptimizationKnobs,
    ParallelGridSpec,
    SingleSurfaceOptimizationConfig,
    VelocityGridSpec,
    benchmark_target_cost,
    benchmark_target_residual,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_flux_tube_geometry_from_gx_eik_reference,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    cyclone_base_case_growth_target,
    compare_geometry_to_gx_eik_reference,
    geometry_to_gx_eik_reference,
    gx_growth_rate_target,
    load_gx_eik_geometry_reference,
    load_gx_growth_rate_reference,
    resample_gx_eik_geometry_reference,
    run_geometry_to_gx_eik_export_gate,
    run_gx_gist_external_eik_suite_gate,
    rosenbluth_hinton_residual,
    rosenbluth_hinton_target,
    run_gx_eik_geometry_gate,
    run_production_cyclone_base_case_gate,
    run_rosenbluth_hinton_plateau_gate,
    run_reduced_cyclone_base_case_gate,
    run_reduced_rosenbluth_hinton_gate,
    run_solver_geometry_to_gx_eik_gate,
    single_surface_benchmark_objective,
)


ROOT = Path(__file__).resolve().parents[1]


def test_named_benchmark_targets_and_costs_are_differentiable():
    rh = rosenbluth_hinton_target()
    cyclone = cyclone_base_case_growth_target()

    np.testing.assert_allclose(rh.reference_value, 0.0711)
    np.testing.assert_allclose(cyclone.reference_value, 0.179)
    assert rh.quantity == "zonal_residual"
    assert cyclone.quantity == "selected_growth_rate"
    cyclone_metadata = dict(cyclone.metadata)
    rh_metadata = dict(rh.metadata)
    assert rh_metadata["n_z"] == 64
    assert rh_metadata["n_vpar"] == 64
    assert rh_metadata["n_mu"] == 16
    assert rh_metadata["disp_vp"] == 0.08
    assert rh_metadata["reference_n_z"] == 128
    assert cyclone_metadata["nperiod"] == 5
    assert cyclone_metadata["n_z"] == 144
    assert cyclone_metadata["n_vpar"] == 64
    assert cyclone_metadata["n_mu"] == 16
    assert rosenbluth_hinton_residual(1.3, 0.05) > rh.reference_value

    value = jnp.asarray(0.2)
    residual = benchmark_target_residual(value, cyclone)
    cost = benchmark_target_cost(value, cyclone)
    gradient = jax.grad(lambda x: benchmark_target_cost(x, cyclone))(value)

    np.testing.assert_allclose(residual, (0.2 - 0.179) / cyclone.tolerance)
    np.testing.assert_allclose(cost, 0.5 * residual**2)
    np.testing.assert_allclose(gradient, residual / cyclone.tolerance)


def test_gx_growth_rate_reference_loads_time_averaged_cyclone_curve():
    pytest.importorskip("netCDF4")
    path = (
        ROOT
        / "relevant-codes/gx/benchmarks/linear/ITG_cyclone/"
        "itg_salpha_adiabatic_electrons_correct.out.nc"
    )

    reference = load_gx_growth_rate_reference(path)
    target = gx_growth_rate_target(
        reference,
        target_ky=0.5,
        name="gx_salpha_adiabatic_ky05",
    )

    assert reference.ky.shape == reference.growth_rate.shape == reference.frequency.shape
    np.testing.assert_allclose(reference.ky[0], 0.05)
    np.testing.assert_allclose(reference.ky[-1], 0.55)
    np.testing.assert_allclose(jnp.max(reference.growth_rate), 0.09302951, rtol=2e-6)
    np.testing.assert_allclose(reference.ky[jnp.argmax(reference.growth_rate)], 0.3)
    np.testing.assert_allclose(target.reference_value, 0.054058794, rtol=2e-6)
    assert target.quantity == "selected_growth_rate"
    assert dict(target.metadata)["matched_ky"] == pytest.approx(0.5)


def test_gx_eik_geometry_reference_loads_vmec_gs2_fixture():
    path = (
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )

    reference = load_gx_eik_geometry_reference(path)

    assert reference.theta.shape == (20001,)
    assert reference.header[0] == pytest.approx(10000.0)
    np.testing.assert_allclose(reference.theta[0], -10.0 * np.pi, rtol=2e-11)
    np.testing.assert_allclose(reference.theta[-1], 10.0 * np.pi, rtol=2e-11)
    np.testing.assert_allclose(reference.bmag[0], reference.bmag[-1], rtol=2e-12)
    assert jnp.all(jnp.isfinite(reference.gds2))
    assert jnp.min(reference.bmag) > 0.0


def test_gx_eik_loader_uses_gist_drift_column_order():
    path = (
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_li383_1.4m.txt_highres_surf12_pol_10_nz0_10000"
    )

    reference = load_gx_eik_geometry_reference(path)

    np.testing.assert_allclose(reference.cvdrift[0], 9.4946168708e-01)
    np.testing.assert_allclose(reference.cvdrift0[0], -2.1124355474e-02)
    np.testing.assert_allclose(reference.gbdrift[0], 8.8374219166e-01)
    np.testing.assert_allclose(reference.gbdrift0[0], -2.1124355474e-02)


def test_gx_eik_geometry_gate_matches_solver_kperp_contract():
    path = (
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    reference = load_gx_eik_geometry_reference(path)
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    sampled = resample_gx_eik_geometry_reference(reference, theta)
    parallel = _parallel_grid_from_theta(theta)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )

    result = run_gx_eik_geometry_gate(sampled, parallel, fourier)

    assert bool(result.passed)
    np.testing.assert_allclose(result.observed_value, 0.0, atol=1.0e-13)
    assert result.target.quantity == "max_abs_kperp2_error"


def test_solver_geometry_to_eik_parity_report_matches_imported_geometry():
    path = (
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    reference = load_gx_eik_geometry_reference(path)
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    sampled = resample_gx_eik_geometry_reference(reference, theta)
    parallel = _parallel_grid_from_theta(theta)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    geometry = build_flux_tube_geometry_from_gx_eik_reference(sampled, parallel)

    report = compare_geometry_to_gx_eik_reference(geometry, sampled, fourier)
    gate = run_solver_geometry_to_gx_eik_gate(geometry, sampled, fourier)

    assert bool(gate.passed)
    assert report.field_names[-1] == "kperp2"
    assert "D_x" in report.field_names[5]
    np.testing.assert_allclose(report.field_errors, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(gate.observed_value, 0.0, atol=1.0e-13)


def test_desc_fixture_geometry_exports_to_gx_eik_contract():
    fixture = ROOT / "fixtures/desc_geometry_dshape_rho05_alpha0.npz"
    data = np.load(fixture)
    parallel = _parallel_grid_from_fixture_z(data["z"])
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    geometry = build_desc_geometry_from_arrays(
        parallel,
        theta=data["theta"],
        phi=data["phi"],
        alpha=data["alpha"],
        rho=data["rho"],
        B=data["B"],
        b_dot_grad_z=data["b_dot_grad_z"],
        grad_psi_sq=data["grad_psi_sq"],
        grad_alpha_sq=data["grad_alpha_sq"],
        grad_psi_dot_grad_alpha=data["grad_psi_dot_grad_alpha"],
        B_cross_gradB_dot_grad_psi=data["B_cross_gradB_dot_grad_psi"],
        B_cross_gradB_dot_grad_alpha=data["B_cross_gradB_dot_grad_alpha"],
        b_cross_kappa_dot_grad_psi=data["b_cross_kappa_dot_grad_psi"],
        b_cross_kappa_dot_grad_alpha=data["b_cross_kappa_dot_grad_alpha"],
    )

    reference = geometry_to_gx_eik_reference(geometry)
    report = compare_geometry_to_gx_eik_reference(
        geometry,
        reference,
        fourier,
        include_mirror_proxy=False,
    )
    gate = run_geometry_to_gx_eik_export_gate(geometry, fourier)

    assert reference.source == "desc:gx-eik-export"
    assert not any(name.startswith("G/") for name in report.field_names)
    assert bool(gate.passed)
    np.testing.assert_allclose(reference.bmag, geometry.B, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference.gradpar, geometry.F, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference.gds22, geometry.g_xx, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference.gds21, geometry.g_xy, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference.gds2, geometry.g_yy, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference.gbdrift0 + reference.cvdrift0, geometry.D_x)
    np.testing.assert_allclose(reference.gbdrift + reference.cvdrift, geometry.D_y)
    np.testing.assert_allclose(report.field_errors, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(gate.observed_value, 0.0, atol=1.0e-13)


def test_external_gist_eik_suite_gate_runs_multiple_stellarator_fixtures():
    paths = (
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000",
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_li383_1.4m.txt_highres_surf12_pol_10_nz0_10000",
        ROOT
        / "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_st_a34_i32v22_beta_35_scaledAUG.txt_highres_surf12_pol_10_nz0_10000",
    )

    gate = run_gx_gist_external_eik_suite_gate(paths, n_theta=17)

    assert bool(gate.passed)
    assert gate.target.name == "gx_gist_external_eik_suite"
    assert dict(gate.target.metadata)["n_references"] == 3
    np.testing.assert_allclose(gate.observed_value, 0.0, atol=1.0e-13)


def test_reduced_rh_and_cyclone_validation_gates_run_and_report_current_gap():
    rh = run_reduced_rosenbluth_hinton_gate(n_z=8, n_vpar=6, n_mu=4, n_steps=5)
    cyclone = run_reduced_cyclone_base_case_gate(n_z=8, n_vpar=6, n_mu=4, n_steps=5)

    assert rh.target.name == "rosenbluth_hinton_q13_eps005"
    assert cyclone.target.name == "cyclone_base_case_gkw_kt05"
    assert jnp.isfinite(rh.observed_value)
    assert jnp.isfinite(cyclone.observed_value)
    assert jnp.isfinite(rh.cost)
    assert jnp.isfinite(cyclone.cost)
    assert not bool(rh.passed)
    assert not bool(cyclone.passed)
    assert "production" in rh.notes
    assert "production" in cyclone.notes
    assert "window" in cyclone.notes
    assert "nperiod=5" in cyclone.notes
    assert "selected ky only" in cyclone.notes


def test_production_cyclone_gate_runs_with_reduced_overrides():
    cyclone = run_production_cyclone_base_case_gate(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
    )

    assert cyclone.target.name == "cyclone_base_case_gkw_kt05"
    assert jnp.isfinite(cyclone.observed_value)
    assert jnp.isfinite(cyclone.cost)
    assert "production-control CBC gate" in cyclone.notes
    assert "nperiod=5" in cyclone.notes


def test_rh_plateau_gate_runs_late_window_metric_without_claiming_pass():
    rh = run_rosenbluth_hinton_plateau_gate(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        t_end=0.05,
        t_start=0.02,
        diagnostic_interval=0.01,
    )

    assert rh.target.name == "rosenbluth_hinton_q13_eps005"
    assert jnp.isfinite(rh.observed_value)
    assert not bool(rh.passed)
    assert "long-time RH plateau gate" in rh.notes
    assert "production" in rh.notes


def test_desc_fixture_can_drive_reduced_benchmark_target_objective():
    fixture = ROOT / "fixtures/desc_geometry_dshape_rho05_alpha0.npz"
    data = np.load(fixture)
    parallel = _parallel_grid_from_fixture_z(data["z"])
    geometry = build_desc_geometry_from_arrays(
        parallel,
        theta=data["theta"],
        phi=data["phi"],
        alpha=data["alpha"],
        rho=data["rho"],
        B=data["B"],
        b_dot_grad_z=data["b_dot_grad_z"],
        grad_psi_sq=data["grad_psi_sq"],
        grad_alpha_sq=data["grad_alpha_sq"],
        grad_psi_dot_grad_alpha=data["grad_psi_dot_grad_alpha"],
        B_cross_gradB_dot_grad_psi=data["B_cross_gradB_dot_grad_psi"],
        B_cross_gradB_dot_grad_alpha=data["B_cross_gradB_dot_grad_alpha"],
        b_cross_kappa_dot_grad_psi=data["b_cross_kappa_dot_grad_psi"],
        b_cross_kappa_dot_grad_alpha=data["b_cross_kappa_dot_grad_alpha"],
    )
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.5, mu_max=1.0))
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.45, ky_values=(0.0, 0.35), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    initial_state = _initial_state(velocity, parallel, fourier)
    target = BenchmarkTarget(
        name="desc_dshape_zero_growth_proxy",
        quantity="selected_growth_rate",
        reference_value=0.0,
        tolerance=1.0,
        source=str(fixture),
    )
    config = SingleSurfaceOptimizationConfig(
        geometry_model="desc",
        dt=0.005,
        n_steps=1,
        selected_ky=1,
        objective_kind="selected_growth",
        store_history=False,
    )
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    knobs = OptimizationKnobs(
        density=0.9,
        temperature=1.2,
        density_gradient=0.8,
        temperature_gradient=2.1,
        q=1.25,
        shat=0.45,
        eps=0.17,
    )

    def objective(temperature_gradient):
        local_knobs = replace(knobs, temperature_gradient=temperature_gradient)
        return single_surface_benchmark_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            target,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
            geometry=geometry,
        ).scalar_objective

    result = single_surface_benchmark_objective(
        knobs,
        velocity,
        parallel,
        fourier,
        initial_state,
        target,
        electron_params=electrons,
        connectivity=connectivity,
        config=config,
        geometry=geometry,
    )
    value, gradient = jax.jit(jax.value_and_grad(objective))(knobs.temperature_gradient)

    assert result.surface_result.geometry.source == "desc"
    assert jnp.isfinite(result.scalar_objective)
    assert jnp.isfinite(result.target_residual)
    np.testing.assert_allclose(value, result.scalar_objective, rtol=2e-12, atol=2e-12)
    assert jnp.isfinite(gradient)


def _parallel_grid_from_fixture_z(z):
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    grid = build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )
    np.testing.assert_allclose(grid.z, z, rtol=2e-12, atol=2e-12)
    return grid


def _parallel_grid_from_theta(theta):
    theta = np.asarray(theta, dtype=float)
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    grid = build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )
    np.testing.assert_allclose(grid.z, z, rtol=2e-12, atol=2e-12)
    return grid


def _initial_state(velocity, parallel, fourier):
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    return 0.01 * (jnp.cos(index / 7.0) + 1j * jnp.sin(index / 9.0))
