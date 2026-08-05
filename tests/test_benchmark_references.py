import csv
import sys
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
    CycloneCoefficientSourceAudit,
    CycloneDiagnosticPackingAudit,
    CycloneIghArakawaAudit,
    CycloneIghArakawaSeriesAudit,
    CycloneMatdatMatrixAudit,
    CycloneProfileOperatorAudit,
    CycloneSelectedKyGapAudit,
    CycloneTermVIIModePackingAudit,
    CycloneTermVIIFieldConventionAudit,
    CycloneVelocitySpaceSliceAudit,
    CycloneVelocitySpaceSliceSeriesAudit,
    CycloneVelocitySpaceSliceSeriesVariantAudit,
    CycloneVparOddSignAudit,
    CycloneTermIFortranAudit,
    CycloneTimeNormalizationAudit,
    CycloneKyScanConventionAudit,
    CycloneKyScanGateReport,
    CycloneSourceTermTrace,
    CycloneTrace,
    ExternalEikProducerReport,
    GxCycloneConventionReport,
    GxCycloneInputReference,
    GkwVelocitySpaceSlice,
    GkwVelocitySpaceSliceSeries,
    GxGrowthRateReference,
    ParallelPhiProfileGateReport,
    ParallelPhiTrace,
    PerKyModeStructureComparisonReport,
    PerKyModeStructureFixture,
    VelocitySliceConventionAudit,
    VelocitySlicePhaseAudit,
    audit_cyclone_selected_ky_gap,
    audit_cyclone_velocity_space_slice,
    audit_cyclone_velocity_space_slice_series,
    audit_cyclone_velocity_space_slice_series_variants,
    audit_parallel_phi_profile_alignment,
    audit_velocity_space_slice_conventions,
    audit_velocity_space_slice_phase_alignment,
    benchmark_target_cost,
    benchmark_target_residual,
    build_desc_gx_eik_reference_from_path,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_flux_tube_geometry_from_gx_eik_reference,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    calibrate_gx_growth_rate_reference_to_target,
    cyclone_base_case_growth_target,
    compare_cyclone_base_case_traces,
    compare_gx_cyclone_input_to_solver_controls,
    compare_geometry_to_gx_eik_reference,
    compare_parallel_phi_traces,
    compare_per_ky_mode_structure_fixtures,
    evaluate_cyclone_ky_scan_convention_audit,
    evaluate_cyclone_ky_scan_gate,
    evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures,
    evaluate_parallel_phi_profile_gate,
    geometry_to_gx_eik_reference,
    gx_growth_rate_target,
    gx_salpha_cyclone_growth_target,
    load_cyclone_trace_csv,
    load_gkw_parallel_phi_trace,
    load_gkw_time_dat_trace,
    load_gkw_velocity_space_slice,
    load_gkw_velocity_space_slice_series,
    load_gx_cyclone_input_reference,
    load_gx_eik_geometry_reference,
    load_gx_growth_rate_reference,
    load_gx_mode_structure_fixture,
    load_per_ky_mode_structure_fixture_csv,
    load_stella_mode_structure_fixture,
    resample_per_ky_mode_structure_fixture,
    resample_gx_eik_geometry_reference,
    run_geometry_to_gx_eik_export_gate,
    run_cyclone_base_case_coefficient_source_audit,
    run_cyclone_base_case_diagnostic_packing_audit,
    run_cyclone_base_case_igh_arakawa_audit,
    run_cyclone_base_case_igh_arakawa_series_audit,
    run_cyclone_base_case_matdat_matrix_audit,
    run_cyclone_base_case_cosin2_gap_audit,
    run_cyclone_base_case_cosin2_velocity_convention_audit,
    run_cyclone_base_case_cosin2_velocity_phase_audit,
    run_cyclone_base_case_cosin2_velocity_series_audit,
    run_cyclone_base_case_cosin2_velocity_series_variant_audit,
    run_cyclone_base_case_cosin2_velocity_slice_audit,
    run_cyclone_base_case_cosin2_term_vii_field_convention_audit,
    run_cyclone_base_case_cosin2_vpar_odd_sign_audit,
    run_desc_gx_eik_external_geometry_gate,
    run_gx_gist_external_eik_suite_gate,
    run_gx_salpha_moment_rhs_mode_structure_fixture,
    run_independent_external_eik_producer_gate,
    run_independent_external_eik_producer_report,
    run_cyclone_base_case_profile_operator_audit,
    run_cyclone_base_case_term_i_fortran_audit,
    run_cyclone_base_case_term_vii_mode_packing_audit,
    run_cyclone_base_case_time_normalization_audit,
    run_cyclone_base_case_term_parity_audit,
    run_cyclone_base_case_source_term_trace,
    run_cyclone_base_case_selected_state_trace,
    run_cyclone_base_case_ky_scan_convention_audit,
    run_cyclone_base_case_ky_scan_gate,
    run_cyclone_base_case_mode_structure_fixture,
    run_cyclone_base_case_parallel_phi_profile_gate,
    run_cyclone_base_case_parallel_phi_trace,
    run_cyclone_base_case_trace,
    run_cyclone_base_case_velocity_space_slice,
    run_cyclone_base_case_velocity_space_slice_series,
    rosenbluth_hinton_residual,
    rosenbluth_hinton_target,
    run_gx_eik_geometry_gate,
    run_production_cyclone_base_case_gate,
    run_production_control_cyclone_ky_scan_convention_audit,
    run_rosenbluth_hinton_plateau_gate,
    run_reduced_cyclone_base_case_gate,
    run_reduced_rosenbluth_hinton_gate,
    run_solver_geometry_to_gx_eik_gate,
    single_surface_benchmark_objective,
    write_cyclone_source_term_trace_csv,
    write_cyclone_ky_scan_convention_audit_csv,
    write_per_ky_mode_structure_fixture_csv,
    write_cyclone_trace_csv,
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
    assert cyclone_metadata["parallel_backend"] == "finite_difference"
    assert cyclone_metadata["parallel_boundary"] == "zero"
    assert cyclone_metadata["parallel_derivative_model"] == "gkw_upwind"
    assert cyclone_metadata["velocity_backend"] == "finite_difference"
    assert rosenbluth_hinton_residual(1.3, 0.05) > rh.reference_value

    value = jnp.asarray(0.2)
    residual = benchmark_target_residual(value, cyclone)
    cost = benchmark_target_cost(value, cyclone)
    gradient = jax.grad(lambda x: benchmark_target_cost(x, cyclone))(value)

    np.testing.assert_allclose(residual, (0.2 - 0.179) / cyclone.tolerance)
    np.testing.assert_allclose(cost, 0.5 * residual**2)
    np.testing.assert_allclose(gradient, residual / cyclone.tolerance)


def test_gx_salpha_cyclone_target_matches_input_metadata():
    target = gx_salpha_cyclone_growth_target()
    metadata = dict(target.metadata)

    assert target.source.endswith("itg_salpha_adiabatic_electrons.in")
    assert metadata["q"] == pytest.approx(1.4)
    assert metadata["shat"] == pytest.approx(0.8)
    assert metadata["epsilon"] == pytest.approx(0.18)
    assert metadata["nperiod"] == 2
    assert metadata["ntheta_per_2pi"] == 32
    assert metadata["n_z_total"] == 96
    assert metadata["n_z"] == 96
    assert metadata["Rmaj_over_Lref"] == pytest.approx(2.77778)
    assert metadata["R_over_Ln"] == pytest.approx(0.8 * 2.77778)
    assert metadata["R_over_LT"] == pytest.approx(2.49 * 2.77778)
    assert metadata["gx_hypercollision_model"] == "kz"


@pytest.mark.external
def test_gx_cyclone_input_reference_records_domain_and_hypercollision_conventions(
    gx_root: Path,
):
    path = gx_root / "benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in"

    reference = load_gx_cyclone_input_reference(path)

    assert isinstance(reference, GxCycloneInputReference)
    assert reference.ntheta_per_2pi == 32
    assert reference.nperiod == 2
    assert reference.n_z_total == 96
    assert reference.nhermite == 48
    assert reference.nlaguerre == 16
    assert reference.boundary == "linked"
    assert reference.geometry == "s-alpha"
    assert reference.hypercollisions_requested
    assert reference.hypercollisions_kz
    assert not reference.hypercollisions_const
    assert reference.nu_hyper_m == pytest.approx(1.0)
    assert reference.p_hyper_m == 20
    assert reference.has_fields_diagnostic
    assert reference.has_moments_diagnostic
    assert not reference.has_eigenfunctions_diagnostic
    np.testing.assert_allclose(reference.ky[:3], [0.0, 0.05, 0.1])
    np.testing.assert_allclose(reference.ky[-1], 0.55)


@pytest.mark.external
def test_gx_cyclone_input_convention_report_separates_numeric_and_physics_gaps(
    gx_root: Path,
):
    path = gx_root / "benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in"
    reference = load_gx_cyclone_input_reference(path)
    target = gx_salpha_cyclone_growth_target()

    current_solver = compare_gx_cyclone_input_to_solver_controls(
        reference,
        target=target,
        ky_values=(0.3, 0.5),
    )
    gx_like_solver = compare_gx_cyclone_input_to_solver_controls(
        reference,
        target=target,
        ky_values=(0.3, 0.5),
        parallel_boundary="linked",
        velocity_representation="hermite_laguerre_moment_rhs",
        hypercollision_model="gx_kz_hypercollision",
        mode_structure_reference="external-per-ky-eigenfunction",
    )
    underresolved = compare_gx_cyclone_input_to_solver_controls(
        reference,
        target=target,
        n_z=32,
        n_vpar=32,
        n_mu=8,
        ky_values=(0.3, 0.5),
        parallel_boundary="linked",
        velocity_representation="hermite_laguerre_moment_rhs",
        hypercollision_model="gx_kz_hypercollision",
        mode_structure_reference="external-per-ky-eigenfunction",
    )

    assert isinstance(current_solver, GxCycloneConventionReport)
    assert not bool(current_solver.passed)
    assert bool(jnp.all(current_solver.metric_passed))
    gap_map = dict(zip(current_solver.gap_names, np.asarray(current_solver.gap_present), strict=True))
    assert gap_map["linked_parallel_boundary_not_enabled"]
    assert gap_map["gx_hermite_laguerre_moment_rhs_not_enabled"]
    assert gap_map["gx_kz_hypercollision_not_enabled"]
    assert gap_map["per_ky_mode_structure_reference_missing"]

    assert bool(gx_like_solver.passed)
    np.testing.assert_allclose(gx_like_solver.metric_values, 0.0, atol=1.0e-12)
    assert not bool(jnp.any(gx_like_solver.gap_present))

    assert not bool(underresolved.passed)
    metric_map = dict(zip(underresolved.metric_names, np.asarray(underresolved.metric_values), strict=True))
    assert metric_map["n_z_total"] == pytest.approx(64.0)
    assert metric_map["n_vpar_or_nhermite"] == pytest.approx(16.0)
    assert metric_map["n_mu_or_nlaguerre"] == pytest.approx(8.0)


def test_gx_growth_rate_reference_loads_synthetic_curve(tmp_path: Path):
    netcdf = pytest.importorskip("netCDF4")
    path = tmp_path / "synthetic_gx_growth.out.nc"
    ky = np.arange(12, dtype=float) * 0.05
    growth = np.asarray(
        [0.0, 0.01, 0.03, 0.05, 0.075, 0.09, 0.09302951, 0.085, 0.07, 0.06, 0.054058794, 0.04]
    )
    frequency = -0.2 - 0.1 * ky
    with netcdf.Dataset(path, "w") as data:
        data.createDimension("time", 4)
        data.createDimension("ky", ky.size)
        data.createDimension("kx", 1)
        data.createDimension("ri", 2)
        grids = data.createGroup("Grids")
        diagnostics = data.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.arange(4, dtype=float)
        grids.createVariable("ky", "f8", ("ky",))[:] = ky
        omega = diagnostics.createVariable(
            "omega_kxkyt", "f8", ("time", "ky", "kx", "ri")
        )
        values = np.zeros((4, ky.size, 1, 2), dtype=float)
        values[..., 0, 0] = frequency
        values[..., 0, 1] = growth
        omega[:] = values

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


def test_gx_big_nc_mode_structure_fixture_loader(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    big_path = tmp_path / "synthetic_gx.big.nc"
    out_path = tmp_path / "synthetic_gx.out.nc"
    ky = np.asarray([0.0, 0.1, 0.3, 0.5], dtype=float)
    kx = np.asarray([0.0], dtype=float)
    z = np.linspace(-np.pi, np.pi, 5)
    times = np.asarray([0.0, 1.0, 2.0], dtype=float)

    with netcdf.Dataset(big_path, "w") as data:
        data.createDimension("time", None)
        data.createDimension("ky", ky.shape[0])
        data.createDimension("kx", kx.shape[0])
        data.createDimension("theta", z.shape[0])
        data.createDimension("ri", 2)
        grids = data.createGroup("Grids")
        diagnostics = data.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = times
        grids.createVariable("ky", "f8", ("ky",))[:] = ky
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        grids.createVariable("theta", "f8", ("theta",))[:] = z
        phi = diagnostics.createVariable(
            "Phi",
            "f8",
            ("time", "ky", "kx", "theta", "ri"),
        )
        values = np.zeros((times.shape[0], ky.shape[0], kx.shape[0], z.shape[0], 2))
        for time_index, _time in enumerate(times):
            for ky_index, _ky in enumerate(ky):
                complex_row = (time_index + 1.0) * (ky_index + 1.0) * np.exp(
                    1j * (ky_index + 1.0) * z
                )
                values[time_index, ky_index, 0, :, 0] = complex_row.real
                values[time_index, ky_index, 0, :, 1] = complex_row.imag
        phi[:] = values

    with netcdf.Dataset(out_path, "w") as data:
        data.createDimension("time", None)
        data.createDimension("ky", ky.shape[0])
        data.createDimension("kx", kx.shape[0])
        data.createDimension("ri", 2)
        grids = data.createGroup("Grids")
        diagnostics = data.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = times
        grids.createVariable("ky", "f8", ("ky",))[:] = ky
        omega = diagnostics.createVariable(
            "omega_kxkyt",
            "f8",
            ("time", "ky", "kx", "ri"),
        )
        omega_values = np.zeros((times.shape[0], ky.shape[0], kx.shape[0], 2))
        omega_values[:, :, 0, 0] = -ky[None, :]
        omega_values[:, :, 0, 1] = 2.0 * ky[None, :]
        omega[:] = omega_values

    fixture = load_gx_mode_structure_fixture(
        big_path,
        growth_reference_path=out_path,
        ky_values=(0.3, 0.5),
        time_index=-1,
        z_scale=1.0 / (2.0 * np.pi),
    )
    observed = replace(
        fixture,
        phi=3.0 * jnp.exp(-0.2j) * fixture.phi,
        source="phase-scaled GX fixture",
    )
    gate = evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures(
        observed,
        fixture,
        growth_tolerance=1.0e-14,
        frequency_tolerance=1.0e-14,
        profile_tolerance=1.0e-12,
        require_profile=True,
    )

    assert isinstance(fixture, PerKyModeStructureFixture)
    np.testing.assert_allclose(fixture.ky, [0.3, 0.5])
    np.testing.assert_allclose(fixture.z, z / (2.0 * np.pi))
    np.testing.assert_allclose(fixture.growth_rate, [0.6, 1.0])
    np.testing.assert_allclose(fixture.frequency, [-0.3, -0.5])
    np.testing.assert_allclose(
        fixture.phi[0],
        values[-1, 2, 0, :, 0] + 1j * values[-1, 2, 0, :, 1],
    )
    assert dict(fixture.metadata)["format"] == "GX Diagnostics/Phi(time,ky,kx,z,ri)"
    assert bool(gate.passed)

    with pytest.raises(ValueError, match="Diagnostics/Phi"):
        load_gx_mode_structure_fixture(out_path)


def test_stella_out_nc_mode_structure_fixture_loader(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    path = tmp_path / "synthetic_stella.out.nc"
    ky = np.asarray([0.0, 0.1, 0.3], dtype=float)
    kx = np.asarray([0.0, 0.2], dtype=float)
    zed = np.linspace(-np.pi, np.pi, 5)
    times = np.asarray([0.0, 1.0, 2.0], dtype=float)

    with netcdf.Dataset(path, "w") as data:
        data.createDimension("t", times.shape[0])
        data.createDimension("tube", 1)
        data.createDimension("zed", zed.shape[0])
        data.createDimension("kx", kx.shape[0])
        data.createDimension("ky", ky.shape[0])
        data.createDimension("ri", 2)
        data.createVariable("t", "f8", ("t",))[:] = times
        data.createVariable("ky", "f8", ("ky",))[:] = ky
        data.createVariable("kx", "f8", ("kx",))[:] = kx
        data.createVariable("zed", "f8", ("zed",))[:] = zed
        phi = data.createVariable(
            "phi_vs_t",
            "f8",
            ("t", "tube", "zed", "kx", "ky", "ri"),
        )
        values = np.zeros((times.shape[0], 1, zed.shape[0], kx.shape[0], ky.shape[0], 2))
        for time_index, _time in enumerate(times):
            for ikx, _kx in enumerate(kx):
                for iky, _ky in enumerate(ky):
                    row = (time_index + 1.0) * (ikx + 1.0) * (iky + 1.0) * np.exp(
                        1j * (iky + 1.0) * zed
                    )
                    values[time_index, 0, :, ikx, iky, 0] = row.real
                    values[time_index, 0, :, ikx, iky, 1] = row.imag
        phi[:] = values
        omega = data.createVariable("omega", "f8", ("t", "kx", "ky", "ri"))
        omega_values = np.zeros((times.shape[0], kx.shape[0], ky.shape[0], 2))
        omega_values[..., 0] = -ky[None, None, :] - kx[None, :, None]
        omega_values[..., 1] = 2.0 * ky[None, None, :]
        omega[:] = omega_values

    fixture = load_stella_mode_structure_fixture(
        path,
        ikx=1,
        ky_values=(0.1, 0.3),
        time_index=-1,
        z_scale=1.0 / (2.0 * np.pi),
    )
    csv_path = tmp_path / "stella_fixture.csv"
    write_per_ky_mode_structure_fixture_csv(csv_path, fixture)
    loaded = load_per_ky_mode_structure_fixture_csv(csv_path)

    assert isinstance(fixture, PerKyModeStructureFixture)
    np.testing.assert_allclose(fixture.ky, [0.1, 0.3])
    np.testing.assert_allclose(fixture.z, zed / (2.0 * np.pi))
    np.testing.assert_allclose(fixture.growth_rate, [0.2, 0.6])
    np.testing.assert_allclose(fixture.frequency, [-0.3, -0.5])
    np.testing.assert_allclose(
        fixture.phi[0],
        values[-1, 0, :, 1, 1, 0] + 1j * values[-1, 0, :, 1, 1, 1],
    )
    assert dict(fixture.metadata)["format"] == "stella phi_vs_t(t,tube,zed,kx,ky,ri)"
    assert fixture.normalization == "stella_out_nc_complex_phi"
    np.testing.assert_allclose(loaded.phi, fixture.phi)

    with pytest.raises(ValueError, match="requested stella ky values"):
        load_stella_mode_structure_fixture(path, ky_values=(0.2,))

    missing_phi = tmp_path / "missing_phi.out.nc"
    with netcdf.Dataset(missing_phi, "w") as data:
        data.createDimension("ky", ky.shape[0])
        data.createDimension("kx", kx.shape[0])
        data.createDimension("zed", zed.shape[0])
        data.createVariable("ky", "f8", ("ky",))[:] = ky
        data.createVariable("kx", "f8", ("kx",))[:] = kx
        data.createVariable("zed", "f8", ("zed",))[:] = zed
    with pytest.raises(ValueError, match="phi_vs_t"):
        load_stella_mode_structure_fixture(missing_phi)


def test_mode_structure_fixture_resampler_handles_periodic_edge_to_center_grid():
    source_z = jnp.asarray([0.0, 0.25, 0.5, 0.75])
    target_z = jnp.asarray([0.125, 0.375, 0.625, 0.875])
    phi = jnp.asarray([[1.0, 1.0j, -1.0, -1.0j]], dtype=jnp.complex128)
    fixture = PerKyModeStructureFixture(
        ky=jnp.asarray([0.3]),
        z=source_z,
        phi=phi,
        growth_rate=jnp.asarray([0.2]),
        frequency=jnp.asarray([-0.1]),
        source="synthetic periodic edge grid",
    )

    with pytest.raises(ValueError, match="outside fixture.z"):
        resample_per_ky_mode_structure_fixture(fixture, target_z)

    resampled = resample_per_ky_mode_structure_fixture(
        fixture,
        target_z,
        periodic=True,
        period=1.0,
    )

    np.testing.assert_allclose(resampled.z, target_z)
    np.testing.assert_allclose(
        resampled.phi,
        jnp.asarray([[0.5 + 0.5j, -0.5 + 0.5j, -0.5 - 0.5j, 0.5 - 0.5j]]),
    )
    assert dict(resampled.metadata)["resampled_periodic"]


def test_cyclone_ky_scan_gate_evaluator_separates_scan_metrics():
    report = evaluate_cyclone_ky_scan_gate(
        ky=(0.2, 0.4),
        matched_reference_ky=(0.2, 0.4),
        observed_growth=(0.11, 0.18),
        reference_growth=(0.10, 0.20),
        observed_frequency=(-0.21, -0.31),
        reference_frequency=(-0.20, -0.30),
        profile_error=(0.01, 0.03),
        growth_tolerance=2.5e-2,
        frequency_tolerance=1.5e-2,
        profile_tolerance=2.0e-2,
        require_profile=True,
        source="synthetic",
    )

    assert isinstance(report, CycloneKyScanGateReport)
    assert not bool(report.passed)
    np.testing.assert_allclose(report.solver_ky, report.ky)
    np.testing.assert_allclose(report.growth_error, jnp.asarray([0.01, -0.02]))
    np.testing.assert_allclose(report.frequency_error, jnp.asarray([-0.01, -0.01]))
    np.testing.assert_array_equal(np.asarray(report.growth_passed), np.asarray([True, True]))
    np.testing.assert_array_equal(np.asarray(report.frequency_passed), np.asarray([True, True]))
    np.testing.assert_array_equal(np.asarray(report.profile_passed), np.asarray([True, False]))
    np.testing.assert_allclose(report.max_growth_error, 0.02)
    np.testing.assert_allclose(report.max_frequency_error, 0.01)
    np.testing.assert_allclose(report.max_profile_error, 0.03)

    without_profile_reference = evaluate_cyclone_ky_scan_gate(
        ky=(0.2,),
        observed_growth=(0.1,),
        reference_growth=(0.1,),
        require_frequency=False,
        require_profile=False,
    )
    assert bool(without_profile_reference.passed)

    optional_metric_errors = evaluate_cyclone_ky_scan_gate(
        ky=(0.2,),
        observed_growth=(0.1,),
        reference_growth=(0.1,),
        observed_frequency=(100.0,),
        reference_frequency=(0.0,),
        profile_error=(100.0,),
        require_frequency=False,
        require_profile=False,
    )
    assert bool(optional_metric_errors.passed)
    np.testing.assert_array_equal(np.asarray(optional_metric_errors.frequency_passed), [True])
    np.testing.assert_array_equal(np.asarray(optional_metric_errors.profile_passed), [True])
    np.testing.assert_allclose(optional_metric_errors.max_frequency_error, 100.0)
    np.testing.assert_allclose(optional_metric_errors.max_profile_error, 100.0)
    np.testing.assert_allclose(without_profile_reference.solver_ky, without_profile_reference.ky)
    assert bool(without_profile_reference.profile_passed[0])


def test_cyclone_ky_scan_convention_audit_ranks_candidates():
    worse = evaluate_cyclone_ky_scan_gate(
        ky=(0.2, 0.4),
        solver_ky=(0.17, 0.34),
        matched_reference_ky=(0.2, 0.4),
        observed_growth=(0.2, 0.1),
        reference_growth=(0.1, 0.1),
        observed_frequency=(0.7, 0.8),
        reference_frequency=(0.2, 0.3),
        growth_tolerance=5.0e-2,
        frequency_tolerance=2.0e-1,
        require_profile=False,
    )
    better = evaluate_cyclone_ky_scan_gate(
        ky=(0.2, 0.4),
        solver_ky=(0.2, 0.4),
        matched_reference_ky=(0.2, 0.4),
        observed_growth=(0.11, 0.09),
        reference_growth=(0.1, 0.1),
        observed_frequency=(0.25, 0.35),
        reference_frequency=(0.2, 0.3),
        growth_tolerance=5.0e-2,
        frequency_tolerance=2.0e-1,
        require_profile=False,
    )

    audit = evaluate_cyclone_ky_scan_convention_audit(
        (worse, better),
        candidate_names=("worse", "better"),
        ky_input_conventions=("k_theta_rhos", "internal_krho"),
        growth_diagnostics=("late_fit", "late_mean_window"),
        normalization_models=("weighted", "gkw_unweighted"),
        observed_frequency_signs=(1.0, -1.0),
        observed_frequency_scales=(1.0, 1.0),
    )

    assert isinstance(audit, CycloneKyScanConventionAudit)
    assert bool(audit.passed)
    assert int(audit.best_index) == 1
    assert audit.candidate_names == ("worse", "better")
    assert audit.growth_diagnostics == ("late_fit", "late_mean_window")
    assert audit.normalization_models == ("weighted", "gkw_unweighted")
    np.testing.assert_allclose(audit.ky, jnp.asarray([0.2, 0.4]))
    np.testing.assert_allclose(audit.solver_ky[1], jnp.asarray([0.2, 0.4]))
    assert float(audit.combined_errors[1]) < float(audit.combined_errors[0])
    np.testing.assert_array_equal(np.asarray(audit.candidate_passed), np.asarray([False, True]))


def test_cyclone_ky_scan_convention_score_respects_ignore_frequency():
    first = evaluate_cyclone_ky_scan_gate(
        ky=(0.5,),
        observed_growth=(0.179,),
        reference_growth=(0.179,),
        observed_frequency=(100.0,),
        reference_frequency=(0.0,),
        growth_tolerance=1.0e-2,
        frequency_tolerance=1.0e-2,
        require_frequency=False,
    )
    second = evaluate_cyclone_ky_scan_gate(
        ky=(0.5,),
        observed_growth=(0.189,),
        reference_growth=(0.179,),
        observed_frequency=(0.0,),
        reference_frequency=(0.0,),
        growth_tolerance=1.0e-2,
        frequency_tolerance=1.0e-2,
        require_frequency=False,
    )

    audit = evaluate_cyclone_ky_scan_convention_audit(
        (first, second),
        candidate_names=("growth_best_frequency_bad", "growth_worse_frequency_best"),
    )

    assert int(audit.best_index) == 0
    assert float(audit.combined_errors[0]) == pytest.approx(0.0)
    assert float(audit.combined_errors[1]) == pytest.approx(1.0)


def test_calibrate_gx_growth_rate_reference_to_selected_cyclone_target():
    reference = GxGrowthRateReference(
        ky=(0.3, 0.5),
        growth_rate=(0.05, 0.1),
        frequency=(0.2, 0.4),
        source="synthetic-gx-scan",
    )
    target = replace(cyclone_base_case_growth_target(), reference_value=0.2)

    calibrated = calibrate_gx_growth_rate_reference_to_target(
        reference,
        target=target,
        target_ky=0.5,
    )
    calibrated_with_frequency = calibrate_gx_growth_rate_reference_to_target(
        reference,
        target=target,
        target_ky=0.5,
        scale_frequency=True,
    )

    np.testing.assert_allclose(calibrated.growth_rate, jnp.asarray([0.1, 0.2]))
    np.testing.assert_allclose(calibrated.frequency, reference.frequency)
    assert calibrated.growth_scale == pytest.approx(2.0)
    assert calibrated.frequency_scale == pytest.approx(1.0)
    np.testing.assert_allclose(
        calibrated_with_frequency.frequency,
        jnp.asarray([0.4, 0.8]),
    )
    assert "growth calibrated" in calibrated.source


@pytest.mark.external
def test_gx_eik_geometry_reference_loads_vmec_gs2_fixture(gx_root: Path):
    path = (
        gx_root
        / "geometry_modules/vmec/tests/"
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


@pytest.mark.external
def test_gx_eik_loader_uses_gist_drift_column_order(gx_root: Path):
    path = (
        gx_root
        / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_li383_1.4m.txt_highres_surf12_pol_10_nz0_10000"
    )

    reference = load_gx_eik_geometry_reference(path)

    np.testing.assert_allclose(reference.cvdrift[0], 9.4946168708e-01)
    np.testing.assert_allclose(reference.cvdrift0[0], -2.1124355474e-02)
    np.testing.assert_allclose(reference.gbdrift[0], 8.8374219166e-01)
    np.testing.assert_allclose(reference.gbdrift0[0], -2.1124355474e-02)


def test_gx_eik_loader_reads_desc_block_eik_fixture():
    path = ROOT / "fixtures/gx_desc_dshape_rho05_alpha0.eik.out"

    reference = load_gx_eik_geometry_reference(path)

    assert reference.theta.shape == (33,)
    assert reference.header[0] == pytest.approx(16.0)
    assert reference.header[2] == pytest.approx(32.0)
    np.testing.assert_allclose(reference.theta[0], -np.pi, rtol=2e-12)
    np.testing.assert_allclose(reference.theta[-1], np.pi, rtol=2e-12)
    np.testing.assert_allclose(reference.bmag[0], 1.1055080508304442)
    assert jnp.all(jnp.isfinite(reference.gds2))


@pytest.mark.external
def test_gx_eik_geometry_gate_matches_solver_kperp_contract(gx_root: Path):
    path = (
        gx_root
        / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    reference = load_gx_eik_geometry_reference(path)
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    sampled = resample_gx_eik_geometry_reference(reference, theta)
    parallel = _parallel_grid_from_theta(theta)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35)))

    result = run_gx_eik_geometry_gate(sampled, parallel, fourier)

    assert bool(result.passed)
    np.testing.assert_allclose(result.observed_value, 0.0, atol=1.0e-13)
    assert result.target.quantity == "max_abs_kperp2_error"


@pytest.mark.external
def test_solver_geometry_to_eik_parity_report_matches_imported_geometry(gx_root: Path):
    path = (
        gx_root
        / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    reference = load_gx_eik_geometry_reference(path)
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    sampled = resample_gx_eik_geometry_reference(reference, theta)
    parallel = _parallel_grid_from_theta(theta)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35)))
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
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35)))
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


@pytest.mark.external
def test_external_gist_eik_suite_gate_runs_multiple_stellarator_fixtures(gx_root: Path):
    paths = (
        gx_root / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000",
        gx_root / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_li383_1.4m.txt_highres_surf12_pol_10_nz0_10000",
        gx_root / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_st_a34_i32v22_beta_35_scaledAUG.txt_highres_surf12_pol_10_nz0_10000",
    )

    names = ("gx-vmec-gist:w7x", "gx-vmec-gist:li383", "gx-vmec-gist:stellarator")
    report = run_independent_external_eik_producer_report(
        paths,
        producer_names=names,
        n_theta=17,
    )
    generic_gate = run_independent_external_eik_producer_gate(
        paths,
        producer_names=names,
        n_theta=17,
    )
    gate = run_gx_gist_external_eik_suite_gate(paths, n_theta=17)

    assert isinstance(report, ExternalEikProducerReport)
    assert bool(report.passed)
    assert report.producer_names == names
    assert report.producer_errors.shape == (3,)
    assert len(report.sources) == 3
    assert report.n_theta == 17
    assert "independent" in report.notes
    assert "matched DESC/GX" in report.notes
    assert bool(generic_gate.passed)
    assert generic_gate.target.name == "independent_external_eik_producer_suite"
    assert bool(gate.passed)
    assert gate.target.name == "gx_gist_external_eik_suite"
    assert dict(gate.target.metadata)["n_references"] == 3
    assert "matched DESC/GX" in gate.notes
    np.testing.assert_allclose(report.producer_errors, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(report.max_abs_error, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(generic_gate.observed_value, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(gate.observed_value, 0.0, atol=1.0e-13)


@pytest.mark.external
def test_desc_gx_eik_reference_matches_external_block_fixture(desc_root: Path):
    if desc_root.exists() and str(desc_root) not in sys.path:
        sys.path.insert(0, str(desc_root))
    try:
        __import__("desc")
    except ModuleNotFoundError:
        pytest.fail(f"DESC could not be imported from configured root: {desc_root}")
    desc_path = desc_root / "desc/examples/DSHAPE_output.h5"
    eik_path = ROOT / "fixtures/gx_desc_dshape_rho05_alpha0.eik.out"

    reference = build_desc_gx_eik_reference_from_path(
        desc_path,
        ntheta=32,
        npol=1,
        rho=0.5,
        alpha=0.0,
    )
    external = load_gx_eik_geometry_reference(eik_path)
    gate = run_desc_gx_eik_external_geometry_gate(desc_path, eik_path)

    np.testing.assert_allclose(reference.bmag, external.bmag, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(reference.gds21, external.gds21, rtol=0.0, atol=2.0e-14)
    assert bool(gate.passed)
    assert gate.target.name == "desc_gx_external_eik_geometry_parity"
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
    assert "velocity_backend=finite_difference" in cyclone.notes
    assert "parallel_boundary=zero" in cyclone.notes


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
    assert "parallel_derivative_model=gkw_upwind" in cyclone.notes
    assert "initial_profile=cosine2" in cyclone.notes
    assert "growth_diagnostic=late_fit" in cyclone.notes


def test_production_cyclone_gate_supports_gkw_time_dat_mean_diagnostic():
    cyclone = run_production_cyclone_base_case_gate(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        growth_diagnostic="late_mean_window",
    )

    assert jnp.isfinite(cyclone.observed_value)
    assert "growth_diagnostic=late_mean_window" in cyclone.notes

    with pytest.raises(ValueError, match="growth_diagnostic"):
        run_production_cyclone_base_case_gate(
            n_z=8,
            n_vpar=6,
            n_mu=4,
            steps_per_window=1,
            n_windows=1,
            growth_diagnostic="unsupported",
        )


def test_production_cyclone_gate_supports_gkw_igh_backend():
    cyclone = run_production_cyclone_base_case_gate(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        parallel_derivative_model="gkw_igh",
    )

    assert jnp.isfinite(cyclone.observed_value)
    assert "parallel_derivative_model=gkw_igh" in cyclone.notes
    assert "velocity_recurrence_rate=0.2" in cyclone.notes


def test_cyclone_ky_scan_gate_runs_reduced_against_synthetic_reference():
    reference = GxGrowthRateReference(
        ky=(0.2, 0.5),
        growth_rate=(0.0, 0.0),
        frequency=(0.0, 0.0),
        source="synthetic-gx-scan",
    )

    report = run_cyclone_base_case_ky_scan_gate(
        reference=reference,
        ky_values=(0.2, 0.5),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        parallel_derivative_model="gkw_igh",
        growth_tolerance=1.0e3,
        frequency_tolerance=1.0e3,
        require_profile=False,
    )

    assert isinstance(report, CycloneKyScanGateReport)
    assert bool(report.passed)
    assert report.ky.shape == (2,)
    assert report.observed_growth.shape == (2,)
    assert report.observed_frequency.shape == (2,)
    assert jnp.all(jnp.isfinite(report.observed_growth))
    assert jnp.all(jnp.isfinite(report.observed_frequency))
    assert jnp.all(report.solver_ky > 0.0)
    assert not np.allclose(np.asarray(report.solver_ky), np.asarray(report.ky))
    np.testing.assert_allclose(report.matched_reference_ky, jnp.asarray([0.2, 0.5]))
    assert "ky_input_convention=k_theta_rhos" in report.notes
    assert "multi-ky Cyclone/ITG scan gate" in report.notes

    internal_report = run_cyclone_base_case_ky_scan_gate(
        reference=reference,
        ky_values=(0.2,),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=1,
        parallel_derivative_model="gkw_igh",
        ky_input_convention="internal_krho",
        observed_frequency_sign=-1.0,
        growth_tolerance=1.0e3,
        frequency_tolerance=1.0e3,
        require_profile=False,
    )
    np.testing.assert_allclose(internal_report.solver_ky, internal_report.ky)
    assert "ky_input_convention=internal_krho" in internal_report.notes
    assert "observed_frequency_sign=-1" in internal_report.notes


def test_cyclone_ky_scan_convention_audit_runs_reduced_candidates():
    reference = GxGrowthRateReference(
        ky=(0.2,),
        growth_rate=(0.0,),
        frequency=(0.0,),
        source="synthetic-gx-scan",
    )

    audit = run_cyclone_base_case_ky_scan_convention_audit(
        reference=reference,
        ky_values=(0.2,),
        ky_input_conventions=("k_theta_rhos",),
        growth_diagnostics=("late_fit", "late_mean_window"),
        normalization_models=("weighted", "gkw_unweighted"),
        observed_frequency_signs=(1.0,),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=1,
        parallel_derivative_model="gkw_igh",
        growth_tolerance=1.0e3,
        frequency_tolerance=1.0e3,
        require_profile=False,
    )

    assert isinstance(audit, CycloneKyScanConventionAudit)
    assert bool(audit.passed)
    assert audit.ky.shape == (1,)
    assert audit.observed_growth.shape == (4, 1)
    assert audit.observed_frequency.shape == (4, 1)
    assert audit.candidate_names == (
        "late_fit:weighted:k_theta_rhos:freq_sign=1:freq_scale=1",
        "late_fit:gkw_unweighted:k_theta_rhos:freq_sign=1:freq_scale=1",
        "late_mean_window:weighted:k_theta_rhos:freq_sign=1:freq_scale=1",
        "late_mean_window:gkw_unweighted:k_theta_rhos:freq_sign=1:freq_scale=1",
    )
    assert audit.ky_input_conventions == ("k_theta_rhos",) * 4
    assert audit.growth_diagnostics == (
        "late_fit",
        "late_fit",
        "late_mean_window",
        "late_mean_window",
    )
    assert audit.normalization_models == (
        "weighted",
        "gkw_unweighted",
        "weighted",
        "gkw_unweighted",
    )
    assert jnp.all(jnp.isfinite(audit.observed_frequency))
    np.testing.assert_allclose(audit.observed_frequency[0], audit.observed_frequency[1])
    assert int(audit.best_index) in (0, 1, 2, 3)
    assert "scan convention audit" in audit.notes


def test_cyclone_ky_scan_convention_audit_csv_writer_records_candidate_rows(tmp_path):
    reference = GxGrowthRateReference(
        ky=(0.2,),
        growth_rate=(0.0,),
        frequency=(0.0,),
        source="synthetic-gx-scan",
    )
    audit = run_cyclone_base_case_ky_scan_convention_audit(
        reference=reference,
        ky_values=(0.2,),
        ky_input_conventions=("k_theta_rhos",),
        growth_diagnostics=("late_fit", "late_mean_window"),
        normalization_models=("gkw_unweighted",),
        observed_frequency_signs=(1.0, -1.0),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=1,
        parallel_derivative_model="gkw_igh",
        growth_tolerance=1.0e3,
        frequency_tolerance=1.0e3,
        require_profile=False,
    )
    path = tmp_path / "ky_scan_audit.csv"

    write_cyclone_ky_scan_convention_audit_csv(path, audit)

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["candidate_name"] == (
        "late_fit:gkw_unweighted:k_theta_rhos:freq_sign=1:freq_scale=1"
    )
    assert rows[0]["growth_diagnostic"] == "late_fit"
    assert rows[0]["normalization_model"] == "gkw_unweighted"
    assert float(rows[0]["requested_ky"]) == pytest.approx(0.2)
    assert float(rows[0]["solver_ky"]) == pytest.approx(float(audit.solver_ky[0, 0]))
    assert {row["best_candidate"] for row in rows} <= {"True", "False"}


def test_production_control_cyclone_ky_scan_convention_audit_runs_reduced_override():
    reference = GxGrowthRateReference(
        ky=(0.2,),
        growth_rate=(0.0,),
        frequency=(0.0,),
        source="synthetic-gx-scan",
    )

    audit = run_production_control_cyclone_ky_scan_convention_audit(
        reference=reference,
        ky_values=(0.2,),
        ky_input_conventions=("k_theta_rhos",),
        growth_diagnostics=("late_mean_window",),
        normalization_models=("gkw_unweighted",),
        observed_frequency_signs=(1.0,),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=1,
        growth_tolerance=1.0e3,
        frequency_tolerance=1.0e3,
        require_profile=False,
    )

    assert isinstance(audit, CycloneKyScanConventionAudit)
    assert bool(audit.passed)
    assert audit.candidate_names == (
        "late_mean_window:gkw_unweighted:k_theta_rhos:freq_sign=1:freq_scale=1",
    )
    assert audit.growth_diagnostics == ("late_mean_window",)
    assert audit.normalization_models == ("gkw_unweighted",)
    assert "production-control multi-ky" in audit.notes


def test_production_cyclone_selected_ky_gate_passes_matched_gkw_control_resolution():
    cyclone = run_production_cyclone_base_case_gate(
        n_z=48,
        n_vpar=32,
        n_mu=8,
        steps_per_window=20,
        n_windows=80,
        parallel_derivative_model="gkw_igh",
        growth_diagnostic="late_mean_window",
        initial_profile="cosine2",
    )

    assert bool(cyclone.passed)
    np.testing.assert_allclose(cyclone.observed_value, 0.17799905626204374, atol=1.0e-8)
    assert "growth_diagnostic=late_mean_window" in cyclone.notes
    assert "parallel_derivative_model=gkw_igh" in cyclone.notes
    assert "production GKW/GX tolerance ladder passed" in cyclone.notes


def test_cyclone_term_parity_audit_covers_gkw_conventions():
    report = run_cyclone_base_case_term_parity_audit(n_z=8, n_vpar=6, n_mu=4)

    assert bool(report.passed)
    assert report.max_abs_error < 5.0e-13
    assert report.term_names == (
        "drift_frequency",
        "equilibrium_drive",
        "drift_field_drive",
        "boundary_map",
        "grid_normalization",
        "rhs_assembly",
    )
    assert "matrix_vs_gkw_parallel_boundary_delta" in report.notes


def test_cyclone_trace_records_window_diagnostics_and_compares():
    trace = run_cyclone_base_case_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
    )
    comparison = compare_cyclone_base_case_traces(trace, trace)

    assert trace.times.shape == (4,)
    np.testing.assert_allclose(trace.times, jnp.asarray([0.0, 0.006, 0.012, 0.018]))
    assert jnp.all(jnp.isfinite(trace.raw_amplitude))
    assert jnp.all(jnp.isfinite(trace.physical_amplitude))
    assert jnp.all(jnp.isfinite(trace.window_growth))
    assert jnp.all(jnp.isfinite(trace.fitted_growth))
    assert abs(float(trace.window_growth[1])) < 10.0
    assert jnp.all(trace.phi_norm > 0.0)
    assert jnp.all(trace.state_norm > 0.0)
    assert jnp.all(trace.rhs_norm > 0.0)
    assert bool(comparison.passed)
    np.testing.assert_allclose(comparison.max_abs_error, 0.0)
    assert "trace-level CBC comparison" in comparison.notes


def test_cyclone_trace_supports_gkw_cosine_initial_profile():
    trace = run_cyclone_base_case_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine",
    )

    assert trace.times.shape == (3,)
    assert jnp.all(jnp.isfinite(trace.physical_amplitude))
    assert "initial_profile=cosine" in trace.notes

    with pytest.raises(ValueError, match="initial_profile"):
        run_cyclone_base_case_trace(
            n_z=8,
            n_vpar=6,
            n_mu=4,
            steps_per_window=1,
            n_windows=1,
            initial_profile="unsupported",
        )


def test_cyclone_source_term_trace_reconstructs_gkw_igh_rhs(tmp_path):
    trace = run_cyclone_base_case_source_term_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_windows=(0, 1, 2),
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    leaves, treedef = jax.tree_util.tree_flatten(trace)
    path = tmp_path / "source_terms.csv"

    write_cyclone_source_term_trace_csv(path, trace)

    assert isinstance(jax.tree_util.tree_unflatten(treedef, leaves), CycloneSourceTermTrace)
    assert trace.times.shape == (3,)
    np.testing.assert_allclose(trace.times, jnp.asarray([0.0, 0.006, 0.012]))
    assert trace.term_norms.shape == (3, len(trace.term_names))
    assert trace.term_names == (
        "gkw_igh_streaming_mirror_recurrence",
        "magnetic_drift",
        "equilibrium_drive",
        "gkw_parallel_field_drive",
        "drift_field_drive",
        "dissipation",
    )
    assert jnp.all(jnp.isfinite(trace.term_norms))
    assert jnp.all(trace.phi_norm > 0.0)
    assert jnp.all(trace.state_norm > 0.0)
    assert jnp.all(trace.rhs_norm > 0.0)
    assert jnp.max(trace.reconstruction_error) < 5.0e-13
    assert path.read_text().splitlines()[0].startswith("time,phi_norm,state_norm")

    post_trace = run_cyclone_base_case_source_term_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_windows=(1, 2),
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
        snapshot_timing="post_normalization",
    )
    assert post_trace.times.shape == (2,)
    assert jnp.max(post_trace.reconstruction_error) < 5.0e-13
    assert "snapshot_timing=post_normalization" in post_trace.notes

    with pytest.raises(ValueError, match="strictly increasing"):
        run_cyclone_base_case_source_term_trace(
            n_z=8,
            n_vpar=6,
            n_mu=4,
            steps_per_window=1,
            output_windows=(1, 1),
        )


def test_cyclone_selected_state_trace_records_post_normalization_snapshots():
    trace = run_cyclone_base_case_selected_state_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_windows=(1, 2),
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
        snapshot_timing="post_normalization",
    )

    assert trace.steps.shape == (2,)
    np.testing.assert_allclose(trace.times, jnp.asarray([0.006, 0.012]))
    assert trace.state.shape == (2, 6, 4, 8)
    assert trace.phi.shape == (2, 8)
    assert jnp.all(jnp.isfinite(trace.state.real))
    assert jnp.all(jnp.isfinite(trace.phi.real))
    assert "snapshot_timing=post_normalization" in trace.notes


def test_per_ky_mode_structure_fixture_csv_roundtrip_and_phase_gate(tmp_path):
    z = jnp.linspace(-1.0, 1.0, 5)
    phi = jnp.asarray(
        [
            jnp.exp(1j * jnp.pi * z),
            (1.0 + 0.25j) * jnp.cos(jnp.pi * z),
        ],
        dtype=jnp.complex128,
    )
    reference = PerKyModeStructureFixture(
        ky=jnp.asarray([0.3, 0.5]),
        z=z,
        phi=phi,
        growth_rate=jnp.asarray([0.11, 0.18]),
        frequency=jnp.asarray([-0.02, -0.03]),
        source="synthetic-reference",
    )
    observed = PerKyModeStructureFixture(
        ky=jnp.asarray([0.3, 0.5]),
        z=z,
        phi=2.5 * jnp.exp(0.7j) * phi,
        growth_rate=jnp.asarray([0.111, 0.179]),
        frequency=jnp.asarray([-0.021, -0.029]),
        source="synthetic-observed",
        normalization="scaled_global_phase",
    )

    path = tmp_path / "mode_structure.csv"
    write_per_ky_mode_structure_fixture_csv(path, reference)
    loaded = load_per_ky_mode_structure_fixture_csv(path)
    report = compare_per_ky_mode_structure_fixtures(
        observed,
        loaded,
        growth_tolerance=2.0e-3,
        frequency_tolerance=2.0e-3,
        phi_tolerance=2.0e-13,
    )

    assert isinstance(loaded, PerKyModeStructureFixture)
    assert isinstance(report, PerKyModeStructureComparisonReport)
    np.testing.assert_allclose(loaded.ky, reference.ky)
    np.testing.assert_allclose(loaded.z, reference.z)
    np.testing.assert_allclose(loaded.phi, reference.phi)
    assert bool(report.passed)
    assert jnp.max(report.phi_direct_error) > 1.0e-1
    assert float(report.max_phi_phase_aligned_error) < 2.0e-13

    scan_gate = evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures(
        observed,
        loaded,
        growth_tolerance=2.0e-3,
        frequency_tolerance=2.0e-3,
        profile_tolerance=2.0e-13,
        require_profile=True,
    )
    assert bool(scan_gate.passed)


def test_solver_mode_structure_fixture_feeds_scan_gate():
    fixture = run_cyclone_base_case_mode_structure_fixture(
        ky_values=(0.3, 0.5),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=1,
        n_windows=2,
        growth_window_fraction=0.0,
        parallel_derivative_model="gkw_igh",
        normalization_model="gkw_unweighted",
        initial_profile="cosine2",
    )
    observed = replace(
        fixture,
        phi=1.7 * jnp.exp(0.31j) * fixture.phi,
        source="phase-scaled solver fixture",
    )

    assert isinstance(fixture, PerKyModeStructureFixture)
    assert fixture.phi.shape == (2, 8)
    assert fixture.growth_rate.shape == fixture.ky.shape
    assert jnp.all(jnp.isfinite(fixture.phi.real))
    assert dict(fixture.metadata)["solver_ky"][0] != pytest.approx(0.3)

    gate = evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures(
        observed,
        fixture,
        growth_tolerance=1.0e-14,
        frequency_tolerance=1.0e-14,
        profile_tolerance=1.0e-12,
        require_profile=True,
    )
    assert bool(gate.passed)
    np.testing.assert_allclose(gate.profile_error, 0.0, atol=1.0e-12)


def test_gx_salpha_moment_rhs_mode_structure_fixture_runs_reduced_gate():
    fixture = run_gx_salpha_moment_rhs_mode_structure_fixture(
        ky_values=(0.3, 0.5),
        n_z=8,
        n_hermite=5,
        n_laguerre=4,
        nperiod=1,
        dt=0.01,
        steps_per_window=1,
        n_windows=3,
        growth_window_fraction=1.0,
        p_hyper_m=2,
    )
    observed = replace(
        fixture,
        phi=0.4 * jnp.exp(-0.2j) * fixture.phi,
        source="phase-scaled moment fixture",
    )

    assert isinstance(fixture, PerKyModeStructureFixture)
    assert fixture.phi.shape == (2, 8)
    assert fixture.normalization == "complex_phi_gx_moment_rhs"
    assert dict(fixture.metadata)["model"] == "reduced_gx_salpha_moment_rhs"
    assert jnp.all(jnp.isfinite(fixture.phi.real))
    assert jnp.all(jnp.isfinite(fixture.growth_rate))
    assert jnp.all(jnp.isfinite(fixture.frequency))

    gate = evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures(
        observed,
        fixture,
        growth_tolerance=1.0e-14,
        frequency_tolerance=1.0e-14,
        profile_tolerance=1.0e-12,
        require_profile=True,
    )
    assert bool(gate.passed)
    np.testing.assert_allclose(gate.profile_error, 0.0, atol=1.0e-12)


def test_cyclone_trace_csv_roundtrip_and_selected_field_comparison(tmp_path):
    trace = run_cyclone_base_case_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
    )
    path = tmp_path / "trace.csv"

    write_cyclone_trace_csv(path, trace)
    loaded = load_cyclone_trace_csv(path, source="roundtrip")
    shifted_raw = replace(loaded, raw_amplitude=2.0 * loaded.raw_amplitude)
    physical_report = compare_cyclone_base_case_traces(
        trace,
        shifted_raw,
        field_names=(
            "times",
            "physical_amplitude",
            "window_growth",
            "fitted_growth",
            "physical_phi_norm",
            "physical_state_norm",
            "physical_rhs_norm",
        ),
    )
    full_report = compare_cyclone_base_case_traces(trace, shifted_raw)

    assert path.read_text().splitlines()[0].startswith("time,raw_amplitude")
    assert loaded.source == "roundtrip"
    np.testing.assert_allclose(loaded.physical_amplitude, trace.physical_amplitude)
    assert bool(physical_report.passed)
    assert not bool(full_report.passed)


def test_gkw_time_dat_trace_loader_reconstructs_relative_amplitude(tmp_path):
    path = tmp_path / "time.dat"
    path.write_text(
        "\n".join(
            (
                "! time growth optional_frequency",
                "0.0 0.0 0.0",
                "1.0 2.0e-1 1.0e-2",
                "2.0 2.0e-1 1.0e-2",
            )
        )
    )

    trace = load_gkw_time_dat_trace(path, source="gkw-fixture")
    comparison = compare_cyclone_base_case_traces(
        trace,
        trace,
        field_names=("times", "physical_amplitude", "window_growth", "fitted_growth"),
    )

    np.testing.assert_allclose(trace.times, jnp.asarray([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(trace.window_growth, jnp.asarray([0.0, 0.2, 0.2]))
    np.testing.assert_allclose(trace.physical_amplitude, jnp.exp(jnp.asarray([0.0, 0.2, 0.4])))
    np.testing.assert_allclose(trace.fitted_growth[-1], 0.2)
    assert trace.source == "gkw-fixture"
    assert "field/state/RHS norms unavailable" in trace.notes
    assert bool(comparison.passed)


def test_gkw_time_dat_trace_loader_rejects_invalid_time_grid(tmp_path):
    path = tmp_path / "time.dat"
    path.write_text("1.0 0.2\n0.5 0.2\n")

    with pytest.raises(ValueError, match="strictly increasing"):
        load_gkw_time_dat_trace(path)


def test_gkw_parallel_phi_trace_loader_compares_row_normalized_profiles(tmp_path):
    phi_path = tmp_path / "parallel_phi.dat"
    time_path = tmp_path / "time.dat"
    phi_path.write_text("1.0 2.0 1.0\n2.0 4.0 2.0\n")
    time_path.write_text("0.1 0.2\n0.2 0.3\n")

    reference = load_gkw_parallel_phi_trace(
        phi_path,
        time_path=time_path,
        z=(-1.0, 0.0, 1.0),
        source="gkw-parphi-fixture",
    )
    scaled = load_gkw_parallel_phi_trace(
        phi_path,
        times=(0.1, 0.2),
        z=(-1.0, 0.0, 1.0),
        source="scaled",
    )
    scaled = replace(scaled, phi_power=10.0 * scaled.phi_power)
    normalized_report = compare_parallel_phi_traces(reference, scaled)
    absolute_report = compare_parallel_phi_traces(
        reference,
        scaled,
        normalize_profiles=False,
    )

    assert reference.times.shape == (2,)
    assert reference.z.shape == (3,)
    assert reference.phi_power.shape == (2, 3)
    assert reference.source == "gkw-parphi-fixture"
    assert "parallel_phi.dat" in reference.notes
    assert bool(normalized_report.passed)
    assert not bool(absolute_report.passed)
    np.testing.assert_allclose(normalized_report.max_abs_error, 0.0)


def test_parallel_phi_profile_audit_detects_output_order_shift():
    observed = ParallelPhiTrace(
        times=(0.1, 0.2),
        z=(0.0, 1.0, 2.0),
        phi_power=((0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
        source="observed",
    )
    shifted_reference = ParallelPhiTrace(
        times=(0.1, 0.2),
        z=(0.0, 1.0, 2.0),
        phi_power=((0.0, 0.0, 1.0), (0.0, 0.0, 2.0)),
        source="shifted-reference",
    )

    audit = audit_parallel_phi_profile_alignment(
        observed,
        shifted_reference,
        tolerance=1.0e-12,
    )

    assert bool(audit.passed)
    assert int(audit.best_shift) == 2
    np.testing.assert_allclose(audit.best_aligned_max_error, 0.0)
    np.testing.assert_allclose(audit.best_shift_profile_errors, 0.0)
    assert float(jnp.max(audit.direct_profile_errors)) > 0.9
    np.testing.assert_allclose(audit.total_power_ratio, jnp.asarray([1.0, 1.0]))
    np.testing.assert_allclose(audit.peak_z_error, jnp.asarray([-1.0, -1.0]))
    np.testing.assert_allclose(audit.second_moment_error, jnp.asarray([0.0, 0.0]))
    assert int(audit.worst_time_index) == 0
    assert int(audit.worst_z_index) == 1
    np.testing.assert_allclose(audit.worst_time, 0.1)
    np.testing.assert_allclose(audit.worst_z, 1.0)
    np.testing.assert_allclose(audit.worst_signed_error, 1.0)
    np.testing.assert_allclose(audit.worst_observed_value, 1.0)
    np.testing.assert_allclose(audit.worst_reference_value, 0.0)
    assert "alignment/normalization audit" in audit.notes


def test_parallel_phi_profile_gate_uses_direct_shape_not_best_shift():
    observed = ParallelPhiTrace(
        times=(0.1, 0.2),
        z=(0.0, 1.0, 2.0),
        phi_power=((0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
        source="observed",
    )
    shifted_reference = ParallelPhiTrace(
        times=(0.1, 0.2),
        z=(0.0, 1.0, 2.0),
        phi_power=((0.0, 0.0, 1.0), (0.0, 0.0, 2.0)),
        source="shifted-reference",
    )

    gate = evaluate_parallel_phi_profile_gate(
        observed,
        shifted_reference,
        profile_tolerance=1.0e-12,
        tolerance_ladder=(2.0, 0.5),
    )
    metrics = dict(zip(gate.metric_names, np.asarray(gate.metric_values), strict=True))

    assert isinstance(gate, ParallelPhiProfileGateReport)
    assert not bool(gate.passed)
    np.testing.assert_allclose(gate.max_abs_error, 1.0)
    np.testing.assert_array_equal(np.asarray(gate.tolerance_passed), np.asarray([True, False]))
    np.testing.assert_allclose(metrics["row_normalized_direct_max"], 1.0)
    np.testing.assert_allclose(metrics["row_normalized_best_aligned_max"], 0.0)
    np.testing.assert_allclose(metrics["best_circular_shift"], 2.0)
    np.testing.assert_allclose(metrics["total_power_ratio_max_deviation"], 0.0)
    assert "direct row-normalized" in gate.notes


def test_cyclone_selected_ky_gap_audit_aligns_post_window_samples():
    solver_trace = CycloneTrace(
        times=(0.0, 1.0, 2.0),
        raw_amplitude=(1.0, np.exp(0.1), np.exp(0.3)),
        physical_amplitude=(1.0, np.exp(0.1), np.exp(0.3)),
        window_growth=(0.0, 0.1, 0.2),
        fitted_growth=(0.0, 0.1, 0.15),
        phi_norm=(0.0, 0.0, 0.0),
        state_norm=(0.0, 0.0, 0.0),
        rhs_norm=(0.0, 0.0, 0.0),
        log_normalization=(0.0, 0.0, 0.0),
        source="solver",
    )
    reference_trace = CycloneTrace(
        times=(1.0, 2.0),
        raw_amplitude=(1.0, np.exp(0.2)),
        physical_amplitude=(1.0, np.exp(0.2)),
        window_growth=(0.1, 0.2),
        fitted_growth=(0.0, 0.2),
        phi_norm=(0.0, 0.0),
        state_norm=(0.0, 0.0),
        rhs_norm=(0.0, 0.0),
        log_normalization=(0.0, 0.0),
        source="reference",
    )
    solver_profile = ParallelPhiTrace(
        times=(1.0, 2.0),
        z=(-0.5, 0.5),
        phi_power=((0.25, 0.75), (0.5, 0.5)),
        source="solver-profile",
    )
    reference_profile = ParallelPhiTrace(
        times=(1.0, 2.0),
        z=(-0.5, 0.5),
        phi_power=((0.25, 0.75), (0.5, 0.5)),
        source="reference-profile",
    )

    audit = audit_cyclone_selected_ky_gap(
        solver_trace,
        reference_trace,
        solver_profile,
        reference_profile,
        growth_tolerance=1.0e-12,
        profile_tolerance=1.0e-12,
    )

    assert isinstance(audit, CycloneSelectedKyGapAudit)
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.times, jnp.asarray([1.0, 2.0]))
    np.testing.assert_allclose(audit.window_growth_delta, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.late_fit_delta, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.max_profile_error, 0.0, atol=1.0e-14)
    assert "selected-ky growth/profile gap audit" in audit.notes


def test_gkw_parallel_phi_loader_reads_matched_selected_ky_fixture():
    path = ROOT / "fixtures/gkw_cyclone_selected_ky_parallel_phi.dat"
    time_path = ROOT / "fixtures/gkw_cyclone_selected_ky_time.dat"
    trace = load_gkw_parallel_phi_trace(path, time_path=time_path)

    assert trace.times.shape == (80,)
    assert trace.phi_power.shape == (80, 48)
    assert jnp.all(jnp.isfinite(trace.phi_power))
    assert jnp.all(trace.phi_power >= 0.0)
    np.testing.assert_allclose(trace.times[0], 0.06)
    np.testing.assert_allclose(trace.times[-1], 4.79997)


def test_cyclone_parallel_phi_trace_records_gkw_style_profiles():
    trace = run_cyclone_base_case_parallel_phi_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
    )
    comparison = compare_parallel_phi_traces(trace, trace)

    assert trace.times.shape == (2,)
    np.testing.assert_allclose(trace.times, jnp.asarray([0.006, 0.012]))
    assert trace.phi_power.shape == (2, 8)
    assert jnp.all(jnp.isfinite(trace.phi_power))
    assert jnp.all(trace.phi_power >= 0.0)
    assert "initial_profile=cosine" in trace.notes
    assert "normalization_model=gkw_unweighted" in trace.notes
    np.testing.assert_allclose(jnp.sum(trace.phi_power, axis=1), 1.0, rtol=2e-12, atol=2e-12)
    assert bool(comparison.passed)

    gkw_igh_trace = run_cyclone_base_case_parallel_phi_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    assert gkw_igh_trace.phi_power.shape == (2, 8)
    assert jnp.all(jnp.isfinite(gkw_igh_trace.phi_power))
    assert "parallel_derivative_model=gkw_igh" in gkw_igh_trace.notes
    assert "velocity_recurrence_rate=0.2" in gkw_igh_trace.notes

    with pytest.raises(ValueError, match="normalization_model"):
        run_cyclone_base_case_parallel_phi_trace(
            n_z=8,
            n_vpar=6,
            n_mu=4,
            steps_per_window=1,
            n_windows=1,
            normalization_model="unsupported",
        )


def test_cosin2_gap_audit_runner_accepts_matched_reduced_fixtures(tmp_path):
    solver_trace = run_cyclone_base_case_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    solver_profile = run_cyclone_base_case_parallel_phi_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    time_path = tmp_path / "time.dat"
    phi_path = tmp_path / "parallel_phi.dat"
    with time_path.open("w") as handle:
        for time, growth in zip(
            np.asarray(solver_trace.times[1:]),
            np.asarray(solver_trace.window_growth[1:]),
            strict=True,
        ):
            handle.write(f"{time:.16e} {growth:.16e}\n")
    with phi_path.open("w") as handle:
        for row in np.asarray(solver_profile.phi_power):
            handle.write(" ".join(f"{value:.16e}" for value in row) + "\n")

    audit = run_cyclone_base_case_cosin2_gap_audit(
        gkw_time_path=time_path,
        gkw_parallel_phi_path=phi_path,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        growth_tolerance=1.0e-10,
        profile_tolerance=1.0e-10,
    )

    assert bool(audit.passed)
    np.testing.assert_allclose(audit.late_mean_delta, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(audit.max_profile_error, 0.0, atol=1.0e-12)


def test_parallel_phi_profile_gate_runner_accepts_matched_reduced_fixture(tmp_path):
    solver_profile = run_cyclone_base_case_parallel_phi_trace(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    time_path = tmp_path / "time.dat"
    phi_path = tmp_path / "parallel_phi.dat"
    with time_path.open("w") as handle:
        for time in np.asarray(solver_profile.times):
            handle.write(f"{time:.16e} 0.0\n")
    with phi_path.open("w") as handle:
        for row in np.asarray(solver_profile.phi_power):
            handle.write(" ".join(f"{value:.16e}" for value in row) + "\n")

    gate = run_cyclone_base_case_parallel_phi_profile_gate(
        gkw_time_path=time_path,
        gkw_parallel_phi_path=phi_path,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        profile_tolerance=1.0e-10,
        tolerance_ladder=(1.0e-8, 1.0e-10),
    )

    assert bool(gate.passed)
    np.testing.assert_allclose(gate.max_abs_error, 0.0, atol=1.0e-12)
    assert bool(jnp.all(gate.tolerance_passed))


def test_gkw_velocity_space_slice_loader_reads_distr_files(tmp_path):
    vpar = np.broadcast_to(np.array([-1.0, 0.0, 1.0]), (2, 3))
    vperp = np.broadcast_to(np.array([[0.25], [0.75]]), (2, 3))
    imag = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    real = -imag
    paths = [tmp_path / f"distr{index}.dat" for index in range(1, 5)]
    for path, values in zip(paths, (vpar, vperp, imag, real), strict=True):
        np.savetxt(path, values)
    time_path = tmp_path / "time.dat"
    time_path.write_text("1.0 0.1\n2.0 0.2\n")

    loaded = load_gkw_velocity_space_slice(*paths, time_path=time_path, source="fixture")

    assert isinstance(loaded, GkwVelocitySpaceSlice)
    np.testing.assert_allclose(loaded.vpar, vpar)
    np.testing.assert_allclose(loaded.vperp, vperp)
    np.testing.assert_allclose(loaded.real_part, real)
    np.testing.assert_allclose(loaded.imag_part, imag)
    np.testing.assert_allclose(loaded.time, 2.0)
    assert "distr*.dat velocity-space slice" in loaded.notes


def test_gkw_velocity_space_slice_loader_reads_matched_cosin2_fixture():
    loaded = load_gkw_velocity_space_slice(
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat",
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat",
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat",
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat",
        time_path=ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat",
    )

    assert loaded.vpar.shape == (8, 32)
    np.testing.assert_allclose(loaded.vpar[0, 0], -2.90625, atol=5.0e-5)
    np.testing.assert_allclose(loaded.vpar[-1, -1], 2.90625, atol=5.0e-5)
    np.testing.assert_allclose(loaded.time, 4.8)
    assert float(jnp.max(jnp.abs(loaded.real_part))) > 0.0
    assert float(jnp.max(jnp.abs(loaded.imag_part))) > 0.0


def test_gkw_velocity_space_slice_series_loader_reads_matched_cosin2_fixture():
    loaded = load_gkw_velocity_space_slice_series(
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr",
        time_path=ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr/time.dat",
    )

    assert isinstance(loaded, GkwVelocitySpaceSliceSeries)
    np.testing.assert_array_equal(loaded.snapshot_indices, np.array([20, 800, 1600]))
    np.testing.assert_allclose(loaded.times, np.array([0.06, 2.4, 4.8]))
    assert loaded.vpar.shape == (3, 8, 32)
    np.testing.assert_allclose(loaded.vpar[0, 0, 0], -2.90625, atol=5.0e-5)
    np.testing.assert_allclose(loaded.vpar[-1, -1, -1], 2.90625, atol=5.0e-5)
    assert float(jnp.max(jnp.abs(loaded.real_part[-1]))) > 0.0
    assert float(jnp.max(jnp.abs(loaded.imag_part[-1]))) > 0.0


def test_cyclone_velocity_space_slice_audit_accepts_matched_slice():
    observed = GkwVelocitySpaceSlice(
        vpar=((0.0, 1.0), (0.0, 1.0)),
        vperp=((0.5, 0.5), (1.5, 1.5)),
        real_part=((0.1, 0.2), (0.3, 0.4)),
        imag_part=((0.0, -0.1), (-0.2, -0.3)),
        time=1.0,
        peak_z=0.25,
        source="observed",
    )
    reference = GkwVelocitySpaceSlice(
        vpar=((0.0, 1.0), (0.0, 1.0)),
        vperp=((0.5, 0.5), (1.5, 1.5)),
        real_part=((0.1, 0.2), (0.3, 0.4)),
        imag_part=((0.0, -0.1), (-0.2, -0.3)),
        time=1.0,
        peak_z=0.25,
        source="reference",
    )

    audit = audit_cyclone_velocity_space_slice(observed, reference, tolerance=1.0e-12)

    assert isinstance(audit, CycloneVelocitySpaceSliceAudit)
    assert bool(audit.passed)
    assert audit.shape == (2, 2)
    np.testing.assert_allclose(audit.complex_max_abs_error, 0.0, atol=1.0e-14)
    assert "velocity-space slice comparison" in audit.notes


def test_cyclone_velocity_space_slice_series_audit_accepts_matched_series():
    values = np.array(
        [
            [[0.1 + 0.2j, 0.3 - 0.4j], [-0.5 + 0.6j, 0.7 + 0.8j]],
            [[0.2 + 0.1j, 0.4 - 0.3j], [-0.6 + 0.5j, 0.8 + 0.7j]],
        ],
        dtype=np.complex128,
    )
    reference = GkwVelocitySpaceSliceSeries(
        times=(0.1, 0.2),
        snapshot_indices=(2, 4),
        vpar=np.broadcast_to(np.array([[0.0, 1.0], [0.0, 1.0]]), values.shape),
        vperp=np.broadcast_to(np.array([[0.5, 0.5], [1.5, 1.5]]), values.shape),
        real_part=np.real(values),
        imag_part=np.imag(values),
        source="reference",
    )

    audit = audit_cyclone_velocity_space_slice_series(
        reference,
        reference,
        tolerance=1.0e-12,
    )

    assert isinstance(audit, CycloneVelocitySpaceSliceSeriesAudit)
    assert bool(audit.passed)
    assert audit.shape == (2, 2)
    np.testing.assert_allclose(audit.max_direct_max_abs_error, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.max_best_max_abs_error, 0.0, atol=1.0e-14)
    assert audit.variant_names[int(audit.best_variant_indices[0])] == audit.variant_names[0]


def test_cyclone_velocity_space_slice_series_variant_audit_accepts_matched_series():
    values = np.array(
        [
            [[0.1 + 0.2j, 0.3 - 0.4j], [-0.5 + 0.6j, 0.7 + 0.8j]],
            [[0.2 + 0.1j, 0.4 - 0.3j], [-0.6 + 0.5j, 0.8 + 0.7j]],
        ],
        dtype=np.complex128,
    )
    reference = GkwVelocitySpaceSliceSeries(
        times=(0.1, 0.2),
        snapshot_indices=(2, 4),
        vpar=np.broadcast_to(np.array([[0.0, 1.0], [0.0, 1.0]]), values.shape),
        vperp=np.broadcast_to(np.array([[0.5, 0.5], [1.5, 1.5]]), values.shape),
        real_part=np.real(values),
        imag_part=np.imag(values),
        source="reference",
    )
    shifted = GkwVelocitySpaceSliceSeries(
        times=reference.times,
        snapshot_indices=reference.snapshot_indices,
        vpar=reference.vpar,
        vperp=reference.vperp,
        real_part=np.real(values + 0.05),
        imag_part=np.imag(values + 0.05),
        source="shifted",
    )

    audit = audit_cyclone_velocity_space_slice_series_variants(
        (reference, shifted),
        reference,
        variant_names=("baseline", "shifted"),
        tolerance=1.0e-12,
    )

    assert isinstance(audit, CycloneVelocitySpaceSliceSeriesVariantAudit)
    assert bool(audit.passed)
    assert audit.direct_max_abs_errors.shape == (2, 2)
    assert audit.best_direct_variant_indices.shape == (2,)
    assert audit.variant_names == ("baseline", "shifted")
    np.testing.assert_allclose(audit.max_baseline_direct_max_abs_error, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.max_best_direct_max_abs_error, 0.0, atol=1.0e-14)


def test_velocity_space_slice_convention_audit_keeps_direct_baseline():
    reference = GkwVelocitySpaceSlice(
        vpar=((0.0, 1.0), (0.0, 1.0)),
        vperp=((0.5, 0.5), (1.5, 1.5)),
        real_part=((0.1, 0.2), (0.3, 0.4)),
        imag_part=((0.0, -0.1), (-0.2, -0.3)),
        source="reference",
    )
    audit = audit_velocity_space_slice_conventions(reference, reference)

    assert isinstance(audit, VelocitySliceConventionAudit)
    assert audit.variant_names[0] == "direct_mu_rows_vpar_columns:identity"
    assert audit.variant_names[int(audit.best_variant_index)] == audit.variant_names[0]
    np.testing.assert_allclose(audit.direct_max_abs_error, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.best_max_abs_error, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.even_max_abs_error, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.odd_same_sign_max_abs_error, 0.0, atol=1.0e-14)
    assert float(audit.odd_opposite_sign_max_abs_error) > 0.0


def test_velocity_space_slice_convention_audit_detects_one_based_axis_shift():
    reference = GkwVelocitySpaceSlice(
        vpar=np.broadcast_to(np.arange(4.0), (3, 4)),
        vperp=np.broadcast_to(np.arange(3.0)[:, None], (3, 4)),
        real_part=np.arange(12.0).reshape(3, 4),
        imag_part=-np.arange(12.0).reshape(3, 4),
        source="reference",
    )
    observed = GkwVelocitySpaceSlice(
        vpar=reference.vpar,
        vperp=reference.vperp,
        real_part=jnp.roll(reference.real_part, 1, axis=1),
        imag_part=jnp.roll(reference.imag_part, 1, axis=1),
        source="observed",
    )

    audit = audit_velocity_space_slice_conventions(observed, reference)

    assert audit.variant_names[int(audit.best_variant_index)] == "roll_vpar_minus_1:identity"
    np.testing.assert_allclose(audit.best_max_abs_error, 0.0, atol=1.0e-14)
    assert float(audit.direct_max_abs_error) > 0.0


def test_velocity_space_slice_phase_audit_detects_global_phase():
    phase = np.exp(0.37j)
    reference_values = np.array(
        [[0.1 + 0.2j, 0.3 - 0.4j], [-0.5 + 0.6j, 0.7 + 0.8j]],
        dtype=np.complex128,
    )
    observed_values = np.conj(phase) * reference_values
    reference = GkwVelocitySpaceSlice(
        vpar=((0.0, 1.0), (0.0, 1.0)),
        vperp=((0.5, 0.5), (1.5, 1.5)),
        real_part=np.real(reference_values),
        imag_part=np.imag(reference_values),
        source="reference",
    )
    observed = GkwVelocitySpaceSlice(
        vpar=reference.vpar,
        vperp=reference.vperp,
        real_part=np.real(observed_values),
        imag_part=np.imag(observed_values),
        source="observed",
    )

    audit = audit_velocity_space_slice_phase_alignment(observed, reference)

    assert isinstance(audit, VelocitySlicePhaseAudit)
    assert audit.variant_names[0] == "direct_mu_rows_vpar_columns:identity"
    assert int(audit.best_phase_variant_index) == 0
    assert int(audit.best_scaled_variant_index) == 0
    np.testing.assert_allclose(audit.unit_phase_factors[0], phase, atol=1.0e-14)
    np.testing.assert_allclose(audit.phase_aligned_max_abs_errors[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(audit.scaled_max_abs_errors[0], 0.0, atol=1.0e-14)


def test_cosin2_velocity_slice_audit_runner_accepts_matched_reduced_fixtures(tmp_path):
    solver_slice = run_cyclone_base_case_velocity_space_slice(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    paths = [tmp_path / f"distr{index}.dat" for index in range(1, 5)]
    for path, values in zip(
        paths,
        (
            solver_slice.vpar,
            solver_slice.vperp,
            solver_slice.imag_part,
            solver_slice.real_part,
        ),
        strict=True,
    ):
        np.savetxt(path, np.asarray(values), fmt="%.16e")
    time_path = tmp_path / "time.dat"
    time_path.write_text(f"{float(solver_slice.time):.16e} 0.0\n")

    audit = run_cyclone_base_case_cosin2_velocity_slice_audit(
        gkw_distr1_path=paths[0],
        gkw_distr2_path=paths[1],
        gkw_distr3_path=paths[2],
        gkw_distr4_path=paths[3],
        gkw_time_path=time_path,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        tolerance=1.0e-10,
        grid_tolerance=1.0e-10,
    )

    assert bool(audit.passed)
    np.testing.assert_allclose(audit.complex_max_abs_error, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(audit.time_error, 0.0, atol=1.0e-14)


def test_cosin2_velocity_convention_audit_runner_accepts_matched_reduced_fixtures(tmp_path):
    solver_slice = run_cyclone_base_case_velocity_space_slice(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )
    paths = [tmp_path / f"distr{index}.dat" for index in range(1, 5)]
    for path, values in zip(
        paths,
        (
            solver_slice.vpar,
            solver_slice.vperp,
            solver_slice.imag_part,
            solver_slice.real_part,
        ),
        strict=True,
    ):
        np.savetxt(path, np.asarray(values), fmt="%.16e")
    time_path = tmp_path / "time.dat"
    time_path.write_text(f"{float(solver_slice.time):.16e} 0.0\n")

    audit = run_cyclone_base_case_cosin2_velocity_convention_audit(
        gkw_distr1_path=paths[0],
        gkw_distr2_path=paths[1],
        gkw_distr3_path=paths[2],
        gkw_distr4_path=paths[3],
        gkw_time_path=time_path,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
    )

    assert audit.variant_names[int(audit.best_variant_index)] == audit.variant_names[0]
    np.testing.assert_allclose(audit.best_max_abs_error, 0.0, atol=1.0e-12)


def test_cosin2_velocity_phase_audit_runner_accepts_matched_reduced_reference():
    reference = run_cyclone_base_case_velocity_space_slice(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )

    audit = run_cyclone_base_case_cosin2_velocity_phase_audit(
        reference_slice=reference,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, VelocitySlicePhaseAudit)
    assert len(leaves) == len(VelocitySlicePhaseAudit._dynamic_fields)
    assert aux is not None
    assert audit.phase_aligned_max_abs_errors.shape == (20,)
    assert audit.variant_names[0] == "direct_mu_rows_vpar_columns:identity"
    assert int(audit.best_phase_variant_index) == 0
    assert int(audit.best_scaled_variant_index) == 0
    np.testing.assert_allclose(audit.best_phase_max_abs_error, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(audit.best_scaled_max_abs_error, 0.0, atol=1.0e-12)
    assert "ky-sign" in audit.notes


def test_cosin2_velocity_series_audit_runner_accepts_matched_reduced_reference():
    reference = run_cyclone_base_case_velocity_space_slice_series(
        snapshot_indices=(2, 6),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )

    audit = run_cyclone_base_case_cosin2_velocity_series_audit(
        reference_series=reference,
        snapshot_indices=(2, 6),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        tolerance=1.0e-10,
        grid_tolerance=1.0e-10,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneVelocitySpaceSliceSeriesAudit)
    assert bool(audit.passed)
    assert len(leaves) == len(CycloneVelocitySpaceSliceSeriesAudit._dynamic_fields)
    assert aux is not None
    np.testing.assert_allclose(audit.max_direct_max_abs_error, 0.0, atol=1.0e-12)
    assert audit.variant_names[0] == "direct_mu_rows_vpar_columns:identity"


def test_cosin2_velocity_series_variant_audit_runner_accepts_matched_reduced_reference():
    reference = run_cyclone_base_case_velocity_space_slice_series(
        snapshot_indices=(2, 6),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )

    audit = run_cyclone_base_case_cosin2_velocity_series_variant_audit(
        reference_series=reference,
        snapshot_indices=(2, 6),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        tolerance=1.0e-10,
        grid_tolerance=1.0e-10,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneVelocitySpaceSliceSeriesVariantAudit)
    assert bool(audit.passed)
    assert len(leaves) == len(CycloneVelocitySpaceSliceSeriesVariantAudit._dynamic_fields)
    assert aux is not None
    assert audit.variant_names[0] == "baseline"
    assert audit.direct_max_abs_errors.shape == (6, 2)
    np.testing.assert_allclose(audit.max_baseline_direct_max_abs_error, 0.0, atol=1.0e-12)


def test_cosin2_vpar_odd_sign_audit_runs_reduced_against_matched_reference():
    reference = run_cyclone_base_case_velocity_space_slice(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )

    audit = run_cyclone_base_case_cosin2_vpar_odd_sign_audit(
        reference_slice=reference,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneVparOddSignAudit)
    assert audit.rhs_variant_names == (
        "baseline",
        "flip_igh",
        "flip_parallel_field_drive",
        "flip_igh_and_parallel_field_drive",
    )
    assert len(leaves) == len(CycloneVparOddSignAudit._dynamic_fields)
    assert aux is not None
    assert audit.direct_max_abs_errors.shape == (4,)
    assert audit.best_layout_max_abs_errors.shape == (4,)
    assert audit.best_layout_names[0] == "direct_mu_rows_vpar_columns:identity"
    assert int(audit.best_direct_variant_index) == 0
    np.testing.assert_allclose(audit.baseline_direct_max_abs_error, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(audit.direct_max_abs_errors[0], 0.0, atol=1.0e-12)
    assert "one-based k=1..N" in audit.notes


def test_cosin2_term_vii_field_convention_audit_runs_reduced_against_matched_reference():
    reference = run_cyclone_base_case_velocity_space_slice(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine2",
        normalization_model="gkw_unweighted",
        parallel_derivative_model="gkw_igh",
    )

    audit = run_cyclone_base_case_cosin2_term_vii_field_convention_audit(
        reference_slice=reference,
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneTermVIIFieldConventionAudit)
    assert audit.variant_names == (
        "baseline",
        "flip_all_field_terms",
        "flip_term_v_and_viii_only",
        "flip_term_vii_only",
        "conjugate_all_field_terms",
        "conjugate_term_vii_only",
        "negative_conjugate_all_field_terms",
        "negative_conjugate_term_vii_only",
    )
    assert audit.term_vii_phi_variants[3] == "negative"
    assert len(leaves) == len(CycloneTermVIIFieldConventionAudit._dynamic_fields)
    assert aux is not None
    assert audit.term_vii_phi_variants[7] == "negative_conjugate"
    assert audit.direct_max_abs_errors.shape == (8,)
    assert audit.best_layout_max_abs_errors.shape == (8,)
    assert audit.best_layout_names[0] == "direct_mu_rows_vpar_columns:identity"
    assert int(audit.best_direct_variant_index) == 0
    np.testing.assert_allclose(audit.baseline_direct_max_abs_error, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(audit.direct_max_abs_errors[0], 0.0, atol=1.0e-12)
    assert "dist.F90::get_phi" in audit.notes
    assert "field-variable" in audit.notes


def test_cyclone_profile_operator_audit_checks_selected_mode_assembly():
    audit = run_cyclone_base_case_profile_operator_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_window=2,
        target_z=0.0,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-10,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneProfileOperatorAudit)
    assert audit.normalized_phi_power.shape == (8,)
    assert audit.z_grid.shape == (8,)
    assert audit.streaming_delta_profile.shape == (8,)
    assert audit.field_drive_delta_profile.shape == (8,)
    assert audit.field_residual_profile.shape == (8,)
    assert len(leaves) == len(CycloneProfileOperatorAudit._dynamic_fields)
    assert aux is not None
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.time, 0.012)
    np.testing.assert_allclose(jnp.sum(audit.normalized_phi_power), 1.0, rtol=2e-12)
    assert 0 <= int(audit.z_index) < 8
    assert jnp.all(jnp.isfinite(audit.normalized_phi_power))
    assert jnp.all(audit.streaming_delta_profile >= 0.0)
    assert jnp.all(audit.field_drive_delta_profile >= 0.0)
    assert audit.local_streaming_delta <= audit.max_streaming_delta
    assert audit.local_field_drive_delta <= audit.max_field_drive_delta
    assert audit.boundary_streaming_delta <= audit.max_streaming_delta
    assert audit.boundary_field_drive_delta <= audit.max_field_drive_delta
    assert audit.field_residual_max < 1.0e-10
    assert audit.field_reconstruction_error < 1.0e-10
    assert audit.rhs_assembly_error < 1.0e-10
    assert "central profile operator audit" in audit.notes

    with pytest.raises(ValueError, match="output_window"):
        run_cyclone_base_case_profile_operator_audit(output_window=0)


def test_cyclone_term_i_fortran_audit_matches_source_reconstruction():
    audit = run_cyclone_base_case_term_i_fortran_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_window=2,
        target_z=0.0,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-11,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneTermIFortranAudit)
    assert audit.term_error_profile.shape == (8,)
    assert audit.coefficient_error_profile.shape == (8,)
    assert audit.current_term_profile.shape == (8,)
    assert audit.reference_term_profile.shape == (8,)
    assert audit.recurrence_speed_error_profile.shape == (8,)
    assert len(leaves) == len(CycloneTermIFortranAudit._dynamic_fields)
    assert aux is not None
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.time, 0.012)
    assert 0 <= int(audit.z_index) < 8
    assert audit.max_term_error < 1.0e-11
    assert audit.local_term_error <= audit.max_term_error
    assert audit.max_coefficient_error < 1.0e-11
    assert audit.local_coefficient_error <= audit.max_coefficient_error
    assert audit.recurrence_speed_max_error < 1.0e-11
    assert audit.sign_selection_error < 1.0e-14
    assert "vpar_grad_df_4d_testnewbc" in audit.notes

    with pytest.raises(ValueError, match="normalization_model"):
        run_cyclone_base_case_term_i_fortran_audit(normalization_model="bad")


def test_cyclone_time_normalization_audit_matches_gkw_sequence():
    audit = run_cyclone_base_case_time_normalization_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=2,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-11,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneTimeNormalizationAudit)
    assert audit.times.shape == (3,)
    assert audit.normalization_factor.shape == (3,)
    assert audit.gkw_window_growth.shape == (3,)
    assert audit.trace_window_growth.shape == (3,)
    assert audit.post_normalization_field_norm.shape == (3,)
    assert audit.field_linearity_error.shape == (3,)
    assert len(leaves) == len(CycloneTimeNormalizationAudit._dynamic_fields)
    assert aux is not None
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.times, jnp.asarray([0.0, 0.006, 0.012]))
    np.testing.assert_allclose(audit.gkw_window_growth, audit.trace_window_growth, atol=1.0e-11)
    np.testing.assert_allclose(audit.post_normalization_field_norm, 1.0, atol=1.0e-11)
    assert audit.rk4_step_error < 1.0e-11
    assert audit.time_grid_error < 1.0e-14
    assert audit.growth_sequence_error < 1.0e-11
    assert audit.max_field_linearity_error < 1.0e-11
    assert "normalise.F90" in audit.notes

    with pytest.raises(ValueError, match="normalization_model"):
        run_cyclone_base_case_time_normalization_audit(normalization_model="bad")


def test_cyclone_diagnostic_packing_audit_matches_gkw_source_layout():
    audit = run_cyclone_base_case_diagnostic_packing_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_window=2,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-11,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneDiagnosticPackingAudit)
    assert audit.parallel_phi_profile.shape == (8,)
    assert audit.packed_parallel_phi_profile.shape == (8,)
    assert audit.ky_spectrum.shape == (1,)
    assert audit.packed_ky_spectrum.shape == (1,)
    assert audit.kx_spectrum.shape == (1,)
    assert audit.packed_kx_spectrum.shape == (1,)
    assert len(leaves) == len(CycloneDiagnosticPackingAudit._dynamic_fields)
    assert aux is not None
    assert audit.output_window == 2
    assert audit.field_offset == 8 * 6 * 4
    assert audit.n_field_values == 8
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.time, 0.012)
    np.testing.assert_allclose(
        audit.parallel_phi_profile,
        audit.packed_parallel_phi_profile,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(audit.ky_spectrum, audit.packed_ky_spectrum, atol=1.0e-11)
    np.testing.assert_allclose(audit.kx_spectrum, audit.packed_kx_spectrum, atol=1.0e-11)
    assert audit.packing_roundtrip_error < 1.0e-11
    assert audit.parallel_phi_error < 1.0e-11
    assert audit.selected_profile_error < 1.0e-11
    assert "diagnostic.F90" in audit.notes

    with pytest.raises(ValueError, match="output_window"):
        run_cyclone_base_case_diagnostic_packing_audit(output_window=0)


def test_cyclone_matdat_matrix_audit_matches_sparse_conventions():
    audit = run_cyclone_base_case_matdat_matrix_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        initial_profile="cosine",
        tolerance=1.0e-10,
        max_size=512,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneMatdatMatrixAudit)
    assert len(leaves) == len(CycloneMatdatMatrixAudit._dynamic_fields)
    assert aux is not None
    assert audit.n_state == 8 * 6 * 4
    assert audit.n_nonzero > 0
    assert audit.n_duplicate_triplets == 2 * audit.n_nonzero
    assert audit.n_real_entries + audit.n_complex_entries == audit.n_nonzero
    assert bool(audit.passed)
    assert audit.matrix_action_error < 1.0e-10
    assert audit.source_max_abs < 1.0e-10
    assert audit.explicit_delta_error < 1.0e-10
    assert audit.compressed_action_error < 1.0e-10
    assert audit.complex_real_split_error < 1.0e-10
    assert audit.linearity_error < 1.0e-10
    assert audit.max_abs_matrix_entry > 0.0
    assert "matdat.F90" in audit.notes

    with pytest.raises(ValueError, match="nonzero_threshold"):
        run_cyclone_base_case_matdat_matrix_audit(nonzero_threshold=-1.0)


def test_cyclone_coefficient_source_audit_matches_gkw_formulas():
    audit = run_cyclone_base_case_coefficient_source_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_window=2,
        target_z=0.0,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-10,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneCoefficientSourceAudit)
    assert audit.term_errors.shape == (5,)
    assert audit.coefficient_errors.shape == (5,)
    assert audit.insertion_errors.shape == (5,)
    assert len(audit.term_names) == 5
    assert len(leaves) == len(CycloneCoefficientSourceAudit._dynamic_fields)
    assert aux is not None
    assert audit.output_window == 2
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.time, 0.012)
    assert 0 <= int(audit.z_index) < 8
    assert audit.max_term_error < 1.0e-10
    assert audit.max_coefficient_error < 1.0e-10
    assert audit.max_insertion_error < 1.0e-10
    assert audit.max_abs_error < 1.0e-10
    assert "vdgradf" in audit.notes
    assert "vpgrphi_3_newbc" in audit.notes

    with pytest.raises(ValueError, match="normalization_model"):
        run_cyclone_base_case_coefficient_source_audit(normalization_model="bad")


def test_cyclone_term_vii_mode_packing_audit_matches_source_path():
    audit = run_cyclone_base_case_term_vii_mode_packing_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_window=2,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-10,
        contrast_tolerance=1.0e-13,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneTermVIIModePackingAudit)
    assert len(leaves) == len(CycloneTermVIIModePackingAudit._dynamic_fields)
    assert aux is not None
    assert audit.output_window == 2
    assert audit.field_offset == 8 * 6 * 4
    assert audit.n_field_values == 8
    assert bool(audit.passed)
    np.testing.assert_allclose(audit.time, 0.012)
    expected_internal_krho = 0.5 / (1.4 / (2.0 * np.pi * 0.19))
    np.testing.assert_allclose(audit.selected_ky, expected_internal_krho, atol=1.0e-12)
    np.testing.assert_allclose(audit.selected_ky, audit.gkw_krho, atol=1.0e-12)
    np.testing.assert_array_equal(np.asarray(audit.ixplus), np.array([-1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(audit.ixminus), np.array([-1], dtype=np.int32))
    assert audit.direct_field_roundtrip_error < 1.0e-10
    assert audit.direct_term_vii_error < 1.0e-10
    assert audit.packed_term_vii_error < 1.0e-10
    assert audit.conjugate_field_pullback_error > 1.0e-13
    assert audit.conjugate_term_vii_delta > 1.0e-13
    assert audit.negative_field_term_vii_delta > 1.0e-13
    assert "mode.F90" in audit.notes
    assert "vpgrphi_3_newbc" in audit.notes

    with pytest.raises(ValueError, match="contrast_tolerance"):
        run_cyclone_base_case_term_vii_mode_packing_audit(contrast_tolerance=0.0)


def test_cyclone_igh_arakawa_audit_quantifies_fused_path_gap():
    audit = run_cyclone_base_case_igh_arakawa_audit(
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        output_window=2,
        target_z=0.0,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-12,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneIghArakawaAudit)
    assert audit.fused_profile.shape == (8,)
    assert audit.separated_profile.shape == (8,)
    assert audit.delta_profile.shape == (8,)
    assert audit.parallel_diffusion_profile.shape == (8,)
    assert audit.velocity_diffusion_profile.shape == (8,)
    assert len(leaves) == len(CycloneIghArakawaAudit._dynamic_fields)
    assert aux is not None
    assert audit.output_window == 2
    assert not bool(audit.passed)
    np.testing.assert_allclose(audit.time, 0.012)
    assert 0 <= int(audit.z_index) < 8
    assert audit.max_delta > 1.0e-12
    assert audit.local_delta <= audit.max_delta
    assert audit.relative_delta > 0.0
    assert audit.max_parallel_diffusion > 0.0
    assert audit.max_velocity_diffusion > 0.0
    assert "ltrapping_arakawa" in audit.notes
    assert "disp_vp=0.2" in audit.notes

    with pytest.raises(ValueError, match="normalization_model"):
        run_cyclone_base_case_igh_arakawa_audit(normalization_model="bad")


def test_cyclone_igh_arakawa_series_audit_samples_multiple_windows():
    audit = run_cyclone_base_case_igh_arakawa_series_audit(
        output_windows=(1, 2),
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        target_z=0.0,
        initial_profile="cosine",
        normalization_model="gkw_unweighted",
        tolerance=1.0e-12,
    )
    leaves, aux = jax.tree_util.tree_flatten(audit)

    assert isinstance(audit, CycloneIghArakawaSeriesAudit)
    assert len(leaves) == len(CycloneIghArakawaSeriesAudit._dynamic_fields)
    assert aux is not None
    assert audit.output_windows.shape == (2,)
    np.testing.assert_array_equal(np.asarray(audit.output_windows), np.array([1, 2]))
    np.testing.assert_allclose(audit.times, np.array([0.006, 0.012]))
    assert not bool(audit.passed)
    assert audit.max_delta.shape == (2,)
    assert audit.relative_delta.shape == (2,)
    assert audit.worst_max_delta == jnp.max(audit.max_delta)
    assert int(audit.worst_window) in (1, 2)
    assert audit.max_parallel_diffusion.shape == (2,)
    assert audit.max_velocity_diffusion.shape == (2,)
    assert "multi-window" in audit.notes

    with pytest.raises(ValueError, match="strictly increasing"):
        run_cyclone_base_case_igh_arakawa_series_audit(output_windows=(2, 1))


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
