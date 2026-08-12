from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fluxtube_gk import (
    FourierGridSpec,
    ParallelGridSpec,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_mode_connectivity,
    build_parallel_grid,
)
from jax_fluxtube_gk.validation.fixture_io import (
    load_eik_geometry_reference,
    resample_eik_geometry_reference,
)
from jax_fluxtube_gk.validation.geometry_parity import (
    ModeBoundaryContractReport,
    StellaratorGeometryPreflightReport,
    build_flux_tube_geometry_from_eik_reference,
    run_mode_boundary_contract,
    run_stellarator_geometry_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def test_desc_fixture_passes_shared_stellarator_geometry_preflight():
    geometry, _parallel = _desc_fixture_geometry()
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )

    report = run_stellarator_geometry_preflight(geometry, fourier)

    assert isinstance(report, StellaratorGeometryPreflightReport)
    assert bool(report.passed)
    assert set(report.check_names) == {
        "finite_geometry_fields",
        "positive_B",
        "positive_metric_diagonal",
        "finite_kperp2",
        "nonnegative_representative_kperp2",
        "eik_export_contract",
        "finite_difference_mirror_force",
    }
    np.testing.assert_allclose(report.eik_export_error, 0.0, atol=1.0e-13)
    assert float(report.mirror_fd_error) < 1.0e-2
    assert float(report.kperp2_min) >= -1.0e-12


def test_stellarator_geometry_preflight_rejects_bad_imported_geometry():
    geometry, _parallel = _desc_fixture_geometry()
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    bad_geometry = replace(geometry, B=-jnp.abs(geometry.B))

    report = run_stellarator_geometry_preflight(bad_geometry, fourier)

    assert not bool(report.passed)
    checks = dict(zip(report.check_names, np.asarray(report.check_passed), strict=True))
    assert not bool(checks["positive_B"])


def test_mode_boundary_contract_checks_zonal_open_chain_and_spacing():
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=7,
            n_ky=3,
            kx_max=0.6,
            ky_values=(0.0, 0.3, 0.5),
            ikxspace=2,
        )
    )
    connectivity = build_mode_connectivity(fourier)

    report = run_mode_boundary_contract(fourier, connectivity)

    assert isinstance(report, ModeBoundaryContractReport)
    assert bool(report.passed)
    assert int(report.open_end_count) > 0
    assert int(report.linked_edge_count) > 0

    bad_connectivity = replace(
        connectivity,
        ixplus=connectivity.ixplus.at[0, fourier.iyzero].set(-1),
    )
    bad_report = run_mode_boundary_contract(fourier, bad_connectivity)
    checks = dict(zip(bad_report.check_names, np.asarray(bad_report.check_passed), strict=True))
    assert not bool(bad_report.passed)
    assert not bool(checks["zonal_identity"])


def test_mode_boundary_contract_is_independent_of_kx_extent():
    base = build_fourier_grid(
        FourierGridSpec(
            n_kx=7,
            n_ky=3,
            kx_max=0.4,
            ky_values=(0.0, 0.3, 0.5),
            ikxspace=2,
        )
    )
    stretched = build_fourier_grid(
        FourierGridSpec(
            n_kx=7,
            n_ky=3,
            kx_max=1.4,
            ky_values=(0.0, 0.3, 0.5),
            ikxspace=2,
        )
    )

    base_connectivity = build_mode_connectivity(base)
    stretched_connectivity = build_mode_connectivity(stretched)
    base_report = run_mode_boundary_contract(base, base_connectivity)
    stretched_report = run_mode_boundary_contract(stretched, stretched_connectivity)

    assert bool(base_report.passed)
    assert bool(stretched_report.passed)
    np.testing.assert_array_equal(base_connectivity.mode_label, stretched_connectivity.mode_label)
    np.testing.assert_array_equal(base_connectivity.ixplus, stretched_connectivity.ixplus)
    np.testing.assert_array_equal(base_connectivity.ixminus, stretched_connectivity.ixminus)


@pytest.mark.external
def test_external_w7x_eik_preflight_handles_two_field_line_lengths(gx_root: Path):
    path = (
        gx_root
        / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    reference = load_eik_geometry_reference(path)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    reports = []
    mean_b = []
    for npol, n_theta in ((1, 17), (2, 33)):
        theta = np.linspace(-np.pi * npol, np.pi * npol, n_theta, endpoint=False)
        sampled = resample_eik_geometry_reference(reference, theta)
        parallel = _parallel_grid_from_theta(theta)
        geometry = build_flux_tube_geometry_from_eik_reference(sampled, parallel)
        report = run_stellarator_geometry_preflight(
            geometry,
            fourier,
            include_mirror_fd_check=False,
        )
        reports.append(report)
        mean_b.append(float(report.field_mean[0]))

    assert all(bool(report.passed) for report in reports)
    assert all(float(report.eik_export_error) <= 1.0e-12 for report in reports)
    assert abs(mean_b[0] - mean_b[1]) / max(abs(mean_b[1]), 1.0e-12) < 2.0e-2


def _desc_fixture_geometry():
    fixture = ROOT / "fixtures/desc_geometry_dshape_rho05_alpha0.npz"
    data = np.load(fixture)
    parallel = _parallel_grid_from_z(data["z"])
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
    return geometry, parallel


def _parallel_grid_from_z(z):
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _parallel_grid_from_theta(theta):
    theta = np.asarray(theta, dtype=float)
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )
