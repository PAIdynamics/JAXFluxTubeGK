from importlib import import_module
from pathlib import Path
import sys
import tomllib

import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    AdiabaticElectronParams,
    BoozerSurface,
    FieldLineSpec,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityBasisSpec,
    VelocityGridSpec,
    adiabatic_quasineutrality_residual,
    build_boozer_parallel_grid,
    build_circular_geometry,
    build_fourier_grid,
    build_hermite_laguerre_basis,
    build_linear_residual_precompute,
    build_parallel_grid,
    build_physical_flux_tube_geometry_from_arrays,
    build_s_alpha_geometry,
    build_velocity_grid,
    evaluate_boozer_magnetic_field,
    integrate_fixed_step,
    k_perp_squared,
    linear_growth_diagnostics,
    linear_residual,
    map_physical_to_internal_geometry,
    max_growth_objective,
    sample_boozer_field_line,
    solve_adiabatic_electron_phi,
)


def _gyaradax_geometry_module(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return import_module("gyaradax.geometry.geom")


def _cell_centered_parallel_grid(n_z: int):
    lower = -0.5 + 0.5 / n_z
    return build_parallel_grid(
        ParallelGridSpec(n_z=n_z, z_min=lower, z_max=lower + 1.0, topology="periodic")
    )


def _ion(**updates):
    base = dict(
        charge=1.0,
        mass=2.0,
        density=0.9,
        temperature=1.3,
        density_gradient=1.7,
        temperature_gradient=2.4,
    )
    base.update(updates)
    return SpeciesParams(**base)


@pytest.mark.external
def test_reduced_gyaradax_geometry_parity_circular_and_s_alpha(gyaradax_root: Path):
    reference_module = _gyaradax_geometry_module(gyaradax_root)
    n_z = 12
    parallel = _cell_centered_parallel_grid(n_z)
    params = GeometryScalarParams(q=1.45, shat=0.65, eps=0.19)
    models = (
        ("circ", build_circular_geometry(parallel, params)),
        ("s-alpha", build_s_alpha_geometry(parallel, params)),
    )

    for reference_model, geometry in models:
        reference = reference_module.compute_geometry(
            q=params.q,
            shat=params.shat,
            eps=params.eps,
            ns=n_z,
            nkx=5,
            nky=3,
            nvpar=4,
            nmu=3,
            geom_type=reference_model,
        )
        np.testing.assert_allclose(geometry.B, reference["bn"], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(geometry.F, reference["ffun"], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(geometry.G, reference["gfun"], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(geometry.E_y, reference["efun"], rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            jnp.stack([geometry.g_yy, geometry.g_xy, geometry.g_xx], axis=-1),
            reference["little_g"],
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            jnp.stack([geometry.D_x, geometry.D_y], axis=-1),
            reference["dfun"][:, :2],
            rtol=2e-12,
            atol=2e-12,
        )


def test_reduced_phi_rhs_parity_fixture_against_explicit_formula():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=5, n_mu=4, vpar_max=1.8, mu_max=1.3))
    parallel = _cell_centered_parallel_grid(10)
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.6, ky_values=(0.0, 0.35))
    )
    geometry = build_circular_geometry(
        parallel,
        GeometryScalarParams(q=1.35, shat=0.55, eps=0.17),
    )
    species = _ion()
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
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
    distribution = 0.02 * (jnp.sin(index / 7.0) + 1j * jnp.cos(index / 11.0))

    phi = solve_adiabatic_electron_phi(distribution, precompute.field)
    residual = adiabatic_quasineutrality_residual(phi, distribution, precompute.field)
    rhs = linear_residual(distribution, precomputed=precompute)
    manual = _manual_single_species_rhs(distribution, phi, precompute.rhs)

    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=2e-12)
    np.testing.assert_allclose(rhs, manual, rtol=3e-12, atol=3e-12)


def test_constant_zonal_mode_is_stationary_in_flat_flux_tube():
    n_z = 8
    parallel = build_boozer_parallel_grid(n_z=n_z, n_turns=1)
    field_line = sample_boozer_field_line(
        BoozerSurface(iota=0.7, B0=1.0),
        FieldLineSpec(rho=0.4, alpha0=0.2),
        parallel,
    )
    ones = jnp.ones_like(field_line.z)
    zeros = jnp.zeros_like(field_line.z)
    physical = build_physical_flux_tube_geometry_from_arrays(
        field_line=field_line,
        B=ones,
        b_dot_grad_z=ones,
        grad_psi_sq=ones,
        grad_alpha_sq=ones,
        grad_psi_dot_grad_alpha=zeros,
        B_cross_gradB_dot_grad_psi=zeros,
        B_cross_gradB_dot_grad_alpha=zeros,
        b_cross_kappa_dot_grad_psi=zeros,
        b_cross_kappa_dot_grad_alpha=zeros,
    )
    geometry = map_physical_to_internal_geometry(physical, parallel)
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=4, n_mu=3, vpar_max=1.2, mu_max=1.0))
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.0,))
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        _ion(density_gradient=0.0, temperature_gradient=0.0),
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=True,
        ),
    )
    distribution = jnp.ones((velocity.vpar.shape[0], velocity.mu.shape[0], n_z, 1, 1)) * 0.03

    rhs = linear_residual(distribution, precomputed=precompute)

    np.testing.assert_allclose(rhs, 0.0, rtol=0, atol=3e-12)


def test_reduced_stellarator_fixture_matches_precomputed_reference_arrays():
    parallel = build_boozer_parallel_grid(n_z=16, n_turns=1)
    surface = BoozerSurface(
        iota=0.62,
        B0=1.1,
        B_cos=(0.08, -0.03),
        B_sin=(0.02, 0.01),
        m_modes=(1, 2),
        n_modes=(0, 1),
        field_periods=3,
    )
    field_line = sample_boozer_field_line(
        surface,
        FieldLineSpec(rho=0.35, alpha0=0.4),
        parallel,
    )
    B = evaluate_boozer_magnetic_field(surface, field_line)
    ones = jnp.ones_like(B)
    physical = build_physical_flux_tube_geometry_from_arrays(
        field_line=field_line,
        B=B,
        b_dot_grad_z=(0.8 + 0.05 * jnp.cos(field_line.z)) * ones,
        grad_psi_sq=(1.2 + 0.1 * jnp.sin(field_line.z)) * ones,
        grad_alpha_sq=(2.0 + 0.2 * jnp.cos(2.0 * field_line.z)) * ones,
        grad_psi_dot_grad_alpha=0.15 * jnp.sin(field_line.z),
        B_cross_gradB_dot_grad_psi=0.2 * jnp.cos(field_line.z),
        B_cross_gradB_dot_grad_alpha=0.3 * jnp.sin(field_line.z),
        b_cross_kappa_dot_grad_psi=0.4 * jnp.cos(2.0 * field_line.z),
        b_cross_kappa_dot_grad_alpha=0.5 * jnp.sin(2.0 * field_line.z),
        source="phase10-precomputed",
    )
    geometry = map_physical_to_internal_geometry(physical, parallel)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.5, ky_values=(0.0, 0.3)))
    kperp2 = k_perp_squared(geometry, fourier)
    expected_D_x = (
        physical.B_cross_gradB_dot_grad_psi
        + physical.B * physical.b_cross_kappa_dot_grad_psi
    ) / physical.B**2
    expected_D_y = (
        physical.B_cross_gradB_dot_grad_alpha
        + physical.B * physical.b_cross_kappa_dot_grad_alpha
    ) / physical.B**2

    np.testing.assert_allclose(field_line.alpha, 0.0, atol=1e-14)
    np.testing.assert_allclose(geometry.B, B, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.D_x, expected_D_x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.D_y, expected_D_y, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(geometry.g_xx, physical.grad_psi_sq, rtol=2e-13, atol=2e-13)
    assert geometry.source == "phase10-precomputed"
    assert jnp.all(jnp.isfinite(kperp2))
    assert jnp.min(kperp2) >= -1.0e-12


def test_parallel_spectral_derivative_converges_with_resolution():
    def derivative_error(n_z):
        grid = build_parallel_grid(
            ParallelGridSpec(n_z=n_z, z_min=0.0, z_max=2.0 * np.pi, topology="periodic")
        )
        values = jnp.exp(0.3 * jnp.sin(grid.z) + 0.2 * jnp.cos(2.0 * grid.z))
        exact = values * (0.3 * jnp.cos(grid.z) - 0.4 * jnp.sin(2.0 * grid.z))
        return jnp.sqrt(jnp.mean(jnp.abs(grid.D_z @ values - exact) ** 2))

    coarse = derivative_error(12)
    fine = derivative_error(24)

    assert fine < coarse * 1.0e-3
    assert fine < 1.0e-8


def test_velocity_space_spectral_derivatives_converge_with_resolution():
    def vparallel_error(n_vpar):
        grid = build_velocity_grid(
            VelocityGridSpec(n_vpar=n_vpar, n_mu=4, vpar_max=2.0, mu_max=1.0)
        )
        vpar = grid.vpar
        values = jnp.exp(0.25 * vpar) + 0.1 * jnp.sin(2.0 * vpar)
        exact = 0.25 * jnp.exp(0.25 * vpar) + 0.2 * jnp.cos(2.0 * vpar)
        return jnp.sqrt(jnp.mean(jnp.abs(grid.D_vpar @ values - exact) ** 2))

    def mu_error(n_mu):
        grid = build_velocity_grid(
            VelocityGridSpec(n_vpar=4, n_mu=n_mu, vpar_max=1.0, mu_max=3.0)
        )
        mu = grid.mu
        values = jnp.exp(-0.35 * mu) + 0.2 * jnp.cos(1.5 * mu)
        exact = -0.35 * jnp.exp(-0.35 * mu) - 0.3 * jnp.sin(1.5 * mu)
        return jnp.sqrt(jnp.mean(jnp.abs(grid.D_mu @ values - exact) ** 2))

    vparallel_coarse = vparallel_error(8)
    vparallel_fine = vparallel_error(16)
    mu_coarse = mu_error(8)
    mu_fine = mu_error(16)

    assert vparallel_fine < vparallel_coarse * 1.0e-5
    assert vparallel_fine < 1.0e-8
    assert mu_fine < mu_coarse * 1.0e-6
    assert mu_fine < 1.0e-10


def test_manufactured_ky_growth_scan_converges_with_resolution():
    peak_ky = 0.41
    reference_growth = 0.18
    curvature = 1.7

    def scan_error(n_ky):
        fourier = build_fourier_grid(
            FourierGridSpec(n_kx=1, n_ky=n_ky, kx_max=0.0, ky_max=1.0)
        )
        growth = reference_growth - curvature * (fourier.ky - peak_ky) ** 2
        return jnp.abs(max_growth_objective(growth) - reference_growth)

    coarse = scan_error(9)
    fine = scan_error(65)

    assert fine < coarse / 20.0
    assert fine < 1.0e-4


@pytest.mark.external
def test_gx_cyclone_input_fixture_maps_to_solver_specs_and_geometry(gx_root: Path):
    path = gx_root / "benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in"
    data = tomllib.loads(path.read_text())
    dimensions = data["Dimensions"]
    domain = data["Domain"]
    geometry_input = data["Geometry"]
    species_input = data["species"]

    n_z = dimensions["ntheta"] * (2 * dimensions["nperiod"] - 1)
    y0 = domain["y0"]
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=dimensions["nkx"],
            n_ky=dimensions["nky"],
            kx_max=0.0,
            ky_max=(dimensions["nky"] - 1) / y0,
        )
    )
    basis = build_hermite_laguerre_basis(
        VelocityBasisSpec(
            n_hermite=dimensions["nhermite"],
            n_laguerre=dimensions["nlaguerre"],
        )
    )
    parallel = _cell_centered_parallel_grid(n_z)
    geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(
            q=geometry_input["qinp"],
            shat=geometry_input["shat"],
            eps=geometry_input["eps"],
        ),
    )
    ion = SpeciesParams(
        charge=species_input["z"][0],
        mass=species_input["mass"][0],
        density=species_input["dens"][0],
        temperature=species_input["temp"][0],
        density_gradient=species_input["fprim"][0],
        temperature_gradient=species_input["tprim"][0],
    )

    np.testing.assert_allclose(fourier.ky[0], 0.0)
    np.testing.assert_allclose(fourier.ky[1], 1.0 / y0)
    np.testing.assert_allclose(fourier.ky[-1], (dimensions["nky"] - 1) / y0)
    np.testing.assert_allclose(fourier.parseval[0], 1.0)
    np.testing.assert_allclose(fourier.parseval[1:], 2.0)
    assert basis.hermite_to_grid.shape == (dimensions["nhermite"], dimensions["nhermite"])
    assert basis.laguerre_to_grid.shape == (dimensions["nlaguerre"], dimensions["nlaguerre"])
    assert geometry.B.shape == (n_z,)
    assert jnp.all(jnp.isfinite(geometry.B))
    assert data["Physics"]["nonlinear_mode"] is False
    assert data["Boltzmann"]["Boltzmann_type"] == "electrons"
    assert data["Dissipation"]["closure_model"] == "none"
    np.testing.assert_allclose(ion.density_gradient, 0.8)
    np.testing.assert_allclose(ion.temperature_gradient, 2.49)


def test_rk4_growth_rate_converges_with_timestep():
    gamma = 0.18
    omega = -0.55
    rate = gamma - 1j * omega
    state0 = jnp.asarray([1.0 + 0.2j])
    final_time = 0.6

    def rhs(state, coefficient):
        return coefficient * state

    def growth_error(n_steps):
        dt = final_time / n_steps
        result = integrate_fixed_step(state0, dt, n_steps, rhs, rate)
        start = result.history[0].reshape(1, 1, 1)
        end = result.state.reshape(1, 1, 1)
        diagnostics = linear_growth_diagnostics(start, end, 0.0, result.times[-1])
        return jnp.abs(diagnostics.growth_rate[0] - gamma)

    coarse = growth_error(10)
    fine = growth_error(20)

    assert fine < coarse / 12.0
    assert fine < 1.0e-9


def _manual_single_species_rhs(distribution, phi, rhs):
    dz_distribution = jnp.einsum("ij,vmjxy->vmixy", rhs.D_z, distribution)
    dv_distribution = jnp.einsum("ij,jmzxy->imzxy", rhs.D_vpar, distribution)
    gyro_phi = rhs.flr_factors.bessel_j0[0] * phi[None, :, :, :]
    dz_gyro_phi = jnp.einsum("ij,mjxy->mixy", rhs.D_z, gyro_phi)
    charge_over_temperature = rhs.charge_over_temperature[0]
    return (
        -rhs.parallel_streaming_coeff[0, :, None, :, None, None] * dz_distribution
        - 1j * rhs.magnetic_drift_frequency[0] * distribution
        + rhs.mirror_force_coeff[0, None, :, :, None, None] * dv_distribution
        + 1j
        * rhs.E_y[None, None, :, None, None]
        * rhs.ky[None, None, None, None, :]
        * gyro_phi[None, :, :, :, :]
        * rhs.maxwellian[0][..., None, None]
        * rhs.drive_factor[0][..., None, None]
        - charge_over_temperature
        * rhs.parallel_streaming_coeff[0, :, None, :, None, None]
        * rhs.maxwellian[0][..., None, None]
        * dz_gyro_phi[None, :, :, :, :]
        - charge_over_temperature
        * 1j
        * rhs.magnetic_drift_frequency[0]
        * rhs.maxwellian[0][..., None, None]
        * gyro_phi[None, :, :, :, :]
        - rhs.perpendicular_damping.reshape((1, 1, 1) + rhs.perpendicular_damping.shape)
        * distribution
    )
