from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    adiabatic_density_numerator,
    adiabatic_quasineutrality_residual,
    adiabatic_quasineutrality_residual_from_density,
    build_adiabatic_quasineutrality_precompute,
    build_fourier_grid,
    build_kinetic_quasineutrality_precompute,
    build_velocity_grid,
    default_adiabatic_electron_params,
    kinetic_quasineutrality_residual,
    kinetic_quasineutrality_residual_from_density,
    kxky_spectrum,
    ky_spectrum,
    mode_amplitude,
    radial_flux_spectrum,
    solve_adiabatic_electron_phi,
    solve_adiabatic_electron_phi_from_density,
    solve_kinetic_electron_phi,
    solve_kinetic_electron_phi_from_density,
    species_flr_factors,
    total_radial_flux,
    velocity_space_integral,
)


def _ion(**updates):
    base = dict(
        charge=1.0,
        mass=2.0,
        density=1.0,
        temperature=1.3,
        density_gradient=0.0,
        temperature_gradient=0.0,
    )
    base.update(updates)
    return SpeciesParams(**base)


def _setup(species=None, *, zonal_correction=True):
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=5, n_mu=4, vpar_max=2.0, mu_max=3.0))
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.8, ky_values=(0.0, 0.5))
    )
    B = jnp.asarray([0.8, 0.95, 1.1, 1.25, 1.05, 0.9])
    kx = fourier.kx[None, :, None]
    ky = fourier.ky[None, None, :]
    z_factor = 1.0 + 0.2 * jnp.arange(B.shape[0])[:, None, None]
    kperp2 = z_factor * (kx**2 + ky**2)
    species = _ion() if species is None else species
    flr = species_flr_factors(species, velocity.mu, B, kperp2)
    electrons = default_adiabatic_electron_params(
        species,
        temperature=1.0,
        zonal_correction=zonal_correction,
    )
    precompute = build_adiabatic_quasineutrality_precompute(
        velocity,
        B,
        flr,
        species,
        electrons,
        fourier_grid=fourier,
        w_z=jnp.linspace(0.7, 1.2, B.shape[0]),
    )
    return velocity, fourier, B, species, precompute


def test_default_electron_params_use_background_charge_neutrality():
    species = (_ion(density=0.8), _ion(charge=2.0, density=0.1))

    electrons = default_adiabatic_electron_params(species, temperature=0.9)

    np.testing.assert_allclose(electrons.density, 1.0)
    np.testing.assert_allclose(electrons.temperature, 0.9)


def test_zero_distribution_gives_zero_phi():
    velocity, fourier, B, _species, precompute = _setup()
    distribution = jnp.zeros((velocity.vpar.shape[0], velocity.mu.shape[0], B.shape[0], 3, 2))

    phi = solve_adiabatic_electron_phi(distribution, precompute)

    assert phi.shape == (B.shape[0], fourier.kx.shape[0], fourier.ky.shape[0])
    np.testing.assert_allclose(phi, 0.0, atol=0.0)


def test_native_phase_space_measure_overrides_tensor_product_weights():
    velocity, fourier, B, species, _precompute = _setup(zonal_correction=False)
    flr = species_flr_factors(species, velocity.mu, B, jnp.zeros((B.size, 3, 2)))
    measure = jnp.arange(B.size * velocity.vpar.size * velocity.mu.size, dtype=float)
    measure = 0.01 * measure.reshape(B.size, velocity.vpar.size, velocity.mu.size) + 0.1
    precompute = build_adiabatic_quasineutrality_precompute(
        velocity,
        B,
        flr,
        species,
        AdiabaticElectronParams(1.0, 1.0, zonal_correction=False),
        fourier_grid=fourier,
        phase_space_measure=measure,
    )
    distribution = jnp.ones((velocity.vpar.size, velocity.mu.size, B.size, 3, 2))

    numerator = adiabatic_density_numerator(distribution, precompute)

    expected = jnp.broadcast_to(
        jnp.sum(measure, axis=(1, 2))[:, None, None], numerator.shape
    )
    np.testing.assert_allclose(numerator, expected)


def test_local_adiabatic_phi_solve_matches_formula_without_zonal_correction():
    velocity, _fourier, B, _species, precompute = _setup(zonal_correction=False)
    distribution = jnp.arange(velocity.vpar.shape[0] * velocity.mu.shape[0] * B.shape[0] * 3 * 2)
    distribution = distribution.reshape(velocity.vpar.shape[0], velocity.mu.shape[0], B.shape[0], 3, 2)
    distribution = (distribution + 1.0) / 100.0

    numerator = adiabatic_density_numerator(distribution, precompute)
    phi = solve_adiabatic_electron_phi(distribution, precompute)
    residual = adiabatic_quasineutrality_residual(phi, distribution, precompute)

    np.testing.assert_allclose(phi, -numerator / precompute.denominator, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=2e-13)


def test_zonal_correction_solves_flux_surface_corrected_equation():
    _velocity, fourier, B, _species, precompute = _setup(zonal_correction=True)
    numerator = jnp.zeros((B.shape[0], fourier.kx.shape[0], fourier.ky.shape[0]))
    z_profile = jnp.linspace(-0.4, 0.8, B.shape[0])
    numerator = numerator.at[:, 0, fourier.iyzero].set(0.2 + z_profile)
    numerator = numerator.at[:, 1, fourier.iyzero].set(-0.1 + 0.3 * z_profile)
    numerator = numerator.at[:, 2, 1].set(0.05 * (1.0 + z_profile))

    local_phi = -numerator / precompute.denominator
    phi = solve_adiabatic_electron_phi_from_density(numerator, precompute)
    residual = adiabatic_quasineutrality_residual_from_density(phi, numerator, precompute)

    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=2e-13)
    assert not np.allclose(phi[:, 0, fourier.iyzero], local_phi[:, 0, fourier.iyzero])
    np.testing.assert_allclose(
        phi[:, fourier.ixzero, fourier.iyzero],
        local_phi[:, fourier.ixzero, fourier.iyzero],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(phi[:, 2, 1], local_phi[:, 2, 1], rtol=2e-13, atol=2e-13)


def test_multi_species_quasineutrality_shapes_and_residual():
    species = (_ion(density=0.8), _ion(charge=2.0, mass=4.0, density=0.1, temperature=2.0))
    velocity, fourier, B, _species, precompute = _setup(species=species, zonal_correction=True)
    base = jnp.arange(2 * velocity.vpar.shape[0] * velocity.mu.shape[0] * B.shape[0] * 3 * 2)
    distribution = base.reshape(2, velocity.vpar.shape[0], velocity.mu.shape[0], B.shape[0], 3, 2)
    distribution = distribution / 200.0 + 0.03

    phi = solve_adiabatic_electron_phi(distribution, precompute)
    residual = adiabatic_quasineutrality_residual(phi, distribution, precompute)

    assert precompute.phi_weight.shape == distribution.shape
    assert phi.shape == (B.shape[0], fourier.kx.shape[0], fourier.ky.shape[0])
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=5e-13)


def test_kinetic_phi_solve_regularizes_constant_mode():
    species = (_ion(), _ion(charge=-1.0, mass=1.0, density=1.0, temperature=0.7))
    velocity, fourier, B, _species, _adiabatic = _setup(zonal_correction=False)
    kx = fourier.kx[None, :, None]
    ky = fourier.ky[None, None, :]
    z_factor = 1.0 + 0.2 * jnp.arange(B.shape[0])[:, None, None]
    kperp2 = z_factor * (kx**2 + ky**2)
    flr = species_flr_factors(species, velocity.mu, B, kperp2)
    precompute = build_kinetic_quasineutrality_precompute(
        velocity,
        B,
        flr,
        species,
        fourier_grid=fourier,
    )
    numerator = jnp.zeros((B.shape[0], fourier.kx.shape[0], fourier.ky.shape[0]))
    z_profile = jnp.linspace(0.1, 0.4, B.shape[0])
    numerator = numerator.at[:, 0, 1].set(z_profile)
    numerator = numerator.at[:, fourier.ixzero, fourier.iyzero].set(99.0)

    phi = solve_kinetic_electron_phi_from_density(numerator, precompute)
    residual = kinetic_quasineutrality_residual_from_density(phi, numerator, precompute)

    np.testing.assert_allclose(phi[:, fourier.ixzero, fourier.iyzero], 0.0, atol=0.0)
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=2e-13)

    distribution = jnp.zeros(
        (2, velocity.vpar.shape[0], velocity.mu.shape[0], B.shape[0], 3, 2)
    )
    phi_from_distribution = solve_kinetic_electron_phi(distribution, precompute)
    residual_from_distribution = kinetic_quasineutrality_residual(
        phi_from_distribution, distribution, precompute
    )
    np.testing.assert_allclose(phi_from_distribution, 0.0, atol=0.0)
    np.testing.assert_allclose(residual_from_distribution, 0.0, atol=0.0)


def test_phi_solve_is_jittable_and_differentiable():
    velocity, _fourier, B, species, precompute = _setup(zonal_correction=False)
    distribution = jnp.ones((velocity.vpar.shape[0], velocity.mu.shape[0], B.shape[0], 3, 2)) * 0.02

    @jax.jit
    def objective(scale):
        phi = solve_adiabatic_electron_phi(scale * distribution, precompute)
        return jnp.sum(jnp.real(phi * jnp.conj(phi)))

    grad_value = jax.grad(objective)(1.2)
    finite_difference = (objective(1.2 + 1.0e-5) - objective(1.2 - 1.0e-5)) / 2.0e-5

    assert jnp.isfinite(objective(1.2))
    np.testing.assert_allclose(grad_value, finite_difference, rtol=3e-5, atol=3e-7)

    modified = replace(species, density=1.1)
    assert modified.density == 1.1


def test_diagnostic_integrals_and_spectra():
    velocity, fourier, B, _species, _precompute = _setup()
    values = jnp.ones((velocity.vpar.shape[0], velocity.mu.shape[0], B.shape[0], 3, 2))

    integral = velocity_space_integral(values, velocity, B)

    expected = jnp.ones((B.shape[0], 3, 2)) * (
        B[:, None, None] * jnp.sum(velocity.w_vpar) * jnp.sum(velocity.w_mu)
    )
    np.testing.assert_allclose(integral, expected, rtol=2e-13, atol=2e-13)

    phi = jnp.ones((B.shape[0], 3, 2)) * (1.0 + 2.0j)
    amplitude = mode_amplitude(phi, w_z=B)
    kxky = kxky_spectrum(phi, w_z=B, parseval=fourier.parseval)
    ky = ky_spectrum(phi, w_z=B, parseval=fourier.parseval)

    np.testing.assert_allclose(amplitude, np.sqrt(5.0), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(kxky[:, 0], 5.0)
    np.testing.assert_allclose(kxky[:, 1], 10.0)
    np.testing.assert_allclose(ky, jnp.sum(kxky, axis=0))

    response = 1j * phi
    flux = radial_flux_spectrum(phi, response, fourier.ky, w_z=B, parseval=fourier.parseval)
    total_flux = total_radial_flux(phi, response, fourier.ky, w_z=B, parseval=fourier.parseval)

    np.testing.assert_allclose(flux[:, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(flux[:, 1], 2.0 * fourier.ky[1] * 5.0)
    np.testing.assert_allclose(total_flux, jnp.sum(flux))


def test_electron_param_validation():
    with np.testing.assert_raises(ValueError):
        AdiabaticElectronParams(density=0.0, temperature=1.0)
    with np.testing.assert_raises(ValueError):
        AdiabaticElectronParams(density=1.0, temperature=-1.0)
