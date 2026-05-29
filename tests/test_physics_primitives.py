from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from stellarator_gk import (
    FLRFactors,
    SpeciesParams,
    bessel_j0,
    equilibrium_gradient_drive_coefficient,
    gamma0,
    magnetic_drift_frequency,
    maxwellian,
    mirror_force_coefficient,
    normalized_energy,
    parallel_streaming_coefficient,
    polarization_argument,
    species_flr_factors,
    thermodynamic_drive_factor,
    thermal_speed,
)


def _ion(**updates):
    base = dict(
        charge=1.0,
        mass=2.0,
        density=0.7,
        temperature=1.5,
        density_gradient=3.0,
        temperature_gradient=4.0,
    )
    base.update(updates)
    return SpeciesParams(**base)


def test_bessel_j0_matches_scipy_and_has_small_argument_limit():
    x = jnp.asarray([0.0, 1.0e-8, 0.5, 1.0, 8.0, 20.0])
    expected = special.j0(np.asarray(x))

    np.testing.assert_allclose(bessel_j0(x), expected, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(bessel_j0(1.0e-6), 1.0 - 1.0e-12 / 4.0, rtol=0, atol=1e-14)


def test_bessel_j0_gradient_matches_minus_j1():
    points = jnp.asarray([0.2, 1.0, 9.0])

    grad_values = jax.vmap(jax.grad(lambda value: bessel_j0(value)))(points)

    np.testing.assert_allclose(grad_values, -special.j1(np.asarray(points)), rtol=4e-7, atol=2e-7)


def test_gamma0_uses_scaled_bessel_and_remains_finite():
    b = jnp.asarray([0.0, 1.0e-8, 0.5, 10.0, 1000.0])

    np.testing.assert_allclose(gamma0(b), special.i0e(np.asarray(b)), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gamma0(1.0e-8), 1.0 - 1.0e-8, rtol=0, atol=1e-14)
    assert jnp.isfinite(gamma0(1000.0))


def test_energy_maxwellian_and_drive_shapes_and_values():
    species = _ion()
    vpar = jnp.asarray([-1.0, 0.5])
    mu = jnp.asarray([0.0, 0.7, 1.2])
    B = jnp.asarray([0.8, 1.1, 1.4, 1.6])

    energy = normalized_energy(vpar, mu, B, species)
    fmax = maxwellian(vpar, mu, B, species)
    drive = thermodynamic_drive_factor(energy, species)

    assert energy.shape == (2, 3, 4)
    assert fmax.shape == energy.shape
    assert drive.shape == energy.shape
    np.testing.assert_allclose(
        energy[1, 2, 3],
        (vpar[1] ** 2 + 2.0 * mu[2] * B[3]) / species.temperature,
    )
    np.testing.assert_allclose(
        fmax[0, 1, 2],
        species.density
        / (np.pi * species.temperature) ** 1.5
        * np.exp(-float(energy[0, 1, 2])),
    )
    np.testing.assert_allclose(
        drive[0, 1, 2],
        species.density_gradient
        + species.temperature_gradient * (float(energy[0, 1, 2]) - 1.5),
    )
    assert jnp.all(fmax > 0.0)


def test_multi_species_energy_and_maxwellian_have_leading_species_axis():
    species = (_ion(), _ion(charge=-1.0, mass=1.0, temperature=0.8, density=0.4))
    vpar = jnp.asarray([-1.0, 0.5])
    mu = jnp.asarray([0.0, 0.7, 1.2])
    B = jnp.asarray([0.8, 1.1, 1.4, 1.6])

    energy = normalized_energy(vpar, mu, B, species)
    fmax = maxwellian(vpar, mu, B, species)
    drive = thermodynamic_drive_factor(energy, species)

    assert energy.shape == (2, 2, 3, 4)
    assert fmax.shape == energy.shape
    assert drive.shape == energy.shape
    np.testing.assert_allclose(energy[0], normalized_energy(vpar, mu, B, species[0]))
    np.testing.assert_allclose(fmax[1], maxwellian(vpar, mu, B, species[1]))


def test_species_flr_factors_zero_mode_and_formula():
    species = _ion()
    mu = jnp.asarray([0.0, 0.7, 1.2])
    B = jnp.asarray([0.8, 1.1, 1.4, 1.6])
    kperp2 = jnp.ones((4, 3, 2)) * 0.25
    kperp2 = kperp2.at[:, 1, 0].set(0.0)

    factors = species_flr_factors(species, mu, B, kperp2)

    assert isinstance(factors, FLRFactors)
    assert factors.bessel_argument.shape == (3, 4, 3, 2)
    assert factors.bessel_j0.shape == factors.bessel_argument.shape
    assert factors.polarization_argument.shape == (4, 3, 2)
    assert factors.gamma0.shape == factors.polarization_argument.shape
    np.testing.assert_allclose(factors.bessel_j0[:, :, 1, 0], 1.0, atol=1e-13)
    np.testing.assert_allclose(factors.gamma0[:, 1, 0], 1.0, atol=1e-13)
    np.testing.assert_allclose(
        factors.polarization_argument,
        polarization_argument(B, kperp2, species),
    )

    rho_factor = species.mass * thermal_speed(species) / abs(species.charge)
    expected_arg = rho_factor * np.sqrt(0.25) * np.sqrt(2.0 * mu[2] / B[3])
    expected_b = 0.5 * (rho_factor / B[3]) ** 2 * 0.25
    np.testing.assert_allclose(factors.bessel_argument[2, 3, 0, 0], expected_arg)
    np.testing.assert_allclose(factors.polarization_argument[3, 0, 0], expected_b)


def test_multi_species_flr_factors_stack_species_axis():
    species = (_ion(), _ion(charge=-1.0, mass=1.0, temperature=0.8, density=0.4))
    mu = jnp.asarray([0.0, 0.7])
    B = jnp.asarray([0.8, 1.1, 1.4])
    kperp2 = jnp.ones((3, 2, 2)) * 0.25

    factors = species_flr_factors(species, mu, B, kperp2)

    assert factors.bessel_j0.shape == (2, 2, 3, 2, 2)
    assert factors.gamma0.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(factors.bessel_j0[0], species_flr_factors(species[0], mu, B, kperp2).bessel_j0)


def test_gradient_drift_mirror_and_parallel_coefficients():
    species = _ion()
    vpar = jnp.asarray([-1.0, 0.5])
    mu = jnp.asarray([0.0, 0.7, 1.2])
    B = jnp.asarray([0.8, 1.1, 1.4, 1.6])
    E_y = jnp.asarray([0.2, 0.3, 0.4, 0.5])
    D_x = jnp.asarray([1.0, 1.2, 1.4, 1.6])
    D_y = jnp.asarray([-0.2, -0.1, 0.1, 0.2])
    G = jnp.asarray([0.1, -0.2, 0.3, -0.4])
    F = jnp.asarray([0.9, 1.0, 1.1, 1.2])
    kx = jnp.asarray([-0.5, 0.0, 0.5])
    ky = jnp.asarray([0.0, 0.7])

    energy = normalized_energy(vpar, mu, B, species)
    fmax = maxwellian(vpar, mu, B, species)
    drive = thermodynamic_drive_factor(energy, species)
    gradient = equilibrium_gradient_drive_coefficient(fmax, drive, E_y, ky)
    drift = magnetic_drift_frequency(vpar, mu, B, D_x, D_y, kx, ky, species)
    mirror = mirror_force_coefficient(mu, B, G, species)
    parallel = parallel_streaming_coefficient(vpar, F, species)

    assert gradient.shape == (2, 3, 4, 2)
    assert drift.shape == (2, 3, 4, 3, 2)
    assert mirror.shape == (3, 4)
    assert parallel.shape == (2, 4)
    np.testing.assert_allclose(
        gradient[1, 2, 3, 1],
        fmax[1, 2, 3] * drive[1, 2, 3] * E_y[3] * ky[1],
    )
    np.testing.assert_allclose(
        drift[1, 2, 3, 0, 1],
        (vpar[1] ** 2 + mu[2] * B[3]) * (kx[0] * D_x[3] + ky[1] * D_y[3]),
    )
    np.testing.assert_allclose(mirror[2, 3], thermal_speed(species) * mu[2] * B[3] * G[3])
    np.testing.assert_allclose(parallel[1, 3], thermal_speed(species) * vpar[1] * F[3])


def test_physics_primitives_are_jittable_and_differentiable():
    species = _ion()
    vpar = jnp.asarray([-1.0, 0.5])
    mu = jnp.asarray([0.2, 0.7, 1.2])
    B = jnp.asarray([0.8, 1.1, 1.4, 1.6])
    kperp2 = jnp.ones((4, 3, 2)) * 0.2
    D_x = jnp.asarray([1.0, 1.2, 1.4, 1.6])
    D_y = jnp.asarray([-0.2, -0.1, 0.1, 0.2])
    kx = jnp.asarray([-0.5, 0.0, 0.5])
    ky = jnp.asarray([0.0, 0.7])

    @jax.jit
    def objective(params):
        energy = normalized_energy(vpar, mu, B, params)
        fmax = maxwellian(vpar, mu, B, params)
        drive = thermodynamic_drive_factor(energy, params)
        flr = species_flr_factors(params, mu, B, kperp2)
        drift = magnetic_drift_frequency(vpar, mu, B, D_x, D_y, kx, ky, params)
        return (
            jnp.sum(fmax * drive)
            + 0.1 * jnp.sum(flr.bessel_j0)
            + 0.05 * jnp.sum(flr.gamma0)
            + 0.01 * jnp.sum(drift)
        )

    grad_species = jax.grad(objective)(species)

    assert jnp.isfinite(objective(species))
    assert jnp.isfinite(grad_species.temperature)
    assert jnp.isfinite(grad_species.mass)
    assert jnp.isfinite(grad_species.charge)

    step = 1.0e-5
    plus = replace(species, temperature=species.temperature + step)
    minus = replace(species, temperature=species.temperature - step)
    finite_difference = (objective(plus) - objective(minus)) / (2.0 * step)
    np.testing.assert_allclose(grad_species.temperature, finite_difference, rtol=2e-5, atol=2e-6)


def test_species_params_rejects_zero_charge():
    with np.testing.assert_raises(ValueError):
        _ion(charge=0.0)
