from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_linear_rhs_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    dissipation,
    drift_field_drive,
    equilibrium_drive,
    linear_residual,
    linear_residual_from_phi,
    magnetic_drift_advection,
    mirror_force,
    parallel_field_drive,
    parallel_streaming,
)


def _ion(**updates):
    base = dict(
        charge=1.0,
        mass=2.0,
        density=0.8,
        temperature=1.4,
        density_gradient=2.0,
        temperature_gradient=3.0,
    )
    base.update(updates)
    return SpeciesParams(**base)


def _setup(species=None, *, n_z=16, zonal_correction=False):
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=7, n_mu=5, vpar_max=2.0, mu_max=1.5))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=n_z, z_min=0.0, z_max=1.0, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.6, ky_values=(0.0, 0.4))
    )
    geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(q=1.3, shat=0.7, eps=0.18),
    )
    species = _ion() if species is None else species
    electrons = AdiabaticElectronParams(
        density=1.0,
        temperature=1.0,
        zonal_correction=zonal_correction,
    )
    rhs_precompute = build_linear_rhs_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
    )
    residual_precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        electron_params=electrons,
    )
    return velocity, parallel, fourier, geometry, species, rhs_precompute, residual_precompute


def test_rhs_precompute_shapes_and_zero_input_terms():
    velocity, parallel, fourier, _geometry, _species, precompute, _residual_precompute = _setup()
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    distribution = jnp.zeros(shape)
    phi = jnp.zeros(shape[2:])

    assert precompute.flr_factors.bessel_j0.shape == (1, shape[1], shape[2], shape[3], shape[4])
    assert precompute.maxwellian.shape == (1, shape[0], shape[1], shape[2])
    assert precompute.magnetic_drift_frequency.shape == (1,) + shape

    terms = [
        parallel_streaming(distribution, precompute.D_z, precompute.parallel_streaming_coeff),
        magnetic_drift_advection(distribution, precompute.magnetic_drift_frequency),
        mirror_force(distribution, precompute.D_vpar, precompute.mirror_force_coeff),
        equilibrium_drive(phi, precompute),
        parallel_field_drive(phi, precompute.D_z, precompute),
        drift_field_drive(phi, precompute),
        dissipation(distribution),
        linear_residual_from_phi(distribution, phi, precompute),
    ]
    for term in terms:
        assert term.shape == shape
        np.testing.assert_allclose(term, 0.0, atol=0.0)


def test_streaming_and_mirror_terms_match_manufactured_derivatives():
    velocity, parallel, _fourier, _geometry, _species, _precompute, _residual_precompute = _setup(
        n_z=32
    )
    z = parallel.z
    vpar = velocity.vpar

    z_profile = jnp.sin(2.0 * jnp.pi * z)
    distribution_z = jnp.ones((vpar.shape[0], velocity.mu.shape[0], z.shape[0], 1, 1))
    distribution_z = distribution_z * z_profile[None, None, :, None, None]
    streaming_coeff = jnp.ones((vpar.shape[0], z.shape[0]))

    streaming = parallel_streaming(distribution_z, parallel.D_z, streaming_coeff)
    expected_streaming = -2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * z)
    expected_streaming = jnp.ones_like(distribution_z) * expected_streaming[None, None, :, None, None]

    v_profile = vpar**4 - 2.0 * vpar**2 + 0.5 * vpar
    distribution_v = jnp.ones_like(distribution_z) * v_profile[:, None, None, None, None]
    mirror_coeff = jnp.ones((velocity.mu.shape[0], z.shape[0]))

    mirror = mirror_force(distribution_v, velocity.D_vpar, mirror_coeff)
    expected_mirror = 4.0 * vpar**3 - 4.0 * vpar + 0.5
    expected_mirror = jnp.ones_like(distribution_v) * expected_mirror[:, None, None, None, None]

    np.testing.assert_allclose(streaming, expected_streaming, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(mirror, expected_mirror, rtol=2e-10, atol=2e-10)


def test_drift_and_field_drive_terms_match_formulae():
    velocity, parallel, fourier, _geometry, species, precompute, _residual_precompute = _setup()
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    distribution = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape) / 100.0
    phi = jnp.arange(np.prod(shape[2:]), dtype=jnp.float64).reshape(shape[2:]) / 50.0

    drift = magnetic_drift_advection(distribution, precompute.magnetic_drift_frequency)
    np.testing.assert_allclose(
        drift,
        -1j * precompute.magnetic_drift_frequency[0] * distribution,
        rtol=2e-13,
        atol=2e-13,
    )

    gyro_phi = precompute.flr_factors.bessel_j0[0] * phi[None, :, :, :]
    expected_equilibrium = (
        1j
        * precompute.ky[None, None, None, None, :]
        * precompute.E_y[None, None, :, None, None]
        * gyro_phi[None, :, :, :, :]
        * precompute.maxwellian[0][..., None, None]
        * precompute.drive_factor[0][..., None, None]
    )
    np.testing.assert_allclose(equilibrium_drive(phi, precompute), expected_equilibrium)

    dz_gyro_phi = jnp.einsum("ij,mjxy->mixy", precompute.D_z, gyro_phi)
    expected_parallel = (
        -(species.charge / species.temperature)
        * precompute.parallel_streaming_coeff[0, :, None, :, None, None]
        * precompute.maxwellian[0][..., None, None]
        * dz_gyro_phi[None, :, :, :, :]
    )
    np.testing.assert_allclose(parallel_field_drive(phi, precompute.D_z, precompute), expected_parallel)

    expected_drift_drive = (
        -(species.charge / species.temperature)
        * 1j
        * precompute.magnetic_drift_frequency[0]
        * precompute.maxwellian[0][..., None, None]
        * gyro_phi[None, :, :, :, :]
    )
    np.testing.assert_allclose(drift_field_drive(phi, precompute), expected_drift_drive)


def test_full_linear_residual_is_linear_and_jittable():
    velocity, parallel, fourier, _geometry, _species, _rhs, precompute = _setup()
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    base = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    df1 = (base + 1.0) / 200.0
    df2 = jnp.sin(base / 13.0) / 50.0
    a = 1.7
    b = -0.4

    lhs = linear_residual(a * df1 + b * df2, precomputed=precompute)
    rhs = a * linear_residual(df1, precomputed=precompute) + b * linear_residual(
        df2,
        precomputed=precompute,
    )
    jit_rhs = jax.jit(lambda values: linear_residual(values, precomputed=precompute))(df1)

    np.testing.assert_allclose(lhs, rhs, rtol=4e-12, atol=4e-12)
    np.testing.assert_allclose(jit_rhs, linear_residual(df1, precomputed=precompute))


def test_linear_residual_reverse_mode_gradient_matches_finite_difference():
    velocity, parallel, fourier, _geometry, _species, _rhs, precompute = _setup()
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    base = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    distribution = 0.01 * jnp.sin(base / 17.0)

    def objective(scale):
        rhs = linear_residual(scale * distribution, precomputed=precompute)
        return jnp.real(jnp.vdot(rhs, rhs))

    scale = 1.2
    step = 1.0e-5
    grad_value = jax.grad(objective)(scale)
    finite_difference = (objective(scale + step) - objective(scale - step)) / (2.0 * step)

    assert jnp.isfinite(objective(scale))
    np.testing.assert_allclose(grad_value, finite_difference, rtol=2e-5, atol=2e-7)


def test_residual_gradients_flow_through_geometry_arrays_and_species_parameters():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=6, n_mu=4, vpar_max=2.0, mu_max=1.5))
    n_z = 12
    z_min = -0.5 + 0.5 / n_z
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=n_z, z_min=z_min, z_max=z_min + 1.0, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.6, ky_values=(0.0, 0.4))
    )
    base_geometry = build_s_alpha_geometry(
        parallel,
        GeometryScalarParams(q=1.3, shat=0.7, eps=0.18),
    )
    species = _ion()
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    distribution = 0.01 * jnp.cos(jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape))

    def objective(geometry_scale, species_params):
        geometry = replace(
            base_geometry,
            B=base_geometry.B * (1.0 + 0.02 * geometry_scale),
            F=base_geometry.F * (1.0 - 0.01 * geometry_scale),
            G=base_geometry.G + 0.01 * geometry_scale,
            E_y=base_geometry.E_y + 0.02 * geometry_scale,
            D_x=base_geometry.D_x * (1.0 + 0.03 * geometry_scale),
            D_y=base_geometry.D_y * (1.0 - 0.02 * geometry_scale),
            g_yy=base_geometry.g_yy * (1.0 + 0.01 * geometry_scale),
        )
        precompute = build_linear_residual_precompute(
            velocity,
            parallel,
            fourier,
            geometry,
            species_params,
            electron_params=electrons,
        )
        rhs = linear_residual(distribution, precomputed=precompute)
        return jnp.real(jnp.vdot(rhs, rhs))

    geometry_scale = 0.4
    grad_geometry, grad_species = jax.grad(objective, argnums=(0, 1))(geometry_scale, species)
    step = 1.0e-5
    t_plus = replace(species, temperature=species.temperature + step)
    t_minus = replace(species, temperature=species.temperature - step)
    finite_difference_geometry = (
        objective(geometry_scale + step, species) - objective(geometry_scale - step, species)
    ) / (2.0 * step)
    finite_difference_t = (objective(geometry_scale, t_plus) - objective(geometry_scale, t_minus)) / (
        2.0 * step
    )

    assert jnp.isfinite(grad_geometry)
    assert jnp.isfinite(grad_species.temperature)
    np.testing.assert_allclose(grad_geometry, finite_difference_geometry, rtol=6e-5, atol=2e-6)
    np.testing.assert_allclose(
        grad_species.temperature,
        finite_difference_t,
        rtol=6e-5,
        atol=2e-6,
    )


def test_multi_species_linear_residual_shape():
    species = (_ion(density=0.7), _ion(charge=2.0, mass=4.0, density=0.15, temperature=2.0))
    velocity, parallel, fourier, _geometry, _species, _rhs, precompute = _setup(species=species)
    shape = (
        2,
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    distribution = jnp.ones(shape) * 0.01

    rhs = linear_residual(distribution, precompute)

    assert rhs.shape == shape
    assert jnp.all(jnp.isfinite(rhs))
