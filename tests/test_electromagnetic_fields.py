import dataclasses

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
    build_fourier_grid,
    build_electromagnetic_field_precompute,
    build_linear_residual_precompute,
    build_parallel_ampere_precompute,
    build_perpendicular_magnetic_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    mixed_to_physical_distribution,
    parallel_ampere_residual,
    physical_to_mixed_distribution,
    solve_electromagnetic_fields,
    solve_parallel_ampere,
    solve_perpendicular_magnetic_fields,
)
from stellarator_gk.tem_validation import _build_tem_system, gyaradax_tem_case_spec


def _setup(beta=0.01):
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=4,
            vpar_max=3.0,
            mu_max=4.5,
            backend="finite_difference",
        )
    )
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=8, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.4, ky_values=(0.0, 0.3))
    )
    geometry = build_s_alpha_geometry(
        parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18)
    )
    species = (
        SpeciesParams(1.0, 1.0, 1.0, 1.0, 2.2, 0.0),
        SpeciesParams(-1.0, 0.01, 1.0, 1.0, 2.2, 6.9),
    )
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    ampere = build_parallel_ampere_precompute(
        velocity,
        geometry,
        fourier,
        species,
        linear.rhs.flr_factors,
        beta=beta,
    )
    return velocity, parallel, fourier, geometry, species, ampere


def test_parallel_ampere_solve_closes_discrete_field_residual():
    velocity, parallel, fourier, _geometry, _species, precompute = _setup()
    shape = (2, velocity.vpar.shape[0], velocity.mu.shape[0], parallel.z.shape[0], 3, 2)
    state = jax.random.normal(jax.random.key(2), shape).astype(jnp.complex128)

    apar = jax.jit(solve_parallel_ampere)(state, precompute)
    residual = parallel_ampere_residual(apar, state, precompute)

    assert apar.shape == (parallel.z.shape[0], fourier.kx.shape[0], fourier.ky.shape[0])
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-12, rtol=0.0)


def test_parallel_ampere_selects_odd_parallel_velocity_current():
    velocity, parallel, _fourier, _geometry, _species, precompute = _setup()
    envelope = jnp.ones((2, 8, 4, parallel.z.shape[0], 3, 2))
    even = (velocity.vpar**2)[None, :, None, None, None, None] * envelope
    odd = velocity.vpar[None, :, None, None, None, None] * envelope

    np.testing.assert_allclose(solve_parallel_ampere(even, precompute), 0.0, atol=1.0e-14)
    assert float(jnp.max(jnp.abs(solve_parallel_ampere(odd, precompute)))) > 0.0


def test_parallel_ampere_vanishes_in_electrostatic_limit():
    velocity, parallel, _fourier, _geometry, _species, precompute = _setup(beta=0.0)
    state = jnp.ones((2, 8, 4, parallel.z.shape[0], 3, 2))

    np.testing.assert_allclose(precompute.source_weight, 0.0, atol=0.0)
    np.testing.assert_allclose(solve_parallel_ampere(state, precompute), 0.0, atol=0.0)


def test_parallel_ampere_beta_response_is_differentiable():
    velocity, parallel, fourier, geometry, species, _precompute = _setup()
    state = velocity.vpar[None, :, None, None, None, None] * jnp.ones(
        (2, 8, 4, parallel.z.shape[0], 3, 2)
    )
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )

    def objective(beta):
        precompute = build_parallel_ampere_precompute(
            velocity, geometry, fourier, species, linear.rhs.flr_factors, beta=beta
        )
        return jnp.sum(jnp.abs(solve_parallel_ampere(state, precompute)) ** 2)

    value = objective(0.01)
    derivative = jax.grad(objective)(0.01)
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))


def test_perpendicular_magnetic_solve_has_electrostatic_limit_and_beta_response():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    state = jax.random.normal(
        jax.random.key(7), (2, 8, 4, parallel.z.shape[0], 3, 2)
    ).astype(jnp.complex128)
    electrostatic = build_perpendicular_magnetic_precompute(
        velocity, geometry, species, linear.rhs.flr_factors, beta=0.0
    )
    electromagnetic = build_perpendicular_magnetic_precompute(
        velocity, geometry, species, linear.rhs.flr_factors, beta=0.01
    )

    phi_es, bpar_es = solve_perpendicular_magnetic_fields(state, electrostatic)
    phi_em, bpar_em = solve_perpendicular_magnetic_fields(state, electromagnetic)

    np.testing.assert_allclose(bpar_es, 0.0, atol=0.0)
    assert np.all(np.isfinite(np.asarray(phi_es)))
    assert np.all(np.isfinite(np.asarray(phi_em)))
    assert float(jnp.max(jnp.abs(bpar_em))) > 0.0


def test_perpendicular_magnetic_response_is_differentiable():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    state = jnp.ones((2, 8, 4, parallel.z.shape[0], 3, 2))

    def objective(beta):
        precompute = build_perpendicular_magnetic_precompute(
            velocity, geometry, species, linear.rhs.flr_factors, beta=beta
        )
        phi, bpar = solve_perpendicular_magnetic_fields(state, precompute)
        return jnp.sum(jnp.abs(phi) ** 2 + jnp.abs(bpar) ** 2)

    assert np.isfinite(float(jax.grad(objective)(0.01)))


def test_electromagnetic_mixed_variable_transform_roundtrips_and_closes_fields():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    precompute = build_electromagnetic_field_precompute(
        velocity, geometry, fourier, species, linear.rhs.flr_factors, beta=0.01
    )
    mixed = jax.random.normal(
        jax.random.key(11), (2, 8, 4, parallel.z.shape[0], 3, 2)
    ).astype(jnp.complex128)

    phi, apar, bpar, physical = solve_electromagnetic_fields(mixed, precompute)
    restored = physical_to_mixed_distribution(physical, apar, precompute.ampere)

    np.testing.assert_allclose(restored, mixed, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        physical,
        mixed_to_physical_distribution(mixed, apar, precompute.ampere),
        rtol=0.0,
        atol=0.0,
    )
    assert phi.shape == apar.shape == bpar.shape == (parallel.z.shape[0], 3, 2)


def test_electromagnetic_field_contract_recovers_kinetic_electrostatic_limit():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()
    linear = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    precompute = build_electromagnetic_field_precompute(
        velocity, geometry, fourier, species, linear.rhs.flr_factors, beta=0.0
    )
    mixed = jax.random.normal(
        jax.random.key(12), (2, 8, 4, parallel.z.shape[0], 3, 2)
    ).astype(jnp.complex128)
    phi, apar, bpar, physical = solve_electromagnetic_fields(mixed, precompute)

    from stellarator_gk import solve_kinetic_electron_phi

    np.testing.assert_allclose(apar, 0.0, atol=0.0)
    np.testing.assert_allclose(bpar, 0.0, atol=0.0)
    np.testing.assert_allclose(physical, mixed, atol=0.0)
    np.testing.assert_allclose(phi, solve_kinetic_electron_phi(mixed, linear.field), rtol=3e-14)


@pytest.mark.external
def test_parallel_ampere_precompute_matches_pinned_gyaradax(gyaradax_root):
    import sys

    if str(gyaradax_root) not in sys.path:
        sys.path.insert(0, str(gyaradax_root))
    from gyaradax.geometry import compute_geometry
    from gyaradax.integrals import precompute_apar
    from gyaradax.params import GKParams
    from gyaradax.solver import linear_precompute

    spec = dataclasses.replace(gyaradax_tem_case_spec(), n_z=8, n_vpar=8, n_mu=4)
    velocity, _parallel, fourier, geometry, species, linear = _build_tem_system(spec)
    observed = build_parallel_ampere_precompute(
        velocity,
        geometry,
        fourier,
        species,
        linear.rhs.flr_factors,
        beta=0.01,
    )
    kthnorm = spec.q / (2.0 * np.pi * spec.eps)
    reference_geometry = compute_geometry(
        q=spec.q,
        shat=spec.shat,
        eps=spec.eps,
        ns=8,
        nvpar=8,
        nmu=4,
        vpar_max=3.0,
        nkx=1,
        nky=1,
        nperiod=2,
        kxmax=spec.ky * kthnorm,
        krhomax=spec.ky * kthnorm,
        geom_type="s-alpha",
    )
    masses = jnp.asarray([1.0, spec.electron_mass])
    params = GKParams(
        adiabatic_electrons=False,
        beta=0.01,
        mas=masses,
        signz=jnp.asarray([1.0, -1.0]),
        tmp=jnp.ones(2),
        de=jnp.ones(2),
        vthrat=jnp.sqrt(1.0 / masses),
        rln=jnp.asarray([spec.density_gradient, spec.density_gradient]),
        rlt=jnp.asarray([spec.ion_temperature_gradient, spec.electron_temperature_gradient]),
        dgrid=1.0,
        nlapar=True,
    )
    reference_weight, reference_denominator, reference_kperp2 = precompute_apar(
        reference_geometry, params
    )

    # optimal-fusion uses a differentiable Cephes J0 approximation while
    # Gyaradax uses jax.scipy.special.bessel_jn.  Their field coefficients
    # agree to the approximation error rather than bitwise precision.
    np.testing.assert_allclose(observed.source_weight, reference_weight, rtol=3e-8, atol=2e-13)
    np.testing.assert_allclose(observed.denominator, reference_denominator, rtol=3e-8, atol=2e-13)
    np.testing.assert_allclose(
        observed.kperp_squared,
        np.asarray(reference_kperp2).reshape(observed.kperp_squared.shape),
        rtol=2e-13,
        atol=2e-13,
    )
    reference_precompute = linear_precompute(reference_geometry, params)
    np.testing.assert_allclose(
        observed.g_to_f_factor,
        reference_precompute["g2f_factor"],
        rtol=3e-8,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        observed.apar_chi_factor,
        reference_precompute["apar_chi_factor"],
        rtol=3e-8,
        atol=2e-13,
    )


@pytest.mark.external
def test_perpendicular_magnetic_precompute_matches_pinned_gyaradax(gyaradax_root):
    import sys

    if str(gyaradax_root) not in sys.path:
        sys.path.insert(0, str(gyaradax_root))
    from gyaradax.geometry import compute_geometry
    from gyaradax.integrals import _species_bessel_gamma, precompute_bpar
    from gyaradax.params import GKParams

    spec = dataclasses.replace(gyaradax_tem_case_spec(), n_z=8, n_vpar=8, n_mu=4)
    velocity, _parallel, _fourier, geometry, species, linear = _build_tem_system(spec)
    observed = build_perpendicular_magnetic_precompute(
        velocity, geometry, species, linear.rhs.flr_factors, beta=0.01
    )
    kthnorm = spec.q / (2.0 * np.pi * spec.eps)
    reference_geometry = compute_geometry(
        q=spec.q,
        shat=spec.shat,
        eps=spec.eps,
        ns=8,
        nvpar=8,
        nmu=4,
        vpar_max=3.0,
        nkx=1,
        nky=1,
        nperiod=2,
        kxmax=spec.ky * kthnorm,
        krhomax=spec.ky * kthnorm,
        geom_type="s-alpha",
    )
    masses = jnp.asarray([1.0, spec.electron_mass])
    params = GKParams(
        adiabatic_electrons=False,
        beta=0.01,
        mas=masses,
        signz=jnp.asarray([1.0, -1.0]),
        tmp=jnp.ones(2),
        de=jnp.ones(2),
        vthrat=jnp.sqrt(1.0 / masses),
        dgrid=1.0,
    )
    species_geometry = dict(reference_geometry)
    species_geometry.update(
        mas=params.mas,
        signz=params.signz,
        tmp=params.tmp,
        de=params.de,
        vthrat=params.vthrat,
    )
    reference_bessel, _ = _species_bessel_gamma(species_geometry)
    species_precompute = {"bessel": reference_bessel}
    reference = precompute_bpar(reference_geometry, params, species_precompute)

    for name, reference_name in (
        ("phi_weight", "phi_weight"),
        ("bpar_weight", "bpar_weight"),
        ("denominator", "phi_diag"),
        ("bpar_chi_factor", "bpar_chi_factor"),
    ):
        np.testing.assert_allclose(
            getattr(observed, name),
            reference[reference_name],
            rtol=5e-7,
            atol=2e-12,
            err_msg=name,
        )
