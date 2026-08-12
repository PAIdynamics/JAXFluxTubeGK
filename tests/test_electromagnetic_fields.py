import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fluxtube_gk import (
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    PerpendicularMagneticPrecompute,
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
    dense_matrix_from_action,
    estimate_linear_cfl_dt,
    integrate_fixed_step,
    linear_residual,
    mixed_to_physical_distribution,
    parallel_ampere_residual,
    physical_to_mixed_distribution,
    rk4_step,
    solve_electromagnetic_fields,
    solve_parallel_ampere,
    solve_perpendicular_magnetic_fields,
)
from jax_fluxtube_gk.tem_validation import (
    _build_tem_system,
    _initial_tem_state,
    gyaradax_tem_case_spec,
)


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

    eager_even = solve_parallel_ampere(even, precompute)
    compiled_even = jax.jit(solve_parallel_ampere)(even, precompute)
    np.testing.assert_array_equal(eager_even, np.zeros_like(eager_even))
    np.testing.assert_array_equal(compiled_even, np.zeros_like(compiled_even))
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


def test_perpendicular_magnetic_floor_preserves_denominator_sign():
    precompute = PerpendicularMagneticPrecompute(
        phi_weight=jnp.ones((1, 1, 1, 1, 1, 1)),
        bpar_weight=jnp.zeros((1, 1, 1, 1, 1, 1)),
        denominator=jnp.asarray([[[-1.0e-20]]]),
        bpar_chi_factor=jnp.zeros((1, 1, 1, 1, 1, 1)),
        beta=jnp.asarray(0.01),
        denominator_floor=1.0e-14,
        n_species=1,
    )

    phi, bpar = solve_perpendicular_magnetic_fields(
        jnp.ones((1, 1, 1, 1, 1)), precompute
    )

    np.testing.assert_allclose(phi, 1.0e14, rtol=1.0e-15)
    np.testing.assert_allclose(bpar, 0.0, atol=0.0)


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

    from jax_fluxtube_gk import solve_kinetic_electron_phi

    np.testing.assert_allclose(apar, 0.0, atol=0.0)
    np.testing.assert_allclose(bpar, 0.0, atol=0.0)
    np.testing.assert_allclose(physical, mixed, atol=0.0)
    np.testing.assert_allclose(phi, solve_kinetic_electron_phi(mixed, linear.field), rtol=3e-14)


def test_electromagnetic_linear_residual_recovers_electrostatic_limit():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()
    electrostatic = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species, field_model="kinetic"
    )
    electromagnetic = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="electromagnetic",
        beta=0.0,
    )
    mixed = jax.random.normal(
        jax.random.key(13), (2, 8, 4, parallel.z.shape[0], 3, 2)
    ).astype(jnp.complex128)

    expected = linear_residual(mixed, precomputed=electrostatic)
    observed = jax.jit(linear_residual)(mixed, precomputed=electromagnetic)

    np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-13)


def test_electromagnetic_linear_residual_has_finite_nonzero_beta_response():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()
    mixed = jax.random.normal(
        jax.random.key(14), (2, 8, 4, parallel.z.shape[0], 3, 2)
    ).astype(jnp.complex128)

    def objective(beta):
        precompute = build_linear_residual_precompute(
            velocity,
            parallel,
            fourier,
            geometry,
            species,
            field_model="electromagnetic",
            beta=beta,
        )
        rhs = linear_residual(mixed, precomputed=precompute)
        return jnp.sum(jnp.abs(rhs) ** 2)

    value = objective(0.01)
    derivative = jax.grad(objective)(0.01)
    assert np.isfinite(float(value))
    assert np.isfinite(float(derivative))
    assert float(jnp.abs(derivative)) > 0.0


def test_electromagnetic_cfl_bound_includes_mixed_field_feedback():
    velocity, parallel, fourier, geometry, species, _ampere = _setup()

    def bound(beta):
        precompute = build_linear_residual_precompute(
            velocity,
            parallel,
            fourier,
            geometry,
            species,
            field_model="electromagnetic",
            beta=beta,
        )
        return estimate_linear_cfl_dt(precompute)

    beta_zero = bound(0.0)
    finite_beta = bound(0.01)
    derivative = jax.grad(bound)(0.01)

    assert np.isfinite(float(beta_zero))
    assert np.isfinite(float(finite_beta))
    assert finite_beta > 0.0
    assert finite_beta < beta_zero
    assert np.isfinite(float(derivative))


def test_electromagnetic_cfl_bound_dominates_small_dense_operator_row_sum():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=2, vpar_max=2.0, mu_max=2.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=3, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,))
    )
    geometry = build_s_alpha_geometry(
        parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18)
    )
    species = (
        SpeciesParams(1.0, 1.0, 1.0, 1.0, 2.2, 0.0),
        SpeciesParams(-1.0, 0.01, 1.0, 1.0, 2.2, 6.9),
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="electromagnetic",
        beta=0.01,
    )
    template = jnp.zeros((2, 3, 2, 3, 1, 1), dtype=jnp.complex128)
    matrix = dense_matrix_from_action(
        lambda state: linear_residual(state, precomputed=precompute), template
    )
    exact_row_sum = jnp.max(jnp.sum(jnp.abs(matrix), axis=1))
    estimated_dt = estimate_linear_cfl_dt(precompute, safety=1.0, rk4_radius=1.0)

    assert estimated_dt <= 1.0 / exact_row_sum


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

    # jax-fluxtube-gk uses a differentiable Cephes J0 approximation while
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


@pytest.mark.external
def test_electromagnetic_rhs_and_rk4_step_match_pinned_gyaradax(gyaradax_root):
    import sys

    if str(gyaradax_root) not in sys.path:
        sys.path.insert(0, str(gyaradax_root))
    from gyaradax.backends import create_ops
    from gyaradax.fields import _compute_fields, g_to_f
    from gyaradax.geometry import compute_geometry
    from gyaradax.params import GKParams
    from gyaradax.solver import default_state, gkstep_single, linear_precompute

    spec = dataclasses.replace(gyaradax_tem_case_spec(), n_z=8, n_vpar=8, n_mu=4)
    velocity, parallel, fourier, geometry, species, electrostatic = _build_tem_system(spec)
    from jax_fluxtube_gk import build_mode_connectivity

    observed_precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        field_model="electromagnetic",
        parallel_recurrence_rate=0.0,
        velocity_recurrence_rate=0.0,
        mode_connectivity=build_mode_connectivity(fourier),
        parallel_derivative_model="gkw_upwind",
        beta=0.01,
    )
    mixed = _initial_tem_state(electrostatic, parallel, spec)
    mixed = mixed * (
        1.0 + 0.1 * velocity.vpar[None, :, None, None, None, None]
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
        nlapar=True,
        nlbpar=True,
        mas=masses,
        signz=jnp.asarray([1.0, -1.0]),
        tmp=jnp.ones(2),
        de=jnp.ones(2),
        vthrat=jnp.sqrt(1.0 / masses),
        rln=jnp.asarray([spec.density_gradient, spec.density_gradient]),
        rlt=jnp.asarray([spec.ion_temperature_gradient, spec.electron_temperature_gradient]),
        disp_par=0.0,
        disp_vp=0.0,
        disp_x=0.0,
        disp_y=0.0,
        drive_scale=1.0,
        dgrid=1.0,
        sgr_dist=float(reference_geometry["sgr_dist"]),
        dvp=float(reference_geometry["dvp"]),
        kxmax=float(np.max(np.abs(np.asarray(reference_geometry["kxrh"])))) or 1.0,
        kymax=float(np.max(np.asarray(reference_geometry["krho"]))) or 1.0,
        backend="jax",
        mixed_precision=False,
        disable_per_ky_norm=True,
    )
    reference_precompute = linear_precompute(reference_geometry, params)
    reference_ops = create_ops(reference_precompute, backend="jax", mixed_precision=False)
    phi_ref, apar_ref, bpar_ref = _compute_fields(
        mixed, reference_geometry, params, reference_precompute
    )
    physical_ref = g_to_f(mixed, apar_ref, params, reference_precompute)
    reference_rhs = reference_ops.linear_rhs(
        physical_ref,
        phi_ref,
        reference_geometry,
        params,
        reference_precompute,
        apar=apar_ref,
        bpar=bpar_ref,
    )
    reference_electrostatic_rhs = reference_ops.linear_rhs(
        physical_ref,
        phi_ref,
        reference_geometry,
        params,
        reference_precompute,
        apar=None,
        bpar=None,
    )
    phi_obs, apar_obs, bpar_obs, physical_obs = solve_electromagnetic_fields(
        mixed, observed_precompute.field
    )
    observed_rhs = linear_residual(mixed, precomputed=observed_precompute)
    from jax_fluxtube_gk import linear_residual_from_phi

    observed_electrostatic_rhs = linear_residual_from_phi(
        physical_obs, phi_obs, observed_precompute.rhs
    )

    np.testing.assert_allclose(phi_obs, phi_ref, rtol=8e-7, atol=2e-12, err_msg="phi")
    np.testing.assert_allclose(apar_obs, apar_ref, rtol=8e-7, atol=2e-12, err_msg="apar")
    np.testing.assert_allclose(bpar_obs, bpar_ref, rtol=8e-7, atol=2e-12, err_msg="bpar")
    np.testing.assert_allclose(
        physical_obs, physical_ref, rtol=8e-7, atol=2e-12, err_msg="physical_f"
    )
    np.testing.assert_allclose(
        observed_rhs - observed_electrostatic_rhs,
        reference_rhs - reference_electrostatic_rhs,
        rtol=2e-6,
        atol=2e-10,
        err_msg="electromagnetic_rhs_increment",
    )
    np.testing.assert_allclose(
        observed_rhs,
        reference_rhs,
        rtol=2e-6,
        atol=2e-10,
        err_msg="full_electromagnetic_rhs",
    )

    dt = 1.0e-4
    observed_next = rk4_step(
        mixed,
        dt,
        lambda state: linear_residual(state, precomputed=observed_precompute),
    )
    reference_next, _, reference_state = gkstep_single(
        mixed,
        reference_geometry,
        params,
        default_state(nky=1),
        reference_precompute,
        ops=reference_ops,
        dt_override=jnp.asarray(dt),
    )
    np.testing.assert_allclose(
        observed_next,
        reference_next,
        rtol=2e-6,
        atol=2e-10,
        err_msg="electromagnetic_rk4_step",
    )

    n_steps = 5
    observed_trajectory = integrate_fixed_step(
        mixed,
        dt,
        n_steps,
        lambda state: linear_residual(state, precomputed=observed_precompute),
        store_history=False,
    ).state
    reference_trajectory = reference_next
    for _ in range(1, n_steps):
        reference_trajectory, _, reference_state = gkstep_single(
            reference_trajectory,
            reference_geometry,
            params,
            reference_state,
            reference_precompute,
            ops=reference_ops,
            dt_override=jnp.asarray(dt),
        )
    np.testing.assert_allclose(
        observed_trajectory,
        reference_trajectory,
        rtol=3e-6,
        atol=3e-10,
        err_msg="electromagnetic_five_step_trajectory",
    )
