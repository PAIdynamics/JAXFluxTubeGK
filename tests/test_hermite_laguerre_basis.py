import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from jax_fluxtube_gk import (
    GxMomentRHSParams,
    VelocityBasisKind,
    VelocityBasisSpec,
    apply_gx_kz_hypercollision,
    apply_hypercollision,
    apply_linked_abs_kz,
    apply_linked_grad_z,
    build_velocity_basis,
    density_moment,
    fluid_moments,
    free_energy_spectrum,
    gamma0,
    gamma0_limit_error,
    gx_moment_adiabatic_phi,
    gx_moment_itg_drive_source,
    gx_moment_linear_rhs,
    gx_kz_hypercollision_hermite_rates,
    gx_kz_hypercollision_prefactor,
    gx_linked_kz_wavenumbers,
    gyroaverage_laguerre_coefficients,
    hypercollision_damping_rates,
    parallel_heat_flux_moment,
    normalized_hermite_functions,
    parallel_flow_moment,
    parallel_temperature_moment,
    perpendicular_heat_flux_moment,
    perpendicular_temperature_moment,
    signed_laguerre_polynomials,
    spectral_to_velocity_grid,
    truncated_gamma0_from_laguerre,
    velocity_grid_to_spectral,
)


def _basis():
    return build_velocity_basis(
        VelocityBasisSpec(
            n_hermite=6,
            n_laguerre=5,
            n_hermite_grid=8,
            n_laguerre_grid=7,
        )
    )


def test_velocity_basis_spec_is_static_pytree_metadata():
    spec = VelocityBasisSpec(n_hermite=5, n_laguerre=4)
    leaves, treedef = jax.tree_util.tree_flatten(spec)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert leaves == []
    assert rebuilt == spec
    assert spec.kind == VelocityBasisKind.HERMITE_LAGUERRE.value


def test_normalized_polynomial_values_follow_gx_convention():
    x = jnp.asarray([-1.0, 0.0, 2.0])

    hermite = normalized_hermite_functions(4, x)
    laguerre = signed_laguerre_polynomials(3, x)

    np.testing.assert_allclose(hermite[0], 1.0)
    np.testing.assert_allclose(hermite[1], x)
    np.testing.assert_allclose(hermite[2], (x**2 - 1.0) / np.sqrt(2.0))
    np.testing.assert_allclose(hermite[3], (x**3 - 3.0 * x) / np.sqrt(6.0))
    np.testing.assert_allclose(laguerre[0], 1.0)
    np.testing.assert_allclose(laguerre[1], x - 1.0)
    np.testing.assert_allclose(laguerre[2], 1.0 - 2.0 * x + 0.5 * x**2)


def test_hermite_and_laguerre_transforms_are_orthonormal():
    basis = _basis()

    hermite_gram = basis.hermite_to_spectral @ basis.hermite_to_grid
    laguerre_gram = basis.laguerre_to_spectral @ basis.laguerre_to_grid

    np.testing.assert_allclose(hermite_gram, jnp.eye(basis.n_hermite), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(laguerre_gram, jnp.eye(basis.n_laguerre), rtol=1e-13, atol=1e-13)


def test_spectral_grid_transform_round_trip_with_trailing_axes():
    basis = _basis()
    coeffs = jnp.arange(basis.n_laguerre * basis.n_hermite * 3, dtype=jnp.float64)
    coeffs = coeffs.reshape(basis.n_laguerre, basis.n_hermite, 3) / 10.0

    values = spectral_to_velocity_grid(coeffs, basis)
    rebuilt = velocity_grid_to_spectral(values, basis)

    assert values.shape == (basis.n_laguerre_grid, basis.n_hermite_grid, 3)
    np.testing.assert_allclose(rebuilt, coeffs, rtol=2e-13, atol=2e-13)


def test_modal_coupling_matrices_match_projected_grid_multiplication():
    basis = _basis()
    coeffs = jnp.arange(basis.n_laguerre * basis.n_hermite, dtype=jnp.float64)
    coeffs = coeffs.reshape(basis.n_laguerre, basis.n_hermite) / 7.0
    values = spectral_to_velocity_grid(coeffs, basis)

    hermite_projected = velocity_grid_to_spectral(values * basis.hermite_nodes[None, :], basis)
    hermite_matrix = jnp.einsum("mn,ln->lm", basis.hermite_v, coeffs)
    laguerre_projected = velocity_grid_to_spectral(values * basis.laguerre_nodes[:, None], basis)
    laguerre_matrix = jnp.einsum("ln,nm->lm", basis.laguerre_x, coeffs)

    np.testing.assert_allclose(hermite_projected, hermite_matrix, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(laguerre_projected, laguerre_matrix, rtol=2e-13, atol=2e-13)


def test_low_order_moment_diagnostics_match_quadrature():
    basis = _basis()
    coeffs = jnp.zeros((basis.n_laguerre, basis.n_hermite, 2), dtype=jnp.float64)
    coeffs = coeffs.at[0, 0].set(jnp.asarray([1.0, 2.0]))
    coeffs = coeffs.at[0, 1].set(jnp.asarray([3.0, 4.0]))
    coeffs = coeffs.at[0, 2].set(jnp.asarray([5.0, 6.0]))
    coeffs = coeffs.at[0, 3].set(jnp.asarray([9.0, 10.0]))
    coeffs = coeffs.at[1, 0].set(jnp.asarray([7.0, 8.0]))
    coeffs = coeffs.at[1, 1].set(jnp.asarray([11.0, 12.0]))

    values = spectral_to_velocity_grid(coeffs, basis)
    density_quad = jnp.einsum("j,v,jvk->k", basis.laguerre_weights, basis.hermite_weights, values)
    flow_quad = jnp.einsum(
        "j,v,v,jvk->k",
        basis.laguerre_weights,
        basis.hermite_weights,
        basis.hermite_nodes,
        values,
    )
    tpar_quad = jnp.einsum(
        "j,v,v,jvk->k",
        basis.laguerre_weights,
        basis.hermite_weights,
        basis.hermite_nodes**2 - 1.0,
        values,
    )
    tperp_quad = jnp.einsum(
        "j,v,j,jvk->k",
        basis.laguerre_weights,
        basis.hermite_weights,
        basis.laguerre_nodes - 1.0,
        values,
    )
    qpar_quad = jnp.einsum(
        "j,v,v,jvk->k",
        basis.laguerre_weights,
        basis.hermite_weights,
        basis.hermite_nodes**3 - 3.0 * basis.hermite_nodes,
        values,
    )
    qperp_quad = jnp.einsum(
        "j,v,j,v,jvk->k",
        basis.laguerre_weights,
        basis.hermite_weights,
        basis.laguerre_nodes - 1.0,
        basis.hermite_nodes,
        values,
    )

    np.testing.assert_allclose(density_moment(coeffs), density_quad, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(parallel_flow_moment(coeffs), flow_quad, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        parallel_temperature_moment(coeffs), tpar_quad, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        perpendicular_temperature_moment(coeffs), tperp_quad, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        parallel_heat_flux_moment(coeffs), qpar_quad, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        perpendicular_heat_flux_moment(coeffs), qperp_quad, rtol=2e-13, atol=2e-13
    )

    moments = fluid_moments(coeffs)
    np.testing.assert_allclose(moments["density"], density_quad, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        moments["parallel_heat_flux"], qpar_quad, rtol=2e-13, atol=2e-13
    )


def test_gyroaverage_laguerre_coefficients_match_quadrature_and_gamma0_limit():
    basis = _basis()
    b = 0.7
    coefficients = gyroaverage_laguerre_coefficients(jnp.asarray(b), basis.n_laguerre)
    psi = np.asarray(signed_laguerre_polynomials(basis.n_laguerre, basis.laguerre_nodes))
    numerical = np.einsum(
        "j,lj,j->l",
        np.asarray(basis.laguerre_weights),
        psi,
        special.j0(np.sqrt(2.0 * np.asarray(basis.laguerre_nodes) * b)),
    )

    np.testing.assert_allclose(coefficients, numerical, rtol=5e-11, atol=5e-12)
    np.testing.assert_allclose(
        truncated_gamma0_from_laguerre(jnp.asarray(b), 30),
        gamma0(jnp.asarray(b)),
        rtol=2e-13,
        atol=2e-13,
    )
    assert abs(float(gamma0_limit_error(jnp.asarray(b), 30))) < 2e-13
    np.testing.assert_allclose(
        gyroaverage_laguerre_coefficients(jnp.asarray(0.0), 4),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        atol=0.0,
    )


def test_gyroaveraged_moment_diagnostics_use_gx_weights():
    coeffs = jnp.arange(5 * 4 * 2, dtype=jnp.float64).reshape(5, 4, 2) / 10.0
    b = jnp.asarray([0.3, 0.9])
    gyro = gyroaverage_laguerre_coefficients(b, coeffs.shape[0])

    density = density_moment(coeffs, gyro)
    flow = parallel_flow_moment(coeffs, gyro)
    tpar = parallel_temperature_moment(coeffs, gyro)
    tperp = perpendicular_temperature_moment(coeffs, gyro)
    qpar = parallel_heat_flux_moment(coeffs, gyro)
    qperp = perpendicular_heat_flux_moment(coeffs, gyro)

    np.testing.assert_allclose(density, jnp.einsum("lk,lk->k", gyro, coeffs[:, 0]))
    np.testing.assert_allclose(flow, jnp.einsum("lk,lk->k", gyro, coeffs[:, 1]))
    np.testing.assert_allclose(tpar, np.sqrt(2.0) * jnp.einsum("lk,lk->k", gyro, coeffs[:, 2]))
    np.testing.assert_allclose(qpar, np.sqrt(6.0) * jnp.einsum("lk,lk->k", gyro, coeffs[:, 3]))

    ell = jnp.arange(coeffs.shape[0], dtype=jnp.float64)[:, None]
    jm1 = jnp.concatenate([jnp.zeros_like(gyro[:1]), gyro[:-1]], axis=0)
    jp1 = jnp.concatenate([gyro[1:], jnp.zeros_like(gyro[:1])], axis=0)
    expected_tperp = jnp.einsum(
        "lk,lk->k",
        ell * jm1 + 2.0 * ell * gyro + (ell + 1.0) * jp1,
        coeffs[:, 0],
    )
    expected_qperp = jnp.einsum(
        "lk,lk->k",
        ell * jm1 + 2.0 * ell * gyro + (ell + 1.0) * jp1,
        coeffs[:, 1],
    )
    np.testing.assert_allclose(tperp, expected_tperp)
    np.testing.assert_allclose(qperp, expected_qperp)


def test_hypercollision_closure_hook_damps_only_selected_moments():
    coeffs = jnp.ones((4, 6, 2), dtype=jnp.float64)
    rates = hypercollision_damping_rates(4, 6, hermite_nu=0.1)
    rhs = apply_hypercollision(coeffs, rates)

    np.testing.assert_allclose(rates[:, :3], 0.0)
    assert jnp.all(rates[:, 3:] > 0.0)
    np.testing.assert_allclose(rhs, -rates[:, :, None] * coeffs)


def test_gx_linked_kz_wavenumbers_follow_cuda_ordering_and_dealiasing():
    kz = gx_linked_kz_wavenumbers(4, n_links=2, z_periods=3.0)
    expected = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, -3.0, -2.0, -1.0]) / 6.0
    np.testing.assert_allclose(kz, expected, rtol=0.0, atol=0.0)

    dealias = gx_linked_kz_wavenumbers(4, n_links=2, z_periods=3.0, dealias=True)
    expected_dealias = jnp.asarray([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, -2.0, -1.0]) / 6.0
    np.testing.assert_allclose(dealias, expected_dealias, rtol=0.0, atol=0.0)


def test_linked_abs_kz_operator_matches_discrete_fourier_mode():
    n_total = 12
    mode = 3
    z_periods = 4.0
    grid = jnp.arange(n_total, dtype=jnp.float64)
    values = jnp.exp(2j * jnp.pi * mode * grid / n_total)

    out = apply_linked_abs_kz(values, z_periods=z_periods)

    np.testing.assert_allclose(out, abs(mode / z_periods) * values, rtol=3e-14, atol=3e-14)


def test_linked_grad_z_operator_matches_discrete_fourier_mode():
    n_total = 12
    mode = 3
    z_periods = 4.0
    grid = jnp.arange(n_total, dtype=jnp.float64)
    values = jnp.exp(2j * jnp.pi * mode * grid / n_total)

    out = apply_linked_grad_z(values, z_periods=z_periods)

    np.testing.assert_allclose(out, 1j * mode / z_periods * values, rtol=3e-14, atol=3e-14)


def test_gx_kz_hypercollision_prefactor_and_rates_match_source_formula():
    prefactor = gx_kz_hypercollision_prefactor(
        6,
        nu_hyper_m=0.7,
        p_hyper_m=2,
        vt=1.5,
        gradpar_abs=0.25,
    )
    expected_prefactor = 0.7 * 2.5 / (5.0**2.5) * 2.3 * 1.5 * 0.25
    np.testing.assert_allclose(prefactor, expected_prefactor, rtol=2e-15, atol=2e-15)

    rates = gx_kz_hypercollision_hermite_rates(
        6,
        nu_hyper_m=0.7,
        p_hyper_m=2,
        vt=1.5,
        gradpar_abs=0.25,
    )
    expected = expected_prefactor * jnp.arange(6, dtype=jnp.float64) ** 2
    expected = expected.at[:3].set(0.0)
    np.testing.assert_allclose(rates, expected, rtol=2e-15, atol=2e-15)


def test_gx_kz_hypercollision_damps_only_high_hermite_modes_then_abs_kz():
    n_total = 8
    mode = 2
    grid = jnp.arange(n_total, dtype=jnp.float64)
    wave = jnp.exp(2j * jnp.pi * mode * grid / n_total)
    coeffs = jnp.zeros((2, 6, n_total), dtype=jnp.complex128)
    coeffs = coeffs.at[:, :, :].set(wave[None, None, :])

    rhs = apply_gx_kz_hypercollision(
        coeffs,
        nu_hyper_m=0.7,
        p_hyper_m=2,
        vt=1.5,
        gradpar_abs=0.25,
        n_links=2,
        z_periods=3.0,
    )
    rates = gx_kz_hypercollision_hermite_rates(
        6,
        nu_hyper_m=0.7,
        p_hyper_m=2,
        vt=1.5,
        gradpar_abs=0.25,
    )
    expected = -rates[None, :, None] * abs(mode / (3.0 * 2.0)) * coeffs

    np.testing.assert_allclose(rhs, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(rhs[:, :3], 0.0, rtol=0.0, atol=3e-14)

    constant = jnp.ones_like(coeffs)
    constant_rhs = apply_gx_kz_hypercollision(
        constant,
        nu_hyper_m=0.7,
        p_hyper_m=2,
        vt=1.5,
        gradpar_abs=0.25,
        n_links=2,
        z_periods=3.0,
    )
    np.testing.assert_allclose(constant_rhs, 0.0, rtol=0.0, atol=3e-14)


def test_gx_kz_hypercollision_is_jittable_and_differentiable():
    n_total = 6
    grid = jnp.arange(n_total, dtype=jnp.float64)
    wave = jnp.cos(2.0 * jnp.pi * grid / n_total)
    base = jnp.zeros((1, 5, n_total), dtype=jnp.float64)
    base = base.at[0, 3].set(wave)
    base = base.at[0, 4].set(0.25 * wave)

    @jax.jit
    def objective(scale):
        rhs = apply_gx_kz_hypercollision(
            scale * base,
            nu_hyper_m=0.4,
            p_hyper_m=2,
            vt=1.2,
            gradpar_abs=0.7,
        )
        return jnp.real(jnp.sum(jnp.abs(rhs) ** 2))

    value = objective(1.3)
    grad = jax.grad(objective)(1.3)

    assert jnp.isfinite(value)
    np.testing.assert_allclose(grad, 2.0 * value / 1.3, rtol=4e-14, atol=4e-14)


def test_gx_moment_field_solve_and_drive_projection_are_consistent():
    basis = build_velocity_basis(VelocityBasisSpec(n_hermite=5, n_laguerre=4))
    params = GxMomentRHSParams(density_gradient=0.8, temperature_gradient=2.49)
    ky = jnp.asarray([0.3, 0.5], dtype=jnp.float64)
    z = jnp.linspace(-1.0, 1.0, 6)
    b = 0.5 * ky[:, None] ** 2 * jnp.ones((ky.shape[0], z.shape[0]))
    phi_target = jnp.stack(
        (
            jnp.cos(jnp.pi * z),
            jnp.sin(jnp.pi * (z + 0.25)),
        )
    ).astype(jnp.complex128)
    gyro = gyroaverage_laguerre_coefficients(b, basis.n_laguerre)
    gamma = truncated_gamma0_from_laguerre(b, basis.n_laguerre)
    density = (params.tau + 1.0 - gamma) * phi_target
    coeffs = jnp.zeros(
        (basis.n_laguerre, basis.n_hermite, ky.shape[0], z.shape[0]),
        dtype=jnp.complex128,
    )
    coeffs = coeffs.at[:, 0].set(gyro * density / gamma)

    solved = gx_moment_adiabatic_phi(coeffs, b, params)
    drive = gx_moment_itg_drive_source(solved, b, ky, basis, params)

    np.testing.assert_allclose(solved, phi_target, rtol=3e-13, atol=3e-13)
    assert drive.shape == coeffs.shape
    assert jnp.max(jnp.abs(drive[:, 0])) > 0.0
    assert jnp.max(jnp.abs(drive[:, 2])) > 0.0
    np.testing.assert_allclose(drive[:, 1], 0.0, rtol=0.0, atol=0.0)


def test_gx_moment_rhs_hypercollision_only_preserves_low_hermite_modes():
    basis = build_velocity_basis(VelocityBasisSpec(n_hermite=6, n_laguerre=3))
    z = jnp.linspace(-1.0, 1.0, 8, endpoint=False)
    ky = jnp.asarray([0.4], dtype=jnp.float64)
    coeffs = jnp.ones((basis.n_laguerre, basis.n_hermite, 1, z.shape[0]), dtype=jnp.complex128)
    params = GxMomentRHSParams(
        density_gradient=0.0,
        temperature_gradient=0.0,
        streaming_scale=0.0,
        drive_scale=0.0,
        include_curvature_drift=False,
        include_hypercollision=True,
        p_hyper_m=2,
        z_periods=2.0,
    )

    rhs = gx_moment_linear_rhs(coeffs, basis, ky, z, b=0.5 * ky[:, None] ** 2, params=params)

    np.testing.assert_allclose(rhs[:, :3], 0.0, rtol=0.0, atol=3e-14)
    np.testing.assert_allclose(rhs[:, 3:], 0.0, rtol=0.0, atol=3e-14)

    wave = jnp.exp(2j * jnp.pi * jnp.arange(z.shape[0]) / z.shape[0])
    coeffs = coeffs.at[:, 3:].set(wave[None, None, None, :])
    rhs = gx_moment_linear_rhs(coeffs, basis, ky, z, b=0.5 * ky[:, None] ** 2, params=params)

    np.testing.assert_allclose(rhs[:, :3], 0.0, rtol=0.0, atol=3e-14)
    assert jnp.max(jnp.abs(rhs[:, 3:])) > 0.0


def test_gx_moment_rhs_is_jittable_and_differentiable():
    basis = build_velocity_basis(VelocityBasisSpec(n_hermite=5, n_laguerre=4))
    z = jnp.linspace(-1.0, 1.0, 8, endpoint=False)
    ky = jnp.asarray([0.3, 0.5], dtype=jnp.float64)
    b = 0.5 * ky[:, None] ** 2 * jnp.ones((ky.shape[0], z.shape[0]))
    base = jnp.zeros((basis.n_laguerre, basis.n_hermite, ky.shape[0], z.shape[0]))
    base = base.at[0, 0].set(jnp.cos(jnp.pi * z)[None, :])
    params = GxMomentRHSParams(z_periods=2.0)

    @jax.jit
    def objective(scale):
        rhs = gx_moment_linear_rhs(scale * base, basis, ky, z, b, params)
        return jnp.real(jnp.sum(jnp.abs(rhs) ** 2))

    value = objective(0.7)
    grad = jax.grad(objective)(0.7)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_free_energy_spectrum_and_jit_gradient_smoke():
    basis = _basis()
    base = jnp.arange(basis.n_laguerre * basis.n_hermite, dtype=jnp.float64)
    base = base.reshape(basis.n_laguerre, basis.n_hermite) / 20.0

    @jax.jit
    def objective(scale):
        coeffs = scale * base
        values = spectral_to_velocity_grid(coeffs, basis)
        rebuilt = velocity_grid_to_spectral(values, basis)
        return jnp.sum(free_energy_spectrum(rebuilt, axis=None))

    np.testing.assert_allclose(objective(1.3), jnp.sum((1.3 * base) ** 2), rtol=2e-13)
    np.testing.assert_allclose(
        jax.grad(objective)(1.3),
        2.0 * 1.3 * jnp.sum(base**2),
        rtol=2e-13,
    )
