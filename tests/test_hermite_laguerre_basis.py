import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from stellarator_gk import (
    VelocityBasisKind,
    VelocityBasisSpec,
    apply_hypercollision,
    build_velocity_basis,
    density_moment,
    fluid_moments,
    free_energy_spectrum,
    gamma0,
    gamma0_limit_error,
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
