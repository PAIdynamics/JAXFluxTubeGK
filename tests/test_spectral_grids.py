import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    ParallelGridSpec,
    VelocityGridSpec,
    build_parallel_grid,
    build_velocity_grid,
    build_velocity_grid_from_nodes,
)


def test_chebyshev_velocity_nodes_weights_and_interval_scaling():
    grid = build_velocity_grid(VelocityGridSpec(n_vpar=8, n_mu=7, vpar_max=2.0, mu_max=3.0))

    np.testing.assert_allclose(grid.vpar[0], -2.0)
    np.testing.assert_allclose(grid.vpar[-1], 2.0)
    np.testing.assert_allclose(grid.mu[0], 0.0)
    np.testing.assert_allclose(grid.mu[-1], 3.0)
    np.testing.assert_allclose(jnp.sum(grid.w_vpar), 4.0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(jnp.sum(grid.w_mu), 3.0, rtol=1e-13, atol=1e-13)


def test_clenshaw_curtis_integrates_low_order_polynomials():
    grid = build_velocity_grid(VelocityGridSpec(n_vpar=9, n_mu=5, vpar_max=1.0, mu_max=2.0))
    x = grid.vpar
    mu = grid.mu

    np.testing.assert_allclose(jnp.sum(grid.w_vpar * x**2), 2.0 / 3.0, rtol=1e-12)
    np.testing.assert_allclose(jnp.sum(grid.w_vpar * x**3), 0.0, atol=1e-13)
    np.testing.assert_allclose(jnp.sum(grid.w_mu * mu**2), 8.0 / 3.0, rtol=1e-12)


def test_chebyshev_derivative_is_exact_for_polynomial_data():
    grid = build_velocity_grid(VelocityGridSpec(n_vpar=8, n_mu=6, vpar_max=1.5, mu_max=2.0))
    x = grid.vpar
    f = x**4 - 2.0 * x**2 + 0.5 * x
    exact = 4.0 * x**3 - 4.0 * x + 0.5

    np.testing.assert_allclose(grid.D_vpar @ f, exact, rtol=1e-10, atol=1e-10)


def test_open_parallel_grid_uses_chebyshev_derivative():
    grid = build_parallel_grid(
        ParallelGridSpec(n_z=8, z_min=-1.0, z_max=1.0, topology="open")
    )
    z = grid.z
    f = z**5 - z**3 + 2.0
    exact = 5.0 * z**4 - 3.0 * z**2

    assert grid.backend == "chebyshev"
    np.testing.assert_allclose(grid.D_z @ f, exact, rtol=1e-10, atol=1e-10)


def test_periodic_parallel_grid_uses_fourier_derivative():
    grid = build_parallel_grid(
        ParallelGridSpec(n_z=32, z_min=0.0, z_max=2.0 * np.pi, topology="periodic")
    )
    z = grid.z
    f = jnp.sin(3.0 * z) - 0.5 * jnp.cos(2.0 * z)
    exact = 3.0 * jnp.cos(3.0 * z) + jnp.sin(2.0 * z)

    assert grid.backend == "fourier"
    np.testing.assert_allclose(grid.D_z @ f, exact, rtol=1e-10, atol=1e-10)


def test_modal_transforms_reconstruct_values():
    grid = build_velocity_grid(VelocityGridSpec(n_vpar=7, n_mu=6, vpar_max=1.0, mu_max=1.0))
    values = grid.vpar**3 - grid.vpar
    coeffs = grid.vpar_modal_transform @ values

    np.testing.assert_allclose(grid.vpar_inverse_modal_transform @ coeffs, values, atol=1e-12)


def test_native_velocity_grid_accepts_arbitrary_monotone_quadrature():
    grid = build_velocity_grid_from_nodes(
        vpar=[-1.5, -0.2, 0.7, 2.0],
        mu=[0.04, 0.3, 1.2],
        w_vpar=[0.2, 0.7, 0.8, 0.3],
        w_mu=[0.1, 0.4, 0.2],
    )

    assert grid.backend == "native"
    np.testing.assert_allclose(grid.D_vpar @ grid.vpar**2, 2.0 * grid.vpar)
    np.testing.assert_allclose(grid.D_mu @ grid.mu**2, 2.0 * grid.mu)


def test_native_velocity_grid_rejects_nonmonotone_nodes():
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        build_velocity_grid_from_nodes(
            vpar=[-1.0, 0.0, 1.0],
            mu=[0.2, 0.1],
            w_vpar=[1.0, 1.0, 1.0],
            w_mu=[1.0, 1.0],
        )
