import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    ParallelGridSpec,
    VelocityGridSpec,
    build_finite_difference_operators,
    build_parallel_grid,
    build_velocity_grid,
)


def test_centered_finite_difference_coefficients():
    ops = build_finite_difference_operators(n=6, spacing=0.5, periodic=False)

    expected_d1_row = np.array([0.0, 1.0 / 6.0, -4.0 / 3.0, 0.0, 4.0 / 3.0, -1.0 / 6.0])
    expected_d4_row = np.array([0.0, 16.0, -64.0, 96.0, -64.0, 16.0])

    np.testing.assert_allclose(ops.D1[3], expected_d1_row)
    np.testing.assert_allclose(ops.D4[3], expected_d4_row)


def test_periodic_finite_difference_wraps_boundary_coefficients():
    ops = build_finite_difference_operators(n=6, spacing=1.0, periodic=True)

    assert ops.D1[0, -2] != 0.0
    assert ops.D1[0, -1] != 0.0
    assert ops.D4[0, -2] != 0.0
    assert ops.D4[0, -1] != 0.0


def test_nonperiodic_finite_difference_zero_fills_outside_domain():
    ops = build_finite_difference_operators(n=6, spacing=1.0, periodic=False)

    assert ops.D1[0, -1] == 0.0
    assert ops.D1[0, -2] == 0.0
    assert ops.D1[0, 1] != 0.0
    assert ops.D1[0, 2] != 0.0


def test_finite_difference_shape_parity_with_spectral_operator():
    n = 8
    spectral = build_parallel_grid(
        ParallelGridSpec(n_z=n, z_min=-1.0, z_max=1.0, topology="open")
    )
    fallback = build_finite_difference_operators(n=n, spacing=2.0 / (n - 1), periodic=False)

    assert fallback.D1.shape == spectral.D_z.shape
    assert fallback.D4.shape == spectral.D_z.shape


def test_periodic_finite_difference_differentiates_smooth_periodic_data():
    n = 64
    length = 2.0 * np.pi
    spacing = length / n
    x = jnp.arange(n) * spacing
    f = jnp.sin(x)
    exact = jnp.cos(x)
    ops = build_finite_difference_operators(n=n, spacing=spacing, periodic=True)

    np.testing.assert_allclose(ops.D1 @ f, exact, rtol=2e-5, atol=2e-5)


def test_finite_difference_velocity_grid_matches_gkw_cell_centers_and_weights():
    grid = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=8,
            n_mu=4,
            vpar_max=3.0,
            mu_max=4.5,
            backend="finite_difference",
        )
    )

    np.testing.assert_allclose(grid.vpar[0], -3.0 + 3.0 / 8.0)
    np.testing.assert_allclose(grid.vpar[-1], 3.0 - 3.0 / 8.0)
    np.testing.assert_allclose(jnp.sum(grid.w_vpar), 6.0)
    np.testing.assert_allclose(jnp.sum(grid.w_mu), 2.0 * np.pi * 4.5)
    assert grid.D_vpar.shape == (8, 8)
    assert grid.D_mu.shape == (4, 4)
    assert grid.backend == "finite_difference"
