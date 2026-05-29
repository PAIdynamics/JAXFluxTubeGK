import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    ParallelGridSpec,
    build_finite_difference_operators,
    build_parallel_grid,
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

