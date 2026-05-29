import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import FourierGridSpec, build_fourier_grid, build_mode_connectivity


def test_centered_kx_and_nonnegative_ky_grid():
    grid = build_fourier_grid(FourierGridSpec(n_kx=5, n_ky=4, kx_max=2.0, ky_max=0.9))

    np.testing.assert_allclose(grid.kx, jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    np.testing.assert_allclose(grid.ky, jnp.array([0.0, 0.3, 0.6, 0.9]))
    np.testing.assert_allclose(grid.parseval, jnp.array([1.0, 2.0, 2.0, 2.0]))
    assert grid.ixzero == 2
    assert grid.iyzero == 0


def test_explicit_ky_values_are_used_exactly():
    grid = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=3, kx_max=1.0, ky_values=(0.0, 0.2, 0.7))
    )

    np.testing.assert_allclose(grid.ky, jnp.array([0.0, 0.2, 0.7]))


def test_single_mode_uses_gkw_nonzero_ky_convention():
    grid = build_fourier_grid(FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_max=0.35))

    np.testing.assert_allclose(grid.kx, jnp.array([0.0]))
    np.testing.assert_allclose(grid.ky, jnp.array([0.35]))
    np.testing.assert_allclose(grid.parseval, jnp.array([2.0]))
    assert grid.ixzero == 0
    assert grid.iyzero == -1


def test_zonal_only_mode_uses_explicit_zero_ky():
    grid = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=1, kx_max=1.0, ky_values=(0.0,))
    )

    np.testing.assert_allclose(grid.ky, jnp.array([0.0]))
    np.testing.assert_allclose(grid.parseval, jnp.array([1.0]))
    assert grid.iyzero == 0


def test_invalid_fourier_specs_raise():
    with pytest.raises(ValueError, match="odd"):
        FourierGridSpec(n_kx=4, n_ky=2, kx_max=1.0, ky_max=1.0)
    with pytest.raises(ValueError, match="ky_max"):
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=1.0, ky_values=(0.0, -1.0))


def test_mode_connectivity_zonal_identity_and_nonzonal_chains():
    grid = build_fourier_grid(FourierGridSpec(n_kx=5, n_ky=3, kx_max=2.0, ky_max=1.0))
    conn = build_mode_connectivity(grid, ikxspace=2, max_shift=2)

    np.testing.assert_array_equal(conn.ixplus[:, 0], np.arange(5))
    np.testing.assert_array_equal(conn.ixminus[:, 0], np.arange(5))
    assert conn.ixzero == 2
    assert conn.iyzero == 0

    np.testing.assert_array_equal(conn.ixplus[:, 1], np.array([2, 3, 4, -1, -1]))
    np.testing.assert_array_equal(conn.ixminus[:, 1], np.array([-1, -1, 0, 1, 2]))

    zero_shift = conn.max_shift
    plus_two = conn.max_shift + 2
    assert bool(conn.valid_shift[zero_shift, 4, 1])
    assert int(conn.kx_shift[zero_shift, 4, 1]) == 4
    assert bool(conn.valid_shift[plus_two, 0, 1])
    assert int(conn.kx_shift[plus_two, 0, 1]) == 4
    assert not bool(conn.valid_shift[plus_two, 2, 1])


def test_gkw_shear_spacing_option():
    grid = build_fourier_grid(
        FourierGridSpec(
            n_kx=5,
            n_ky=3,
            kx_max=99.0,
            ky_max=1.0,
            ikxspace=5,
            q=2.0,
            shat=0.5,
            eps=0.2,
            use_gkw_shear_spacing=True,
        )
    )
    expected_spacing = abs(2.0 * 0.5 * 0.5 / (0.2 * 5.0))

    np.testing.assert_allclose(grid.kx, expected_spacing * jnp.array([-2, -1, 0, 1, 2]))
