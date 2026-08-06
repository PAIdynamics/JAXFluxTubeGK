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
    build_exb_pseudospectral_precompute,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    estimate_nonlinear_exb_dt,
    estimate_nonlinear_residual_dt,
    exb_pseudospectral_bracket,
    linear_residual,
    integrate_nonlinear_adaptive,
    nonlinear_exb_term,
    nonlinear_residual,
)


def _grid():
    grid = build_fourier_grid(
        FourierGridSpec(n_kx=5, n_ky=3, kx_max=2.0, ky_values=(0.0, 1.0, 2.0))
    )
    return grid, build_exb_pseudospectral_precompute(grid)


def test_manufactured_cosine_bracket_has_expected_retained_coefficients():
    grid, precompute = _grid()
    a = jnp.zeros((5, 3), dtype=jnp.complex128)
    b = jnp.zeros_like(a)
    a = a.at[1, 0].set(0.5).at[3, 0].set(0.5)
    b = b.at[grid.ixzero, 1].set(0.5)

    bracket = exb_pseudospectral_bracket(a, b, precompute)
    expected = jnp.zeros_like(bracket).at[1, 1].set(0.25).at[3, 1].set(-0.25)

    np.testing.assert_allclose(bracket, expected, atol=2.0e-15, rtol=0.0)


def test_bracket_is_antisymmetric_and_constant_field_vanishes():
    _grid_value, precompute = _grid()
    key_a, key_b = jax.random.split(jax.random.key(3))
    a = jax.random.normal(key_a, (5, 3)) + 1j * jax.random.normal(key_b, (5, 3))
    # Enforce reality of the ky=0 line under kx -> -kx.
    a = a.at[:, 0].set(
        jnp.asarray([jnp.conj(a[4, 0]), jnp.conj(a[3, 0]), a[2, 0].real, a[3, 0], a[4, 0]])
    )
    b = jnp.roll(a, 1, axis=0)
    b = b.at[:, 0].set(
        jnp.asarray([jnp.conj(b[4, 0]), jnp.conj(b[3, 0]), b[2, 0].real, b[3, 0], b[4, 0]])
    )
    constant = jnp.zeros_like(a).at[2, 0].set(2.0)

    ab = jax.jit(exb_pseudospectral_bracket)(a, b, precompute)
    ba = exb_pseudospectral_bracket(b, a, precompute)

    np.testing.assert_allclose(ab, -ba, atol=2.0e-13, rtol=2.0e-13)
    np.testing.assert_allclose(
        exb_pseudospectral_bracket(constant, b, precompute), 0.0, atol=2.0e-14
    )


def test_nonlinear_cfl_scales_inversely_with_potential_amplitude():
    _grid_value, precompute = _grid()
    phi = jnp.zeros((5, 3), dtype=jnp.complex128).at[3, 1].set(0.2)

    dt = estimate_nonlinear_exb_dt(phi, precompute)
    doubled_dt = estimate_nonlinear_exb_dt(2.0 * phi, precompute)

    assert np.isfinite(float(dt))
    np.testing.assert_allclose(doubled_dt, 0.5 * dt, rtol=2.0e-13)


def test_nonlinear_precompute_rejects_linear_only_fourier_grids():
    no_zonal = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=1.0, ky_values=(0.3, 0.6))
    )
    with pytest.raises(ValueError, match="kx=0 and ky=0"):
        build_exb_pseudospectral_precompute(no_zonal)


def test_nonlinear_residual_adds_gyroaveraged_exb_term():
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=4, n_mu=3, vpar_max=3.0, mu_max=4.0)
    )
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=4, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.4, ky_values=(0.0, 0.3))
    )
    geometry = build_s_alpha_geometry(parallel, GeometryScalarParams())
    species = SpeciesParams(1.0, 1.0, 1.0, 1.0, 2.0, 3.0)
    linear_precompute = build_linear_residual_precompute(
        velocity, parallel, fourier, geometry, species
    )
    nonlinear_precompute = build_exb_pseudospectral_precompute(fourier)
    shape = (4, 3, 4, 3, 2)
    state = jax.random.normal(jax.random.key(11), shape).astype(jnp.complex128)
    phi = jax.random.normal(jax.random.key(12), (4, 3, 2)).astype(jnp.complex128)

    total = nonlinear_residual(
        state, linear_precompute, nonlinear_precompute, phi=phi
    )
    expected = linear_residual(
        state, precomputed=linear_precompute, phi=phi
    ) + nonlinear_exb_term(
        state, phi, linear_precompute.rhs, nonlinear_precompute
    )

    np.testing.assert_allclose(total, expected, atol=2.0e-12, rtol=2.0e-12)

    selected_dt = estimate_nonlinear_residual_dt(
        state, linear_precompute, nonlinear_precompute
    )
    assert np.isfinite(float(selected_dt))
    assert float(selected_dt) > 0.0

    zero_result = integrate_nonlinear_adaptive(
        jnp.zeros_like(state),
        1.0e-3,
        linear_precompute,
        nonlinear_precompute,
    )
    np.testing.assert_allclose(zero_result.state, 0.0, atol=0.0)
    np.testing.assert_allclose(zero_result.times[-1], 1.0e-3, atol=0.0)
