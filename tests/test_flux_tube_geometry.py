import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fluxtube_gk import (
    BoozerSurface,
    FieldLineSpec,
    FourierGridSpec,
    build_boozer_parallel_grid,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_physical_flux_tube_geometry_from_arrays,
    evaluate_boozer_magnetic_field,
    k_perp_squared,
    map_physical_to_internal_geometry,
    sample_boozer_field_line,
)


def _constant_physical_geometry(n_z=24):
    parallel = build_boozer_parallel_grid(n_z=n_z, n_turns=1)
    surface = BoozerSurface(iota=0.7, B0=1.0)
    field_line = sample_boozer_field_line(surface, FieldLineSpec(rho=0.4), parallel)
    ones = jnp.ones_like(field_line.z)
    zeros = jnp.zeros_like(field_line.z)
    physical = build_physical_flux_tube_geometry_from_arrays(
        field_line=field_line,
        B=ones,
        b_dot_grad_z=ones,
        grad_psi_sq=ones,
        grad_alpha_sq=ones,
        grad_psi_dot_grad_alpha=zeros,
        B_cross_gradB_dot_grad_psi=zeros,
        B_cross_gradB_dot_grad_alpha=zeros,
        b_cross_kappa_dot_grad_psi=zeros,
        b_cross_kappa_dot_grad_alpha=zeros,
    )
    return parallel, physical


def _desc_like_arrays(parallel, amplitude=0.08):
    z = parallel.z
    ones = jnp.ones_like(z)
    return {
        "theta": 0.2 + 0.7 * z,
        "phi": z,
        "rho": 0.45,
        "alpha": 0.2 * ones,
        "B": 1.0 + amplitude * jnp.cos(z),
        "b_dot_grad_z": 1.1 + 0.03 * jnp.sin(z),
        "grad_psi_sq": 1.2 + 0.04 * jnp.cos(z),
        "grad_alpha_sq": 1.5 + 0.03 * jnp.sin(2.0 * z),
        "grad_psi_dot_grad_alpha": 0.05 * jnp.sin(z),
        "B_cross_gradB_dot_grad_psi": 0.02 * jnp.cos(z),
        "B_cross_gradB_dot_grad_alpha": 0.03 * jnp.sin(z),
        "b_cross_kappa_dot_grad_psi": 0.04 * ones,
        "b_cross_kappa_dot_grad_alpha": 0.05 * jnp.cos(z),
    }


def test_boozer_parallel_grid_spans_multiple_turns():
    grid = build_boozer_parallel_grid(n_z=32, n_turns=3)

    np.testing.assert_allclose(jnp.sum(grid.w_z), 6.0 * np.pi, rtol=1e-13)
    assert grid.backend == "fourier"
    assert grid.topology == "periodic"


def test_sample_boozer_field_line_preserves_alpha_label():
    grid = build_boozer_parallel_grid(n_z=16, n_turns=2)
    surface = BoozerSurface(iota=0.43, B0=1.0)
    spec = FieldLineSpec(rho=0.25, alpha0=0.31)
    field_line = sample_boozer_field_line(surface, spec, grid)

    np.testing.assert_allclose(field_line.alpha, spec.alpha0, atol=1e-14)
    np.testing.assert_allclose(field_line.theta, spec.alpha0 + surface.iota * field_line.phi)
    np.testing.assert_allclose(field_line.rho, 0.25)


def test_evaluate_boozer_magnetic_field_fourier_surface():
    grid = build_boozer_parallel_grid(n_z=16, n_turns=1)
    surface = BoozerSurface(
        iota=0.5,
        B0=1.2,
        B_cos=(0.1,),
        B_sin=(0.05,),
        m_modes=(1,),
        n_modes=(0,),
    )
    field_line = sample_boozer_field_line(surface, FieldLineSpec(rho=0.5), grid)
    B = evaluate_boozer_magnetic_field(surface, field_line)
    expected = 1.2 + 0.1 * jnp.cos(field_line.theta) + 0.05 * jnp.sin(field_line.theta)

    np.testing.assert_allclose(B, expected, rtol=1e-13, atol=1e-13)


def test_precomputed_constant_geometry_maps_to_flat_circular_limit():
    parallel, physical = _constant_physical_geometry()
    geometry = map_physical_to_internal_geometry(physical, parallel)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=5, n_ky=3, kx_max=1.0, ky_max=0.6))
    kperp2 = k_perp_squared(geometry, fourier)
    expected = fourier.kx[None, :, None] ** 2 + fourier.ky[None, None, :] ** 2

    np.testing.assert_allclose(geometry.B, 1.0)
    np.testing.assert_allclose(geometry.F, 1.0)
    np.testing.assert_allclose(geometry.G, 0.0, atol=1e-13)
    np.testing.assert_allclose(geometry.D_x, 0.0)
    np.testing.assert_allclose(geometry.D_y, 0.0)
    np.testing.assert_allclose(kperp2, jnp.broadcast_to(expected, kperp2.shape), atol=1e-13)
    assert jnp.min(kperp2) >= -1e-13


def test_physical_to_internal_geometry_maps_drifts_and_mirror_term():
    parallel = build_boozer_parallel_grid(n_z=48, n_turns=1)
    surface = BoozerSurface(iota=0.8, B0=1.0)
    field_line = sample_boozer_field_line(surface, FieldLineSpec(rho=0.5), parallel)
    B = 1.0 + 0.1 * jnp.cos(field_line.z)
    ones = jnp.ones_like(B)
    physical = build_physical_flux_tube_geometry_from_arrays(
        field_line=field_line,
        B=B,
        b_dot_grad_z=2.0 * ones,
        grad_psi_sq=1.5 * ones,
        grad_alpha_sq=3.0 * ones,
        grad_psi_dot_grad_alpha=0.2 * ones,
        B_cross_gradB_dot_grad_psi=0.3 * ones,
        B_cross_gradB_dot_grad_alpha=0.4 * ones,
        b_cross_kappa_dot_grad_psi=0.5 * ones,
        b_cross_kappa_dot_grad_alpha=0.6 * ones,
        equilibrium_drive_scale=0.7 * ones,
    )
    geometry = map_physical_to_internal_geometry(physical, parallel)
    expected_G = -0.2 * jnp.sin(field_line.z) / B
    expected_D_x = (0.3 + B * 0.5) / B**2
    expected_D_y = (0.4 + B * 0.6) / B**2

    np.testing.assert_allclose(geometry.G, expected_G, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(geometry.D_x, expected_D_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(geometry.D_y, expected_D_y, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(geometry.D_x_gradB, 0.3 / B**2)
    np.testing.assert_allclose(geometry.D_y_gradB, 0.4 / B**2)
    np.testing.assert_allclose(geometry.D_x_curvature, 0.5 / B)
    np.testing.assert_allclose(geometry.D_y_curvature, 0.6 / B)
    np.testing.assert_allclose(geometry.E_y, 0.7, rtol=1e-13, atol=1e-13)


def test_desc_geometry_builder_maps_arrays_and_validates_shapes():
    parallel = build_boozer_parallel_grid(n_z=32, n_turns=1)
    arrays = _desc_like_arrays(parallel)
    geometry = build_desc_geometry_from_arrays(parallel, **arrays)
    B = arrays["B"]
    expected_G = arrays["b_dot_grad_z"] * (parallel.D_z @ B) / B
    expected_D_x = (
        arrays["B_cross_gradB_dot_grad_psi"] + B * arrays["b_cross_kappa_dot_grad_psi"]
    ) / B**2
    expected_D_y = (
        arrays["B_cross_gradB_dot_grad_alpha"] + B * arrays["b_cross_kappa_dot_grad_alpha"]
    ) / B**2

    assert geometry.source == "desc"
    np.testing.assert_allclose(geometry.rho, arrays["rho"])
    np.testing.assert_allclose(geometry.G, expected_G, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(geometry.D_x, expected_D_x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(geometry.D_y, expected_D_y, rtol=1e-13, atol=1e-13)

    bad_arrays = dict(arrays)
    bad_arrays["B"] = arrays["B"][:-1]
    with pytest.raises(ValueError, match="B must have shape"):
        build_desc_geometry_from_arrays(parallel, **bad_arrays)


def test_flux_tube_adapter_is_jittable_and_differentiable_through_arrays():
    parallel = build_boozer_parallel_grid(n_z=32, n_turns=1)
    surface = BoozerSurface(iota=0.9, B0=1.0)
    field_line = sample_boozer_field_line(surface, FieldLineSpec(rho=0.55), parallel)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.8, ky_max=0.4))

    @jax.jit
    def objective(amplitude):
        B = 1.0 + amplitude * jnp.cos(field_line.z)
        ones = jnp.ones_like(B)
        zeros = jnp.zeros_like(B)
        physical = build_physical_flux_tube_geometry_from_arrays(
            field_line=field_line,
            B=B,
            b_dot_grad_z=ones,
            grad_psi_sq=ones,
            grad_alpha_sq=(1.0 + amplitude) * ones,
            grad_psi_dot_grad_alpha=zeros,
            B_cross_gradB_dot_grad_psi=zeros,
            B_cross_gradB_dot_grad_alpha=zeros,
            b_cross_kappa_dot_grad_psi=zeros,
            b_cross_kappa_dot_grad_alpha=zeros,
        )
        geometry = map_physical_to_internal_geometry(physical, parallel)
        return jnp.sum(geometry.B**2) + 0.01 * jnp.sum(k_perp_squared(geometry, fourier))

    amplitude = 0.12
    grad_value = jax.grad(objective)(amplitude)
    step = 1e-5
    finite_difference = (objective(amplitude + step) - objective(amplitude - step)) / (
        2.0 * step
    )

    assert jnp.isfinite(objective(amplitude))
    assert jnp.isfinite(grad_value)
    np.testing.assert_allclose(grad_value, finite_difference, rtol=5e-5, atol=5e-5)


def test_desc_geometry_builder_is_differentiable_through_supplied_arrays():
    parallel = build_boozer_parallel_grid(n_z=32, n_turns=1)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.8, ky_max=0.4))

    @jax.jit
    def objective(amplitude):
        geometry = build_desc_geometry_from_arrays(
            parallel,
            **_desc_like_arrays(parallel, amplitude=amplitude),
        )
        return jnp.sum(geometry.B**2) + 0.01 * jnp.sum(k_perp_squared(geometry, fourier))

    amplitude = 0.09
    grad_value = jax.grad(objective)(amplitude)
    step = 1e-5
    finite_difference = (objective(amplitude + step) - objective(amplitude - step)) / (
        2.0 * step
    )

    assert jnp.isfinite(objective(amplitude))
    assert jnp.isfinite(grad_value)
    np.testing.assert_allclose(grad_value, finite_difference, rtol=5e-5, atol=5e-5)
