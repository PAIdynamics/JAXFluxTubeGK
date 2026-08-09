from dataclasses import replace
from importlib import import_module
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fluxtube_gk import (
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    build_circular_geometry,
    build_fourier_grid,
    build_parallel_grid,
    build_s_alpha_geometry,
    k_perp_squared,
)


def _gyaradax_geometry_module(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return import_module("gyaradax.geometry.geom")


def _gkw_cell_centered_parallel_grid(n_z: int, nperiod: int = 1):
    sgrmax = nperiod - 0.5
    lower = -sgrmax + sgrmax / n_z
    upper = lower + 2.0 * sgrmax
    return build_parallel_grid(
        ParallelGridSpec(n_z=n_z, z_min=lower, z_max=upper, topology="periodic")
    )


def test_s_alpha_geometry_shapes_and_finite_values():
    parallel = _gkw_cell_centered_parallel_grid(16)
    params = GeometryScalarParams(q=1.4, shat=0.8, eps=0.18)
    geometry = build_s_alpha_geometry(parallel, params)

    assert geometry.model == "s-alpha"
    assert geometry.B.shape == (16,)
    assert geometry.exb_tensor.shape == (16, 3, 3)
    assert geometry.magnetic_drift_tensor.shape == (16, 3)
    assert geometry.g_xx.shape == geometry.g_xy.shape == geometry.g_yy.shape == (16,)
    assert jnp.all(jnp.isfinite(geometry.B))
    assert jnp.all(geometry.B > 0.0)
    np.testing.assert_allclose(geometry.w_z, parallel.w_z)


def test_circular_metric_is_symmetric_positive_for_perpendicular_block():
    parallel = _gkw_cell_centered_parallel_grid(24)
    params = GeometryScalarParams(q=1.7, shat=0.6, eps=0.16)
    geometry = build_circular_geometry(parallel, params)

    determinant = geometry.g_xx * geometry.g_yy - geometry.g_xy**2

    assert geometry.model == "circular"
    assert jnp.all(geometry.g_xx > 0.0)
    assert jnp.all(determinant > 0.0)
    np.testing.assert_allclose(
        geometry.exb_tensor + jnp.swapaxes(geometry.exb_tensor, 1, 2),
        0.0,
        atol=1e-13,
    )


def test_k_perp_squared_shape_and_nonnegative_metric_response():
    parallel = _gkw_cell_centered_parallel_grid(18)
    params = GeometryScalarParams(q=1.5, shat=0.7, eps=0.2)
    geometry = build_s_alpha_geometry(parallel, params)
    fourier = build_fourier_grid(FourierGridSpec(n_kx=5, n_ky=4, kx_max=1.0, ky_max=0.8))

    kperp2 = k_perp_squared(geometry, fourier)

    assert kperp2.shape == (18, 5, 4)
    assert jnp.min(kperp2) >= -1e-12
    np.testing.assert_allclose(kperp2[:, fourier.ixzero, fourier.iyzero], 0.0, atol=1e-13)


def test_circular_geometry_is_jittable_and_differentiable():
    parallel = _gkw_cell_centered_parallel_grid(12)
    params = GeometryScalarParams(q=1.6, shat=0.5, eps=0.17)

    @jax.jit
    def geometry_objective(geom_params):
        geometry = build_circular_geometry(parallel, geom_params)
        return jnp.sum(geometry.B + 0.05 * geometry.g_yy + 0.01 * geometry.D_y)

    grad_params = jax.grad(geometry_objective)(params)

    assert jnp.isfinite(geometry_objective(params))
    assert jnp.isfinite(grad_params.q)
    assert jnp.isfinite(grad_params.shat)
    assert jnp.isfinite(grad_params.eps)

    step = 1e-5
    for name in ("q", "shat", "eps"):
        plus = replace(params, **{name: getattr(params, name) + step})
        minus = replace(params, **{name: getattr(params, name) - step})
        finite_difference = (geometry_objective(plus) - geometry_objective(minus)) / (2.0 * step)
        np.testing.assert_allclose(
            getattr(grad_params, name),
            finite_difference,
            rtol=5e-4,
            atol=5e-5,
        )


@pytest.mark.external
def test_s_alpha_geometry_matches_gyaradax_reference(gyaradax_root: Path):
    gyaradax_geometry = _gyaradax_geometry_module(gyaradax_root)
    n_z = 10
    parallel = _gkw_cell_centered_parallel_grid(n_z)
    params = GeometryScalarParams(q=1.45, shat=0.65, eps=0.19)

    geometry = build_s_alpha_geometry(parallel, params)
    reference = gyaradax_geometry.compute_geometry(
        q=params.q,
        shat=params.shat,
        eps=params.eps,
        ns=n_z,
        nkx=5,
        nky=3,
        nvpar=4,
        nmu=3,
        geom_type="s-alpha",
    )

    np.testing.assert_allclose(geometry.B, reference["bn"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(geometry.F, reference["ffun"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(geometry.G, reference["gfun"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(geometry.E_y, reference["efun"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        jnp.stack([geometry.g_yy, geometry.g_xy, geometry.g_xx], axis=-1),
        reference["little_g"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.stack([geometry.D_x, geometry.D_y], axis=-1),
        reference["dfun"][:, :2],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.external
def test_circular_geometry_matches_gyaradax_reference(gyaradax_root: Path):
    gyaradax_geometry = _gyaradax_geometry_module(gyaradax_root)
    n_z = 10
    parallel = _gkw_cell_centered_parallel_grid(n_z)
    params = GeometryScalarParams(q=1.45, shat=0.65, eps=0.19)

    geometry = build_circular_geometry(parallel, params)
    reference = gyaradax_geometry.compute_geometry(
        q=params.q,
        shat=params.shat,
        eps=params.eps,
        ns=n_z,
        nkx=5,
        nky=3,
        nvpar=4,
        nmu=3,
        geom_type="circ",
    )

    np.testing.assert_allclose(geometry.B, reference["bn"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(geometry.F, reference["ffun"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(geometry.G, reference["gfun"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(geometry.E_y, reference["efun"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        jnp.stack([geometry.g_yy, geometry.g_xy, geometry.g_xx], axis=-1),
        reference["little_g"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.stack([geometry.D_x, geometry.D_y], axis=-1),
        reference["dfun"][:, :2],
        rtol=1e-12,
        atol=1e-12,
    )
