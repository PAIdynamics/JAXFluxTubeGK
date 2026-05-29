from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from stellarator_gk import (
    DESC_GEOMETRY_COMPUTE_KEYS,
    build_boozer_parallel_grid,
    build_desc_geometry_from_arrays,
    build_desc_geometry_from_equilibrium,
    build_desc_geometry_from_path,
    desc_geometry_arrays_from_data,
    desc_geometry_arrays_from_equilibrium,
    desc_geometry_arrays_from_path,
    load_desc_equilibrium,
)


class _FakeDescGrid:
    def __init__(self, nodes):
        self.nodes = nodes


class _FakeEquilibrium:
    def __init__(self):
        self.compute_calls = []

    def compute(self, names, grid):
        self.compute_calls.append(tuple(names))
        return _fake_desc_data(grid.nodes[:, 2])


class _FakeLoader:
    def __init__(self, loaded):
        self.loaded = loaded
        self.calls = []

    def __call__(self, path, *, file_format=None):
        self.calls.append((path, file_format))
        return self.loaded


def _fake_get_rtz_grid(eq, rho, alpha, zeta, *, coordinates, iota):
    assert coordinates == "raz"
    np.testing.assert_allclose(iota, 0.72)
    zeta = np.asarray(zeta)
    theta = alpha + iota * zeta
    nodes = np.column_stack([np.full_like(zeta, rho), theta, zeta])
    return _FakeDescGrid(nodes)


def _fake_desc_data(zeta):
    zeta = jnp.asarray(zeta)
    B = 1.3 + 0.05 * jnp.cos(zeta)
    B_vector = jnp.column_stack(
        [
            0.1 * jnp.sin(zeta),
            0.2 * jnp.ones_like(zeta),
            B,
        ]
    )
    grad_B = jnp.column_stack(
        [
            0.3 + 0.02 * jnp.cos(zeta),
            0.4 + 0.01 * jnp.sin(zeta),
            0.2 * jnp.ones_like(zeta),
        ]
    )
    grad_psi = jnp.column_stack(
        [
            0.7 + 0.02 * jnp.sin(zeta),
            0.1 * jnp.ones_like(zeta),
            0.05 * jnp.cos(zeta),
        ]
    )
    grad_alpha = jnp.column_stack(
        [
            0.2 * jnp.ones_like(zeta),
            0.6 + 0.03 * jnp.cos(zeta),
            0.04 * jnp.sin(zeta),
        ]
    )
    kappa = jnp.column_stack(
        [
            0.02 * jnp.cos(zeta),
            0.03 * jnp.sin(zeta),
            0.05 * jnp.ones_like(zeta),
        ]
    )
    return {
        "|B|": B,
        "B": B_vector,
        "B^zeta": 0.9 * B,
        "grad(|B|)": grad_B,
        "grad(psi)": grad_psi,
        "grad(alpha)": grad_alpha,
        "b": B_vector / B[:, None],
        "kappa": kappa,
        "theta_PEST": 0.1 + 0.72 * zeta,
        "phi": zeta,
        "alpha": 0.1 * jnp.ones_like(zeta),
    }


def test_desc_geometry_arrays_from_data_maps_required_quantities():
    zeta = jnp.linspace(-1.0, 1.0, 7)
    data = _fake_desc_data(zeta)
    arrays = desc_geometry_arrays_from_data(data, zeta=zeta, rho=0.5, alpha=0.1)
    B = data["|B|"]
    B_cross_gradB = jnp.cross(data["B"], data["grad(|B|)"], axis=-1)
    b_cross_kappa = jnp.cross(data["b"], data["kappa"], axis=-1)

    def dot(left, right):
        return jnp.sum(left * right, axis=-1)

    assert set(arrays) == {
        "theta",
        "phi",
        "rho",
        "alpha",
        "B",
        "b_dot_grad_z",
        "grad_psi_sq",
        "grad_alpha_sq",
        "grad_psi_dot_grad_alpha",
        "B_cross_gradB_dot_grad_psi",
        "B_cross_gradB_dot_grad_alpha",
        "b_cross_kappa_dot_grad_psi",
        "b_cross_kappa_dot_grad_alpha",
    }
    np.testing.assert_allclose(arrays["B"], B)
    np.testing.assert_allclose(arrays["b_dot_grad_z"], 0.9)
    np.testing.assert_allclose(arrays["grad_psi_sq"], dot(data["grad(psi)"], data["grad(psi)"]))
    np.testing.assert_allclose(
        arrays["grad_psi_dot_grad_alpha"],
        dot(data["grad(psi)"], data["grad(alpha)"]),
    )
    np.testing.assert_allclose(
        arrays["B_cross_gradB_dot_grad_alpha"],
        dot(B_cross_gradB, data["grad(alpha)"]),
    )
    np.testing.assert_allclose(
        arrays["b_cross_kappa_dot_grad_psi"],
        dot(b_cross_kappa, data["grad(psi)"]),
    )

    bad = dict(data)
    bad.pop("kappa")
    with pytest.raises(KeyError, match="kappa"):
        desc_geometry_arrays_from_data(bad, zeta=zeta, rho=0.5, alpha=0.1)


def test_desc_geometry_arrays_from_equilibrium_uses_field_line_grid():
    parallel = build_boozer_parallel_grid(n_z=9, n_turns=1)
    eq = _FakeEquilibrium()
    arrays = desc_geometry_arrays_from_equilibrium(
        eq,
        parallel,
        rho=0.5,
        alpha=0.1,
        iota=0.72,
        get_rtz_grid=_fake_get_rtz_grid,
    )

    assert eq.compute_calls == [tuple(DESC_GEOMETRY_COMPUTE_KEYS)]
    assert arrays["B"].shape == parallel.z.shape
    np.testing.assert_allclose(arrays["rho"], 0.5)
    assert jnp.all(jnp.isfinite(arrays["grad_alpha_sq"]))


def test_build_desc_geometry_from_equilibrium_returns_internal_geometry():
    parallel = build_boozer_parallel_grid(n_z=9, n_turns=1)
    geometry = build_desc_geometry_from_equilibrium(
        _FakeEquilibrium(),
        parallel,
        rho=0.5,
        alpha=0.1,
        iota=0.72,
        get_rtz_grid=_fake_get_rtz_grid,
    )

    assert geometry.source == "desc"
    assert geometry.B.shape == parallel.z.shape
    assert geometry.F.shape == parallel.z.shape
    assert jnp.all(jnp.isfinite(geometry.G))
    assert jnp.all(geometry.g_xx > 0.0)


def test_load_desc_equilibrium_selects_equilibrium_from_path_like_family():
    eq0 = _FakeEquilibrium()
    eq1 = _FakeEquilibrium()
    loader = _FakeLoader([eq0, eq1])

    loaded = load_desc_equilibrium("fake_family.h5", file_format="hdf5", index=1, loader=loader)

    assert loaded is eq1
    assert loader.calls == [("fake_family.h5", "hdf5")]


def test_desc_geometry_arrays_from_path_uses_loader_and_field_line_grid():
    parallel = build_boozer_parallel_grid(n_z=9, n_turns=1)
    eq = _FakeEquilibrium()
    loader = _FakeLoader([eq])

    arrays = desc_geometry_arrays_from_path(
        "fake_family.h5",
        parallel,
        rho=0.5,
        alpha=0.1,
        iota=0.72,
        file_format="hdf5",
        loader=loader,
        get_rtz_grid=_fake_get_rtz_grid,
    )

    assert loader.calls == [("fake_family.h5", "hdf5")]
    assert eq.compute_calls == [tuple(DESC_GEOMETRY_COMPUTE_KEYS)]
    assert arrays["B"].shape == parallel.z.shape
    assert jnp.all(jnp.isfinite(arrays["grad_psi_sq"]))


def test_build_desc_geometry_from_path_returns_internal_geometry():
    parallel = build_boozer_parallel_grid(n_z=9, n_turns=1)
    geometry = build_desc_geometry_from_path(
        "fake_family.h5",
        parallel,
        rho=0.5,
        alpha=0.1,
        iota=0.72,
        loader=_FakeLoader([_FakeEquilibrium()]),
        get_rtz_grid=_fake_get_rtz_grid,
    )

    assert geometry.source == "desc"
    assert geometry.B.shape == parallel.z.shape
    assert jnp.all(jnp.isfinite(geometry.G))


def test_extracted_desc_fixture_loads_through_geometry_contract():
    fixture = Path(__file__).resolve().parents[1] / "fixtures/desc_geometry_dshape_rho05_alpha0.npz"
    data = np.load(fixture)
    parallel = build_boozer_parallel_grid(n_z=data["z"].shape[0], n_turns=1)
    np.testing.assert_allclose(parallel.z, data["z"], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(parallel.w_z, data["w_z"], rtol=0.0, atol=1e-14)

    geometry = build_desc_geometry_from_arrays(
        parallel,
        theta=data["theta"],
        phi=data["phi"],
        rho=data["rho"],
        alpha=data["alpha"],
        B=data["B"],
        b_dot_grad_z=data["b_dot_grad_z"],
        grad_psi_sq=data["grad_psi_sq"],
        grad_alpha_sq=data["grad_alpha_sq"],
        grad_psi_dot_grad_alpha=data["grad_psi_dot_grad_alpha"],
        B_cross_gradB_dot_grad_psi=data["B_cross_gradB_dot_grad_psi"],
        B_cross_gradB_dot_grad_alpha=data["B_cross_gradB_dot_grad_alpha"],
        b_cross_kappa_dot_grad_psi=data["b_cross_kappa_dot_grad_psi"],
        b_cross_kappa_dot_grad_alpha=data["b_cross_kappa_dot_grad_alpha"],
    )

    assert geometry.source == "desc"
    assert geometry.B.shape == parallel.z.shape
    assert jnp.all(jnp.isfinite(geometry.G))
    assert jnp.all(geometry.B > 0.0)
    assert jnp.all(geometry.g_xx > 0.0)
    assert jnp.all(geometry.g_yy > 0.0)
    np.testing.assert_allclose(np.mean(data["B"]), 0.21211534648269842, rtol=1e-13)
    np.testing.assert_allclose(np.mean(data["b_dot_grad_z"]), 0.27287213902435115, rtol=1e-13)
