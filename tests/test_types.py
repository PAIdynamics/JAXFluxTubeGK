import jax
import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    FourierGridSpec,
    GeometryScalarParams,
    SolverControls,
    SpeciesParams,
    VelocityGridSpec,
    build_fourier_grid,
    build_velocity_grid,
)


def test_species_params_is_pytree_and_jittable():
    species = SpeciesParams(
        charge=1.0,
        mass=2.0,
        density=0.7,
        temperature=1.5,
        density_gradient=3.0,
        temperature_gradient=4.0,
        kinetic=True,
    )

    leaves, treedef = jax.tree_util.tree_flatten(species)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt == species
    assert len(leaves) == 6

    @jax.jit
    def pressure_like(params):
        return params.density * params.temperature

    np.testing.assert_allclose(pressure_like(species), 1.05)


def test_solver_controls_are_static_pytree_metadata():
    controls = SolverControls(dtype="float64", derivative_backend="chebyshev")
    leaves, treedef = jax.tree_util.tree_flatten(controls)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert leaves == []
    assert rebuilt == controls


def test_geometry_scalar_params_are_differentiable_leaves():
    params = GeometryScalarParams(q=1.3, shat=0.8, eps=0.2, iota=0.9)

    @jax.grad
    def objective(geom):
        return geom.q * geom.shat + geom.eps * geom.iota

    grad_params = objective(params)

    np.testing.assert_allclose(grad_params.q, 0.8)
    np.testing.assert_allclose(grad_params.shat, 1.3)
    np.testing.assert_allclose(grad_params.eps, 0.9)
    np.testing.assert_allclose(grad_params.iota, 0.2)


def test_grid_objects_keep_arrays_as_leaves():
    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=5, n_mu=4, vpar_max=2.0, mu_max=3.0)
    )
    leaves = jax.tree_util.tree_leaves(velocity)

    assert len(leaves) == 10
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_fourier_grid_static_indices_survive_tree_round_trip():
    grid = build_fourier_grid(FourierGridSpec(n_kx=5, n_ky=3, kx_max=1.0, ky_max=0.6))
    leaves, treedef = jax.tree_util.tree_flatten(grid)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.ixzero == 2
    assert rebuilt.iyzero == 0
    np.testing.assert_allclose(jnp.asarray(rebuilt.kx), jnp.asarray(grid.kx))
