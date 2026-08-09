"""Print a reduced single-surface optimization loop iteration by iteration.

Run from the repository root:

    uv run --extra dev python examples/optimization_loop.py --iterations 5
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    OptimizationKnobs,
    ParallelGridSpec,
    SingleSurfaceOptimizationConfig,
    VelocityGridSpec,
    build_fourier_grid,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    single_surface_objective,
)


def main() -> None:
    args = _parse_args()
    velocity, parallel, fourier, connectivity, initial_state = _build_problem()
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    config = SingleSurfaceOptimizationConfig(
        geometry_model="circular",
        dt=args.dt,
        n_steps=args.steps,
        selected_ky=args.ky_index,
        objective_kind="selected_growth",
        store_history=False,
    )
    knobs = OptimizationKnobs(
        density=0.9,
        temperature=1.2,
        density_gradient=0.8,
        temperature_gradient=2.1,
        q=1.25,
        shat=0.45,
        eps=0.17,
        rho=0.45,
        alpha=0.2,
        beta=0.03,
        pressure_gradient=0.04,
        equilibrium_coefficients=(0.04, -0.015),
    )

    def objective(local_knobs):
        result = single_surface_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
        )
        return result.scalar_objective, result

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    print("# reduced fixed-topology jax_fluxtube_gk optimization loop")
    print(f"# iterations={args.iterations} learning_rate={args.learning_rate:g}")
    print(
        "iter objective selected_growth max_growth q shat "
        "R_over_L_T R_over_L_n coeff0 coeff1"
    )
    for iteration in range(args.iterations):
        (value, result), gradient = value_and_grad(knobs)
        print(_format_iteration(iteration, value, result, knobs))
        knobs = _gradient_step(knobs, gradient, args.learning_rate)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--ky-index", type=int, default=1)
    return parser.parse_args()


def _build_problem():
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.5, mu_max=1.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=5, z_min=-0.5 + 0.5 / 5, z_max=0.5 + 0.5 / 5)
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.5, ky_values=(0.0, 0.35), ikxspace=2)
    )
    connectivity = build_mode_connectivity(fourier)
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    initial_state = 0.01 * (jnp.cos(index / 6.0) + 1j * jnp.sin(index / 8.0))
    return velocity, parallel, fourier, connectivity, initial_state


def _gradient_step(knobs: OptimizationKnobs, gradient: OptimizationKnobs, learning_rate: float):
    updated = jax.tree_util.tree_map(
        lambda parameter, grad: parameter - learning_rate * grad,
        knobs,
        gradient,
    )
    return replace(
        updated,
        density=jnp.maximum(updated.density, 1.0e-3),
        temperature=jnp.maximum(updated.temperature, 1.0e-3),
        eps=jnp.maximum(updated.eps, 1.0e-3),
        q=jnp.maximum(updated.q, 1.0e-3),
    )


def _format_iteration(iteration, value, result, knobs) -> str:
    coeffs = jnp.pad(knobs.equilibrium_coefficients, (0, max(0, 2 - knobs.equilibrium_coefficients.shape[0])))
    return (
        f"{iteration:04d} "
        f"{float(value): .8e} "
        f"{float(result.values.selected_growth_rate): .8e} "
        f"{float(result.values.max_growth_rate): .8e} "
        f"{float(knobs.q): .8e} "
        f"{float(knobs.shat): .8e} "
        f"{float(knobs.temperature_gradient): .8e} "
        f"{float(knobs.density_gradient): .8e} "
        f"{float(coeffs[0]): .8e} "
        f"{float(coeffs[1]): .8e}"
    )


if __name__ == "__main__":
    main()
