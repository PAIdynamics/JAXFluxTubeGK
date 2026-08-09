"""Print a reduced benchmark-target optimization loop on the DESC DSHAPE fixture.

Run from the repository root:

    uv run --extra dev python examples/desc_fixture_optimization_loop.py --iterations 5
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    BenchmarkTarget,
    FourierGridSpec,
    OptimizationKnobs,
    ParallelGridSpec,
    SingleSurfaceOptimizationConfig,
    VelocityGridSpec,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    single_surface_benchmark_objective,
)


def main() -> None:
    args = _parse_args()
    readiness = _load_readiness_status(args.readiness_gate)
    if args.require_production_ready and not readiness["passed"]:
        raise SystemExit(
            "production DESC optimization is blocked: "
            f"{readiness['status']} in {args.readiness_gate}"
        )
    fixture = np.load(args.fixture)
    parallel = _parallel_grid_from_fixture_z(fixture["z"])
    geometry = _geometry_from_fixture(fixture, parallel)
    velocity, fourier, connectivity, initial_state = _build_problem(parallel)
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    target = BenchmarkTarget(
        name="desc_dshape_selected_growth_target",
        quantity="selected_growth_rate",
        reference_value=args.target_growth,
        tolerance=args.target_tolerance,
        source=str(args.fixture),
    )
    config = SingleSurfaceOptimizationConfig(
        geometry_model="desc",
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
    )

    def objective(local_knobs):
        result = single_surface_benchmark_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            target,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
            geometry=geometry,
        )
        return result.scalar_objective, result

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))

    print("# reduced DESC DSHAPE benchmark-target optimization loop")
    print(f"# fixture={args.fixture}")
    print(
        "# production_readiness="
        f"{readiness['status']} desc_optimization_status={readiness['desc_optimization_status']}"
    )
    print(
        f"# iterations={args.iterations} learning_rate={args.learning_rate:g} "
        f"target_growth={args.target_growth:g}"
    )
    print("iter cost residual observed_growth max_growth R_over_L_T R_over_L_n density temperature")
    for iteration in range(args.iterations):
        (value, result), gradient = value_and_grad(knobs)
        print(_format_iteration(iteration, value, result, knobs))
        knobs = _gradient_step(knobs, gradient, args.learning_rate)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/desc_geometry_dshape_rho05_alpha0.npz"),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--target-growth", type=float, default=0.0)
    parser.add_argument("--target-tolerance", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--ky-index", type=int, default=1)
    parser.add_argument(
        "--readiness-gate",
        type=Path,
        default=Path("fixtures/w7x_itg_convergence_study/production_readiness_gate.json"),
    )
    parser.add_argument("--require-production-ready", action="store_true")
    return parser.parse_args()


def _load_readiness_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "status": "missing_production_readiness_gate",
            "passed": False,
            "desc_optimization_status": "keep_reduced_until_readiness_gate_exists",
        }
    payload = json.loads(path.read_text())
    return {
        "status": payload.get("status", "missing_status"),
        "passed": bool(payload.get("passed")),
        "desc_optimization_status": payload.get(
            "desc_optimization_status",
            "keep_reduced_until_w7x_external_parity_and_production_timing_pass",
        ),
    }


def _parallel_grid_from_fixture_z(z):
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _geometry_from_fixture(fixture, parallel):
    return build_desc_geometry_from_arrays(
        parallel,
        theta=fixture["theta"],
        phi=fixture["phi"],
        alpha=fixture["alpha"],
        rho=fixture["rho"],
        B=fixture["B"],
        b_dot_grad_z=fixture["b_dot_grad_z"],
        grad_psi_sq=fixture["grad_psi_sq"],
        grad_alpha_sq=fixture["grad_alpha_sq"],
        grad_psi_dot_grad_alpha=fixture["grad_psi_dot_grad_alpha"],
        B_cross_gradB_dot_grad_psi=fixture["B_cross_gradB_dot_grad_psi"],
        B_cross_gradB_dot_grad_alpha=fixture["B_cross_gradB_dot_grad_alpha"],
        b_cross_kappa_dot_grad_psi=fixture["b_cross_kappa_dot_grad_psi"],
        b_cross_kappa_dot_grad_alpha=fixture["b_cross_kappa_dot_grad_alpha"],
    )


def _build_problem(parallel):
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.5, mu_max=1.0))
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.45, ky_values=(0.0, 0.35), ikxspace=2)
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
    initial_state = 0.01 * (jnp.cos(index / 7.0) + 1j * jnp.sin(index / 9.0))
    return velocity, fourier, connectivity, initial_state


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
    )


def _format_iteration(iteration, value, result, knobs) -> str:
    values = result.surface_result.values
    return (
        f"{iteration:04d} "
        f"{float(value): .8e} "
        f"{float(result.target_residual): .8e} "
        f"{float(result.observed_value): .8e} "
        f"{float(values.max_growth_rate): .8e} "
        f"{float(knobs.temperature_gradient): .8e} "
        f"{float(knobs.density_gradient): .8e} "
        f"{float(knobs.density): .8e} "
        f"{float(knobs.temperature): .8e}"
    )


if __name__ == "__main__":
    main()
