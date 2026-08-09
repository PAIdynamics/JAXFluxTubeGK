"""Run a reduced outer finite-difference W7-X design loop through VMEC++.

This is a real-provider integration demonstration, not an end-to-end AD or
production shape-optimization claim. Generated equilibria and records remain in
explicit external scratch storage.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jax_fluxtube_gk import (
    DesignObjectiveSpec,
    FourierGridSpec,
    GeometryRequest,
    OptimizationKnobs,
    SingleSurfaceOptimizationConfig,
    VelocityGridSpec,
    VmecppGeometryProvider,
    assert_fixed_optimization_topology,
    build_fourier_grid,
    build_mode_connectivity,
    build_optimization_topology_contract,
    build_velocity_grid,
    design_objective,
    internal_geometry_from_result,
    resolve_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/private/tmp/jax-fluxtube-gk-vmecpp-w7x-design/design_loop.json")
STATUS = "reduced_outer_finite_difference_real_vmecpp_not_end_to_end_ad"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    vmecpp = importlib.import_module("vmecpp")
    report = run_vmecpp_w7x_design_loop(
        args,
        named_loader=vmecpp.named_configuration,
        runner=vmecpp.run,
    )
    print(f"PASS: {report['status']}")
    print(args.output)
    return 0


def run_vmecpp_w7x_design_loop(args, *, named_loader, runner) -> dict[str, object]:
    """Evaluate central differences through fresh VMEC++ solves and take steps."""

    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("VMEC++ design-loop output must be outside the repository")
    if args.iterations < 1 or args.finite_difference_step <= 0.0:
        raise ValueError("iterations and finite_difference_step must be positive")
    request = GeometryRequest(
        configuration=args.configuration,
        radial_value=args.rho,
        alpha=args.alpha,
        n_z=args.n_z,
        z_min=-np.pi * args.field_periods,
        z_max=np.pi * args.field_periods,
        field_periods=args.field_periods,
    )
    base_input = named_loader(args.configuration)
    scale = float(args.initial_scale)
    reference_topology = None
    rows = []
    provenance = None
    for iteration in range(args.iterations):
        center = _evaluate_scale(args, base_input, scale, request, runner)
        plus = _evaluate_scale(
            args, base_input, scale + args.finite_difference_step, request, runner
        )
        minus = _evaluate_scale(
            args, base_input, scale - args.finite_difference_step, request, runner
        )
        if reference_topology is None:
            reference_topology = center["topology"]
        for evaluation in (center, plus, minus):
            assert_fixed_optimization_topology(reference_topology, evaluation["topology"])
        gradient = (plus["objective"] - minus["objective"]) / (
            2.0 * args.finite_difference_step
        )
        next_scale = float(
            np.clip(
                scale - args.learning_rate * gradient,
                args.minimum_scale,
                args.maximum_scale,
            )
        )
        rows.append(
            {
                "iteration": iteration,
                "boundary_scale": scale,
                "objective": center["objective"],
                "growth_rate": center["growth_rate"],
                "frequency": center["frequency"],
                "finite_difference_gradient": gradient,
                "next_boundary_scale": next_scale,
            }
        )
        provenance = center["provenance"]
        scale = next_scale
    report = {
        "schema_version": 1,
        "status": STATUS,
        "passed": True,
        "configuration": args.configuration,
        "provider": "vmecpp",
        "provider_differentiable": False,
        "gradient_method": "central_finite_difference_across_fresh_vmecpp_solves",
        "boundary_harmonic": {"m": args.m, "n": args.n},
        "controls": {
            "rho": args.rho,
            "alpha": args.alpha,
            "n_z": args.n_z,
            "field_periods": args.field_periods,
            "n_vpar": args.n_vpar,
            "n_mu": args.n_mu,
            "ky": args.ky,
            "dt": args.dt,
            "n_steps": args.n_steps,
        },
        "provenance": provenance,
        "topology": asdict(reference_topology),
        "iterations": rows,
        "claims": {
            "real_mhd_provider_loop": True,
            "end_to_end_mhd_autodiff": False,
            "full_shape_optimization": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluate_scale(args, base_input, scale, request, runner):
    vmec_input = _scaled_boundary_input(base_input, args.m, args.n, scale)
    result = resolve_geometry(
        VmecppGeometryProvider(
            vmec_input=vmec_input,
            runner=runner,
            max_threads=args.max_threads,
            revision=args.vmecpp_revision,
        ),
        request,
    )
    geometry = internal_geometry_from_result(result)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=args.n_vpar,
            n_mu=args.n_mu,
            vpar_max=args.vpar_max,
            mu_max=args.mu_max,
        )
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(args.ky,))
    )
    connectivity = build_mode_connectivity(fourier)
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        result.parallel_grid.z.shape[0],
        1,
        1,
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    initial_state = args.initial_amplitude * (
        jnp.cos(index / 7.0) + 1j * jnp.sin(index / 11.0)
    )
    objective = design_objective(
        OptimizationKnobs(
            density_gradient=args.density_gradient,
            temperature_gradient=args.temperature_gradient,
        ),
        velocity,
        result.parallel_grid,
        fourier,
        initial_state,
        DesignObjectiveSpec(selected_ky=0),
        connectivity=connectivity,
        config=SingleSurfaceOptimizationConfig(
            geometry_model="precomputed",
            dt=args.dt,
            n_steps=args.n_steps,
            selected_ky=0,
            objective_kind="selected_growth",
            store_history=False,
        ),
        geometry=geometry,
    )
    topology = build_optimization_topology_contract(
        velocity,
        result.parallel_grid,
        fourier,
        connectivity=connectivity,
        geometry_metadata=result.metadata,
    )
    return {
        "objective": float(objective.scalar_objective),
        "growth_rate": float(objective.selected_growth_rate),
        "frequency": float(objective.selected_frequency),
        "topology": topology,
        "provenance": asdict(result.metadata.provenance),
    }


def _scaled_boundary_input(base_input, m: int, n: int, scale: float):
    vmec_input = copy.deepcopy(base_input)
    if not 0 <= m < int(vmec_input.mpol) or not -int(vmec_input.ntor) <= n <= int(
        vmec_input.ntor
    ):
        raise ValueError("requested boundary harmonic is outside VMEC mpol/ntor")
    index = int(vmec_input.ntor) + n
    vmec_input.rbc[m, index] *= scale
    vmec_input.zbs[m, index] *= scale
    return vmec_input


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="w7x-standard")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--initial-scale", type=float, default=1.0)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--minimum-scale", type=float, default=0.95)
    parser.add_argument("--maximum-scale", type=float, default=1.05)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n-z", type=int, default=32)
    parser.add_argument("--field-periods", type=float, default=1.0)
    parser.add_argument("--n-vpar", type=int, default=3)
    parser.add_argument("--n-mu", type=int, default=3)
    parser.add_argument("--vpar-max", type=float, default=2.0)
    parser.add_argument("--mu-max", type=float, default=1.5)
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--density-gradient", type=float, default=0.8)
    parser.add_argument("--temperature-gradient", type=float, default=2.1)
    parser.add_argument("--initial-amplitude", type=float, default=1.0e-2)
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument("--vmecpp-revision", default="unknown")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
