"""Write a guarded W7-X CPU-timing artifact.

By default this command refuses to make a production timing claim until the
external W7-X parity gate passes.  Use ``--allow-pending-external-parity`` for a
development timing run that is explicitly labeled as not production-validated.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    benchmark_linear_residual,
    build_flux_tube_geometry_from_gx_eik_reference,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    estimate_linear_memory_from_dimensions,
    format_bytes,
    load_gx_eik_geometry_reference,
    resample_gx_eik_geometry_reference,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERGENCE_DIR = ROOT / "fixtures/w7x_itg_convergence_study"
DEFAULT_READINESS_GATE = DEFAULT_CONVERGENCE_DIR / "production_readiness_gate.json"
DEFAULT_OUTPUT = DEFAULT_CONVERGENCE_DIR / "production_cpu_timing.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from stellarator_gk.external import announce_external_path

    if args.eik_reference is not None:
        announce_external_path("GX/GIST eik", args.eik_reference)
    artifact = run_w7x_cpu_timing(args)
    print(f"{'PASS' if artifact['passed'] else 'OPEN'}: {artifact['status']}")
    print(args.output)
    return 0 if artifact["passed"] or not args.require_pass else 2


def run_w7x_cpu_timing(args: argparse.Namespace) -> dict[str, object]:
    """Build and optionally time the W7-X production-control residual."""

    parity_ready = external_parity_ready(args.readiness_gate)
    if args.require_external_parity and not parity_ready and not args.allow_pending_external_parity:
        artifact = _blocked_artifact(args, parity_ready)
        _write_json(args.output, artifact)
        return artifact

    controls = timing_controls(args)
    memory = estimate_linear_memory_from_dimensions(
        n_vpar=controls["n_vpar"],
        n_mu=controls["n_mu"],
        n_z=controls["n_z"],
        n_kx=controls["n_kx"],
        n_ky=len(controls["ky_values"]),
        n_steps=0,
        store_history=False,
    )
    artifact: dict[str, object] = {
        "benchmark_name": "w7x_itg_cpu_timing",
        "status": "production_timing_validated" if parity_ready else "development_timing_only",
        "passed": bool(parity_ready),
        "external_parity_ready": bool(parity_ready),
        "production_claim": bool(parity_ready),
        "controls": _jsonify(controls),
        "memory_estimate": _memory_payload(memory),
        "timing": None,
    }
    if args.estimate_only:
        artifact["status"] = (
            "production_timing_estimate_only" if parity_ready else "development_estimate_only"
        )
        artifact["passed"] = False
        _write_json(args.output, artifact)
        return artifact

    if args.preset == "stella-production":
        artifact["timing"] = _benchmark_stella_production_scan(args)
        artifact["timing_scope"] = (
            "end_to_end_including_geometry_load_jax_compile_and_diagnostics"
        )
        _write_json(args.output, artifact)
        return artifact

    build_start = perf_counter()
    distribution, precompute = build_timing_problem(args, controls)
    build_seconds = perf_counter() - build_start
    benchmark = benchmark_linear_residual(
        distribution,
        precompute,
        repeats=args.repeats,
    )
    rhs_per_rk4_step = 4
    total_rhs_calls = rhs_per_rk4_step * args.steps_per_window * args.n_windows
    artifact["timing"] = {
        "build_seconds": float(build_seconds),
        "compile_seconds": benchmark.compile_seconds,
        "mean_execute_seconds_per_rhs": benchmark.mean_execute_seconds,
        "best_execute_seconds_per_rhs": benchmark.best_execute_seconds,
        "repeats": benchmark.repeats,
        "estimated_rk4_rhs_calls": total_rhs_calls,
        "estimated_rk4_window_seconds_best": benchmark.best_execute_seconds
        * total_rhs_calls,
        "state_bytes": benchmark.state_bytes,
        "coefficient_bytes": benchmark.coefficient_bytes,
    }
    _write_json(args.output, artifact)
    return artifact


def build_timing_problem(args: argparse.Namespace, controls: dict[str, object]):
    """Assemble a deterministic W7-X timing state and residual precompute."""

    if args.eik_reference is None:
        raise ValueError("a production timing run requires --eik-reference")
    eik = load_gx_eik_geometry_reference(args.eik_reference)
    theta = np.linspace(
        -np.pi * controls["field_line_periods"],
        np.pi * controls["field_line_periods"],
        controls["n_z"],
        endpoint=False,
    )
    parallel = _parallel_grid_from_theta(theta)
    sampled = resample_gx_eik_geometry_reference(eik, theta)
    geometry = build_flux_tube_geometry_from_gx_eik_reference(sampled, parallel)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=controls["n_vpar"],
            n_mu=controls["n_mu"],
            vpar_max=controls["vpar_max"],
            mu_max=controls["mu_max"],
        )
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=controls["n_kx"],
            n_ky=len(controls["ky_values"]),
            kx_max=controls["kx_max"],
            ky_values=tuple(controls["ky_values"]),
            ikxspace=controls["ikxspace"],
        )
    )
    connectivity = build_mode_connectivity(fourier)
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        SpeciesParams(
            charge=1.0,
            mass=1.0,
            density=1.0,
            temperature=1.0,
            density_gradient=1.0,
            temperature_gradient=3.0,
        ),
        electron_params=AdiabaticElectronParams(
            density=1.0,
            temperature=1.0,
            zonal_correction=False,
        ),
        mode_connectivity=connectivity,
    )
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    distribution = args.initial_amplitude * (
        jnp.cos(index / 17.0) + 1j * jnp.sin(index / 19.0)
    )
    return distribution, precompute


def external_parity_ready(readiness_gate: Path) -> bool:
    """Return whether the W7-X external mode-structure gate has passed."""

    if not readiness_gate.exists():
        return False
    payload = json.loads(readiness_gate.read_text())
    external = payload.get("external_mode_structure_gate", {})
    return bool(external.get("passed"))


def timing_controls(args: argparse.Namespace) -> dict[str, object]:
    """Return timing dimensions for the selected preset."""

    if args.preset == "smoke":
        defaults = {
            "n_z": 17,
            "field_line_periods": 1,
            "n_kx": 1,
            "kx_max": 0.0,
            "ikxspace": 1,
            "n_vpar": 4,
            "n_mu": 3,
            "ky_values": (0.1, 0.2),
            "vpar_max": 2.0,
            "mu_max": 1.5,
        }
    elif args.preset == "stella-production":
        defaults = {
            "n_z": 256,
            "field_line_periods": 1,
            "n_kx": 1,
            "kx_max": 0.0,
            "ikxspace": 1,
            "n_vpar": 32,
            "n_mu": 8,
            "ky_values": (0.3,),
            "vpar_max": 3.0,
            "mu_max": 4.916958697837631,
        }
    else:
        defaults = {
            "n_z": 256,
            "field_line_periods": 6,
            "n_kx": 1,
            "kx_max": 0.0,
            "ikxspace": 1,
            "n_vpar": 16,
            "n_mu": 8,
            "ky_values": tuple(np.linspace(0.0, args.ky_max, args.n_ky)),
            "vpar_max": 2.0,
            "mu_max": 1.5,
        }
    for key in (
        "n_z",
        "field_line_periods",
        "n_kx",
        "kx_max",
        "ikxspace",
        "n_vpar",
        "n_mu",
        "vpar_max",
        "mu_max",
    ):
        override = getattr(args, key)
        if override is not None:
            defaults[key] = override
    if args.ky_values is not None:
        defaults["ky_values"] = _parse_float_tuple(args.ky_values)
    return defaults


def _benchmark_stella_production_scan(args: argparse.Namespace) -> dict[str, object]:
    if args.stella_geometry is None:
        raise ValueError("stella-production timing requires --stella-geometry")
    module_path = ROOT / "examples/run_stellarator_linear_scan.py"
    spec = importlib.util.spec_from_file_location(
        "_optimal_fusion_stella_production_timing_scan", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load stellarator scan from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="optimal-fusion-w7x-timing-") as scratch:
        scan_args = _stella_scan_args(args, Path(scratch))
        start = perf_counter()
        exit_code = module.main(scan_args)
        wall_seconds = perf_counter() - start
    if exit_code != 0:
        raise RuntimeError(f"source-matched W7-X timing scan exited with {exit_code}")
    n_steps = args.steps_per_window * args.n_windows
    return {
        "wall_seconds": float(wall_seconds),
        "steps": int(n_steps),
        "windows": int(args.n_windows),
        "seconds_per_step_including_compile": float(wall_seconds / n_steps),
        "scratch_artifacts_retained": False,
        "algorithm": "ssp_rk3_then_stella_cubic_mirror_then_implicit_response",
    }


def _stella_scan_args(args: argparse.Namespace, output_dir: Path) -> list[str]:
    controls = timing_controls(args)
    return [
        "--geometry-source",
        "stella-geometry",
        "--stella-geometry",
        str(args.stella_geometry),
        "--output-dir",
        str(output_dir),
        "--n-kx",
        str(controls["n_kx"]),
        "--kx-max",
        str(controls["kx_max"]),
        "--ikxspace",
        str(controls["ikxspace"]),
        "--ky-values",
        ",".join(str(value) for value in controls["ky_values"]),
        "--n-vpar",
        str(controls["n_vpar"]),
        "--n-mu",
        str(controls["n_mu"]),
        "--vpar-max",
        str(controls["vpar_max"]),
        "--mu-max",
        str(controls["mu_max"]),
        "--velocity-backend",
        "midpoint_gauss_laguerre",
        "--velocity-measure-normalization",
        "full_gyroangle",
        "--mirror-advance",
        "semi_lagrangian",
        "--mirror-interpolation",
        "stella_cubic",
        "--parallel-advance",
        "stella_implicit",
        "--initial-condition",
        "stella_maxwellian",
        "--dt",
        "0.1",
        "--steps-per-window",
        str(args.steps_per_window),
        "--n-windows",
        str(args.n_windows),
    ]


def _blocked_artifact(args: argparse.Namespace, parity_ready: bool) -> dict[str, object]:
    controls = timing_controls(args)
    memory = estimate_linear_memory_from_dimensions(
        n_vpar=controls["n_vpar"],
        n_mu=controls["n_mu"],
        n_z=controls["n_z"],
        n_kx=controls["n_kx"],
        n_ky=len(controls["ky_values"]),
        n_steps=0,
        store_history=False,
    )
    return {
        "benchmark_name": "w7x_itg_cpu_timing",
        "status": "blocked_until_external_parity_passes",
        "passed": False,
        "external_parity_ready": bool(parity_ready),
        "production_claim": False,
        "controls": _jsonify(controls),
        "memory_estimate": _memory_payload(memory),
        "timing": None,
        "next_required_artifact": "fixtures/w7x_itg_external_mode_structure_fixture.csv",
    }


def _parallel_grid_from_theta(theta):
    theta = np.asarray(theta, dtype=float)
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _memory_payload(memory) -> dict[str, object]:
    return {
        "state_shape": memory.state_shape,
        "field_shape": memory.field_shape,
        "state_bytes": memory.state_bytes,
        "field_bytes": memory.field_bytes,
        "coefficient_bytes": memory.coefficient_bytes,
        "history_bytes": memory.history_bytes,
        "total_bytes": memory.total_bytes,
        "total_bytes_human": format_bytes(memory.total_bytes),
        "store_history": memory.store_history,
    }


def _jsonify(value):
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--readiness-gate", type=Path, default=DEFAULT_READINESS_GATE)
    parser.add_argument("--eik-reference", type=Path)
    parser.add_argument(
        "--preset",
        choices=("production-control", "stella-production", "smoke"),
        default="production-control",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--allow-pending-external-parity", action="store_true")
    parser.add_argument("--require-external-parity", action="store_true", default=True)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps-per-window", type=int, default=1)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--initial-amplitude", type=float, default=1.0e-2)
    parser.add_argument("--n-z", type=int)
    parser.add_argument("--field-line-periods", type=int)
    parser.add_argument("--n-kx", type=int)
    parser.add_argument("--kx-max", type=float)
    parser.add_argument("--ikxspace", type=int)
    parser.add_argument("--n-vpar", type=int)
    parser.add_argument("--n-mu", type=int)
    parser.add_argument("--vpar-max", type=float)
    parser.add_argument("--mu-max", type=float)
    parser.add_argument("--n-ky", type=int, default=28)
    parser.add_argument("--ky-max", type=float, default=2.7)
    parser.add_argument("--ky-values")
    parser.add_argument("--stella-geometry", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
