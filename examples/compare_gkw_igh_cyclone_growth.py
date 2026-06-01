"""Compare the fused-GKW-igh Cyclone growth history with matched GKW time.dat.

Run from the repository root:

    uv run --extra dev python examples/compare_gkw_igh_cyclone_growth.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from stellarator_gk import (
    load_gkw_time_dat_trace,
    run_cyclone_base_case_trace,
    write_cyclone_trace_csv,
)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)

    solver = run_cyclone_base_case_trace(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        parallel_derivative_model=args.parallel_derivative_model,
        velocity_recurrence_rate=args.velocity_recurrence_rate,
    )
    reference = load_gkw_time_dat_trace(
        args.gkw_time,
        source="gkw:time.dat",
        notes="matched selected-ky production-control run",
    )
    metrics = _growth_metrics(solver, reference, args.growth_window_fraction)
    _write_csv(args.output, metrics, args)
    write_cyclone_trace_csv(args.trace_output, solver)

    late_fit = metrics["late_fit_growth"]
    late_mean = metrics["late_mean_window_growth"]
    print(
        "OPEN: "
        f"late_fit_solver={late_fit['solver']:.8e}, "
        f"late_fit_gkw={late_fit['gkw']:.8e}, "
        f"late_mean_solver={late_mean['solver']:.8e}, "
        f"late_mean_gkw={late_mean['gkw']:.8e}, "
        f"parallel_derivative_model={args.parallel_derivative_model}"
    )
    print(args.output)
    print(args.trace_output)


def _growth_metrics(solver, reference, start_fraction: float) -> dict[str, dict[str, float]]:
    solver_window_growth = jnp.asarray(solver.window_growth[1:], dtype=jnp.float64)
    reference_window_growth = jnp.asarray(reference.window_growth, dtype=jnp.float64)
    return {
        "final_window_growth": {
            "gkw": float(reference_window_growth[-1]),
            "solver": float(solver_window_growth[-1]),
        },
        "late_mean_window_growth": {
            "gkw": float(_late_mean(reference_window_growth, start_fraction)),
            "solver": float(_late_mean(solver_window_growth, start_fraction)),
        },
        "late_fit_growth": {
            "gkw": float(
                _late_fit(
                    reference.times,
                    reference.physical_amplitude,
                    start_fraction=start_fraction,
                )
            ),
            "solver": float(
                _late_fit(
                    solver.times,
                    solver.physical_amplitude,
                    start_fraction=start_fraction,
                )
            ),
        },
    }


def _late_mean(values, start_fraction: float):
    values = jnp.asarray(values, dtype=jnp.float64)
    start = max(0, min(int(values.shape[0] * start_fraction), values.shape[0] - 1))
    return jnp.mean(values[start:])


def _late_fit(times, amplitudes, *, start_fraction: float):
    times = jnp.asarray(times, dtype=jnp.float64)
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.float64)
    start = max(0, min(int(times.shape[0] * start_fraction), times.shape[0] - 2))
    times = times[start:]
    log_amplitudes = jnp.log(jnp.maximum(amplitudes[start:], jnp.asarray(1.0e-300)))
    centered_time = times - jnp.mean(times)
    centered_log = log_amplitudes - jnp.mean(log_amplitudes)
    return jnp.sum(centered_time * centered_log) / jnp.sum(centered_time**2)


def _write_csv(path: Path, metrics: dict[str, dict[str, float]], args) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "reference_trace",
                "metric",
                "gkw",
                "solver",
                "delta_gkw_minus_solver",
                "parallel_derivative_model",
                "initial_profile",
                "normalization_model",
            )
        )
        for metric, values in metrics.items():
            writer.writerow(
                (
                    "gkw_igh_backend",
                    metric,
                    values["gkw"],
                    values["solver"],
                    values["gkw"] - values["solver"],
                    args.parallel_derivative_model,
                    args.initial_profile,
                    args.normalization_model,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkw-time",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_time.dat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gkw_igh_cyclone_selected_ky_time_comparison.csv"),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("figures/gkw_igh_cyclone_selected_ky_time_trace.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--initial-profile", choices=("cosine2", "cosine"), default="cosine")
    parser.add_argument(
        "--normalization-model",
        choices=("weighted", "gkw_unweighted"),
        default="gkw_unweighted",
    )
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("gkw_upwind", "gkw_igh"),
        default="gkw_igh",
    )
    parser.add_argument("--velocity-recurrence-rate", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
