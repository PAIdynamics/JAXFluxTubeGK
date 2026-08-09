"""Compare patched GKW compact state trace against the solver source trace.

Run from the repository root:

    uv run --extra dev python examples/compare_gkw_state_trace.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

from jax_fluxtube_gk import (
    compare_gkw_state_trace_to_source_term_trace,
    load_gkw_state_trace,
    run_cyclone_base_case_source_term_trace,
)


def main() -> None:
    args = _parse_args()
    gkw_trace = load_gkw_state_trace(args.gkw_state_trace)
    solver_trace = run_cyclone_base_case_source_term_trace(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        output_windows=tuple(range(int(gkw_trace.times.shape[0]) + 1)),
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        parallel_derivative_model=args.parallel_derivative_model,
        snapshot_timing="post_normalization",
    )
    report = compare_gkw_state_trace_to_source_term_trace(
        gkw_trace,
        solver_trace,
        tolerance=args.tolerance,
    )
    _write_csv(args.output, report, gkw_trace, solver_trace)
    status = "PASS" if bool(report.passed) else "OPEN"
    print(
        f"{status}: max_abs_error={float(report.max_abs_error):.8e}, "
        f"state_norm_error={float(report.field_errors[1]):.8e}, "
        f"phi_norm_error={float(report.field_errors[2]):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, report, gkw_trace, solver_trace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    solver_state = solver_trace.state_norm[1:]
    solver_phi = solver_trace.phi_norm[1:]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "value"))
        writer.writerow(("max_abs_error", float(report.max_abs_error)))
        for name, value in zip(report.field_names, report.field_errors, strict=True):
            writer.writerow((f"{name}_error", float(value)))
        writer.writerow(("gkw_final_state_norm", float(gkw_trace.state_norm[-1])))
        writer.writerow(("solver_final_state_norm", float(solver_state[-1])))
        writer.writerow(("gkw_final_phi_norm", float(gkw_trace.phi_norm[-1])))
        writer.writerow(("solver_final_phi_norm", float(solver_phi[-1])))
        writer.writerow(("passed", bool(report.passed)))
        writer.writerow(("notes", report.notes))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkw-state-trace",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_state_trace.dat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gkw_cosin2_cyclone_state_trace_comparison.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--initial-profile", choices=("cosine2", "cosine"), default="cosine2")
    parser.add_argument(
        "--normalization-model",
        choices=("weighted", "gkw_unweighted"),
        default="gkw_unweighted",
    )
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("gkw_upwind", "gkw_igh", "matrix"),
        default="gkw_igh",
    )
    parser.add_argument("--tolerance", type=float, default=5.0e-3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
