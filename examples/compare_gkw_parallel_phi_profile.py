"""Compare matched GKW ``parallel_phi.dat`` with the solver profile trace.

Run from the repository root:

    uv run --extra dev python examples/compare_gkw_parallel_phi_profile.py
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

import numpy as np

from stellarator_gk import (
    audit_parallel_phi_profile_alignment,
    compare_parallel_phi_traces,
    load_gkw_parallel_phi_trace,
    run_cyclone_base_case_parallel_phi_trace,
)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    solver = run_cyclone_base_case_parallel_phi_trace(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
    )
    reference = load_gkw_parallel_phi_trace(
        args.gkw_parallel_phi,
        time_path=args.gkw_time,
        z=np.asarray(solver.z),
        source="gkw:parallel_phi.dat",
        notes="matched selected-ky production-control run",
    )
    report = compare_parallel_phi_traces(
        solver,
        reference,
        tolerance=args.tolerance,
        time_tolerance=args.time_tolerance,
        normalize_profiles=args.normalize_profiles,
    )
    audit = audit_parallel_phi_profile_alignment(
        solver,
        reference,
        tolerance=args.tolerance,
        time_tolerance=args.time_tolerance,
        normalize_profiles=args.normalize_profiles,
    )
    _write_csv(args.output, solver, reference, report, audit)
    status = "PASS" if bool(report.passed) else "OPEN"
    print(
        f"{status}: max_abs_profile_error={float(report.max_abs_error):.8e}, "
        f"best_aligned={float(audit.best_aligned_max_error):.8e}, "
        f"best_shift={int(audit.best_shift)}, "
        f"time_error={float(report.time_error):.8e}, tolerance={args.tolerance:.8e}"
    )
    print(args.output)


def _write_csv(path: Path, solver, reference, report, audit) -> None:
    solver_profiles = _profiles_for_csv(np.asarray(solver.phi_power), report.normalized_profiles)
    reference_profiles = _profiles_for_csv(
        np.asarray(reference.phi_power),
        report.normalized_profiles,
    )
    raw_solver = np.asarray(solver.phi_power)
    raw_reference = np.asarray(reference.phi_power)
    direct_errors = np.asarray(audit.direct_profile_errors)
    reversed_errors = np.asarray(audit.reversed_profile_errors)
    shifted_errors = np.asarray(audit.best_shift_profile_errors)
    total_power_ratio = np.asarray(audit.total_power_ratio)
    center_error = np.asarray(audit.center_of_power_error)
    edge_error = np.asarray(audit.edge_fraction_error)
    peak_z_error = np.asarray(audit.peak_z_error)
    second_moment_error = np.asarray(audit.second_moment_error)
    z = np.asarray(solver.z)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "time",
                "max_abs_profile_error",
                "reversed_profile_error",
                "best_shift_profile_error",
                "best_global_shift",
                "solver_peak_z",
                "gkw_peak_z",
                "solver_peak_value",
                "gkw_peak_value",
                "solver_total_power",
                "gkw_total_power",
                "total_power_ratio",
                "center_of_power_error",
                "edge_fraction_error",
                "peak_z_error",
                "second_moment_error",
                "global_worst_time",
                "global_worst_z",
                "global_worst_signed_profile_error",
                "global_worst_solver_value",
                "global_worst_gkw_value",
                "normalized_profiles",
                "status",
                "notes",
            )
        )
        status = "PASS" if bool(report.passed) else "OPEN"
        for index, time in enumerate(np.asarray(solver.times)):
            solver_peak = int(np.argmax(solver_profiles[index]))
            reference_peak = int(np.argmax(reference_profiles[index]))
            writer.writerow(
                (
                    float(time),
                    float(direct_errors[index]),
                    float(reversed_errors[index]),
                    float(shifted_errors[index]),
                    int(audit.best_shift),
                    float(z[solver_peak]),
                    float(z[reference_peak]),
                    float(solver_profiles[index, solver_peak]),
                    float(reference_profiles[index, reference_peak]),
                    float(np.sum(raw_solver[index])),
                    float(np.sum(raw_reference[index])),
                    float(total_power_ratio[index]),
                    float(center_error[index]),
                    float(edge_error[index]),
                    float(peak_z_error[index]),
                    float(second_moment_error[index]),
                    float(audit.worst_time),
                    float(audit.worst_z),
                    float(audit.worst_signed_error),
                    float(audit.worst_observed_value),
                    float(audit.worst_reference_value),
                    bool(report.normalized_profiles),
                    status,
                    audit.notes,
                )
            )


def _profiles_for_csv(values, normalize_profiles: bool) -> np.ndarray:
    profiles = np.asarray(values, dtype=float)
    if not normalize_profiles:
        return profiles
    row_sum = np.maximum(np.sum(profiles, axis=1, keepdims=True), 1.0e-300)
    return profiles / row_sum


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkw-parallel-phi",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_parallel_phi.dat"),
    )
    parser.add_argument(
        "--gkw-time",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_time.dat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gkw_cyclone_parallel_phi_profile_comparison.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--initial-profile", choices=("cosine2", "cosine"), default="cosine")
    parser.add_argument(
        "--normalization-model",
        choices=("weighted", "gkw_unweighted"),
        default="gkw_unweighted",
    )
    parser.add_argument("--tolerance", type=float, default=2.0e-2)
    parser.add_argument("--time-tolerance", type=float, default=5.0e-5)
    parser.add_argument(
        "--absolute-profiles",
        action="store_true",
        help="compare raw |phi|^2 instead of row-normalized profile shapes",
    )
    args = parser.parse_args()
    args.normalize_profiles = not args.absolute_profiles
    return args


if __name__ == "__main__":
    main()
