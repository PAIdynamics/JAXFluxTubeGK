"""Audit the remaining selected-ky gap against patched GKW ``cosin2`` fixtures.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_cosin2_gap.py
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

from jax_fluxtube_gk import run_cyclone_base_case_cosin2_gap_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_cosin2_gap_audit(
        gkw_time_path=args.gkw_time,
        gkw_parallel_phi_path=args.gkw_parallel_phi,
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        parallel_derivative_model=args.parallel_derivative_model,
        normalization_model=args.normalization_model,
        growth_window_fraction=args.growth_window_fraction,
        growth_tolerance=args.growth_tolerance,
        profile_tolerance=args.profile_tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: late_fit_delta={float(audit.late_fit_delta):.8e}, "
        f"late_mean_delta={float(audit.late_mean_delta):.8e}, "
        f"max_profile_error={float(audit.max_profile_error):.8e}, "
        f"worst_time={float(audit.worst_time):.8e}, "
        f"worst_z={float(audit.worst_z):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "metric",
                "value",
            )
        )
        for name in (
            "solver_late_fit_growth",
            "reference_late_fit_growth",
            "late_fit_delta",
            "solver_late_mean_growth",
            "reference_late_mean_growth",
            "late_mean_delta",
            "final_window_growth_delta",
            "first_profile_error",
            "max_profile_error",
            "late_profile_error",
            "worst_time",
            "worst_z",
            "worst_signed_profile_error",
            "total_power_ratio_mean",
            "total_power_ratio_max_deviation",
            "passed",
        ):
            value = getattr(audit, name)
            writer.writerow((name, bool(value) if name == "passed" else float(value)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkw-time",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"),
    )
    parser.add_argument(
        "--gkw-parallel-phi",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gkw_cosin2_cyclone_gap_audit.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--growth-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--profile-tolerance", type=float, default=2.0e-2)
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("gkw_upwind", "gkw_igh"),
        default="gkw_igh",
    )
    parser.add_argument(
        "--normalization-model",
        choices=("weighted", "gkw_unweighted"),
        default="gkw_unweighted",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
