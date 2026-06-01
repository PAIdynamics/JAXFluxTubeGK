"""Audit selected-ky velocity-space slices against multi-time patched GKW output.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_cosin2_velocity_series.py
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

from stellarator_gk import run_cyclone_base_case_cosin2_velocity_series_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_cosin2_velocity_series_audit(
        gkw_directory=args.gkw_directory,
        gkw_time_path=args.gkw_time,
        snapshot_indices=tuple(args.snapshot_indices) if args.snapshot_indices else None,
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        parallel_derivative_model=args.parallel_derivative_model,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
        grid_tolerance=args.grid_tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    best_final = audit.variant_names[int(audit.best_variant_indices[-1])]
    print(
        f"{status}: first_direct={float(audit.first_direct_max_abs_error):.8e}, "
        f"last_direct={float(audit.last_direct_max_abs_error):.8e}, "
        f"max_direct={float(audit.max_direct_max_abs_error):.8e}, "
        f"max_best={float(audit.max_best_max_abs_error):.8e}, "
        f"final_best_convention={best_final}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "snapshot_index",
                "time",
                "vpar_error",
                "vperp_error",
                "time_error",
                "direct_max_abs_error",
                "direct_l2_error",
                "direct_relative_l2_error",
                "best_variant",
                "best_max_abs_error",
                "best_l2_error",
                "passed",
            )
        )
        for row in range(len(audit.snapshot_indices)):
            writer.writerow(
                (
                    int(audit.snapshot_indices[row]),
                    float(audit.times[row]),
                    float(audit.vpar_errors[row]),
                    float(audit.vperp_errors[row]),
                    float(audit.time_errors[row]),
                    float(audit.direct_max_abs_errors[row]),
                    float(audit.direct_l2_errors[row]),
                    float(audit.direct_relative_l2_errors[row]),
                    audit.variant_names[int(audit.best_variant_indices[row])],
                    float(audit.best_max_abs_errors[row]),
                    float(audit.best_l2_errors[row]),
                    bool(audit.passed),
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkw-directory",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr"),
    )
    parser.add_argument(
        "--gkw-time",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_multitime_distr/time.dat"),
    )
    parser.add_argument("--snapshot-indices", type=int, nargs="*")
    parser.add_argument("--output", type=Path, default=Path("figures/gkw_cosin2_cyclone_velocity_series_audit.csv"))
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--parallel-derivative-model", default="gkw_igh")
    parser.add_argument("--normalization-model", default="gkw_unweighted")
    parser.add_argument("--tolerance", type=float, default=2.0e-2)
    parser.add_argument("--grid-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
