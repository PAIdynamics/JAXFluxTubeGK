"""Audit multi-time velocity slices across Term VII and gkw_igh variants.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_cosin2_velocity_series_variants.py
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

from stellarator_gk import run_cyclone_base_case_cosin2_velocity_series_variant_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_cosin2_velocity_series_variant_audit(
        gkw_directory=args.gkw_directory,
        gkw_time_path=args.gkw_time,
        snapshot_indices=tuple(args.snapshot_indices) if args.snapshot_indices else None,
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
        grid_tolerance=args.grid_tolerance,
    )
    _write_csv(args.output, audit)
    mid = _snapshot_position(audit, args.report_snapshot)
    best = int(audit.best_direct_variant_indices[mid])
    print(
        "OPEN: "
        f"snapshot={int(audit.snapshot_indices[mid])}, "
        f"baseline_direct={float(audit.baseline_direct_max_abs_errors[mid]):.8e}, "
        f"best_variant={audit.variant_names[best]}, "
        f"best_direct={float(audit.best_direct_max_abs_errors[mid]):.8e}, "
        f"max_baseline={float(audit.max_baseline_direct_max_abs_error):.8e}, "
        f"max_best={float(audit.max_best_direct_max_abs_error):.8e}"
    )
    print(args.output)


def _snapshot_position(audit, requested: int) -> int:
    indices = [int(value) for value in audit.snapshot_indices]
    if requested in indices:
        return indices.index(requested)
    return len(indices) // 2


def _write_csv(path: Path, audit) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "variant",
                "snapshot_index",
                "time",
                "direct_max_abs_error",
                "direct_l2_error",
                "direct_relative_l2_error",
                "best_layout",
                "best_layout_max_abs_error",
                "best_layout_l2_error",
                "is_best_direct_at_snapshot",
                "passed",
            )
        )
        for variant_index, variant_name in enumerate(audit.variant_names):
            for snapshot_position in range(len(audit.snapshot_indices)):
                layout_index = int(
                    audit.best_layout_variant_indices[variant_index, snapshot_position]
                )
                writer.writerow(
                    (
                        variant_name,
                        int(audit.snapshot_indices[snapshot_position]),
                        float(audit.times[snapshot_position]),
                        float(audit.direct_max_abs_errors[variant_index, snapshot_position]),
                        float(audit.direct_l2_errors[variant_index, snapshot_position]),
                        float(audit.direct_relative_l2_errors[variant_index, snapshot_position]),
                        audit.layout_variant_names[layout_index],
                        float(
                            audit.best_layout_max_abs_errors[variant_index, snapshot_position]
                        ),
                        float(audit.best_layout_l2_errors[variant_index, snapshot_position]),
                        variant_index == int(audit.best_direct_variant_indices[snapshot_position]),
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
    parser.add_argument("--report-snapshot", type=int, default=800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gkw_cosin2_cyclone_velocity_series_variant_audit.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--normalization-model", default="gkw_unweighted")
    parser.add_argument("--tolerance", type=float, default=2.0e-2)
    parser.add_argument("--grid-tolerance", type=float, default=1.0e-4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
