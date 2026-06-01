"""Audit the Cyclone RK4 window and GKW normalization sequence.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_time_normalization.py
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

from stellarator_gk import run_cyclone_base_case_time_normalization_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_time_normalization_audit(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: rk4_step_error={float(audit.rk4_step_error):.8e}, "
        f"growth_sequence_error={float(audit.growth_sequence_error):.8e}, "
        f"post_normalization_error={float(audit.post_normalization_error):.8e}, "
        f"field_linearity_error={float(audit.max_field_linearity_error):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    times = np.asarray(audit.times)
    factors = np.asarray(audit.normalization_factor)
    source_growth = np.asarray(audit.gkw_window_growth)
    trace_growth = np.asarray(audit.trace_window_growth)
    post_norm = np.asarray(audit.post_normalization_field_norm)
    field_linearity = np.asarray(audit.field_linearity_error)
    status = "PASS" if bool(audit.passed) else "OPEN"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "time",
                "normalization_factor",
                "gkw_window_growth",
                "trace_window_growth",
                "post_normalization_field_norm",
                "field_linearity_error",
                "rk4_step_error",
                "time_grid_error",
                "growth_sequence_error",
                "post_normalization_error",
                "max_field_linearity_error",
                "status",
                "notes",
            )
        )
        for index in range(times.shape[0]):
            writer.writerow(
                (
                    float(times[index]),
                    float(factors[index]),
                    float(source_growth[index]),
                    float(trace_growth[index]),
                    float(post_norm[index]),
                    float(field_linearity[index]),
                    float(audit.rk4_step_error),
                    float(audit.time_grid_error),
                    float(audit.growth_sequence_error),
                    float(audit.post_normalization_error),
                    float(audit.max_field_linearity_error),
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_time_normalization_audit.csv"),
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
    parser.add_argument("--tolerance", type=float, default=5.0e-12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
