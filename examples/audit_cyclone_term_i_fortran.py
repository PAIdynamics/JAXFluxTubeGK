"""Compare the selected-mode Term I operator with GKW Fortran formulas.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_term_i_fortran.py
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

from stellarator_gk import run_cyclone_base_case_term_i_fortran_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_term_i_fortran_audit(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        output_window=args.output_window,
        target_z=args.target_z,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: time={float(audit.time):.8e}, z={float(audit.z):.8e}, "
        f"max_term_error={float(audit.max_term_error):.8e}, "
        f"max_coefficient_error={float(audit.max_coefficient_error):.8e}, "
        f"recurrence_speed_error={float(audit.recurrence_speed_max_error):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    term_error = np.asarray(audit.term_error_profile)
    coefficient_error = np.asarray(audit.coefficient_error_profile)
    current_term = np.asarray(audit.current_term_profile)
    reference_term = np.asarray(audit.reference_term_profile)
    recurrence_error = np.asarray(audit.recurrence_speed_error_profile)
    target_index = int(audit.z_index)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "z_index",
                "term_error_profile",
                "coefficient_error_profile",
                "current_term_profile",
                "reference_term_profile",
                "recurrence_speed_error_profile",
                "is_target_z",
                "time",
                "target_z",
                "local_term_error",
                "max_term_error",
                "local_coefficient_error",
                "max_coefficient_error",
                "recurrence_speed_max_error",
                "sign_selection_error",
                "status",
                "notes",
            )
        )
        status = "PASS" if bool(audit.passed) else "OPEN"
        for index in range(term_error.shape[0]):
            writer.writerow(
                (
                    index,
                    float(term_error[index]),
                    float(coefficient_error[index]),
                    float(current_term[index]),
                    float(reference_term[index]),
                    float(recurrence_error[index]),
                    index == target_index,
                    float(audit.time),
                    float(audit.z),
                    float(audit.local_term_error),
                    float(audit.max_term_error),
                    float(audit.local_coefficient_error),
                    float(audit.max_coefficient_error),
                    float(audit.recurrence_speed_max_error),
                    float(audit.sign_selection_error),
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_term_i_fortran_audit.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--output-window", type=int, default=62)
    parser.add_argument("--target-z", type=float, default=0.09375)
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
