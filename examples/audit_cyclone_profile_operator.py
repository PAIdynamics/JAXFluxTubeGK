"""Audit selected-mode operators at the localized Cyclone profile mismatch.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_profile_operator.py
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

from stellarator_gk import run_cyclone_base_case_profile_operator_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_profile_operator_audit(
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
        f"local_streaming_delta={float(audit.local_streaming_delta):.8e}, "
        f"local_field_drive_delta={float(audit.local_field_drive_delta):.8e}, "
        f"field_residual={float(audit.field_residual_max):.8e}, "
        f"rhs_assembly={float(audit.rhs_assembly_error):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    profile = np.asarray(audit.normalized_phi_power)
    z_grid = np.asarray(audit.z_grid)
    streaming_delta = np.asarray(audit.streaming_delta_profile)
    field_delta = np.asarray(audit.field_drive_delta_profile)
    field_residual = np.asarray(audit.field_residual_profile)
    target_index = int(audit.z_index)
    n_z = profile.shape[0]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "z_index",
                "z",
                "normalized_phi_power",
                "streaming_delta_profile",
                "field_drive_delta_profile",
                "field_residual_profile",
                "is_target_z",
                "time",
                "target_z",
                "peak_z",
                "second_moment",
                "local_streaming_delta",
                "max_streaming_delta",
                "boundary_streaming_delta",
                "local_field_drive_delta",
                "max_field_drive_delta",
                "boundary_field_drive_delta",
                "field_residual_max",
                "field_reconstruction_error",
                "rhs_assembly_error",
                "status",
                "notes",
            )
        )
        status = "PASS" if bool(audit.passed) else "OPEN"
        for index in range(n_z):
            writer.writerow(
                (
                    index,
                    float(z_grid[index]),
                    float(profile[index]),
                    float(streaming_delta[index]),
                    float(field_delta[index]),
                    float(field_residual[index]),
                    index == target_index,
                    float(audit.time),
                    float(audit.z),
                    float(audit.peak_z),
                    float(audit.second_moment),
                    float(audit.local_streaming_delta),
                    float(audit.max_streaming_delta),
                    float(audit.boundary_streaming_delta),
                    float(audit.local_field_drive_delta),
                    float(audit.max_field_drive_delta),
                    float(audit.boundary_field_drive_delta),
                    float(audit.field_residual_max),
                    float(audit.field_reconstruction_error),
                    float(audit.rhs_assembly_error),
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_profile_operator_audit.csv"),
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
    parser.add_argument("--tolerance", type=float, default=5.0e-11)
    return parser.parse_args()


if __name__ == "__main__":
    main()
