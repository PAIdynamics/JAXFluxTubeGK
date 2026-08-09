"""Audit GKW's fused ltrapping_arakawa path over multiple windows.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_igh_arakawa_series.py
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

import numpy as np

from jax_fluxtube_gk import run_cyclone_base_case_igh_arakawa_series_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_igh_arakawa_series_audit(
        output_windows=tuple(args.output_windows),
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        target_z=args.target_z,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: worst_window={int(audit.worst_window)}, "
        f"worst_max_delta={float(audit.worst_max_delta):.8e}, "
        f"z={float(audit.z):.8e}, z_index={int(audit.z_index)}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    status = "PASS" if bool(audit.passed) else "OPEN"
    output_windows = np.asarray(audit.output_windows)
    times = np.asarray(audit.times)
    local_delta = np.asarray(audit.local_delta)
    max_delta = np.asarray(audit.max_delta)
    relative_delta = np.asarray(audit.relative_delta)
    max_parallel_diffusion = np.asarray(audit.max_parallel_diffusion)
    max_velocity_diffusion = np.asarray(audit.max_velocity_diffusion)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "output_window",
                "time",
                "local_delta",
                "max_delta",
                "relative_delta",
                "max_parallel_diffusion",
                "max_velocity_diffusion",
                "worst_window",
                "worst_max_delta",
                "target_z",
                "target_z_index",
                "status",
                "notes",
            )
        )
        for index, window in enumerate(output_windows):
            writer.writerow(
                (
                    int(window),
                    float(times[index]),
                    float(local_delta[index]),
                    float(max_delta[index]),
                    float(relative_delta[index]),
                    float(max_parallel_diffusion[index]),
                    float(max_velocity_diffusion[index]),
                    int(audit.worst_window),
                    float(audit.worst_max_delta),
                    float(audit.z),
                    int(audit.z_index),
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_igh_arakawa_series_audit.csv"),
    )
    parser.add_argument(
        "--output-windows",
        type=int,
        nargs="+",
        default=(1, 40, 80),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
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
