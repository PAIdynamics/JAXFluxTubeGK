"""Audit GKW's fused ltrapping_arakawa Term I/IV path.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_igh_arakawa.py
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

from stellarator_gk import run_cyclone_base_case_igh_arakawa_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_igh_arakawa_audit(
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
        f"local_delta={float(audit.local_delta):.8e}, "
        f"max_delta={float(audit.max_delta):.8e}, "
        f"relative_delta={float(audit.relative_delta):.8e}, "
        f"max_parallel_diffusion={float(audit.max_parallel_diffusion):.8e}, "
        f"max_velocity_diffusion={float(audit.max_velocity_diffusion):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    fused = np.asarray(audit.fused_profile)
    separated = np.asarray(audit.separated_profile)
    delta = np.asarray(audit.delta_profile)
    parallel_diffusion = np.asarray(audit.parallel_diffusion_profile)
    velocity_diffusion = np.asarray(audit.velocity_diffusion_profile)
    target_index = int(audit.z_index)
    status = "PASS" if bool(audit.passed) else "OPEN"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "z_index",
                "fused_profile",
                "separated_profile",
                "delta_profile",
                "parallel_diffusion_profile",
                "velocity_diffusion_profile",
                "is_target_z",
                "time",
                "target_z",
                "local_delta",
                "max_delta",
                "relative_delta",
                "max_parallel_diffusion",
                "max_velocity_diffusion",
                "status",
                "notes",
            )
        )
        for index in range(delta.shape[0]):
            writer.writerow(
                (
                    index,
                    float(fused[index]),
                    float(separated[index]),
                    float(delta[index]),
                    float(parallel_diffusion[index]),
                    float(velocity_diffusion[index]),
                    index == target_index,
                    float(audit.time),
                    float(audit.z),
                    float(audit.local_delta),
                    float(audit.max_delta),
                    float(audit.relative_delta),
                    float(audit.max_parallel_diffusion),
                    float(audit.max_velocity_diffusion),
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_igh_arakawa_audit.csv"),
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
