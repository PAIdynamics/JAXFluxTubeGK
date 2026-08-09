"""Audit GKW field packing and diagnostic-output formulas for Cyclone.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_diagnostic_packing.py
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

from jax_fluxtube_gk import run_cyclone_base_case_diagnostic_packing_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_diagnostic_packing_audit(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        output_window=args.output_window,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: time={float(audit.time):.8e}, "
        f"packing_roundtrip_error={float(audit.packing_roundtrip_error):.8e}, "
        f"parallel_phi_error={float(audit.parallel_phi_error):.8e}, "
        f"ky_spectrum_error={float(audit.ky_spectrum_error):.8e}, "
        f"kx_spectrum_error={float(audit.kx_spectrum_error):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    profile = np.asarray(audit.parallel_phi_profile)
    packed_profile = np.asarray(audit.packed_parallel_phi_profile)
    status = "PASS" if bool(audit.passed) else "OPEN"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "z_index",
                "parallel_phi_profile",
                "packed_parallel_phi_profile",
                "time",
                "output_window",
                "field_offset",
                "n_field_values",
                "packing_roundtrip_error",
                "parallel_phi_error",
                "selected_profile_error",
                "ky_spectrum_error",
                "kx_spectrum_error",
                "status",
                "notes",
            )
        )
        for index in range(profile.shape[0]):
            writer.writerow(
                (
                    index,
                    float(profile[index]),
                    float(packed_profile[index]),
                    float(audit.time),
                    audit.output_window,
                    audit.field_offset,
                    audit.n_field_values,
                    float(audit.packing_roundtrip_error),
                    float(audit.parallel_phi_error),
                    float(audit.selected_profile_error),
                    float(audit.ky_spectrum_error),
                    float(audit.kx_spectrum_error),
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_diagnostic_packing_audit.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--output-window", type=int, default=62)
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
