"""Audit selected Cyclone Term VII mode and field-packing conventions.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_term_vii_mode_packing.py
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

from stellarator_gk import run_cyclone_base_case_term_vii_mode_packing_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_term_vii_mode_packing_audit(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        output_window=args.output_window,
        initial_profile=args.initial_profile,
        normalization_model=args.normalization_model,
        tolerance=args.tolerance,
        contrast_tolerance=args.contrast_tolerance,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: time={float(audit.time):.8e}, "
        f"selected_ky={float(audit.selected_ky):.8e}, "
        f"direct_term_vii_error={float(audit.direct_term_vii_error):.8e}, "
        f"conjugate_delta={float(audit.conjugate_term_vii_delta):.8e}, "
        f"negative_delta={float(audit.negative_field_term_vii_delta):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    status = "PASS" if bool(audit.passed) else "OPEN"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "metric",
                "value",
                "time",
                "selected_ky",
                "gkw_krho",
                "ixplus",
                "ixminus",
                "status",
                "notes",
            )
        )
        metrics = (
            ("direct_field_roundtrip_error", audit.direct_field_roundtrip_error),
            ("conjugate_field_pullback_error", audit.conjugate_field_pullback_error),
            ("direct_term_vii_error", audit.direct_term_vii_error),
            ("packed_term_vii_error", audit.packed_term_vii_error),
            ("conjugate_term_vii_delta", audit.conjugate_term_vii_delta),
            ("negative_field_term_vii_delta", audit.negative_field_term_vii_delta),
            ("positive_ky_error", audit.positive_ky_error),
        )
        ixplus = " ".join(str(int(value)) for value in np.asarray(audit.ixplus))
        ixminus = " ".join(str(int(value)) for value in np.asarray(audit.ixminus))
        for name, value in metrics:
            writer.writerow(
                (
                    name,
                    float(value),
                    float(audit.time),
                    float(audit.selected_ky),
                    float(audit.gkw_krho),
                    ixplus,
                    ixminus,
                    status,
                    audit.notes,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_term_vii_mode_packing_audit.csv"),
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
    parser.add_argument("--tolerance", type=float, default=5.0e-11)
    parser.add_argument("--contrast-tolerance", type=float, default=1.0e-12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
