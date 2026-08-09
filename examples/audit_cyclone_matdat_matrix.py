"""Audit reduced Cyclone residuals against GKW matdat matrix conventions.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_matdat_matrix.py
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

from jax_fluxtube_gk import run_cyclone_base_case_matdat_matrix_audit


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = run_cyclone_base_case_matdat_matrix_audit(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        initial_profile=args.initial_profile,
        tolerance=args.tolerance,
        nonzero_threshold=args.nonzero_threshold,
        max_size=args.max_size,
    )
    _write_csv(args.output, audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: n_state={audit.n_state}, n_nonzero={audit.n_nonzero}, "
        f"matrix_action_error={float(audit.matrix_action_error):.8e}, "
        f"source_max_abs={float(audit.source_max_abs):.8e}, "
        f"explicit_delta_error={float(audit.explicit_delta_error):.8e}, "
        f"compressed_action_error={float(audit.compressed_action_error):.8e}, "
        f"complex_real_split_error={float(audit.complex_real_split_error):.8e}"
    )
    print(args.output)


def _write_csv(path: Path, audit) -> None:
    status = "PASS" if bool(audit.passed) else "OPEN"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "n_state",
                "n_nonzero",
                "n_duplicate_triplets",
                "n_real_entries",
                "n_complex_entries",
                "matrix_action_error",
                "source_max_abs",
                "explicit_delta_error",
                "compressed_action_error",
                "complex_real_split_error",
                "linearity_error",
                "max_abs_matrix_entry",
                "status",
                "notes",
            )
        )
        writer.writerow(
            (
                audit.n_state,
                audit.n_nonzero,
                audit.n_duplicate_triplets,
                audit.n_real_entries,
                audit.n_complex_entries,
                float(audit.matrix_action_error),
                float(audit.source_max_abs),
                float(audit.explicit_delta_error),
                float(audit.compressed_action_error),
                float(audit.complex_real_split_error),
                float(audit.linearity_error),
                float(audit.max_abs_matrix_entry),
                status,
                audit.notes,
            )
        )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_matdat_matrix_audit.csv"),
    )
    parser.add_argument("--n-z", type=int, default=8)
    parser.add_argument("--n-vpar", type=int, default=6)
    parser.add_argument("--n-mu", type=int, default=4)
    parser.add_argument("--initial-profile", choices=("cosine2", "cosine"), default="cosine")
    parser.add_argument("--tolerance", type=float, default=5.0e-11)
    parser.add_argument("--nonzero-threshold", type=float, default=1.0e-14)
    parser.add_argument("--max-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    main()
