"""Audit the selected-ky velocity-space slice against patched GKW ``distr*.dat``.

Run from the repository root:

    uv run --extra dev python examples/audit_cyclone_cosin2_velocity_slice.py
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

from stellarator_gk import (
    audit_cyclone_velocity_space_slice,
    audit_velocity_space_slice_conventions,
    load_gkw_velocity_space_slice,
    run_cyclone_base_case_velocity_space_slice,
)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.conventions_output.parent.mkdir(parents=True, exist_ok=True)
    observed = run_cyclone_base_case_velocity_space_slice(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        initial_profile="cosine2",
        parallel_derivative_model=args.parallel_derivative_model,
        normalization_model=args.normalization_model,
    )
    reference = load_gkw_velocity_space_slice(
        args.gkw_distr1,
        args.gkw_distr2,
        args.gkw_distr3,
        args.gkw_distr4,
        time_path=args.gkw_time,
        source="gkw:cosin2:distr*.dat",
        notes="patched selected-ky production-control cosin2 final output",
    )
    audit = audit_cyclone_velocity_space_slice(
        observed,
        reference,
        tolerance=args.tolerance,
        grid_tolerance=args.grid_tolerance,
    )
    convention_audit = audit_velocity_space_slice_conventions(observed, reference)
    _write_csv(args.output, audit)
    _write_convention_csv(args.conventions_output, convention_audit)
    status = "PASS" if bool(audit.passed) else "OPEN"
    best_name = convention_audit.variant_names[int(convention_audit.best_variant_index)]
    print(
        f"{status}: complex_max_abs_error={float(audit.complex_max_abs_error):.8e}, "
        f"complex_relative_l2_error={float(audit.complex_relative_l2_error):.8e}, "
        f"vpar_error={float(audit.vpar_error):.8e}, "
        f"vperp_error={float(audit.vperp_error):.8e}, "
        f"best_convention={best_name}"
    )
    print(args.output)
    print(args.conventions_output)


def _write_csv(path: Path, audit) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("metric", "value"))
        for name in (
            "vpar_error",
            "vperp_error",
            "real_max_abs_error",
            "imag_max_abs_error",
            "complex_max_abs_error",
            "complex_l2_error",
            "complex_relative_l2_error",
            "observed_l2_norm",
            "reference_l2_norm",
            "time_error",
            "peak_z_error",
            "passed",
        ):
            value = getattr(audit, name)
            writer.writerow((name, bool(value) if name == "passed" else float(value)))


def _write_convention_csv(path: Path, audit) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("variant", "max_abs_error", "l2_error", "is_best"))
        best = int(audit.best_variant_index)
        for index, name in enumerate(audit.variant_names):
            writer.writerow(
                (
                    name,
                    float(audit.max_abs_errors[index]),
                    float(audit.l2_errors[index]),
                    index == best,
                )
            )
        for name, max_error, l2_error in (
            ("even_part_error", audit.even_max_abs_error, audit.even_l2_error),
            (
                "odd_same_sign_error",
                audit.odd_same_sign_max_abs_error,
                audit.odd_same_sign_l2_error,
            ),
            (
                "odd_opposite_sign_error",
                audit.odd_opposite_sign_max_abs_error,
                audit.odd_opposite_sign_l2_error,
            ),
        ):
            writer.writerow((name, float(max_error), float(l2_error), False))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gkw-distr1",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_distr1.dat"),
    )
    parser.add_argument(
        "--gkw-distr2",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_distr2.dat"),
    )
    parser.add_argument(
        "--gkw-distr3",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_distr3.dat"),
    )
    parser.add_argument(
        "--gkw-distr4",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_distr4.dat"),
    )
    parser.add_argument(
        "--gkw-time",
        type=Path,
        default=Path("fixtures/gkw_cyclone_selected_ky_cosin2_time.dat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gkw_cosin2_cyclone_velocity_slice_audit.csv"),
    )
    parser.add_argument(
        "--conventions-output",
        type=Path,
        default=Path("figures/gkw_cosin2_cyclone_velocity_slice_conventions.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=2.0e-2)
    parser.add_argument("--grid-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("gkw_upwind", "gkw_igh"),
        default="gkw_igh",
    )
    parser.add_argument(
        "--normalization-model",
        choices=("weighted", "gkw_unweighted"),
        default="gkw_unweighted",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
