"""Audit Term VII field-variable conventions against patched GKW ``cosin2`` slices.

Run with:

    uv run --extra dev python examples/audit_cyclone_cosin2_term_vii_field_conventions.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from jax_fluxtube_gk import run_cyclone_base_case_cosin2_term_vii_field_convention_audit


def main() -> None:
    args = _parse_args()
    audit = run_cyclone_base_case_cosin2_term_vii_field_convention_audit(
        gkw_distr1_path=args.gkw_distr1,
        gkw_distr2_path=args.gkw_distr2,
        gkw_distr3_path=args.gkw_distr3,
        gkw_distr4_path=args.gkw_distr4,
        gkw_time_path=args.gkw_time,
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_direct = int(audit.best_direct_variant_index)
    best_layout = int(audit.best_layout_variant_index)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "term_v_phi_variant",
                "term_vii_phi_variant",
                "term_viii_phi_variant",
                "direct_max_abs_error",
                "direct_l2_error",
                "direct_relative_l2_error",
                "best_layout",
                "best_layout_max_abs_error",
                "best_layout_l2_error",
                "peak_z",
                "is_best_direct",
                "is_best_layout",
            ),
        )
        writer.writeheader()
        for index, name in enumerate(audit.variant_names):
            writer.writerow(
                {
                    "variant": name,
                    "term_v_phi_variant": audit.term_v_phi_variants[index],
                    "term_vii_phi_variant": audit.term_vii_phi_variants[index],
                    "term_viii_phi_variant": audit.term_viii_phi_variants[index],
                    "direct_max_abs_error": f"{float(audit.direct_max_abs_errors[index]):.16e}",
                    "direct_l2_error": f"{float(audit.direct_l2_errors[index]):.16e}",
                    "direct_relative_l2_error": (
                        f"{float(audit.direct_relative_l2_errors[index]):.16e}"
                    ),
                    "best_layout": audit.best_layout_names[index],
                    "best_layout_max_abs_error": (
                        f"{float(audit.best_layout_max_abs_errors[index]):.16e}"
                    ),
                    "best_layout_l2_error": f"{float(audit.best_layout_l2_errors[index]):.16e}",
                    "peak_z": f"{float(audit.peak_z_values[index]):.16e}",
                    "is_best_direct": index == best_direct,
                    "is_best_layout": index == best_layout,
                }
            )

    print(
        "OPEN: "
        f"best_direct={audit.variant_names[best_direct]}, "
        f"direct_max_abs_error={float(audit.direct_max_abs_errors[best_direct]):.8e}, "
        f"baseline_direct_max_abs_error={float(audit.baseline_direct_max_abs_error):.8e}, "
        "term_vii_only_direct_max_abs_error="
        f"{float(audit.term_vii_only_direct_max_abs_error):.8e}, "
        f"all_field_direct_max_abs_error={float(audit.all_field_direct_max_abs_error):.8e}, "
        "term_vii_only_improvement_factor="
        f"{float(audit.term_vii_only_improvement_factor):.6f}"
    )
    print(args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Term VII field-variable conventions against GKW distr*.dat."
    )
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
        default=Path("figures/gkw_cosin2_cyclone_term_vii_field_convention_audit.csv"),
    )
    parser.add_argument("--n-z", type=int, default=48)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--n-windows", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    np.set_printoptions(precision=8)
    main()
