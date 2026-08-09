"""Audit velocity-slice global phase alignment against patched GKW ``cosin2`` slices.

Run with:

    uv run --extra dev python examples/audit_cyclone_cosin2_velocity_phase.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from jax_fluxtube_gk import run_cyclone_base_case_cosin2_velocity_phase_audit


def main() -> None:
    args = _parse_args()
    audit = run_cyclone_base_case_cosin2_velocity_phase_audit(
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
    best_phase = int(audit.best_phase_variant_index)
    best_scaled = int(audit.best_scaled_variant_index)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "variant",
                "unit_phase_real",
                "unit_phase_imag",
                "unit_phase_angle",
                "phase_aligned_max_abs_error",
                "phase_aligned_l2_error",
                "phase_aligned_relative_l2_error",
                "complex_scale_real",
                "complex_scale_imag",
                "complex_scale_abs",
                "scaled_max_abs_error",
                "scaled_l2_error",
                "scaled_relative_l2_error",
                "is_best_phase",
                "is_best_scaled",
            ),
        )
        writer.writeheader()
        for index, name in enumerate(audit.variant_names):
            scale = complex(audit.complex_scale_factors[index])
            phase = complex(audit.unit_phase_factors[index])
            writer.writerow(
                {
                    "variant": name,
                    "unit_phase_real": f"{phase.real:.16e}",
                    "unit_phase_imag": f"{phase.imag:.16e}",
                    "unit_phase_angle": f"{float(audit.unit_phase_angles[index]):.16e}",
                    "phase_aligned_max_abs_error": (
                        f"{float(audit.phase_aligned_max_abs_errors[index]):.16e}"
                    ),
                    "phase_aligned_l2_error": (
                        f"{float(audit.phase_aligned_l2_errors[index]):.16e}"
                    ),
                    "phase_aligned_relative_l2_error": (
                        f"{float(audit.phase_aligned_relative_l2_errors[index]):.16e}"
                    ),
                    "complex_scale_real": f"{scale.real:.16e}",
                    "complex_scale_imag": f"{scale.imag:.16e}",
                    "complex_scale_abs": f"{abs(scale):.16e}",
                    "scaled_max_abs_error": f"{float(audit.scaled_max_abs_errors[index]):.16e}",
                    "scaled_l2_error": f"{float(audit.scaled_l2_errors[index]):.16e}",
                    "scaled_relative_l2_error": (
                        f"{float(audit.scaled_relative_l2_errors[index]):.16e}"
                    ),
                    "is_best_phase": index == best_phase,
                    "is_best_scaled": index == best_scaled,
                }
            )

    print(
        "OPEN: "
        f"best_phase={audit.variant_names[best_phase]}, "
        f"best_phase_max_abs_error={float(audit.best_phase_max_abs_error):.8e}, "
        f"direct_phase_max_abs_error={float(audit.direct_phase_max_abs_error):.8e}, "
        "reverse_vpar_phase_max_abs_error="
        f"{float(audit.reverse_vpar_phase_max_abs_error):.8e}, "
        f"best_scaled={audit.variant_names[best_scaled]}, "
        f"best_scaled_max_abs_error={float(audit.best_scaled_max_abs_error):.8e}"
    )
    print(args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit global phase alignment for GKW cosin2 velocity slices."
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
        default=Path("figures/gkw_cosin2_cyclone_velocity_phase_audit.csv"),
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
