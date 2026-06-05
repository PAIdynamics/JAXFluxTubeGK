"""Run the Cyclone multi-ky scan convention calibration audit.

Reduced smoke run from the repository root:

    uv run --extra dev python examples/run_cyclone_ky_scan_calibration.py

Production-control scan:

    uv run --extra dev python examples/run_cyclone_ky_scan_calibration.py \
        --profile production-control --calibrate-reference-growth

GX s-alpha input-control scan:

    uv run --extra dev python examples/run_cyclone_ky_scan_calibration.py \
        --profile gx-salpha-input --target-convention gx-salpha \
        --ky-input-conventions internal_krho --calibrate-reference-growth
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from stellarator_gk import (
    cyclone_base_case_growth_target,
    gx_salpha_cyclone_growth_target,
    run_production_control_cyclone_ky_scan_convention_audit,
    write_cyclone_ky_scan_convention_audit_csv,
)


def main() -> None:
    args = _parse_args()
    profile_defaults = _profile_defaults(args.profile)
    target = _target(args.target_convention)
    nperiod = args.nperiod
    if nperiod is None and args.target_convention == "gx-salpha":
        nperiod = 2
    audit = run_production_control_cyclone_ky_scan_convention_audit(
        ky_values=_parse_float_tuple(args.ky_values, allow_all=True),
        ky_input_conventions=_parse_str_tuple(args.ky_input_conventions),
        growth_diagnostics=_parse_str_tuple(args.growth_diagnostics),
        normalization_models=_parse_str_tuple(args.normalization_models),
        observed_frequency_signs=_parse_float_tuple(args.observed_frequency_signs),
        observed_frequency_scales=_parse_float_tuple(args.observed_frequency_scales),
        n_z=args.n_z or profile_defaults["n_z"],
        n_vpar=args.n_vpar or profile_defaults["n_vpar"],
        n_mu=args.n_mu or profile_defaults["n_mu"],
        steps_per_window=args.steps_per_window or profile_defaults["steps_per_window"],
        n_windows=args.n_windows or profile_defaults["n_windows"],
        nperiod=5 if nperiod is None else nperiod,
        growth_window_fraction=args.growth_window_fraction,
        growth_tolerance=args.growth_tolerance,
        frequency_tolerance=args.frequency_tolerance,
        profile_tolerance=args.profile_tolerance,
        require_frequency=not args.ignore_frequency,
        require_profile=args.require_profile,
        parallel_derivative_model=args.parallel_derivative_model,
        initial_profile=args.initial_profile,
        target=target,
        calibrate_reference_growth=args.calibrate_reference_growth,
        reference_calibration_ky=args.reference_calibration_ky,
        reference_calibration_growth=args.reference_calibration_growth,
        scale_reference_frequency_with_growth=args.scale_reference_frequency_with_growth,
    )
    write_cyclone_ky_scan_convention_audit_csv(args.output, audit)
    best = int(np.asarray(audit.best_index))
    status = "PASS" if bool(audit.passed) else "OPEN"
    print(
        f"{status}: best={audit.candidate_names[best]}, "
        f"max_growth_error={float(audit.max_growth_errors[best]):.8e}, "
        f"max_frequency_error={float(audit.max_frequency_errors[best]):.8e}, "
        f"combined_error={float(audit.combined_errors[best]):.8e}"
    )
    print(args.output)


def _profile_defaults(profile: str) -> dict[str, int]:
    if profile == "reduced-smoke":
        return {"n_z": 8, "n_vpar": 6, "n_mu": 4, "steps_per_window": 2, "n_windows": 2}
    if profile == "production-control":
        return {"n_z": 48, "n_vpar": 32, "n_mu": 8, "steps_per_window": 20, "n_windows": 80}
    if profile == "gx-salpha-input":
        return {"n_z": 96, "n_vpar": 48, "n_mu": 16, "steps_per_window": 20, "n_windows": 80}
    raise ValueError(f"unsupported profile {profile!r}")


def _target(convention: str):
    if convention == "gkw":
        return cyclone_base_case_growth_target()
    if convention == "gx-salpha":
        return gx_salpha_cyclone_growth_target()
    raise ValueError(f"unsupported target convention {convention!r}")


def _parse_float_tuple(value: str, *, allow_all: bool = False):
    if allow_all and value.strip().lower() == "all":
        return None
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("reduced-smoke", "production-control", "gx-salpha-input"),
        default="reduced-smoke",
    )
    parser.add_argument("--ky-values", default="0.3,0.5")
    parser.add_argument("--target-convention", choices=("gkw", "gx-salpha"), default="gkw")
    parser.add_argument("--ky-input-conventions", default="k_theta_rhos")
    parser.add_argument("--growth-diagnostics", default="late_fit,late_mean_window")
    parser.add_argument("--normalization-models", default="gkw_unweighted")
    parser.add_argument("--observed-frequency-signs", default="1,-1")
    parser.add_argument("--observed-frequency-scales", default="1")
    parser.add_argument("--n-z", type=int)
    parser.add_argument("--n-vpar", type=int)
    parser.add_argument("--n-mu", type=int)
    parser.add_argument("--nperiod", type=int)
    parser.add_argument("--steps-per-window", type=int)
    parser.add_argument("--n-windows", type=int)
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--growth-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--frequency-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--profile-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--ignore-frequency", action="store_true")
    parser.add_argument("--require-profile", action="store_true")
    parser.add_argument("--calibrate-reference-growth", action="store_true")
    parser.add_argument("--reference-calibration-ky", type=float)
    parser.add_argument("--reference-calibration-growth", type=float)
    parser.add_argument("--scale-reference-frequency-with-growth", action="store_true")
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("gkw_upwind", "gkw_igh"),
        default="gkw_igh",
    )
    parser.add_argument(
        "--initial-profile",
        choices=("cosine", "cosine2"),
        default="cosine2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/cyclone_ky_scan_convention_audit.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
