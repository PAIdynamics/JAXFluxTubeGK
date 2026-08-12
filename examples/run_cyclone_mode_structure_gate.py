"""Run the Cyclone multi-ky complex mode-structure gate.

This command compares a solver-produced per-ky fixture against either a
portable fixture CSV, a reduced GX-style Hermite-Laguerre moment-RHS fixture,
or a retained GX full-field .big.nc diagnostic.  A reduced smoke run can
compare the solver fixture to itself:

    uv run --extra dev python examples/run_cyclone_mode_structure_gate.py \
        --self-check --profile reduced-smoke

With a real GX run:

    uv run --extra dev python examples/run_cyclone_mode_structure_gate.py \
        --gx-big-output path/to/itg_salpha_adiabatic_electrons.big.nc \
        --gx-growth-output path/to/itg_salpha_adiabatic_electrons.out.nc \
        --profile gx-salpha-input --target-convention gx-salpha \
        --ky-input-convention internal_krho --gx-z-coordinate theta_over_2pi \
        --resample-reference-to-solver-z --periodic-z

With a fixture exported from any external code:

    uv run --extra dev python examples/run_cyclone_mode_structure_gate.py \
        --reference-fixture fixtures/external_mode_structure.csv \
        --profile production-control --require-profile

With the built-in reduced GX-style moment-RHS reference:

    uv run --extra dev python examples/run_cyclone_mode_structure_gate.py \
        --reference-moment-rhs --profile reduced-smoke --require-profile
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_fluxtube_gk.validation.cyclone_gkw import (
    calibrate_gx_growth_rate_reference_to_target,
    compare_per_ky_mode_structure_fixtures,
    cyclone_base_case_growth_target,
    evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures,
    gx_salpha_cyclone_growth_target,
    run_cyclone_base_case_mode_structure_fixture,
    run_s_alpha_moment_rhs_mode_structure_fixture,
)
from jax_fluxtube_gk.validation.fixture_io import (
    load_gx_growth_rate_reference,
    load_gx_mode_structure_fixture,
    load_per_ky_mode_structure_fixture_csv,
    resample_per_ky_mode_structure_fixture,
    write_per_ky_mode_structure_fixture_csv,
)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_defaults = _profile_defaults(args.profile)
    target = _target(args.target_convention)
    reference = None
    if args.gx_growth_output is not None:
        reference = load_gx_growth_rate_reference(args.gx_growth_output)
        if args.calibrate_reference_growth:
            reference = calibrate_gx_growth_rate_reference_to_target(
                reference,
                target=target,
                target_ky=args.reference_calibration_ky,
                target_growth=args.reference_calibration_growth,
                scale_frequency=args.scale_reference_frequency_with_growth,
            )
    nperiod = args.nperiod
    if nperiod is None and args.target_convention == "gx-salpha":
        nperiod = 2

    solver = run_cyclone_base_case_mode_structure_fixture(
        reference=reference,
        ky_values=_parse_float_tuple(args.ky_values),
        n_z=args.n_z or profile_defaults["n_z"],
        n_vpar=args.n_vpar or profile_defaults["n_vpar"],
        n_mu=args.n_mu or profile_defaults["n_mu"],
        nperiod=5 if nperiod is None else nperiod,
        steps_per_window=args.steps_per_window or profile_defaults["steps_per_window"],
        n_windows=args.n_windows or profile_defaults["n_windows"],
        growth_window_fraction=args.growth_window_fraction,
        growth_diagnostic=args.growth_diagnostic,
        ky_input_convention=args.ky_input_convention,
        observed_frequency_sign=args.observed_frequency_sign,
        observed_frequency_scale=args.observed_frequency_scale,
        growth_tolerance=args.growth_tolerance,
        parallel_derivative_model=args.parallel_derivative_model,
        normalization_model=args.normalization_model,
        initial_profile=args.initial_profile,
        target=target,
    )
    reference_source_count = sum(
        (
            bool(args.self_check),
            args.reference_fixture is not None,
            bool(args.reference_moment_rhs),
            args.gx_big_output is not None,
        )
    )
    if reference_source_count != 1:
        raise ValueError(
            "choose exactly one reference source: --self-check, "
            "--reference-fixture, --reference-moment-rhs, or --gx-big-output"
        )

    if args.self_check:
        reference_fixture = replace(solver, source="self-check reference")
    elif args.reference_fixture is not None:
        reference_fixture = load_per_ky_mode_structure_fixture_csv(
            args.reference_fixture,
            metadata=(
                ("format", "per_ky_mode_structure_csv"),
                ("path", str(args.reference_fixture)),
            ),
        )
        if args.resample_reference_to_solver_z:
            reference_fixture = resample_per_ky_mode_structure_fixture(
                reference_fixture,
                solver.z,
                periodic=args.periodic_z,
                period=args.z_period,
            )
    elif args.reference_moment_rhs:
        moment_defaults = _moment_profile_defaults(args.profile)
        reference_fixture = run_s_alpha_moment_rhs_mode_structure_fixture(
            ky_values=_parse_float_tuple(args.ky_values),
            n_z=args.moment_n_z or args.n_z or moment_defaults["n_z"],
            n_hermite=args.moment_n_hermite or moment_defaults["n_hermite"],
            n_laguerre=args.moment_n_laguerre or moment_defaults["n_laguerre"],
            nperiod=5 if nperiod is None else nperiod,
            dt=args.moment_dt,
            steps_per_window=args.moment_steps_per_window
            or moment_defaults["steps_per_window"],
            n_windows=args.moment_n_windows or moment_defaults["n_windows"],
            growth_window_fraction=args.moment_growth_window_fraction,
            density_gradient=args.moment_density_gradient,
            temperature_gradient=args.moment_temperature_gradient,
            tau=args.moment_tau,
            magnetic_shear=args.moment_magnetic_shear,
            drift_scale=args.moment_drift_scale,
            drive_scale=args.moment_drive_scale,
            streaming_scale=args.moment_streaming_scale,
            nu_hyper_m=args.moment_nu_hyper_m,
            p_hyper_m=args.moment_p_hyper_m,
            normalize_each_window=not args.moment_no_normalize_each_window,
            initial_profile=args.moment_initial_profile,
            initial_width=args.moment_initial_width,
        )
        if args.resample_reference_to_solver_z:
            reference_fixture = resample_per_ky_mode_structure_fixture(
                reference_fixture,
                solver.z,
                periodic=args.periodic_z,
                period=args.z_period,
            )
    else:
        reference_fixture = load_gx_mode_structure_fixture(
            args.gx_big_output,
            growth_reference_path=args.gx_growth_output,
            ikx=args.ikx,
            time_index=args.time_index,
            ky_values=_parse_float_tuple(args.ky_values),
            average_fraction=args.average_fraction,
            drop_zonal=not args.keep_zonal,
            z_scale=_gx_z_scale(args.gx_z_coordinate),
        )
        if args.resample_reference_to_solver_z:
            reference_fixture = resample_per_ky_mode_structure_fixture(
                reference_fixture,
                solver.z,
                periodic=args.periodic_z,
                period=args.z_period,
            )

    gate = evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures(
        solver,
        reference_fixture,
        growth_tolerance=args.growth_tolerance,
        frequency_tolerance=args.frequency_tolerance,
        profile_tolerance=args.profile_tolerance,
        ky_tolerance=args.ky_tolerance,
        z_tolerance=args.z_tolerance,
        require_frequency=not args.ignore_frequency,
        require_profile=args.require_profile,
    )
    comparison = compare_per_ky_mode_structure_fixtures(
        solver,
        reference_fixture,
        growth_tolerance=args.growth_tolerance,
        frequency_tolerance=args.frequency_tolerance,
        phi_tolerance=args.profile_tolerance,
        ky_tolerance=args.ky_tolerance,
        z_tolerance=args.z_tolerance,
        require_frequency=not args.ignore_frequency,
        require_phi=args.require_profile,
    )

    solver_path = output_dir / "solver_mode_structure_fixture.csv"
    reference_path = output_dir / "reference_mode_structure_fixture.csv"
    report_path = output_dir / "mode_structure_gate.csv"
    write_per_ky_mode_structure_fixture_csv(solver_path, solver)
    write_per_ky_mode_structure_fixture_csv(reference_path, reference_fixture)
    _write_report_csv(report_path, comparison, gate)

    status = "PASS" if bool(gate.passed) else "OPEN"
    print(
        f"{status}: max_growth_error={float(gate.max_growth_error):.8e}, "
        f"max_frequency_error={float(gate.max_frequency_error):.8e}, "
        f"max_profile_error={float(gate.max_profile_error):.8e}"
    )
    print(report_path)


def _write_report_csv(path: Path, comparison, gate) -> None:
    ky = np.asarray(comparison.ky, dtype=float)
    matched_ky = np.asarray(comparison.matched_reference_ky, dtype=float)
    fields = {
        "solver_ky": np.asarray(gate.solver_ky, dtype=float),
        "observed_growth": np.asarray(comparison.observed_growth, dtype=float),
        "reference_growth": np.asarray(comparison.reference_growth, dtype=float),
        "growth_error": np.asarray(comparison.growth_error, dtype=float),
        "observed_frequency": np.asarray(comparison.observed_frequency, dtype=float),
        "reference_frequency": np.asarray(comparison.reference_frequency, dtype=float),
        "frequency_error": np.asarray(comparison.frequency_error, dtype=float),
        "phi_direct_error": np.asarray(comparison.phi_direct_error, dtype=float),
        "phi_phase_aligned_error": np.asarray(comparison.phi_phase_aligned_error, dtype=float),
        "growth_passed": np.asarray(comparison.growth_passed, dtype=bool),
        "frequency_passed": np.asarray(comparison.frequency_passed, dtype=bool),
        "phi_passed": np.asarray(comparison.phi_passed, dtype=bool),
    }
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ky",
                "matched_reference_ky",
                "solver_ky",
                "observed_growth",
                "reference_growth",
                "growth_error",
                "observed_frequency",
                "reference_frequency",
                "frequency_error",
                "phi_direct_error",
                "phi_phase_aligned_error",
                "growth_passed",
                "frequency_passed",
                "phi_passed",
                "scan_gate_passed",
            )
        )
        for index, ky_value in enumerate(ky):
            writer.writerow(
                (
                    ky_value,
                    matched_ky[index],
                    fields["solver_ky"][index],
                    fields["observed_growth"][index],
                    fields["reference_growth"][index],
                    fields["growth_error"][index],
                    fields["observed_frequency"][index],
                    fields["reference_frequency"][index],
                    fields["frequency_error"][index],
                    fields["phi_direct_error"][index],
                    fields["phi_phase_aligned_error"][index],
                    bool(fields["growth_passed"][index]),
                    bool(fields["frequency_passed"][index]),
                    bool(fields["phi_passed"][index]),
                    bool(gate.passed),
                )
            )


def _profile_defaults(profile: str) -> dict[str, int]:
    if profile == "reduced-smoke":
        return {"n_z": 8, "n_vpar": 6, "n_mu": 4, "steps_per_window": 1, "n_windows": 2}
    if profile == "production-control":
        return {"n_z": 48, "n_vpar": 32, "n_mu": 8, "steps_per_window": 20, "n_windows": 80}
    if profile == "gx-salpha-input":
        return {"n_z": 96, "n_vpar": 48, "n_mu": 16, "steps_per_window": 20, "n_windows": 80}
    raise ValueError(f"unsupported profile {profile!r}")


def _moment_profile_defaults(profile: str) -> dict[str, int]:
    if profile == "reduced-smoke":
        return {"n_z": 8, "n_hermite": 5, "n_laguerre": 4, "steps_per_window": 1, "n_windows": 3}
    if profile == "production-control":
        return {
            "n_z": 48,
            "n_hermite": 32,
            "n_laguerre": 8,
            "steps_per_window": 5,
            "n_windows": 40,
        }
    if profile == "gx-salpha-input":
        return {
            "n_z": 96,
            "n_hermite": 48,
            "n_laguerre": 16,
            "steps_per_window": 5,
            "n_windows": 40,
        }
    raise ValueError(f"unsupported profile {profile!r}")


def _target(convention: str):
    if convention == "gkw":
        return cyclone_base_case_growth_target()
    if convention == "gx-salpha":
        return gx_salpha_cyclone_growth_target()
    raise ValueError(f"unsupported target convention {convention!r}")


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _gx_z_scale(name: str) -> float:
    if name == "theta":
        return 1.0
    if name == "theta_over_2pi":
        return 1.0 / (2.0 * np.pi)
    raise ValueError(f"unsupported GX z-coordinate convention {name!r}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gx-big-output", type=Path)
    parser.add_argument("--gx-growth-output", type=Path)
    parser.add_argument("--reference-fixture", type=Path)
    parser.add_argument("--reference-moment-rhs", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--profile", choices=("reduced-smoke", "production-control", "gx-salpha-input"), default="reduced-smoke")
    parser.add_argument("--target-convention", choices=("gkw", "gx-salpha"), default="gkw")
    parser.add_argument("--ky-values", default="0.3,0.5")
    parser.add_argument("--ky-input-convention", choices=("k_theta_rhos", "internal_krho"), default="k_theta_rhos")
    parser.add_argument("--n-z", type=int)
    parser.add_argument("--n-vpar", type=int)
    parser.add_argument("--n-mu", type=int)
    parser.add_argument("--nperiod", type=int)
    parser.add_argument("--steps-per-window", type=int)
    parser.add_argument("--n-windows", type=int)
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--growth-diagnostic", choices=("late_fit", "late_mean_window"), default="late_fit")
    parser.add_argument("--growth-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--frequency-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--profile-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--ky-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--z-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--ignore-frequency", action="store_true")
    parser.add_argument("--require-profile", action="store_true")
    parser.add_argument("--calibrate-reference-growth", action="store_true")
    parser.add_argument("--reference-calibration-ky", type=float)
    parser.add_argument("--reference-calibration-growth", type=float)
    parser.add_argument("--scale-reference-frequency-with-growth", action="store_true")
    parser.add_argument("--observed-frequency-sign", type=float, default=1.0)
    parser.add_argument("--observed-frequency-scale", type=float, default=1.0)
    parser.add_argument("--ikx", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--average-fraction", type=float, default=0.5)
    parser.add_argument("--keep-zonal", action="store_true")
    parser.add_argument("--gx-z-coordinate", choices=("theta", "theta_over_2pi"), default="theta_over_2pi")
    parser.add_argument("--resample-reference-to-solver-z", action="store_true")
    parser.add_argument("--periodic-z", action="store_true")
    parser.add_argument("--z-period", type=float)
    parser.add_argument("--parallel-derivative-model", choices=("gkw_upwind", "gkw_igh"), default="gkw_igh")
    parser.add_argument("--normalization-model", choices=("weighted", "gkw_unweighted"), default="gkw_unweighted")
    parser.add_argument("--initial-profile", choices=("cosine", "cosine2"), default="cosine2")
    parser.add_argument("--moment-n-z", type=int)
    parser.add_argument("--moment-n-hermite", type=int)
    parser.add_argument("--moment-n-laguerre", type=int)
    parser.add_argument("--moment-dt", type=float, default=0.02)
    parser.add_argument("--moment-steps-per-window", type=int)
    parser.add_argument("--moment-n-windows", type=int)
    parser.add_argument("--moment-growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--moment-density-gradient", type=float, default=0.8)
    parser.add_argument("--moment-temperature-gradient", type=float, default=2.49)
    parser.add_argument("--moment-tau", type=float, default=1.0)
    parser.add_argument("--moment-magnetic-shear", type=float, default=0.8)
    parser.add_argument("--moment-drift-scale", type=float, default=0.18)
    parser.add_argument("--moment-drive-scale", type=float, default=1.0)
    parser.add_argument("--moment-streaming-scale", type=float, default=1.0)
    parser.add_argument("--moment-nu-hyper-m", type=float, default=1.0)
    parser.add_argument("--moment-p-hyper-m", type=int)
    parser.add_argument("--moment-no-normalize-each-window", action="store_true")
    parser.add_argument(
        "--moment-initial-profile",
        choices=("gaussian", "cosine", "cosine2"),
        default="gaussian",
    )
    parser.add_argument("--moment-initial-width", type=float, default=0.35)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/cyclone_mode_structure_gate"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
