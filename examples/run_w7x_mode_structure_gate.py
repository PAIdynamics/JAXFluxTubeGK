"""Run the W7-X external mode-structure parity gate.

The command compares the solver W7-X per-ky fixture against a matched external
GX/GKW/GS2/stella fixture.  If the external fixture is missing, it writes a
machine-readable pending report that points to the prepared GX run workflow.

Examples from the repository root:

    uv run python examples/run_w7x_mode_structure_gate.py

    uv run python examples/run_w7x_mode_structure_gate.py \
        --reference-fixture fixtures/w7x_itg_external_mode_structure_fixture.csv \
        --run-solver --solver-preset gx-production-shape \
        --resample-reference-to-observed-z
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from stellarator_gk import (
    PerKyModeStructureFixture,
    compare_per_ky_mode_structure_fixtures,
    load_per_ky_mode_structure_fixture_csv,
    resample_per_ky_mode_structure_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVED = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"
DEFAULT_REFERENCE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_OUTPUT_DIR = ROOT / "figures/w7x_itg_mode_structure_gate"
DEFAULT_GX_PREP_METADATA = (
    ROOT / "fixtures/gx_w7x_mode_structure_run/mode_structure_run_metadata.json"
)
DEFAULT_EIK_REFERENCE = (
    ROOT
    / "relevant-codes/gx/geometry_modules/vmec/tests/"
    "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.reference_fixture.exists():
        _write_pending_report(args)
        print(f"PENDING: missing external W7-X fixture {args.reference_fixture}")
        print(args.output_dir / "gate_status.json")
        return 2 if args.require_reference else 0

    observed_path = _observed_fixture_path(args)
    observed = load_per_ky_mode_structure_fixture_csv(observed_path)
    reference = load_per_ky_mode_structure_fixture_csv(args.reference_fixture)
    ky_values = _parse_float_tuple(args.ky_values)
    observed = _select_ky_values(observed, ky_values, args.ky_tolerance)
    reference = _select_ky_values(reference, ky_values, args.ky_tolerance)

    if args.resample_reference_to_observed_z and args.resample_observed_to_reference_z:
        raise ValueError("choose at most one z-resampling direction")
    if args.resample_reference_to_observed_z:
        reference = resample_per_ky_mode_structure_fixture(
            reference,
            observed.z,
            periodic=args.periodic_z,
            period=args.z_period,
        )
    if args.resample_observed_to_reference_z:
        observed = resample_per_ky_mode_structure_fixture(
            observed,
            reference.z,
            periodic=args.periodic_z,
            period=args.z_period,
        )

    comparison = compare_per_ky_mode_structure_fixtures(
        observed,
        reference,
        growth_tolerance=args.growth_tolerance,
        frequency_tolerance=args.frequency_tolerance,
        phi_tolerance=args.profile_tolerance,
        ky_tolerance=args.ky_tolerance,
        z_tolerance=args.z_tolerance,
        require_frequency=not args.ignore_frequency,
        require_phi=not args.no_require_profile,
    )
    _write_comparison_outputs(args, observed_path, comparison)
    status = "PASS" if bool(comparison.passed) else "OPEN"
    print(
        f"{status}: max_growth_error={float(comparison.max_growth_error):.8e}, "
        f"max_frequency_error={float(comparison.max_frequency_error):.8e}, "
        f"max_profile_error={float(comparison.max_phi_phase_aligned_error):.8e}"
    )
    print(args.output_dir / "gate_status.json")
    return 0


def _observed_fixture_path(args) -> Path:
    if not args.run_solver:
        if not args.observed_fixture.exists():
            raise FileNotFoundError(args.observed_fixture)
        return args.observed_fixture
    from examples.run_stellarator_linear_scan import main as run_scan

    solver_run_dir = args.output_dir / "solver_run"
    controls = _solver_controls(args)
    run_scan(
        [
            "--geometry-source",
            "eik",
            "--eik-reference",
            str(args.eik_reference),
            "--output-dir",
            str(solver_run_dir),
            "--n-z",
            str(controls["n_z"]),
            "--field-line-periods",
            str(controls["field_line_periods"]),
            "--ky-values",
            args.ky_values,
            "--n-kx",
            str(controls["n_kx"]),
            "--kx-max",
            str(controls["kx_max"]),
            "--ikxspace",
            str(controls["ikxspace"]),
            "--n-vpar",
            str(controls["n_vpar"]),
            "--n-mu",
            str(controls["n_mu"]),
            "--vpar-max",
            str(controls["vpar_max"]),
            "--mu-max",
            str(controls["mu_max"]),
            "--density",
            "1.0",
            "--temperature",
            "1.0",
            "--density-gradient",
            "1.0",
            "--temperature-gradient",
            "3.0",
            "--electron-density",
            "1.0",
            "--electron-temperature",
            "1.0",
            "--dt",
            str(controls["dt"]),
            "--steps-per-window",
            str(controls["steps_per_window"]),
            "--n-windows",
            str(controls["n_windows"]),
            "--growth-diagnostic",
            args.growth_diagnostic,
            "--growth-window-fraction",
            str(args.growth_window_fraction),
        ]
    )
    return solver_run_dir / "mode_structures.csv"


def _solver_controls(args) -> dict[str, float | int]:
    if args.solver_preset == "reduced":
        defaults = {
            "n_z": 33,
            "field_line_periods": 1,
            "n_kx": 3,
            "kx_max": 0.3,
            "ikxspace": 2,
            "n_vpar": 4,
            "n_mu": 4,
            "vpar_max": 2.0,
            "mu_max": 1.5,
            "dt": 0.002,
            "steps_per_window": 1,
            "n_windows": 6,
        }
    else:
        defaults = {
            "n_z": 256,
            "field_line_periods": 6,
            "n_kx": 1,
            "kx_max": 0.0,
            "ikxspace": 1,
            "n_vpar": 16,
            "n_mu": 8,
            "vpar_max": 2.0,
            "mu_max": 1.5,
            "dt": 0.002,
            "steps_per_window": 1,
            "n_windows": 6,
        }
    for name in defaults:
        override = getattr(args, name)
        if override is not None:
            defaults[name] = override
    return defaults


def _select_ky_values(
    fixture: PerKyModeStructureFixture,
    ky_values: tuple[float, ...],
    tolerance: float,
) -> PerKyModeStructureFixture:
    ky = np.asarray(fixture.ky, dtype=float)
    selected_indices = []
    for value in ky_values:
        index = int(np.argmin(np.abs(ky - value)))
        if abs(ky[index] - value) > tolerance:
            raise ValueError(
                f"requested ky={value} is not present in {fixture.source}; "
                f"nearest ky={ky[index]}"
            )
        selected_indices.append(index)
    indices = np.asarray(selected_indices, dtype=int)
    return PerKyModeStructureFixture(
        ky=np.asarray(fixture.ky)[indices],
        z=np.asarray(fixture.z),
        phi=np.asarray(fixture.phi)[indices],
        growth_rate=np.asarray(fixture.growth_rate)[indices],
        frequency=np.asarray(fixture.frequency)[indices],
        source=fixture.source,
        normalization=fixture.normalization,
        metadata=fixture.metadata
        + (
            ("ky_filter", ",".join(str(value) for value in ky_values)),
            ("ky_filter_tolerance", tolerance),
        ),
    )


def _write_pending_report(args) -> None:
    prep = _load_json_if_present(args.gx_prep_metadata)
    _write_json(
        args.output_dir / "gate_status.json",
        {
            "benchmark_name": "w7x_itg_external_mode_structure_gate",
            "status": "pending_external_reference",
            "passed": False,
            "reference_fixture": _display_path(args.reference_fixture),
            "observed_fixture": _display_path(args.observed_fixture),
            "ky_values": _parse_float_tuple(args.ky_values),
            "next_required_artifact": _display_path(args.reference_fixture),
            "gx_prep_metadata": _display_path(args.gx_prep_metadata),
            "gx_export_command": prep.get("export_command") if prep else None,
            "gx_run_command": prep.get("run_command") if prep else None,
        },
    )


def _write_comparison_outputs(args, observed_path: Path, comparison) -> None:
    ky = np.asarray(comparison.ky, dtype=float)
    matched_ky = np.asarray(comparison.matched_reference_ky, dtype=float)
    report_csv = args.output_dir / "mode_structure_gate.csv"
    with report_csv.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ky",
                "matched_reference_ky",
                "observed_growth",
                "reference_growth",
                "growth_error",
                "observed_frequency",
                "reference_frequency",
                "frequency_error",
                "phi_phase_aligned_error",
                "growth_passed",
                "frequency_passed",
                "phi_passed",
            )
        )
        for index, value in enumerate(ky):
            writer.writerow(
                (
                    value,
                    matched_ky[index],
                    float(comparison.observed_growth[index]),
                    float(comparison.reference_growth[index]),
                    float(comparison.growth_error[index]),
                    float(comparison.observed_frequency[index]),
                    float(comparison.reference_frequency[index]),
                    float(comparison.frequency_error[index]),
                    float(comparison.phi_phase_aligned_error[index]),
                    bool(comparison.growth_passed[index]),
                    bool(comparison.frequency_passed[index]),
                    bool(comparison.phi_passed[index]),
                )
            )
    _write_json(
        args.output_dir / "gate_status.json",
        {
            "benchmark_name": "w7x_itg_external_mode_structure_gate",
            "status": "pass" if bool(comparison.passed) else "open",
            "passed": bool(comparison.passed),
            "observed_fixture": _display_path(observed_path),
            "reference_fixture": _display_path(args.reference_fixture),
            "report_csv": _display_path(report_csv),
            "ky_values": _parse_float_tuple(args.ky_values),
            "max_growth_error": float(comparison.max_growth_error),
            "max_frequency_error": float(comparison.max_frequency_error),
            "max_profile_error": float(comparison.max_phi_phase_aligned_error),
            "growth_tolerance": args.growth_tolerance,
            "frequency_tolerance": args.frequency_tolerance,
            "profile_tolerance": args.profile_tolerance,
            "require_frequency": not args.ignore_frequency,
            "require_profile": not args.no_require_profile,
            "run_solver": bool(args.run_solver),
            "solver_preset": args.solver_preset if args.run_solver else None,
        },
    )


def _load_json_if_present(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-fixture", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--observed-fixture", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gx-prep-metadata", type=Path, default=DEFAULT_GX_PREP_METADATA)
    parser.add_argument("--ky-values", default="0.1,0.2,0.3")
    parser.add_argument("--growth-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--frequency-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--profile-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--ky-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--z-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--ignore-frequency", action="store_true")
    parser.add_argument("--no-require-profile", action="store_true")
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--resample-reference-to-observed-z", action="store_true")
    parser.add_argument("--resample-observed-to-reference-z", action="store_true")
    parser.add_argument("--periodic-z", action="store_true")
    parser.add_argument("--z-period", type=float)
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument(
        "--solver-preset",
        choices=("reduced", "gx-production-shape"),
        default="reduced",
    )
    parser.add_argument("--eik-reference", type=Path, default=DEFAULT_EIK_REFERENCE)
    parser.add_argument("--n-z", type=int)
    parser.add_argument("--field-line-periods", type=int)
    parser.add_argument("--n-kx", type=int)
    parser.add_argument("--kx-max", type=float)
    parser.add_argument("--ikxspace", type=int)
    parser.add_argument("--n-vpar", type=int)
    parser.add_argument("--n-mu", type=int)
    parser.add_argument("--vpar-max", type=float)
    parser.add_argument("--mu-max", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--steps-per-window", type=int)
    parser.add_argument("--n-windows", type=int)
    parser.add_argument(
        "--growth-diagnostic",
        choices=("late_fit", "late_mean_window"),
        default="late_fit",
    )
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
