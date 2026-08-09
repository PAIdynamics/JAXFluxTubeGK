"""Compare two per-ky complex mode-structure fixture CSV files.

Run from the repository root, for example:

    uv run --extra dev python examples/compare_mode_structure_fixtures.py \
        --observed figures/solver_mode_structure.csv \
        --reference fixtures/external_mode_structure.csv
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_fluxtube_gk import (
    PerKyModeStructureFixture,
    compare_per_ky_mode_structure_fixtures,
    evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures,
    load_per_ky_mode_structure_fixture_csv,
    resample_per_ky_mode_structure_fixture,
)


def main() -> None:
    args = _parse_args()
    observed = load_per_ky_mode_structure_fixture_csv(args.observed)
    reference = load_per_ky_mode_structure_fixture_csv(args.reference)
    if args.ky_values:
        ky_values = _parse_float_tuple(args.ky_values)
        observed = _select_ky_values(observed, ky_values, args.ky_tolerance)
        reference = _select_ky_values(reference, ky_values, args.ky_tolerance)
    observed, reference = _apply_z_resampling(observed, reference, args)
    comparison = compare_per_ky_mode_structure_fixtures(
        observed,
        reference,
        growth_tolerance=args.growth_tolerance,
        frequency_tolerance=args.frequency_tolerance,
        phi_tolerance=args.profile_tolerance,
        ky_tolerance=args.ky_tolerance,
        z_tolerance=args.z_tolerance,
        require_frequency=not args.ignore_frequency,
        require_phi=args.require_profile,
    )
    gate = evaluate_cyclone_ky_scan_gate_from_mode_structure_fixtures(
        observed,
        reference,
        growth_tolerance=args.growth_tolerance,
        frequency_tolerance=args.frequency_tolerance,
        profile_tolerance=args.profile_tolerance,
        ky_tolerance=args.ky_tolerance,
        z_tolerance=args.z_tolerance,
        require_frequency=not args.ignore_frequency,
        require_profile=args.require_profile,
    )
    _write_csv(args.output, comparison, gate)
    status = "PASS" if bool(gate.passed) else "OPEN"
    print(
        f"{status}: max_growth_error={float(gate.max_growth_error):.8e}, "
        f"max_frequency_error={float(gate.max_frequency_error):.8e}, "
        f"max_profile_error={float(gate.max_profile_error):.8e}"
    )
    print(args.output)


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


def _apply_z_resampling(
    observed: PerKyModeStructureFixture,
    reference: PerKyModeStructureFixture,
    args,
) -> tuple[PerKyModeStructureFixture, PerKyModeStructureFixture]:
    """Optionally interpolate one fixture onto the other's parallel grid."""

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
    return observed, reference


def _write_csv(path: Path, comparison, gate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ky = np.asarray(comparison.ky, dtype=float)
    matched_ky = np.asarray(comparison.matched_reference_ky, dtype=float)
    rows = {
        "observed_growth": np.asarray(comparison.observed_growth, dtype=float),
        "reference_growth": np.asarray(comparison.reference_growth, dtype=float),
        "growth_error": np.asarray(comparison.growth_error, dtype=float),
        "observed_frequency": np.asarray(comparison.observed_frequency, dtype=float),
        "reference_frequency": np.asarray(comparison.reference_frequency, dtype=float),
        "frequency_error": np.asarray(comparison.frequency_error, dtype=float),
        "phi_direct_error": np.asarray(comparison.phi_direct_error, dtype=float),
        "phi_phase_aligned_error": np.asarray(
            comparison.phi_phase_aligned_error,
            dtype=float,
        ),
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
                    rows["observed_growth"][index],
                    rows["reference_growth"][index],
                    rows["growth_error"][index],
                    rows["observed_frequency"][index],
                    rows["reference_frequency"][index],
                    rows["frequency_error"][index],
                    rows["phi_direct_error"][index],
                    rows["phi_phase_aligned_error"][index],
                    bool(rows["growth_passed"][index]),
                    bool(rows["frequency_passed"][index]),
                    bool(rows["phi_passed"][index]),
                    bool(gate.passed),
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/per_ky_mode_structure_comparison.csv"),
    )
    parser.add_argument("--growth-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--frequency-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--profile-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--ky-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--z-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--ky-values", help="optional comma-separated ky subset to compare")
    parser.add_argument("--ignore-frequency", action="store_true")
    parser.add_argument("--require-profile", action="store_true")
    parser.add_argument("--resample-reference-to-observed-z", action="store_true")
    parser.add_argument("--resample-observed-to-reference-z", action="store_true")
    parser.add_argument("--periodic-z", action="store_true")
    parser.add_argument("--z-period", type=float)
    return parser.parse_args()


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


if __name__ == "__main__":
    main()
