"""Audit W7-X stella frequency/profile convention hypotheses.

This diagnostic is intentionally convention-level only.  It compares the
long-time stella-matched solver fixture against the exported stella fixture
under simple frequency sign/scale and complex-profile transformations before
changing RHS or velocity-space terms.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from stellarator_gk import (
    PerKyModeStructureFixture,
    load_per_ky_mode_structure_fixture_csv,
    resample_per_ky_mode_structure_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVED = (
    ROOT / "fixtures/w7x_itg_stella_matched_time_ladder/runs/time_200/mode_structures.csv"
)
DEFAULT_REFERENCE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_OUTPUT = (
    ROOT / "fixtures/w7x_itg_stella_matched_time_ladder/frequency_profile_convention_audit.json"
)
DEFAULT_OUTPUT_CSV = (
    ROOT / "fixtures/w7x_itg_stella_matched_time_ladder/frequency_profile_convention_audit.csv"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_w7x_stella_frequency_profile_convention_audit(
        observed_fixture=args.observed_fixture,
        reference_fixture=args.reference_fixture,
        output=args.output,
        output_csv=args.output_csv,
        ky_values=_parse_float_tuple(args.ky_values),
        ky_tolerance=args.ky_tolerance,
        periodic_z=args.periodic_z,
        z_period=args.z_period,
    )
    print(
        "OPEN: "
        f"best_frequency={report['summary']['best_frequency_variant']} "
        f"max_frequency_error={report['summary']['best_frequency_max_abs_error']:.8e}, "
        f"best_profile={report['summary']['best_profile_variant']} "
        f"max_profile_error={report['summary']['best_profile_max_abs_error']:.8e}"
    )
    print(args.output)
    return 0


def run_w7x_stella_frequency_profile_convention_audit(
    *,
    observed_fixture: Path = DEFAULT_OBSERVED,
    reference_fixture: Path = DEFAULT_REFERENCE,
    output: Path = DEFAULT_OUTPUT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    ky_values: tuple[float, ...] = (0.1, 0.2, 0.3),
    ky_tolerance: float = 1.0e-6,
    periodic_z: bool = False,
    z_period: float | None = None,
) -> dict[str, Any]:
    """Write and return convention-level frequency/profile diagnostics."""

    observed = _select_ky_values(
        load_per_ky_mode_structure_fixture_csv(observed_fixture),
        ky_values,
        ky_tolerance,
    )
    reference = _select_ky_values(
        load_per_ky_mode_structure_fixture_csv(reference_fixture),
        ky_values,
        ky_tolerance,
    )
    if observed.z.shape != reference.z.shape or not np.allclose(
        np.asarray(observed.z),
        np.asarray(reference.z),
        rtol=0.0,
        atol=1.0e-12,
    ):
        reference = resample_per_ky_mode_structure_fixture(
            reference,
            observed.z,
            periodic=periodic_z,
            period=z_period,
        )

    frequency_variants = _frequency_variant_errors(observed, reference)
    profile_variants = _profile_variant_errors(observed, reference)
    direct_profile = profile_variants["direct_reference"]
    direct_frequency = frequency_variants["direct_reference"]
    best_frequency_name, best_frequency = min(
        frequency_variants.items(),
        key=lambda item: item[1]["max_abs_error"],
    )
    best_profile_name, best_profile = min(
        profile_variants.items(),
        key=lambda item: item[1]["max_phase_aligned_error"],
    )
    rows = _per_ky_rows(
        observed,
        reference,
        direct_frequency=direct_frequency,
        direct_profile=direct_profile,
        best_frequency_name=best_frequency_name,
        best_frequency=best_frequency,
        best_profile_name=best_profile_name,
        best_profile=best_profile,
    )
    report = {
        "benchmark_name": "w7x_stella_frequency_profile_convention_audit",
        "status": "open",
        "passed": False,
        "observed_fixture": _display_path(observed_fixture),
        "reference_fixture": _display_path(reference_fixture),
        "ky_values": list(ky_values),
        "frequency_variants": frequency_variants,
        "profile_variants": profile_variants,
        "summary": {
            "best_frequency_variant": best_frequency_name,
            "best_frequency_max_abs_error": best_frequency["max_abs_error"],
            "direct_frequency_max_abs_error": direct_frequency["max_abs_error"],
            "best_profile_variant": best_profile_name,
            "best_profile_max_abs_error": best_profile["max_phase_aligned_error"],
            "direct_profile_max_abs_error": direct_profile["max_phase_aligned_error"],
            "frequency_sign_flip_improves": (
                frequency_variants["negated_reference"]["max_abs_error"]
                < direct_frequency["max_abs_error"]
            ),
            "profile_conjugation_improves": (
                profile_variants["conjugate_reference"]["max_phase_aligned_error"]
                < direct_profile["max_phase_aligned_error"]
            ),
            "profile_z_reversal_improves": (
                profile_variants["reverse_z_reference"]["max_phase_aligned_error"]
                < direct_profile["max_phase_aligned_error"]
            ),
            "next_action": (
                "inspect velocity/RHS terms; simple frequency/profile convention "
                "transforms do not close the W7-X gate"
            ),
        },
        "per_ky": rows,
    }
    _write_json(output, report)
    _write_csv(output_csv, rows)
    return report


def _frequency_variant_errors(
    observed: PerKyModeStructureFixture,
    reference: PerKyModeStructureFixture,
) -> dict[str, dict[str, Any]]:
    observed_frequency = np.asarray(observed.frequency, dtype=float)
    reference_frequency = np.asarray(reference.frequency, dtype=float)
    variants: dict[str, np.ndarray] = {
        "direct_reference": reference_frequency,
        "negated_reference": -reference_frequency,
    }
    denominator = float(np.dot(reference_frequency, reference_frequency))
    scale = (
        float(np.dot(observed_frequency, reference_frequency) / denominator)
        if denominator > 0.0
        else 0.0
    )
    variants["least_squares_scaled_reference"] = scale * reference_frequency
    matrix = np.column_stack([reference_frequency, np.ones_like(reference_frequency)])
    affine, *_ = np.linalg.lstsq(matrix, observed_frequency, rcond=None)
    variants["least_squares_affine_reference"] = matrix @ affine
    return {
        name: _frequency_error_summary(
            observed_frequency,
            predicted,
            scale=scale if name == "least_squares_scaled_reference" else None,
            affine=tuple(float(value) for value in affine)
            if name == "least_squares_affine_reference"
            else None,
        )
        for name, predicted in variants.items()
    }


def _frequency_error_summary(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    scale: float | None = None,
    affine: tuple[float, float] | None = None,
) -> dict[str, Any]:
    error = observed - predicted
    payload: dict[str, Any] = {
        "predicted_frequency": [float(value) for value in predicted],
        "frequency_error": [float(value) for value in error],
        "max_abs_error": float(np.max(np.abs(error))),
        "rms_error": float(np.sqrt(np.mean(error**2))),
    }
    if scale is not None:
        payload["scale"] = float(scale)
    if affine is not None:
        payload["scale"] = float(affine[0])
        payload["offset"] = float(affine[1])
    return payload


def _profile_variant_errors(
    observed: PerKyModeStructureFixture,
    reference: PerKyModeStructureFixture,
) -> dict[str, dict[str, Any]]:
    z = np.asarray(observed.z, dtype=float)
    reference_phi = np.asarray(reference.phi, dtype=np.complex128)
    reversed_phi = _periodic_sample_rows(reference_phi, z, -z)
    variants = {
        "direct_reference": reference_phi,
        "conjugate_reference": np.conj(reference_phi),
        "reverse_z_reference": reversed_phi,
        "conjugate_reverse_z_reference": np.conj(reversed_phi),
    }
    summaries = {
        name: _profile_error_summary(observed.phi, phi)
        for name, phi in variants.items()
    }
    common_shift = _best_common_circular_shift(observed.phi, reference_phi)
    per_ky_shift = _best_per_ky_circular_shift(observed.phi, reference_phi)
    summaries["best_common_circular_shift"] = common_shift
    summaries["best_per_ky_circular_shift"] = per_ky_shift
    return summaries


def _profile_error_summary(observed_phi, reference_phi) -> dict[str, Any]:
    observed = _row_l2_normalize_complex(np.asarray(observed_phi, dtype=np.complex128))
    reference = _row_l2_normalize_complex(np.asarray(reference_phi, dtype=np.complex128))
    errors, scales = _row_phase_aligned_max_abs_errors(observed, reference)
    return {
        "phase_aligned_error": [float(value) for value in errors],
        "max_phase_aligned_error": float(np.max(errors)),
        "alignment_scale": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in scales
        ],
    }


def _best_common_circular_shift(observed_phi, reference_phi) -> dict[str, Any]:
    reference = np.asarray(reference_phi, dtype=np.complex128)
    best_shift = 0
    best_summary = _profile_error_summary(observed_phi, reference)
    for shift in range(1, reference.shape[1]):
        summary = _profile_error_summary(observed_phi, np.roll(reference, shift, axis=1))
        if summary["max_phase_aligned_error"] < best_summary["max_phase_aligned_error"]:
            best_shift = shift
            best_summary = summary
    return {"shift_index": int(best_shift), **best_summary}


def _best_per_ky_circular_shift(observed_phi, reference_phi) -> dict[str, Any]:
    observed = np.asarray(observed_phi, dtype=np.complex128)
    reference = np.asarray(reference_phi, dtype=np.complex128)
    shifts = []
    errors = []
    for ky_index in range(reference.shape[0]):
        best_shift = 0
        best_error = np.inf
        for shift in range(reference.shape[1]):
            summary = _profile_error_summary(
                observed[ky_index : ky_index + 1],
                np.roll(reference[ky_index : ky_index + 1], shift, axis=1),
            )
            error = summary["max_phase_aligned_error"]
            if error < best_error:
                best_shift = shift
                best_error = error
        shifts.append(int(best_shift))
        errors.append(float(best_error))
    return {
        "shift_index": shifts,
        "phase_aligned_error": errors,
        "max_phase_aligned_error": float(max(errors)),
    }


def _periodic_sample_rows(phi: np.ndarray, z: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    period = 1.0
    source_z = z
    interp_z = np.concatenate([source_z, [source_z[0] + period]])
    target_eval = ((target_z - source_z[0]) % period) + source_z[0]
    out = np.empty((phi.shape[0], target_z.shape[0]), dtype=np.complex128)
    for index, row in enumerate(phi):
        interp_row = np.concatenate([row, row[:1]])
        out[index] = np.interp(target_eval, interp_z, interp_row.real) + 1j * np.interp(
            target_eval,
            interp_z,
            interp_row.imag,
        )
    return out


def _row_l2_normalize_complex(values: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.sum(np.abs(values) ** 2, axis=1))
    safe = np.where(norms > 0.0, norms, 1.0)
    return np.where(norms[:, None] > 0.0, values / safe[:, None], values)


def _row_phase_aligned_max_abs_errors(
    observed: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.einsum("ij,ij->i", np.conjugate(reference), observed)
    denominator = np.einsum("ij,ij->i", np.conjugate(reference), reference)
    scale = np.where(np.abs(denominator) > 0.0, numerator / denominator, 0.0)
    errors = np.max(np.abs(observed - scale[:, None] * reference), axis=1)
    return errors, scale


def _per_ky_rows(
    observed: PerKyModeStructureFixture,
    reference: PerKyModeStructureFixture,
    *,
    direct_frequency: dict[str, Any],
    direct_profile: dict[str, Any],
    best_frequency_name: str,
    best_frequency: dict[str, Any],
    best_profile_name: str,
    best_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    ky = np.asarray(observed.ky, dtype=float)
    observed_frequency = np.asarray(observed.frequency, dtype=float)
    reference_frequency = np.asarray(reference.frequency, dtype=float)
    rows = []
    for index, ky_value in enumerate(ky):
        rows.append(
            {
                "ky": float(ky_value),
                "observed_frequency": float(observed_frequency[index]),
                "reference_frequency": float(reference_frequency[index]),
                "direct_frequency_error": float(direct_frequency["frequency_error"][index]),
                "best_frequency_variant": best_frequency_name,
                "best_frequency_error": float(best_frequency["frequency_error"][index]),
                "direct_profile_error": float(
                    direct_profile["phase_aligned_error"][index]
                ),
                "best_profile_variant": best_profile_name,
                "best_profile_error": float(best_profile["phase_aligned_error"][index]),
            }
        )
    return rows


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "ky",
        "observed_frequency",
        "reference_frequency",
        "direct_frequency_error",
        "best_frequency_variant",
        "best_frequency_error",
        "direct_profile_error",
        "best_profile_variant",
        "best_profile_error",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-fixture", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--reference-fixture", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--ky-values", default="0.1,0.2,0.3")
    parser.add_argument("--ky-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--periodic-z", action="store_true")
    parser.add_argument("--z-period", type=float)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
