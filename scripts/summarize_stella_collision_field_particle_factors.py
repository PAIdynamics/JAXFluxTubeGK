"""Validate stella's pair-resolved scalar/basis field-particle factorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_field_particle_components import _read_values


FACTOR_COLUMNS = (
    "iv",
    "imu",
    "iky",
    "ikx",
    "iz",
    "tube",
    "target",
    "background",
    "l",
    "m",
    "j",
    "vpa",
    "mu",
    "psi_re",
    "psi_im",
    "basis",
    "rhs_re",
    "rhs_im",
)


def summarize_factor_trace(
    factor_path: Path,
    aggregate_path: Path,
    *,
    expected_revision: str,
    reconstruction_tolerance: float = 2.0e-12,
) -> dict[str, object]:
    """Require `psi * response_basis` and its sum to match native stella."""

    factors = _read_values(
        factor_path,
        columns=len(FACTOR_COLUMNS),
        schema="stellarator_gk_stella_collision_fieldpart_factors_v1",
    )
    aggregate = _read_values(
        aggregate_path,
        columns=13,
        schema="stellarator_gk_stella_collision_fieldpart_trace_v1",
    )
    indices = factors[:, :11]
    if not np.array_equal(indices, np.rint(indices)):
        raise ValueError("factor trace indices must be integers")
    if np.unique(indices, axis=0).shape[0] != factors.shape[0]:
        raise ValueError("factor trace contains duplicate phase/component rows")

    psi = factors[:, 13] + 1j * factors[:, 14]
    basis = factors[:, 15]
    rhs = factors[:, 16] + 1j * factors[:, 17]
    factorization_scale = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    factorization_error = float(np.linalg.norm(psi * basis - rhs) / factorization_scale)
    if factorization_error > reconstruction_tolerance:
        raise ValueError(
            "scalar response and basis do not reconstruct factor RHS: "
            f"relative L2={factorization_error:.6g}"
        )

    phase_keys, inverse, counts = np.unique(
        indices[:, :7], axis=0, return_inverse=True, return_counts=True
    )
    labels = np.unique(indices[:, 7:11], axis=0)
    if np.unique(counts).size != 1 or int(counts[0]) != labels.shape[0]:
        raise ValueError("phase-space rows do not contain every pair/component label")
    reconstructed = np.zeros(phase_keys.shape[0], dtype=complex)
    np.add.at(reconstructed, inverse, rhs)

    aggregate_indices = aggregate[:, :7]
    aggregate_order = np.lexsort(aggregate_indices[:, ::-1].T)
    if not np.array_equal(phase_keys, aggregate_indices[aggregate_order]):
        raise ValueError("factor and aggregate phase-space grids differ")
    aggregate_rhs = (aggregate[:, 11] + 1j * aggregate[:, 12])[aggregate_order]
    aggregate_scale = max(float(np.linalg.norm(aggregate_rhs)), np.finfo(float).tiny)
    aggregate_error = float(np.linalg.norm(reconstructed - aggregate_rhs) / aggregate_scale)
    if aggregate_error > reconstruction_tolerance:
        raise ValueError(
            "pair/component factors do not reconstruct aggregate RHS: "
            f"relative L2={aggregate_error:.6g}"
        )

    pair_metrics = {}
    for target, background in np.unique(indices[:, 6:8], axis=0).astype(int):
        selected = (indices[:, 6] == target) & (indices[:, 7] == background)
        pair_metrics[f"target_{target}_background_{background}"] = {
            "rows": int(np.count_nonzero(selected)),
            "rhs_l2": float(np.linalg.norm(rhs[selected])),
            "psi_l2": float(np.linalg.norm(psi[selected])),
            "basis_l2": float(np.linalg.norm(basis[selected])),
        }
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_field_particle_factor_trace",
        "status": "native_low_rank_factorization_passed",
        "stella_source_revision": expected_revision,
        "factor_trace": str(Path(factor_path).resolve()),
        "aggregate_trace": str(Path(aggregate_path).resolve()),
        "phase_space_rows": int(phase_keys.shape[0]),
        "factors_per_row": int(counts[0]),
        "factor_labels": labels.astype(int).tolist(),
        "metrics": {
            "psi_basis_to_factor_relative_l2": factorization_error,
            "factor_sum_to_aggregate_relative_l2": aggregate_error,
            "aggregate_rhs_l2": float(np.linalg.norm(aggregate_rhs)),
        },
        "pair_metrics": pair_metrics,
        "scope": "native low-rank factor contract; local coefficient construction pending",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_factor_trace(
        args.factors,
        args.aggregate,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
