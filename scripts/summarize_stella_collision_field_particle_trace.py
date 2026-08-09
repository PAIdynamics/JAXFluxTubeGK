"""Validate and summarize a signed stella field-particle collision trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COLUMNS = (
    "iv",
    "imu",
    "iky",
    "ikx",
    "iz",
    "tube",
    "species",
    "vpa",
    "mu",
    "before_re",
    "before_im",
    "rhs_re",
    "rhs_im",
)


def summarize_trace(trace_path: Path, *, expected_revision: str) -> dict[str, object]:
    """Return fail-closed norms and indexing metadata for a native trace."""

    trace_path = Path(trace_path)
    header = trace_path.read_text(encoding="utf-8").splitlines()[:2]
    if not header or header[0].strip() != (
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_trace_v1"
    ):
        raise ValueError("unsupported or missing collision trace schema")
    values = np.loadtxt(trace_path)
    values = np.atleast_2d(values)
    if values.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} trace columns, found {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("collision trace contains non-finite values")
    indices = values[:, :7]
    if not np.array_equal(indices, np.rint(indices)):
        raise ValueError("collision trace indices must be integers")
    if np.unique(indices, axis=0).shape[0] != values.shape[0]:
        raise ValueError("collision trace contains duplicate phase-space rows")

    before = values[:, 9] + 1j * values[:, 10]
    rhs = values[:, 11] + 1j * values[:, 12]
    before_l2 = float(np.linalg.norm(before))
    rhs_l2 = float(np.linalg.norm(rhs))
    if before_l2 <= 0.0:
        raise ValueError("collision trace input has zero norm")
    if rhs_l2 <= 0.0:
        raise ValueError("collision trace field-particle RHS has zero norm")
    species_metrics: dict[str, dict[str, float | int]] = {}
    for species in sorted(set(indices[:, 6].astype(int))):
        selected = indices[:, 6] == species
        species_rhs = rhs[selected]
        species_metrics[str(species)] = {
            "rows": int(np.count_nonzero(selected)),
            "rhs_l2": float(np.linalg.norm(species_rhs)),
            "rhs_max_abs": float(np.max(np.abs(species_rhs))),
        }
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_signed_field_particle_trace",
        "status": "signed_native_trace_passed",
        "scope": "aggregate signed native field-particle RHS; common-grid local parity pending",
        "stella_source_revision": expected_revision,
        "trace": str(trace_path.resolve()),
        "rows": int(values.shape[0]),
        "grid": {
            "nvpa": int(np.unique(indices[:, 0]).size),
            "nmu": int(np.unique(indices[:, 1]).size),
            "nky": int(np.unique(indices[:, 2]).size),
            "nkx": int(np.unique(indices[:, 3]).size),
            "nz": int(np.unique(indices[:, 4]).size),
            "ntube": int(np.unique(indices[:, 5]).size),
            "nspecies": int(np.unique(indices[:, 6]).size),
        },
        "metrics": {
            "input_l2": before_l2,
            "field_particle_rhs_l2": rhs_l2,
            "field_particle_rhs_max_abs": float(np.max(np.abs(rhs))),
            "field_particle_rhs_to_input_l2": rhs_l2 / before_l2,
            "nonzero_rhs_fraction": float(np.count_nonzero(rhs) / rhs.size),
        },
        "species_metrics": species_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_trace(args.trace, expected_revision=args.expected_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
