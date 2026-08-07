"""Validate stella's signed Laguerre--Legendre field-particle components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMPONENT_COLUMNS = (
    "iv",
    "imu",
    "iky",
    "ikx",
    "iz",
    "tube",
    "species",
    "l",
    "m",
    "j",
    "vpa",
    "mu",
    "before_re",
    "before_im",
    "rhs_re",
    "rhs_im",
)


def _read_values(path: Path, *, columns: int, schema: str) -> np.ndarray:
    path = Path(path)
    first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != f"# schema={schema}":
        raise ValueError(f"unsupported or missing trace schema in {path}")
    values = np.atleast_2d(np.loadtxt(path))
    if values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f"invalid signed collision trace {path}")
    return values


def summarize_component_trace(
    component_path: Path,
    aggregate_path: Path,
    *,
    expected_revision: str,
    reconstruction_tolerance: float = 2.0e-12,
) -> dict[str, object]:
    """Require the `(j,l,m)` components to reconstruct the aggregate action."""

    components = _read_values(
        component_path,
        columns=len(COMPONENT_COLUMNS),
        schema="stellarator_gk_stella_collision_fieldpart_components_v1",
    )
    aggregate = _read_values(
        aggregate_path,
        columns=13,
        schema="stellarator_gk_stella_collision_fieldpart_trace_v1",
    )
    component_indices = components[:, :10]
    aggregate_indices = aggregate[:, :7]
    if not np.array_equal(component_indices, np.rint(component_indices)):
        raise ValueError("component trace indices must be integers")
    if np.unique(component_indices, axis=0).shape[0] != components.shape[0]:
        raise ValueError("component trace contains duplicate phase/component rows")

    phase_keys, inverse, counts = np.unique(
        component_indices[:, :7], axis=0, return_inverse=True, return_counts=True
    )
    if np.unique(counts).size != 1:
        raise ValueError("phase-space rows do not have a uniform component count")
    labels = np.unique(component_indices[:, 7:10], axis=0)
    if int(counts[0]) != labels.shape[0]:
        raise ValueError("phase-space rows do not contain every component label")
    if np.unique(aggregate_indices, axis=0).shape[0] != aggregate.shape[0]:
        raise ValueError("aggregate trace contains duplicate phase-space rows")
    aggregate_order = np.lexsort(aggregate_indices[:, ::-1].T)
    if not np.array_equal(phase_keys, aggregate_indices[aggregate_order]):
        raise ValueError("component and aggregate phase-space grids differ")

    component_rhs = components[:, 14] + 1j * components[:, 15]
    reconstructed = np.zeros(phase_keys.shape[0], dtype=complex)
    np.add.at(reconstructed, inverse, component_rhs)
    aggregate_rhs = (aggregate[:, 11] + 1j * aggregate[:, 12])[aggregate_order]
    denominator = max(float(np.linalg.norm(aggregate_rhs)), np.finfo(float).tiny)
    relative_l2 = float(np.linalg.norm(reconstructed - aggregate_rhs) / denominator)
    if relative_l2 > reconstruction_tolerance:
        raise ValueError(
            "Laguerre--Legendre components do not reconstruct aggregate RHS: "
            f"relative L2={relative_l2:.6g}"
        )

    metrics = {}
    for label in labels.astype(int):
        selected = np.all(component_indices[:, 7:10] == label, axis=1)
        key = f"l{label[0]}_m{label[1]}_j{label[2]}"
        metrics[key] = {
            "rows": int(np.count_nonzero(selected)),
            "rhs_l2": float(np.linalg.norm(component_rhs[selected])),
            "rhs_max_abs": float(np.max(np.abs(component_rhs[selected]))),
        }
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_laguerre_legendre_component_trace",
        "status": "native_component_reconstruction_passed",
        "stella_source_revision": expected_revision,
        "component_trace": str(Path(component_path).resolve()),
        "aggregate_trace": str(Path(aggregate_path).resolve()),
        "phase_space_rows": int(phase_keys.shape[0]),
        "components_per_row": int(counts[0]),
        "component_labels": labels.astype(int).tolist(),
        "metrics": {
            "component_sum_to_aggregate_relative_l2": relative_l2,
            "aggregate_rhs_l2": float(np.linalg.norm(aggregate_rhs)),
        },
        "component_metrics": metrics,
        "scope": "native coefficient-action decomposition; local coefficient parity pending",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_component_trace(
        args.components,
        args.aggregate,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
