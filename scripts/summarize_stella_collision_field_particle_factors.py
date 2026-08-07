"""Validate stella's pair-resolved scalar/basis field-particle factorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_field_particle_components import _read_values
from stellarator_gk import (
    build_laguerre_legendre_collision_precompute,
    laguerre_legendre_collision_components_from_moments,
)


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


def _axis_lookup(values: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.unique(values.astype(int))
    return labels, {int(value): index for index, value in enumerate(labels)}


def replay_factor_trace_with_local_kernel(factors: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Replay native scalar responses and bases through the JAX low-rank kernel."""

    import jax

    if not jax.config.x64_enabled:
        raise RuntimeError("factor replay requires JAX_ENABLE_X64=1")
    axes = {
        "iv": _axis_lookup(factors[:, 0]),
        "imu": _axis_lookup(factors[:, 1]),
        "iky": _axis_lookup(factors[:, 2]),
        "ikx": _axis_lookup(factors[:, 3]),
        "iz": _axis_lookup(factors[:, 4]),
        "target": _axis_lookup(factors[:, 6]),
        "background": _axis_lookup(factors[:, 7]),
    }
    if axes["target"][0].size != axes["background"][0].size:
        raise ValueError("factor trace target/background species grids differ")
    labels = tuple(
        tuple(int(value) for value in row)
        for row in np.unique(factors[:, 8:11], axis=0)
    )
    label_lookup = {label: index for index, label in enumerate(labels)}
    shape = (
        axes["target"][0].size,
        axes["background"][0].size,
        len(labels),
        axes["iv"][0].size,
        axes["imu"][0].size,
        axes["iz"][0].size,
        axes["ikx"][0].size,
        axes["iky"][0].size,
    )
    response = np.full(shape, np.nan)
    moments = np.full((shape[0], shape[1], shape[2], *shape[5:]), np.nan + 0j)
    for row in factors:
        target = axes["target"][1][int(row[6])]
        background = axes["background"][1][int(row[7])]
        component = label_lookup[tuple(int(value) for value in row[8:11])]
        iv = axes["iv"][1][int(row[0])]
        imu = axes["imu"][1][int(row[1])]
        iz = axes["iz"][1][int(row[4])]
        ikx = axes["ikx"][1][int(row[3])]
        iky = axes["iky"][1][int(row[2])]
        response[target, background, component, iv, imu, iz, ikx, iky] = row[15]
        moment_index = (target, background, component, iz, ikx, iky)
        psi = row[13] + 1j * row[14]
        previous = moments[moment_index]
        if np.isfinite(previous) and previous != psi:
            raise ValueError("factor trace psi varies across a velocity-space response")
        moments[moment_index] = psi
    if not np.isfinite(response).all() or not np.isfinite(moments).all():
        raise ValueError("factor trace does not span a dense pair/component grid")
    precompute = build_laguerre_legendre_collision_precompute(
        np.zeros_like(response),
        response,
        component_labels=labels,
    )
    components = laguerre_legendre_collision_components_from_moments(
        moments, precompute
    )
    return np.asarray(components).sum(axis=1), axes


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

    local_action, axes = replay_factor_trace_with_local_kernel(factors)
    local_rows = np.empty(aggregate.shape[0], dtype=complex)
    for row_index, row in enumerate(aggregate):
        target = axes["target"][1][int(row[6])]
        iv = axes["iv"][1][int(row[0])]
        imu = axes["imu"][1][int(row[1])]
        iz = axes["iz"][1][int(row[4])]
        ikx = axes["ikx"][1][int(row[3])]
        iky = axes["iky"][1][int(row[2])]
        local_rows[row_index] = local_action[target, iv, imu, iz, ikx, iky]
    native_rows = aggregate[:, 11] + 1j * aggregate[:, 12]
    local_error = float(np.linalg.norm(local_rows - native_rows) / aggregate_scale)
    if local_error > reconstruction_tolerance:
        raise ValueError(
            "local JAX factor replay does not match native aggregate RHS: "
            f"relative L2={local_error:.6g}"
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
            "local_jax_replay_to_native_relative_l2": local_error,
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
