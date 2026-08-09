"""Validate stella's unfactorized implicit test-particle collision matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA = "# schema=jax_fluxtube_gk_stella_collision_test_particle_matrix_v1"
COLUMNS = (
    "iky",
    "ikx",
    "iz",
    "species",
    "row",
    "col",
    "matrix_re",
    "matrix_im",
    "kperp2",
    "code_dt",
)


def _dense_matrices(values: np.ndarray) -> dict[tuple[int, int, int, int], np.ndarray]:
    indices = values[:, :6]
    if not np.array_equal(indices, np.rint(indices)):
        raise ValueError("matrix trace indices must be integers")
    if np.unique(indices, axis=0).shape[0] != values.shape[0]:
        raise ValueError("matrix trace contains duplicate entries")

    size = int(np.max(indices[:, 4:6]))
    if size <= 0 or np.min(indices[:, 4:6]) != 1:
        raise ValueError("matrix row and column indices must be one-based")
    matrices: dict[tuple[int, int, int, int], np.ndarray] = {}
    for line in values:
        key = tuple(int(item) for item in line[:4])
        matrix = matrices.setdefault(key, np.zeros((size, size), dtype=np.complex128))
        row, col = (int(item) - 1 for item in line[4:6])
        matrix[row, col] = line[6] + 1j * line[7]
    return matrices


def summarize_matrix_trace(trace_path: Path, *, expected_revision: str) -> dict[str, object]:
    """Check storage reconstruction and isolate stella's gyro-diffusion diagonal."""

    trace_path = Path(trace_path)
    header = trace_path.read_text(encoding="utf-8").splitlines()[:2]
    if not header or header[0].strip() != SCHEMA:
        raise ValueError("unsupported or missing collision matrix trace schema")
    values = np.atleast_2d(np.loadtxt(trace_path))
    if values.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} trace columns, found {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("collision matrix trace contains non-finite values")
    if np.any(values[:, 8] < 0.0) or np.any(values[:, 9] <= 0.0):
        raise ValueError("kperp2 must be nonnegative and code_dt must be positive")
    if not np.allclose(values[:, 9], values[0, 9], rtol=0.0, atol=0.0):
        raise ValueError("collision matrix trace contains inconsistent timesteps")

    matrices = _dense_matrices(values)
    kperp_by_key: dict[tuple[int, int, int, int], float] = {}
    for key in matrices:
        selected = np.all(values[:, :4] == np.asarray(key), axis=1)
        kperp = np.unique(values[selected, 8])
        if kperp.size != 1:
            raise ValueError(f"matrix {key} contains inconsistent kperp2 values")
        kperp_by_key[key] = float(kperp[0])

    imaginary_max = max(float(np.max(np.abs(matrix.imag))) for matrix in matrices.values())
    nonzero_bandwidth = max(
        (
            int(np.max(np.abs(np.argwhere(np.abs(matrix) > 0.0)[:, 0]
                              - np.argwhere(np.abs(matrix) > 0.0)[:, 1])))
            if np.count_nonzero(matrix) else 0
        )
        for matrix in matrices.values()
    )

    offdiagonal_residual = 0.0
    linearity_residual = 0.0
    base_modes = 0
    compared_modes = 0
    for iz, species in sorted({(key[2], key[3]) for key in matrices}):
        group = [key for key in matrices if key[2:] == (iz, species)]
        base_key = min(group, key=kperp_by_key.__getitem__)
        base_kperp = kperp_by_key[base_key]
        if base_kperp != 0.0:
            raise ValueError(f"no kperp2=0 base matrix for iz={iz}, species={species}")
        base_modes += 1
        slopes: list[np.ndarray] = []
        for key in group:
            kperp = kperp_by_key[key]
            if kperp == 0.0:
                if not np.array_equal(matrices[key], matrices[base_key]):
                    raise ValueError("duplicate kperp2=0 modes have different matrices")
                continue
            delta = matrices[key] - matrices[base_key]
            offdiagonal = delta.copy()
            np.fill_diagonal(offdiagonal, 0.0)
            offdiagonal_residual = max(
                offdiagonal_residual, float(np.max(np.abs(offdiagonal)))
            )
            slopes.append(np.diag(delta) / kperp)
            compared_modes += 1
        if slopes:
            reference = slopes[0]
            linearity_residual = max(
                linearity_residual,
                *(float(np.max(np.abs(slope - reference))) for slope in slopes[1:]),
            )

    if offdiagonal_residual > 1.0e-13:
        raise ValueError("kperp-dependent test-particle correction is not diagonal")
    if linearity_residual > 1.0e-10:
        raise ValueError("test-particle diagonal correction is not linear in kperp2")

    return {
        "schema_version": 1,
        "benchmark": "stella_collision_test_particle_matrix",
        "status": "native_test_particle_matrix_structure_passed",
        "scope": (
            "unfactorized I-dt*C_test matrix; local coefficient parity pending"
        ),
        "stella_source_revision": expected_revision,
        "trace": str(trace_path.resolve()),
        "rows": int(values.shape[0]),
        "matrices": len(matrices),
        "matrix_size": next(iter(matrices.values())).shape[0],
        "grid": {
            "nky": int(np.unique(values[:, 0]).size),
            "nkx": int(np.unique(values[:, 1]).size),
            "nz": int(np.unique(values[:, 2]).size),
            "nspecies": int(np.unique(values[:, 3]).size),
        },
        "metrics": {
            "code_dt": float(values[0, 9]),
            "imaginary_max_abs": imaginary_max,
            "nonzero_bandwidth": nonzero_bandwidth,
            "zero_kperp_base_matrices": base_modes,
            "nonzero_kperp_matrices_compared": compared_modes,
            "gyro_offdiagonal_max_abs": offdiagonal_residual,
            "gyro_kperp_linearity_max_abs": linearity_residual,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_matrix_trace(args.trace, expected_revision=args.expected_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
