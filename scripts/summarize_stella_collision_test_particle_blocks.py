"""Validate native stella velocity-block packing against its dense matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_test_particle_matrix import (
    COLUMNS as MATRIX_COLUMNS,
    SCHEMA as MATRIX_SCHEMA,
    _dense_matrices,
)
from stellarator_gk import assemble_stella_test_particle_blocks


SCHEMA = "# schema=stellarator_gk_stella_collision_test_particle_blocks_v1"
COLUMNS = (
    "iz",
    "target",
    "background",
    "iv",
    "row_mu",
    "col_mu",
    "lower_re",
    "lower_im",
    "diagonal_re",
    "diagonal_im",
    "upper_re",
    "upper_im",
    "code_dt",
)


def _read(path: Path, schema: str, columns: int) -> np.ndarray:
    path = Path(path)
    header = path.read_text(encoding="utf-8").splitlines()[:1]
    if not header or header[0].strip() != schema:
        raise ValueError(f"unsupported or missing trace schema in {path}")
    values = np.atleast_2d(np.loadtxt(path))
    if values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f"invalid trace values in {path}")
    return values


def _block_arrays(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = values[:, :6]
    if not np.array_equal(indices, np.rint(indices)):
        raise ValueError("block trace indices must be integers")
    if np.unique(indices, axis=0).shape[0] != values.shape[0]:
        raise ValueError("block trace contains duplicate entries")
    axes = [np.unique(values[:, index].astype(int)) for index in range(6)]
    if any(not np.array_equal(axis, np.arange(axis[0], axis[-1] + 1)) for axis in axes):
        raise ValueError("block trace axes must be contiguous")
    expected_rows = int(np.prod([axis.size for axis in axes]))
    if values.shape[0] != expected_rows:
        raise ValueError("block trace does not span its Cartesian index grid")
    z_values, target_values, background_values, v_values, row_values, col_values = axes
    if not np.array_equal(row_values, col_values):
        raise ValueError("row_mu and col_mu axes differ")
    shape = (
        target_values.size,
        background_values.size,
        v_values.size,
        row_values.size,
        col_values.size,
        z_values.size,
    )
    arrays = [np.empty(shape, dtype=np.complex128) for _ in range(3)]
    lookups = [{value: index for index, value in enumerate(axis)} for axis in axes]
    for line in values:
        iz, target, background, iv, row_mu, col_mu = (
            lookups[index][int(line[index])] for index in range(6)
        )
        destination = (target, background, iv, row_mu, col_mu, iz)
        for array, column in zip(arrays, range(6, 12, 2), strict=True):
            array[destination] = line[column] + 1j * line[column + 1]
    return *arrays, z_values, target_values


def summarize_block_trace(
    block_trace: Path,
    matrix_trace: Path,
    *,
    expected_revision: str,
    absolute_tolerance: float = 2.0e-12,
) -> dict[str, object]:
    """Compare independently packed native blocks with zero-kperp matrices."""

    blocks = _read(block_trace, SCHEMA, len(COLUMNS))
    matrices_values = _read(matrix_trace, MATRIX_SCHEMA, len(MATRIX_COLUMNS))
    if np.any(blocks[:, 12] <= 0.0) or np.unique(blocks[:, 12]).size != 1:
        raise ValueError("block trace must contain one positive timestep")
    lower, diagonal, upper, z_values, target_values = _block_arrays(blocks)
    local = np.asarray(assemble_stella_test_particle_blocks(lower, diagonal, upper))
    matrices = _dense_matrices(matrices_values)

    errors: list[np.ndarray] = []
    native_values: list[np.ndarray] = []
    compared = 0
    for iz_index, iz in enumerate(z_values):
        for target_index, target in enumerate(target_values):
            keys = [key for key in matrices if key[2:] == (int(iz), int(target))]
            if not keys:
                raise ValueError(f"matrix trace lacks iz={iz}, species={target}")
            zero_keys = []
            for key in keys:
                selected = np.all(matrices_values[:, :4] == np.asarray(key), axis=1)
                kperp = np.unique(matrices_values[selected, 8])
                if kperp.size != 1:
                    raise ValueError(f"matrix {key} contains inconsistent kperp2")
                if kperp[0] == 0.0:
                    zero_keys.append(key)
            if not zero_keys:
                raise ValueError(f"no zero-kperp matrix for iz={iz}, species={target}")
            reference = matrices[zero_keys[0]].real
            if any(
                not np.array_equal(matrices[key], matrices[zero_keys[0]]) for key in zero_keys[1:]
            ):
                raise ValueError("duplicate zero-kperp matrices differ")
            errors.append(local[iz_index, target_index] - reference)
            native_values.append(reference)
            compared += 1

    error = np.concatenate([item.ravel() for item in errors])
    native = np.concatenate([item.ravel() for item in native_values])
    maximum = float(np.max(np.abs(error)))
    relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(native), np.finfo(float).tiny))
    if maximum > absolute_tolerance:
        raise ValueError("native block assembly does not reproduce zero-kperp matrices")
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_test_particle_blocks",
        "status": "native_velocity_block_assembly_passed",
        "scope": "zero-kperp I-dt*C_test packing; local block coefficient generation pending",
        "stella_source_revision": expected_revision,
        "block_trace": str(Path(block_trace).resolve()),
        "matrix_trace": str(Path(matrix_trace).resolve()),
        "rows": int(blocks.shape[0]),
        "matrices_compared": compared,
        "metrics": {"relative_l2": relative_l2, "max_abs": maximum},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_block_trace(
        args.blocks,
        args.matrix,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
