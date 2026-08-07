"""Replay a native stella implicit collision update from independent traces."""

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


FIELD_SCHEMA = "# schema=stellarator_gk_stella_collision_fieldpart_trace_v1"
FINAL_SCHEMA = "# schema=stellarator_gk_stella_collision_final_state_v1"


def _read_trace(path: Path, schema: str, columns: int) -> np.ndarray:
    path = Path(path)
    header = path.read_text(encoding="utf-8").splitlines()[:1]
    if not header or header[0].strip() != schema:
        raise ValueError(f"unsupported or missing trace schema in {path}")
    values = np.atleast_2d(np.loadtxt(path))
    if values.shape[1] != columns:
        raise ValueError(f"expected {columns} columns in {path}, found {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {path}")
    return values


def replay_implicit_collision(
    matrix_trace: Path,
    field_particle_trace: Path,
    final_state_trace: Path,
    *,
    expected_revision: str,
) -> dict[str, object]:
    """Reconstruct ``solve(I-dt*C_tp, input + dt*C_fp)`` mode by mode."""

    matrix_values = _read_trace(matrix_trace, MATRIX_SCHEMA, len(MATRIX_COLUMNS))
    field_values = _read_trace(field_particle_trace, FIELD_SCHEMA, 13)
    final_values = _read_trace(final_state_trace, FINAL_SCHEMA, 11)
    matrices = _dense_matrices(matrix_values)
    dt_values = np.unique(matrix_values[:, 9])
    if dt_values.size != 1 or dt_values[0] <= 0.0:
        raise ValueError("matrix trace must contain one positive timestep")
    dt = float(dt_values[0])

    if not np.array_equal(field_values[:, :7], final_values[:, :7]):
        raise ValueError("field-particle and final-state traces have different phase-space rows")
    if np.unique(final_values[:, :7], axis=0).shape[0] != final_values.shape[0]:
        raise ValueError("final-state trace contains duplicate phase-space rows")
    if not np.allclose(field_values[:, 9:11], final_values[:, 7:9], rtol=0.0, atol=0.0):
        raise ValueError("traces do not contain an identical collision input state")

    predicted_parts = []
    observed_parts = []
    input_only_parts = []
    mode_keys = sorted(
        {tuple(int(value) for value in row[2:7]) for row in final_values}
    )
    for iky, ikx, iz, tube, species in mode_keys:
        selected = np.all(
            final_values[:, 2:7] == np.asarray((iky, ikx, iz, tube, species)),
            axis=1,
        )
        final_rows = final_values[selected]
        field_rows = field_values[selected]
        order = np.lexsort((final_rows[:, 1], final_rows[:, 0]))
        final_rows = final_rows[order]
        field_rows = field_rows[order]
        state = final_rows[:, 7] + 1j * final_rows[:, 8]
        observed = final_rows[:, 9] + 1j * final_rows[:, 10]
        field_rhs = field_rows[:, 11] + 1j * field_rows[:, 12]
        matrix_key = (iky, ikx, iz, species)
        if matrix_key not in matrices:
            raise ValueError(f"missing test-particle matrix {matrix_key}")
        matrix = matrices[matrix_key]
        if matrix.shape[0] != state.size:
            raise ValueError(f"matrix {matrix_key} is incompatible with its state")
        predicted_parts.append(np.linalg.solve(matrix, state + dt * field_rhs))
        input_only_parts.append(np.linalg.solve(matrix, state))
        observed_parts.append(observed)

    predicted = np.concatenate(predicted_parts)
    observed = np.concatenate(observed_parts)
    input_only = np.concatenate(input_only_parts)
    error = predicted - observed
    observed_l2 = float(np.linalg.norm(observed))
    if observed_l2 == 0.0:
        raise ValueError("native final collision state has zero norm")
    return {
        "schema_version": 1,
        "benchmark": "stella_complete_implicit_collision_replay",
        "status": "native_implicit_collision_replay_passed",
        "scope": (
            "native matrix and solved field-particle RHS replay; independent matrix construction pending"
        ),
        "stella_source_revision": expected_revision,
        "matrix_trace": str(Path(matrix_trace).resolve()),
        "field_particle_trace": str(Path(field_particle_trace).resolve()),
        "final_state_trace": str(Path(final_state_trace).resolve()),
        "rows": int(final_values.shape[0]),
        "modes": len(mode_keys),
        "metrics": {
            "code_dt": dt,
            "native_output_l2": observed_l2,
            "replay_relative_l2": float(np.linalg.norm(error) / observed_l2),
            "replay_max_abs": float(np.max(np.abs(error))),
            "field_particle_effect_relative_l2": float(
                np.linalg.norm(predicted - input_only) / observed_l2
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--field-particle", type=Path, required=True)
    parser.add_argument("--final-state", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = replay_implicit_collision(
        args.matrix,
        args.field_particle,
        args.final_state,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
