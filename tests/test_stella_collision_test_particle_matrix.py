import numpy as np

from scripts.summarize_stella_collision_test_particle_blocks import (
    SCHEMA as BLOCK_SCHEMA,
    summarize_block_trace,
)
from scripts.summarize_stella_collision_test_particle_matrix import (
    SCHEMA,
    summarize_matrix_trace,
)


def _write_matrix_trace(path, *, gyro_offdiagonal=0.0):
    rows = []
    base = np.array([[1.2, -0.1], [0.3, 1.4]])
    gyro_slope = np.diag([0.5, 0.8])
    gyro_slope[0, 1] = gyro_offdiagonal
    for ikx, kperp2 in ((1, 0.0), (2, 0.25), (3, 0.5)):
        matrix = base + kperp2 * gyro_slope
        for col in range(2):
            for row in range(2):
                rows.append([1, ikx, 0, 1, row + 1, col + 1, matrix[row, col], 0.0, kperp2, 0.01])
    path.write_text(SCHEMA + "\n# iky ikx iz species row col matrix_re matrix_im kperp2 code_dt\n")
    with path.open("a") as stream:
        np.savetxt(stream, np.asarray(rows))


def test_matrix_trace_isolates_diagonal_kperp_correction(tmp_path):
    trace = tmp_path / "matrix.dat"
    _write_matrix_trace(trace)

    report = summarize_matrix_trace(trace, expected_revision="abc123")

    assert report["status"] == "native_test_particle_matrix_structure_passed"
    assert report["matrices"] == 3
    assert report["matrix_size"] == 2
    assert report["metrics"]["gyro_offdiagonal_max_abs"] == 0.0
    assert report["metrics"]["gyro_kperp_linearity_max_abs"] == 0.0


def test_matrix_trace_rejects_kperp_dependent_offdiagonal(tmp_path):
    trace = tmp_path / "matrix.dat"
    _write_matrix_trace(trace, gyro_offdiagonal=0.1)

    with np.testing.assert_raises_regex(ValueError, "not diagonal"):
        summarize_matrix_trace(trace, expected_revision="abc123")


def test_block_trace_reconstructs_zero_kperp_matrix(tmp_path):
    blocks = tmp_path / "blocks.dat"
    block_rows = []
    lower = (0.0, -0.2)
    diagonal = (0.3, 0.4)
    upper = (0.1, 0.0)
    for iv in range(2):
        block_rows.append(
            [
                0,
                1,
                1,
                iv + 1,
                1,
                1,
                lower[iv],
                0.0,
                diagonal[iv],
                0.0,
                upper[iv],
                0.0,
                0.01,
            ]
        )
    blocks.write_text(BLOCK_SCHEMA + "\n# columns\n")
    with blocks.open("a") as stream:
        np.savetxt(stream, np.asarray(block_rows))

    matrix = tmp_path / "matrix.dat"
    expected = np.asarray([[1.3, 0.1], [-0.2, 1.4]])
    rows = []
    for column in range(2):
        for row in range(2):
            rows.append([1, 1, 0, 1, row + 1, column + 1, expected[row, column], 0.0, 0.0, 0.01])
    matrix.write_text(SCHEMA + "\n# columns\n")
    with matrix.open("a") as stream:
        np.savetxt(stream, np.asarray(rows))

    report = summarize_block_trace(blocks, matrix, expected_revision="abc123")

    assert report["status"] == "native_velocity_block_assembly_passed"
    assert report["rows"] == 2
    assert report["matrices_compared"] == 1
    assert report["metrics"] == {"relative_l2": 0.0, "max_abs": 0.0}
