import numpy as np

from scripts.replay_stella_implicit_collision_state import replay_implicit_collision
from scripts.summarize_stella_collision_test_particle_matrix import SCHEMA


def test_replays_native_implicit_collision_update(tmp_path):
    matrix = np.array([[1.2, -0.1], [0.3, 1.4]])
    state = np.array([1.0 + 0.2j, -0.4 + 0.7j])
    rhs = np.array([0.3 - 0.1j, -0.2 + 0.4j])
    dt = 0.01
    output = np.linalg.solve(matrix, state + dt * rhs)

    matrix_trace = tmp_path / "matrix.dat"
    matrix_trace.write_text(
        SCHEMA + "\n# iky ikx iz species row col matrix_re matrix_im kperp2 code_dt\n"
    )
    with matrix_trace.open("a") as stream:
        for col in range(2):
            for row in range(2):
                stream.write(
                    f"1 1 0 1 {row + 1} {col + 1} "
                    f"{matrix[row, col]} 0 0 {dt}\n"
                )

    field_trace = tmp_path / "field.dat"
    final_trace = tmp_path / "final.dat"
    field_trace.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_trace_v1\n# columns\n"
    )
    final_trace.write_text(
        "# schema=stellarator_gk_stella_collision_final_state_v1\n# columns\n"
    )
    with field_trace.open("a") as field_stream, final_trace.open("a") as final_stream:
        for iv in range(2):
            field_stream.write(
                f"{iv + 1} 1 1 1 0 1 1 0 0 "
                f"{state[iv].real} {state[iv].imag} {rhs[iv].real} {rhs[iv].imag}\n"
            )
            final_stream.write(
                f"{iv + 1} 1 1 1 0 1 1 {state[iv].real} {state[iv].imag} "
                f"{output[iv].real} {output[iv].imag}\n"
            )

    report = replay_implicit_collision(
        matrix_trace,
        field_trace,
        final_trace,
        expected_revision="abc123",
    )

    assert report["status"] == "native_implicit_collision_replay_passed"
    assert report["metrics"]["replay_relative_l2"] < 1.0e-15
    assert report["metrics"]["field_particle_effect_relative_l2"] > 0.0
