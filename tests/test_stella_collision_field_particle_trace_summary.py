from pathlib import Path

import pytest

from scripts.summarize_stella_collision_field_particle_trace import summarize_trace


def _write_trace(path: Path, *, duplicate: bool = False) -> None:
    rows = [
        "1 1 1 1 -1 1 1 -1.0 0.5 1.0 0.0 0.2 -0.1",
        "1 1 1 1 -1 1 2 -1.0 0.5 2.0 0.0 -0.4 0.3",
    ]
    if duplicate:
        rows[1] = rows[0]
    path.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_trace_v1\n"
        "# iv imu iky ikx iz tube species vpa mu before_re before_im rhs_re rhs_im\n"
        + "\n".join(rows)
        + "\n"
    )


def test_trace_summary_records_signed_action_and_species_split(tmp_path):
    trace = tmp_path / "trace.dat"
    _write_trace(trace)

    report = summarize_trace(trace, expected_revision="564ca09")

    assert report["status"] == "signed_native_trace_passed"
    assert report["rows"] == 2
    assert report["grid"]["nspecies"] == 2
    assert report["metrics"]["field_particle_rhs_l2"] == pytest.approx(0.5477225575)
    assert report["species_metrics"]["1"]["rhs_l2"] == pytest.approx(0.2236067977)


def test_trace_summary_rejects_duplicate_phase_space_rows(tmp_path):
    trace = tmp_path / "trace.dat"
    _write_trace(trace, duplicate=True)

    with pytest.raises(ValueError, match="duplicate"):
        summarize_trace(trace, expected_revision="564ca09")
