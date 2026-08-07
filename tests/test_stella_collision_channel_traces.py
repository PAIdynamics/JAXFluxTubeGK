from pathlib import Path

import pytest

from scripts.run_stella_collision_channel_traces import CHANNELS, summarize_channel_traces


def _write_trace(path: Path, before: float, rhs: float) -> Path:
    path.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_trace_v1\n"
        "# iv imu iky ikx iz tube species vpa mu before_re before_im rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 -1.0 0.5 {before} 0.0 {rhs} 0.0\n"
    )
    return path


def test_pair_resolved_summary_requires_common_input_and_reports_closure(tmp_path):
    rhs = {
        "all": 1.0,
        "ion_ion": 0.1,
        "ion_electron": 0.2,
        "electron_electron": 0.3,
        "electron_ion": 0.4,
    }
    paths = {
        name: _write_trace(tmp_path / f"{name}.dat", 2.0, rhs[name]) for name in CHANNELS
    }

    report = summarize_channel_traces(paths, expected_revision="564ca09")

    assert report["status"] == "pair_resolved_native_traces_passed"
    assert report["metrics"]["identical_input_state"]
    assert report["metrics"]["isolated_sum_to_full_relative_l2"] == pytest.approx(0.0)


def test_pair_resolved_summary_rejects_input_mismatch(tmp_path):
    paths = {
        name: _write_trace(tmp_path / f"{name}.dat", 2.0, 0.2) for name in CHANNELS
    }
    _write_trace(paths["electron_ion"], 3.0, 0.2)

    with pytest.raises(ValueError, match="identical input"):
        summarize_channel_traces(paths, expected_revision="564ca09")
