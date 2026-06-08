from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OBSERVED = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"


def test_w7x_mode_structure_gate_reports_pending_missing_external_fixture(tmp_path):
    output_dir = tmp_path / "pending_gate"
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "examples/run_w7x_mode_structure_gate.py",
            "--reference-fixture",
            str(tmp_path / "missing_external.csv"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PENDING" in result.stdout
    status = json.loads((output_dir / "gate_status.json").read_text())
    assert status["status"] == "pending_external_reference"
    assert not status["passed"]
    assert status["next_required_artifact"].endswith("missing_external.csv")
    assert "export_gx_mode_structure_fixture.py" in status["gx_export_command"]


def test_w7x_mode_structure_gate_passes_committed_fixture_self_check(tmp_path):
    output_dir = tmp_path / "self_gate"
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "examples/run_w7x_mode_structure_gate.py",
            "--observed-fixture",
            str(OBSERVED),
            "--reference-fixture",
            str(OBSERVED),
            "--ky-values",
            "0.1,0.3",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS" in result.stdout
    status = json.loads((output_dir / "gate_status.json").read_text())
    assert status["status"] == "pass"
    assert status["passed"]
    assert status["ky_values"] == [0.1, 0.3]
    assert status["max_growth_error"] == 0.0
    assert status["max_frequency_error"] == 0.0
    assert status["max_profile_error"] == 0.0
    report_lines = (output_dir / "mode_structure_gate.csv").read_text().splitlines()
    assert len(report_lines) == 3
