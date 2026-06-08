import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_stellarator_linear_scan_example_writes_machine_readable_outputs(tmp_path):
    output_dir = tmp_path / "stellarator_scan"
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "examples/run_stellarator_linear_scan.py",
            "--output-dir",
            str(output_dir),
            "--steps-per-window",
            "1",
            "--n-windows",
            "1",
            "--dt",
            "0.002",
            "--ky-values",
            "0.0,0.35",
            "--n-vpar",
            "3",
            "--n-mu",
            "3",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS: stellarator linear scan" in result.stdout
    expected = {
        "geometry_audit.json",
        "geometry_audit.csv",
        "ky_growth.csv",
        "mode_structures.csv",
        "convergence_metadata.json",
        "convergence_history.csv",
        "quasilinear_proxy.json",
        "run_config.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    audit = json.loads((output_dir / "geometry_audit.json").read_text())
    assert audit["passed"]
    assert audit["checks"]["gx_eik_export_gate"]
    assert audit["geometry"]["geometry_source"] == "fixture"

    with (output_dir / "ky_growth.csv").open(newline="") as handle:
        ky_rows = tuple(csv.DictReader(handle))
    assert [float(row["ky"]) for row in ky_rows] == [0.0, 0.35]
    assert all(np.isfinite(float(row["growth_rate"])) for row in ky_rows)
    assert float(ky_rows[0]["quasilinear_contribution"]) == 0.0

    with (output_dir / "mode_structures.csv").open(newline="") as handle:
        mode_rows = tuple(csv.DictReader(handle))
    assert len(mode_rows) == 2 * int(audit["z"]["n_z"])
    assert {"ky", "z", "phi_real", "phi_imag", "growth_rate", "frequency"}.issubset(
        mode_rows[0]
    )

    metadata = json.loads((output_dir / "convergence_metadata.json").read_text())
    assert metadata["finite_growth"]
    assert metadata["finite_frequency"]
    assert metadata["steps_per_window"] == 1
