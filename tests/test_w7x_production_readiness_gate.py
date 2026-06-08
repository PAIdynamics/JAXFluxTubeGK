from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERGENCE = ROOT / "fixtures/w7x_itg_convergence_study"
OBSERVED = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"


def _load_module():
    path = ROOT / "scripts/run_w7x_production_readiness_gate.py"
    spec = importlib.util.spec_from_file_location("run_w7x_production_readiness_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_w7x_production_readiness_gate_blocks_on_missing_external_reference(tmp_path):
    module = _load_module()
    output = tmp_path / "production_readiness_gate.json"

    report = module.run_w7x_production_readiness_gate(
        convergence_dir=CONVERGENCE,
        observed_fixture=OBSERVED,
        reference_fixture=tmp_path / "missing_external.csv",
        external_gate_dir=tmp_path / "external_gate",
        output_path=output,
        production_timing_path=tmp_path / "production_cpu_timing.json",
    )

    written = json.loads(output.read_text())
    assert report == written
    assert report["status"] == "blocked_external_reference"
    assert not report["passed"]
    assert report["reduced_convergence_regression"]["passed"]
    assert report["external_mode_structure_gate"]["status"] == "pending_external_reference"
    assert not report["production_cpu_timing"]["passed"]
    assert report["desc_optimization_status"].startswith("keep_reduced")
    assert any("external W7-X mode-structure fixture" in item for item in report["required_actions"])


def test_w7x_production_readiness_gate_blocks_on_missing_production_timing_after_parity(
    tmp_path,
):
    module = _load_module()

    report = module.run_w7x_production_readiness_gate(
        convergence_dir=CONVERGENCE,
        observed_fixture=OBSERVED,
        reference_fixture=OBSERVED,
        external_gate_dir=tmp_path / "external_gate",
        output_path=tmp_path / "production_readiness_gate.json",
        production_timing_path=tmp_path / "production_cpu_timing.json",
        ky_values="0.1,0.3",
    )

    assert report["status"] == "missing_production_cpu_timing_artifact"
    assert not report["passed"]
    assert report["reduced_convergence_regression"]["passed"]
    assert report["external_mode_structure_gate"]["passed"]
    assert report["external_mode_structure_gate"]["max_growth_error"] == 0.0
    assert report["production_cpu_timing"]["status"] == "missing_production_cpu_timing_artifact"
    assert any("production-control CPU timing" in item for item in report["required_actions"])
