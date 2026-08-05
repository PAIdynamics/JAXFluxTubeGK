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


def _write_single_ky_fixture(path: Path, *, growth_rate: float) -> None:
    path.write_text(
        "\n".join(
            (
                "ky,z_index,z,phi_real,phi_imag,growth_rate,frequency,normalization,source",
                f"0.1,0,0.0,1.0,0.0,{growth_rate},0.0,test,synthetic",
                f"0.1,1,1.0,1.0,0.0,{growth_rate},0.0,test,synthetic",
                "",
            )
        )
    )


def test_w7x_production_readiness_gate_blocks_on_missing_external_reference(
    tmp_path, monkeypatch
):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT],
    )
    monkeypatch.delitem(sys.modules, "examples", raising=False)
    monkeypatch.delitem(sys.modules, "examples.run_w7x_mode_structure_gate", raising=False)
    output = tmp_path / "production_readiness_gate.json"
    production_timing = tmp_path / "production_cpu_timing.json"
    production_timing.write_text(
        json.dumps(
            {
                "benchmark_name": "w7x_itg_cpu_timing",
                "status": "blocked_until_external_parity_passes",
                "passed": False,
            }
        )
    )

    report = module.run_w7x_production_readiness_gate(
        convergence_dir=CONVERGENCE,
        observed_fixture=OBSERVED,
        reference_fixture=tmp_path / "missing_external.csv",
        external_gate_dir=tmp_path / "external_gate",
        output_path=output,
        production_timing_path=production_timing,
    )

    written = json.loads(output.read_text())
    assert report == written
    assert report["status"] == "blocked_external_reference"
    assert not report["passed"]
    assert report["reduced_convergence_regression"]["passed"]
    assert report["external_mode_structure_gate"]["status"] == "pending_external_reference"
    assert not report["production_cpu_timing"]["passed"]
    assert report["production_cpu_timing"]["artifact_exists"]
    assert report["production_cpu_timing"]["artifact_status"] == (
        "blocked_until_external_parity_passes"
    )
    assert report["desc_optimization_status"].startswith("keep_reduced")
    assert any("external W7-X mode-structure fixture" in item for item in report["required_actions"])


def test_w7x_production_readiness_gate_blocks_on_open_external_parity(tmp_path):
    module = _load_module()
    observed = tmp_path / "observed.csv"
    reference = tmp_path / "reference.csv"
    _write_single_ky_fixture(observed, growth_rate=0.0)
    _write_single_ky_fixture(reference, growth_rate=0.1)

    report = module.run_w7x_production_readiness_gate(
        convergence_dir=CONVERGENCE,
        observed_fixture=observed,
        reference_fixture=reference,
        external_gate_dir=tmp_path / "external_gate",
        output_path=tmp_path / "production_readiness_gate.json",
        production_timing_path=tmp_path / "production_cpu_timing.json",
        ky_values="0.1",
    )

    assert report["status"] == "blocked_external_mode_structure_parity"
    assert not report["passed"]
    assert report["reduced_convergence_regression"]["passed"]
    assert report["external_mode_structure_gate"]["status"] == "open"
    assert report["external_mode_structure_gate"]["max_growth_error"] == 0.1
    assert any("mode-structure parity gap" in item for item in report["required_actions"])
    assert not any("export a matched external" in item for item in report["required_actions"])


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
