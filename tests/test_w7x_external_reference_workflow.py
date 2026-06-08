from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts/run_w7x_external_reference_workflow.py"
    spec = importlib.util.spec_from_file_location("run_w7x_external_reference_workflow", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metadata(tmp_path: Path, *, external_fixture: Path | None = None) -> Path:
    run_dir = tmp_path / "gx_run"
    run_dir.mkdir()
    prepared_input = run_dir / "itg_w7x_adiabatic_electrons.in"
    vmec_source = tmp_path / "wout_w7x.nc"
    prepared_input.write_text("[Diagnostics]\nfields = true\n")
    vmec_source.write_bytes(b"vmec")
    payload = {
        "benchmark_name": "synthetic_w7x_external_reference",
        "prepared_input": str(prepared_input),
        "vmec_source": str(vmec_source),
        "vmec_destination": str(run_dir / "wout_w7x.nc"),
        "gx_big_output": str(run_dir / "itg_w7x_adiabatic_electrons.big.nc"),
        "gx_growth_output": str(run_dir / "itg_w7x_adiabatic_electrons.out.nc"),
        "external_fixture": str(external_fixture or (tmp_path / "external.csv")),
        "gx_executable": "path/to/gx",
        "ky_values": "0.1,0.2",
        "gx_z_coordinate": "theta_over_2pi",
    }
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(payload))
    return metadata


def test_w7x_external_reference_workflow_reports_missing_gx_executable(tmp_path):
    module = _load_module()
    metadata = _metadata(tmp_path)
    status = tmp_path / "status.json"

    exit_code = module.main(
        [
            "--metadata",
            str(metadata),
            "--status-output",
            str(status),
        ]
    )

    report = json.loads(status.read_text())
    assert exit_code == 0
    assert report["status"] == "blocked_missing_gx_executable"
    assert not report["passed"]
    assert report["prepared_input_exists"]
    assert report["vmec_source_exists"]
    assert not report["vmec_destination_exists"]
    assert not report["gx_executable_exists"]
    assert "build GX" in report["required_actions"][0]


def test_w7x_external_reference_workflow_copy_vmec_updates_status(tmp_path):
    module = _load_module()
    metadata = _metadata(tmp_path)
    status = tmp_path / "status.json"

    module.main(
        [
            "--metadata",
            str(metadata),
            "--status-output",
            str(status),
            "--copy-vmec",
        ]
    )

    report = json.loads(status.read_text())
    assert report["status"] == "blocked_missing_gx_executable"
    assert report["copy_vmec"]
    assert report["vmec_destination_exists"]
    assert (tmp_path / "gx_run/wout_w7x.nc").read_bytes() == b"vmec"


def test_w7x_external_reference_workflow_passes_when_external_fixture_exists(tmp_path):
    module = _load_module()
    external = tmp_path / "external.csv"
    external.write_text("metadata_key,metadata_value\n")
    metadata = _metadata(tmp_path, external_fixture=external)
    status = tmp_path / "status.json"

    module.main(
        [
            "--metadata",
            str(metadata),
            "--status-output",
            str(status),
        ]
    )

    report = json.loads(status.read_text())
    assert report["status"] == "external_fixture_available"
    assert report["passed"]
    assert report["external_fixture_exists"]


def test_committed_w7x_external_reference_status_records_current_blocker():
    status = json.loads(
        (ROOT / "fixtures/gx_w7x_mode_structure_run/external_reference_status.json").read_text()
    )

    assert status["status"] == "blocked_missing_gx_executable"
    assert not status["passed"]
    assert status["prepared_input_exists"]
    assert status["vmec_source_exists"]
    assert not status["gx_big_output_exists"]
    assert not status["gx_growth_output_exists"]
    assert not status["external_fixture_exists"]
