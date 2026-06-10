from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
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


def _load_packager():
    path = ROOT / "scripts/package_w7x_external_reference_bundle.py"
    spec = importlib.util.spec_from_file_location("package_w7x_external_reference_bundle", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_ingest():
    path = ROOT / "scripts/ingest_w7x_external_reference_outputs.py"
    spec = importlib.util.spec_from_file_location("ingest_w7x_external_reference_outputs", path)
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
            "--bundle-path",
            str(tmp_path / "bundle.tar.gz"),
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
    assert not report["local_bundle_exists"]
    assert "package the external-run inputs" in report["required_actions"][0]
    assert "build GX" in report["required_actions"][1]
    assert "package_w7x_external_reference_bundle.py" in report["bundle_command"]
    assert "ingest_returned_outputs.sh" in report["returned_outputs_ingest_command"]
    assert "gx_on_path" in report["local_capability"]


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
            "--bundle-path",
            str(tmp_path / "bundle.tar.gz"),
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
    assert status["local_bundle_exists"]
    assert status["local_bundle"] == (
        "fixtures/gx_w7x_mode_structure_run/w7x_external_reference_bundle.tar.gz"
    )
    assert "package_w7x_external_reference_bundle.py" in status["bundle_command"]
    assert "ingest_returned_outputs.sh" in status["returned_outputs_ingest_command"]
    assert "local_capability" in status
    assert any("run_external_reference.sh" in item for item in status["required_actions"])
    assert any("ingest_returned_outputs.sh" in item for item in status["required_actions"])


def test_committed_w7x_external_handoff_scripts_encode_blocker_commands():
    run_script = (
        ROOT / "fixtures/gx_w7x_mode_structure_run/run_external_reference.sh"
    ).read_text()
    timing_script = (
        ROOT / "fixtures/gx_w7x_mode_structure_run/run_production_timing_after_parity.sh"
    ).read_text()
    ingest_script = (
        ROOT / "fixtures/gx_w7x_mode_structure_run/ingest_returned_outputs.sh"
    ).read_text()

    assert "GX_EXECUTABLE" in run_script
    assert "run_w7x_external_reference_workflow.py" in run_script
    assert "--copy-vmec" in run_script
    assert "--run-gx" in run_script
    assert "--require-pass" in run_script
    assert "run_w7x_production_readiness_gate.py" in run_script

    assert "run_w7x_production_cpu_timing.py" in timing_script
    assert "run_w7x_production_readiness_gate.py --require-pass" in timing_script
    assert "ingest_w7x_external_reference_outputs.py" in ingest_script


def test_w7x_external_reference_bundle_contains_manifest_and_handoff(tmp_path):
    module = _load_packager()
    metadata = _metadata(tmp_path)
    run_dir = tmp_path / "gx_run"
    (run_dir / "README.md").write_text("external run readme\n")
    (run_dir / "run_external_reference.sh").write_text("#!/usr/bin/env bash\n")
    (run_dir / "ingest_returned_outputs.sh").write_text("#!/usr/bin/env bash\n")
    (run_dir / "run_production_timing_after_parity.sh").write_text("#!/usr/bin/env bash\n")
    output = tmp_path / "bundle.tar.gz"

    manifest = module.package_w7x_external_reference_bundle(
        metadata_path=metadata,
        output_path=output,
    )

    assert output.exists()
    assert manifest["include_vmec"]
    assert manifest["requires_repository_root"]
    assert "tar -xzf" in manifest["unpack_command"]
    assert "GX_EXECUTABLE=/path/to/gx" in manifest["external_run_command"]
    assert "ingest_returned_outputs.sh" in manifest["returned_outputs_ingest_command"]
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert "BUNDLE_MANIFEST.json" in names
        assert "fixtures/gx_w7x_mode_structure_run/itg_w7x_adiabatic_electrons.in" in names
        assert "fixtures/gx_w7x_mode_structure_run/wout_w7x.nc" in names
        assert "fixtures/gx_w7x_mode_structure_run/ingest_returned_outputs.sh" in names
        embedded = json.loads(archive.extractfile("BUNDLE_MANIFEST.json").read())
    assert embedded["files"] == manifest["files"]
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])


def test_w7x_external_reference_ingest_reports_missing_outputs(tmp_path):
    module = _load_ingest()
    metadata = _metadata(tmp_path)
    status = tmp_path / "ingest_status.json"

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
    assert report["status"] == "blocked_missing_external_outputs"
    assert not report["passed"]
    assert report["missing_outputs"] == ["gx_big_output", "gx_growth_output"]
    assert "copy the returned GX" in report["required_actions"][0]


def test_w7x_external_reference_ingest_copies_and_runs_return_path(
    tmp_path,
    monkeypatch,
):
    module = _load_ingest()
    metadata = _metadata(tmp_path)
    returned_big = tmp_path / "returned.big.nc"
    returned_growth = tmp_path / "returned.out.nc"
    returned_big.write_bytes(b"big")
    returned_growth.write_bytes(b"growth")
    status = tmp_path / "ingest_status.json"

    def fake_export(paths, metadata_payload):
        paths["external_fixture"].write_text("metadata_key,metadata_value\n")
        return {
            "status": "external_fixture_exported",
            "ky_count": 2,
            "n_z": 5,
            "metadata_name": metadata_payload["benchmark_name"],
        }

    monkeypatch.setattr(module, "_export_external_fixture", fake_export)
    monkeypatch.setattr(
        module,
        "_run_mode_structure_gate",
        lambda paths, args: {"passed": True, "status": "pass"},
    )
    monkeypatch.setattr(
        module,
        "_run_production_readiness_gate",
        lambda paths, args: {
            "passed": False,
            "status": "missing_production_cpu_timing_artifact",
            "required_actions": ["run true production-control CPU timing"],
        },
    )

    exit_code = module.main(
        [
            "--metadata",
            str(metadata),
            "--status-output",
            str(status),
            "--gx-big-output",
            str(returned_big),
            "--gx-growth-output",
            str(returned_growth),
            "--copy-outputs",
            "--resample-reference-to-observed-z",
        ]
    )

    report = json.loads(status.read_text())
    assert exit_code == 0
    assert report["status"] == "external_parity_passed_readiness_open"
    assert not report["passed"]
    assert report["external_fixture_exists"]
    assert report["external_mode_structure_gate"]["passed"]
    assert report["production_readiness_gate"]["status"] == (
        "missing_production_cpu_timing_artifact"
    )
    assert len(report["copied_outputs"]) == 2
    assert (tmp_path / "gx_run/itg_w7x_adiabatic_electrons.big.nc").read_bytes() == b"big"
    assert (tmp_path / "gx_run/itg_w7x_adiabatic_electrons.out.nc").read_bytes() == b"growth"
