from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts/prepare_stella_w7x_reference_run.py"
    spec = importlib.util.spec_from_file_location("prepare_stella_w7x_reference_run", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_stella_w7x_reference_run_writes_matched_input(tmp_path):
    module = _load_module()
    vmec = tmp_path / "wout_w7x.nc"
    eik = tmp_path / "w7x.eik"
    vmec.write_bytes(b"vmec")
    eik.write_text("10000 10.0 -0.01955439 1.0 1.25 0.1 0.538 2.65\n")

    metadata = module.prepare_stella_w7x_reference_run(
        output_dir=tmp_path / "stella_run",
        vmec_file=vmec,
        eik_reference=eik,
        stella_executable=tmp_path / "stella",
        external_fixture=tmp_path / "external.csv",
        observed_fixture=tmp_path / "observed.csv",
        comparison_output=tmp_path / "comparison.csv",
        torflux=0.64,
        alpha=0.0,
        gx_npol=6.0,
        nfp=5.0,
        q_value=None,
        ky_values="0.0,0.1,0.2,0.3",
        export_ky_values="0.1,0.2,0.3",
        nzed=256,
        nmu=8,
        nvgrid=16,
        tend=200.0,
        delt=0.1,
        nwrite=10,
        average_fraction=0.5,
        copy_vmec=True,
        overwrite=False,
    )

    run_dir = tmp_path / "stella_run"
    input_text = (run_dir / "stella_w7x_adiabatic_electrons.in").read_text()
    assert metadata["nfield_periods"] == pytest.approx(37.5)
    assert metadata["grid"]["ky_values"] == [0.0, 0.1, 0.2, 0.3]
    assert metadata["grid"]["stella_vpa_points"] == 32
    assert metadata["species"]["electron_model"] == "adiabatic"
    assert metadata["run_command"] == f"bash {run_dir / 'run_stella_reference.sh'}"
    assert (run_dir / "wout_w7x.nc").read_bytes() == b"vmec"
    assert "geometry_option = 'vmec'" in input_text
    assert "torflux = 0.64" in input_text
    assert "nfield_periods = 37.5" in input_text
    assert "nspec = 1" in input_text
    assert "type = 'ion'" in input_text
    assert "adiabatic_option = 'field-line-average-term'" in input_text
    assert "aky_min = 0" in input_text
    assert "aky_max = 0.3" in input_text
    assert "naky = 4" in input_text
    assert "nzed = 256" in input_text
    assert "nmu = 8" in input_text
    assert "nvgrid = 16" in input_text
    assert "tend = 200" in input_text
    assert "delt = 0.1" in input_text
    assert "--average-fraction 0.5" in metadata["export_command"]
    assert "--stella-z-coordinate zed_over_2pi" in metadata["export_command"]
    assert "--ky-values 0.1,0.2,0.3" in metadata["comparison_command"]
    assert "--resample-reference-to-observed-z" in metadata["comparison_command"]


def test_prepare_stella_w7x_reference_run_rejects_unmatched_export_ky(tmp_path):
    module = _load_module()
    vmec = tmp_path / "wout_w7x.nc"
    eik = tmp_path / "w7x.eik"
    vmec.write_bytes(b"vmec")
    eik.write_text("10000 10.0 -0.01955439 1.0 1.25 0.1 0.538 2.65\n")

    with pytest.raises(ValueError, match="export_ky_values"):
        module.prepare_stella_w7x_reference_run(
            output_dir=tmp_path / "stella_run",
            vmec_file=vmec,
            eik_reference=eik,
            stella_executable=tmp_path / "stella",
            external_fixture=tmp_path / "external.csv",
            observed_fixture=tmp_path / "observed.csv",
            comparison_output=tmp_path / "comparison.csv",
            torflux=0.64,
            alpha=0.0,
            gx_npol=6.0,
            nfp=5.0,
            q_value=None,
            ky_values="0.0,0.1,0.2,0.3",
            export_ky_values="0.1,0.4",
            nzed=256,
            nmu=8,
            nvgrid=16,
            tend=200.0,
            delt=0.1,
            nwrite=10,
            average_fraction=0.5,
        )


def test_committed_stella_w7x_reference_input_matches_production_gate():
    metadata = json.loads(
        (ROOT / "fixtures/stella_w7x_mode_structure_run/mode_structure_run_metadata.json").read_text()
    )
    input_text = (
        ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.in"
    ).read_text()
    run_script = (ROOT / "fixtures/stella_w7x_mode_structure_run/run_stella_reference.sh").read_text()
    export_script = (
        ROOT / "fixtures/stella_w7x_mode_structure_run/export_stella_fixture.sh"
    ).read_text()

    assert metadata["benchmark_name"] == "w7x_itg_external_stella_mode_structure_reference"
    assert metadata["status"] == "prepared_stella_run_pending_execution"
    assert metadata["vmec_source"] == "dependency://gx/benchmarks/linear/ITG_w7x/wout_w7x.nc"
    assert metadata["vmec_source_sha256"] == (
        "7d3bc31a4dd599b30619444da740c64fba27be997a2aaa25f5f211387616ab86"
    )
    assert metadata["geometry"]["torflux"] == 0.64
    assert metadata["geometry"]["alpha0"] == 0.0
    assert metadata["geometry"]["field_line_match"] == "nfield_periods = gx_npol * q_eik * nfp"
    assert metadata["eik_q_used"] == pytest.approx(1.158412)
    assert metadata["gx_npol"] == 6.0
    assert metadata["nfp"] == 5.0
    assert metadata["nfield_periods"] == pytest.approx(34.75236)
    assert metadata["species"]["electron_model"] == "adiabatic"
    assert metadata["species"]["adiabatic_option"] == "field-line-average-term"
    assert metadata["grid"]["ky_values"] == [0.0, 0.1, 0.2, 0.3]
    assert metadata["grid"]["export_ky_values"] == [0.1, 0.2, 0.3]
    assert metadata["grid"]["nzed"] == 256
    assert metadata["grid"]["nmu"] == 8
    assert metadata["grid"]["nvgrid"] == 16
    assert metadata["time"]["tend"] == 200.0
    assert metadata["time"]["delt"] == 0.1
    assert metadata["time"]["growth_average_fraction"] == 0.5
    assert "nfield_periods = 34.75236" in input_text
    assert "nspec = 1" in input_text
    assert "adiabatic_option = 'field-line-average-term'" in input_text
    assert "STELLA_EXECUTABLE" in run_script
    assert "stella_w7x_adiabatic_electrons.in" in run_script
    assert "export_stella_mode_structure_fixture.py" in export_script
    assert "--stella-z-coordinate zed_over_2pi" in export_script
    assert "--resample-reference-to-observed-z" in metadata["comparison_command"]
    assert "run_w7x_production_readiness_gate.py" in export_script
