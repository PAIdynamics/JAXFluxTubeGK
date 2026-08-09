import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jax_fluxtube_gk import load_per_ky_mode_structure_fixture_csv


ROOT = Path(__file__).resolve().parents[1]
W7X_FIXTURE = ROOT / "fixtures/w7x_itg_reduced_benchmark"


def test_committed_w7x_reduced_benchmark_fixture_contract():
    metadata = json.loads((W7X_FIXTURE / "benchmark_metadata.json").read_text())
    expected_files = set(metadata["artifact_contract"]["required_files"])

    assert expected_files.issubset({path.name for path in W7X_FIXTURE.iterdir()})
    assert metadata["benchmark_name"] == "w7x_itg_adiabatic_electrons_reduced"
    assert metadata["validation_status"] == (
        "real_external_geometry_internal_reduced_solver_regression"
    )
    assert metadata["external_growth_frequency_mode_structure_reference"] is None
    assert metadata["external_reference_workflow"]["status"] == (
        "prepared_gx_run_pending_external_execution"
    )
    assert metadata["external_reference_workflow"]["run_prep_dir"] == (
        "fixtures/gx_w7x_mode_structure_run"
    )
    assert metadata["gx_input_reference"]["vmec_file"] == "wout_w7x.nc"
    assert metadata["gx_input_reference"]["torflux"] == pytest.approx(0.64)
    assert metadata["gx_input_reference"]["ion_temperature_gradient"] == pytest.approx(3.0)

    audit = json.loads((W7X_FIXTURE / "geometry_audit.json").read_text())
    assert audit["passed"]
    assert audit["geometry"]["geometry_source"] == "eik"
    assert audit["geometry"]["rho"] is None
    assert audit["geometry"]["radial_coordinate"] == "external_eik_table"
    assert not audit["mirror_fd_check_enabled"]
    assert audit["checks"]["gx_eik_export_contract"]
    assert audit["field_stats"]["B"]["min"] > 0.0
    assert audit["kperp2"]["min"] >= -1.0e-12

    convergence = json.loads((W7X_FIXTURE / "convergence_metadata.json").read_text())
    assert convergence["finite_growth"]
    assert convergence["finite_frequency"]
    assert convergence["n_windows"] == metadata["reduced_solver_controls"]["n_windows"]

    fixture = load_per_ky_mode_structure_fixture_csv(W7X_FIXTURE / "mode_structures.csv")
    np.testing.assert_allclose(fixture.ky, metadata["reduced_solver_controls"]["ky_values"])
    assert fixture.phi.shape == (
        len(metadata["reduced_solver_controls"]["ky_values"]),
        metadata["reduced_solver_controls"]["n_z"],
    )
    assert fixture.source == "stellarator-linear-scan:eik"
    assert fixture.normalization == "unit_mode_chain_amplitude"
    assert np.all(np.isfinite(np.asarray(fixture.phi.real)))
    assert np.all(np.isfinite(np.asarray(fixture.phi.imag)))

    with (W7X_FIXTURE / "ky_growth.csv").open(newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    assert [float(row["ky"]) for row in rows] == metadata["reduced_solver_controls"]["ky_values"]
    np.testing.assert_allclose(
        [float(row["growth_rate"]) for row in rows],
        np.asarray(fixture.growth_rate),
    )
    np.testing.assert_allclose(
        [float(row["frequency"]) for row in rows],
        np.asarray(fixture.frequency),
    )
    assert all(np.isfinite(float(row["kperp2_average"])) for row in rows)


@pytest.mark.external
def test_stellarator_linear_scan_example_accepts_gx_gist_eik_source(
    tmp_path,
    gx_root: Path,
):
    eik_reference = (
        gx_root
        / "geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )

    output_dir = tmp_path / "w7x_eik_scan"
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "examples/run_stellarator_linear_scan.py",
            "--geometry-source",
            "eik",
            "--eik-reference",
            str(eik_reference),
            "--output-dir",
            str(output_dir),
            "--n-z",
            "9",
            "--ky-values",
            "0.0,0.1",
            "--n-vpar",
            "3",
            "--n-mu",
            "3",
            "--dt",
            "0.001",
            "--steps-per-window",
            "1",
            "--n-windows",
            "1",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS: stellarator linear scan" in result.stdout
    audit = json.loads((output_dir / "geometry_audit.json").read_text())
    assert audit["passed"]
    assert audit["geometry"]["geometry_source"] == "eik"
    assert audit["geometry"]["n_z"] == 9
    assert not audit["mirror_fd_check_enabled"]
