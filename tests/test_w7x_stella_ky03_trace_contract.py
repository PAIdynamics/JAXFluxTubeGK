from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "fixtures/w7x_ky03_stella_trace_contract"


def _load_module():
    path = ROOT / "scripts/audit_w7x_stella_ky03_trace_contract.py"
    spec = importlib.util.spec_from_file_location("audit_w7x_stella_ky03_trace_contract", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_stella_output(path: Path) -> None:
    netcdf = pytest.importorskip("netCDF4")
    with netcdf.Dataset(path, mode="w") as data:
        data.createDimension("ky", 2)
        data.createDimension("kx", 1)
        data.createDimension("zed", 3)
        data.createDimension("alpha", 1)
        data.createDimension("t", 1)
        data.createDimension("tube", 1)
        data.createDimension("ri", 2)

        ky = data.createVariable("ky", "f8", ("ky",))
        ky[:] = np.asarray([0.1, 0.3])
        kx = data.createVariable("kx", "f8", ("kx",))
        kx[:] = np.asarray([0.0])
        zed = data.createVariable("zed", "f8", ("zed",))
        zed[:] = np.asarray([-np.pi, 0.0, np.pi])
        time = data.createVariable("t", "f8", ("t",))
        time[:] = np.asarray([12.5])

        bmag = data.createVariable("bmag", "f8", ("zed", "alpha"))
        bmag[:, 0] = np.asarray([1.0, 1.2, 1.0])
        b_dot_gradz = data.createVariable("b_dot_gradz", "f8", ("zed", "alpha"))
        b_dot_gradz[:, 0] = 2.0 * np.pi * np.asarray([0.5, 0.6, 0.5])
        drift = data.createVariable("B_times_gradB_dot_grady", "f8", ("zed", "alpha"))
        drift[:, 0] = np.asarray([0.7, 0.8, -0.7])

        kperp2 = data.createVariable("kperp2", "f8", ("zed", "alpha", "kx", "ky"))
        kperp2[:, 0, 0, 0] = np.asarray([10.0, 11.0, 10.0])
        kperp2[:, 0, 0, 1] = np.asarray([3.0, 4.0, 3.0])

        phi = data.createVariable("phi_vs_t", "f8", ("t", "tube", "zed", "kx", "ky", "ri"))
        phi[0, 0, :, 0, 1, 0] = np.asarray([10.0, 20.0, 30.0])
        phi[0, 0, :, 0, 1, 1] = np.asarray([1.0, 2.0, 3.0])

        g2 = data.createVariable("g2_vs_zkykxs", "f8", ("t", "tube", "kx", "ky", "zed", "alpha"))
        g2[:] = 0.0


def test_standard_stella_trace_loader_drops_periodic_endpoint(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "synthetic.out.nc"
    _write_synthetic_stella_output(path)

    trace = module.load_standard_stella_trace_summary(
        path,
        ky=0.3,
        z_scale=1.0 / (2.0 * np.pi),
    )
    arrays = trace["arrays"]

    assert trace["ky_index_python"] == 1
    assert trace["ky_index_fortran"] == 2
    assert trace["kx_index_fortran"] == 1
    assert trace["time_final"] == 12.5
    assert tuple(arrays["z"]) == (-0.5, 0.0)
    assert tuple(arrays["bmag"]) == (1.0, 1.2)
    assert tuple(arrays["b_dot_gradz"] / (2.0 * np.pi)) == pytest.approx((0.5, 0.6))
    assert tuple(arrays["kperp2"]) == (3.0, 4.0)
    assert tuple(arrays["phi"]) == (10.0 + 1.0j, 20.0 + 2.0j)
    assert trace["optional_trace_shapes"]["g2_vs_zkykxs"] == (1, 1, 1, 2, 3, 1)


def test_standard_stella_output_is_not_a_complex_rhs_trace(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "synthetic.out.nc"
    _write_synthetic_stella_output(path)
    trace = module.load_standard_stella_trace_summary(
        path,
        ky=0.3,
        z_scale=1.0 / (2.0 * np.pi),
    )

    availability = module.build_trace_availability(trace)

    assert availability["standard_output_has_complex_phi"] is True
    assert availability["standard_output_has_geometry"] is True
    assert availability["standard_output_distribution_energy_variables"] == ["g2_vs_zkykxs"]
    assert availability["standard_output_has_complex_distribution"] is False
    assert availability["standard_output_has_per_term_rhs"] is False
    assert "rhs_parallel_streaming" in " ".join(availability["missing_for_true_term_parity"])


def test_committed_stella_trace_contract_records_rhs_blocker():
    status = json.loads((CONTRACT / "stella_ky03_trace_contract_status.json").read_text())
    patch_plan = (CONTRACT / "stella_rhs_trace_patch_plan.md").read_text()

    assert status["status"] == "blocked_missing_complex_stella_rhs_trace"
    assert status["focus_ky"] == 0.3
    assert status["comparison_row_count"] == 256
    assert status["geometry_summary"]["direct_geometry_contract_passed"] is True
    assert status["geometry_summary"]["strict_out_nc_precision_contract_passed"] is False
    assert status["geometry_summary"]["max_abs_F_error"] < 1.0e-6
    assert status["geometry_summary"]["max_abs_B_error"] < 7.0e-4
    assert status["availability"]["standard_output_has_complex_distribution"] is False
    assert status["availability"]["standard_output_has_per_term_rhs"] is False
    assert status["dominant_solver_rhs_terms"][0]["term"] == "parallel_streaming"
    assert "add_explicit_gyrokinetic_terms" in patch_plan
    assert "advance_parallel_streaming_explicit" in patch_plan
    assert "advance_mirror_explicit" in patch_plan
