from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "fixtures/w7x_ky03_stella_rhs_trace_comparison"


def _load_module():
    path = ROOT / "scripts/compare_w7x_stella_rhs_trace_to_solver_balance.py"
    spec = importlib.util.spec_from_file_location("compare_w7x_stella_rhs_trace", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_term_comparison_converts_stella_rhs_dt_units():
    module = _load_module()
    stella_summary = {
        "code_dts": [0.5],
        "term_summaries": [
            {"record": "pdf_g", "term": "input_pdf", "l2_norm": 1.0, "max_abs": 1.0},
            {"record": "rhs_total", "term": "total", "l2_norm": 4.0, "max_abs": 2.0},
            {"record": "rhs_delta", "term": "parallel_streaming", "l2_norm": 2.0, "max_abs": 1.0},
            {"record": "rhs_delta", "term": "mirror_force", "l2_norm": 1.0, "max_abs": 0.5},
            {"record": "rhs_delta", "term": "magnetic_drift_y", "l2_norm": 1.0, "max_abs": 0.5},
            {"record": "rhs_delta", "term": "magnetic_drift_x", "l2_norm": 0.0, "max_abs": 0.0},
            {"record": "rhs_delta", "term": "equilibrium_drive_wstar", "l2_norm": 3.0, "max_abs": 1.5},
        ],
    }
    solver_rows = {
        "parallel_streaming": {"rhs_l2": 10.0, "rhs_fraction_of_total_l2": 0.5, "total_rhs_l2": 20.0},
        "parallel_field_drive": {"rhs_l2": 2.0, "rhs_fraction_of_total_l2": 0.1, "total_rhs_l2": 20.0},
        "mirror_force": {"rhs_l2": 1.0, "rhs_fraction_of_total_l2": 0.05, "total_rhs_l2": 20.0},
        "magnetic_drift": {"rhs_l2": 4.0, "rhs_fraction_of_total_l2": 0.2, "total_rhs_l2": 20.0},
        "drift_field_drive": {"rhs_l2": 1.0, "rhs_fraction_of_total_l2": 0.05, "total_rhs_l2": 20.0},
        "equilibrium_drive": {"rhs_l2": 3.0, "rhs_fraction_of_total_l2": 0.15, "total_rhs_l2": 20.0},
    }

    rows = module._term_comparison_rows(stella_summary, solver_rows)
    by_group = {row["comparison_group"]: row for row in rows}

    assert by_group["total_rhs"]["stella_rhs_l2_continuous"] == 8.0
    assert by_group["parallel_streaming_bundle"]["stella_rhs_l2_continuous"] == 4.0
    assert by_group["parallel_streaming_bundle"]["stella_rhs_fraction_of_total_l2"] == 0.5
    assert by_group["parallel_streaming_bundle"]["solver_rhs_fraction_sum_of_total_l2"] == 0.6
    assert by_group["equilibrium_drive"]["stella_rhs_fraction_of_total_l2"] == 0.75


def test_array_contract_reports_shape_and_fixture_blockers(tmp_path: Path):
    module = _load_module()
    geometry_csv = tmp_path / "geometry.csv"
    geometry_csv.write_text("z\n0\n1\n", encoding="utf-8")
    stella_summary = {
        "trace_format": "stellarator_gk_stella_rhs_trace_v1",
        "rhs_units": "stella_native_rhs_times_code_dt",
        "total_rows": 10,
        "steps": [20],
        "code_dts": [0.1],
        "term_summaries": [
            {
                "record": "pdf_g",
                "term": "input_pdf",
                "iz_range": [-1, 1],
                "iv_range": [1, 4],
                "imu_range": [1, 2],
                "vpa_range": [-3.0, 3.0],
                "mu_range": [0.1, 1.0],
            },
            {
                "record": "rhs_total",
                "term": "total",
                "rows": 24,
                "iz_range": [-1, 1],
                "iv_range": [1, 4],
                "imu_range": [1, 2],
                "vpa_range": [-3.0, 3.0],
                "mu_range": [0.1, 1.0],
            },
        ],
    }
    solver_status = {"case_summaries": [{"case": "case", "n_vpar": 2, "n_mu": 2}]}
    solver_metadata = {"geometry_balance_csv": str(geometry_csv)}
    solver_velocity = {
        "vpar_range": [-1.0, 1.0],
        "mu_range": [0.1, 1.0],
    }
    solver_rows = {"parallel_streaming": {"rhs_l2": 1.0}}

    contract = module._array_contract_payload(
        stella_summary,
        solver_rows=solver_rows,
        solver_status=solver_status,
        solver_metadata=solver_metadata,
        solver_velocity=solver_velocity,
        raw_trace_used=False,
        trace_path=Path("missing.dat"),
    )

    assert contract["direct_array_parity_ready"] is False
    assert contract["stella_n_z_raw"] == 3
    assert contract["stella_n_z_after_endpoint_drop"] == 2
    assert any("duplicate periodic z endpoint" in item for item in contract["array_parity_blockers"])
    assert any("different n_vpar" in item for item in contract["array_parity_blockers"])
    assert any("scalar term summaries" in item for item in contract["array_parity_blockers"])
    assert any("velocity quadrature weights" in item for item in contract["array_parity_blockers"])


def test_array_contract_accepts_v2_velocity_weight_columns(tmp_path: Path):
    module = _load_module()
    geometry_csv = tmp_path / "geometry.csv"
    geometry_csv.write_text("z\n0\n1\n", encoding="utf-8")
    stella_summary = {
        "trace_format": "stellarator_gk_stella_rhs_trace_v2",
        "rhs_units": "stella_native_rhs_times_code_dt",
        "total_rows": 10,
        "steps": [20],
        "code_dts": [0.1],
        "velocity_weight_columns_present": True,
        "term_summaries": [
            {
                "record": "pdf_g",
                "term": "input_pdf",
                "iz_range": [-1, 1],
                "iv_range": [1, 2],
                "imu_range": [1, 2],
                "vpa_range": [-1.0, 1.0],
                "mu_range": [0.1, 1.0],
                "velocity_weight_columns_present": True,
                "wgts_vpa_range": [0.1, 0.2],
                "wgts_mu_range": [0.3, 0.4],
            },
            {
                "record": "rhs_total",
                "term": "total",
                "rows": 8,
                "iz_range": [-1, 1],
                "iv_range": [1, 2],
                "imu_range": [1, 2],
                "vpa_range": [-1.0, 1.0],
                "mu_range": [0.1, 1.0],
            },
        ],
    }
    solver_status = {"case_summaries": [{"case": "case", "n_vpar": 2, "n_mu": 2}]}
    solver_metadata = {"geometry_balance_csv": str(geometry_csv)}
    solver_velocity = {
        "vpar_range": [-1.0, 1.0],
        "mu_range": [0.1, 1.0],
    }
    solver_rows = {"parallel_streaming": {"rhs_l2": 1.0}}

    contract = module._array_contract_payload(
        stella_summary,
        solver_rows=solver_rows,
        solver_status=solver_status,
        solver_metadata=solver_metadata,
        solver_velocity=solver_velocity,
        raw_trace_used=True,
        trace_path=Path("trace.dat"),
    )

    assert contract["stella_velocity_weight_columns_present"] is True
    assert contract["stella_wgts_vpa_range"] == [0.1, 0.2]
    assert not any("velocity quadrature weights" in item for item in contract["array_parity_blockers"])


def test_committed_stella_rhs_trace_comparison_contract():
    status = json.loads((COMPARISON / "stella_solver_rhs_trace_comparison_status.json").read_text())
    contract = json.loads((COMPARISON / "array_contract.json").read_text())
    rows = tuple(csv.DictReader((COMPARISON / "term_norm_comparison.csv").open()))
    by_group = {row["comparison_group"]: row for row in rows}

    assert status["status"] == "blocked_array_contract_mismatch"
    assert status["raw_trace_used"] is True
    assert status["stella_required_record_terms_present"] is True
    assert status["direct_array_parity_ready"] is False
    assert contract["stella_n_z_raw"] == 257
    assert contract["stella_n_z_after_endpoint_drop"] == 256
    assert contract["stella_n_vpar"] == 32
    assert contract["solver_case"]["n_vpar"] == 16
    assert float(by_group["parallel_streaming_bundle"]["stella_rhs_fraction_of_total_l2"]) > 0.0
    assert float(by_group["mirror_force"]["stella_rhs_fraction_of_total_l2"]) > 0.0
    assert float(by_group["total_rhs"]["stella_rhs_fraction_of_total_l2"]) == 1.0
