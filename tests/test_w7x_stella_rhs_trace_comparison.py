from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


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


def test_solver_term_reader_aliases_open_chain_streaming_names(tmp_path):
    module = _load_module()
    path = tmp_path / "terms.csv"
    path.write_text(
        "term,rhs_l2,rhs_fraction_of_total_l2,total_rhs_l2\n"
        "gkw_parallel_streaming_recurrence,2,0.2,10\n"
        "gkw_parallel_field_drive,3,0.3,10\n",
        encoding="utf-8",
    )

    rows = module._read_solver_term_rows(path)

    assert rows["parallel_streaming"] == rows["gkw_parallel_streaming_recurrence"]
    assert rows["parallel_field_drive"] == rows["gkw_parallel_field_drive"]


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
    assert contract["stella_endpoint_drop_applied"] is True
    assert not any(
        "duplicate periodic z endpoint" in item for item in contract["array_parity_blockers"]
    )
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


def test_drop_stella_periodic_endpoint_trims_requested_array_axis():
    module = _load_module()
    z_indices = np.arange(-2, 3)
    state = np.arange(2 * 5 * 3).reshape(2, 5, 3)

    trimmed_z, trimmed_state = module.drop_stella_periodic_endpoint(
        z_indices, state, axis=1
    )

    np.testing.assert_array_equal(trimmed_z, np.arange(-2, 2))
    np.testing.assert_array_equal(trimmed_state, state[:, :-1, :])


def test_phase_space_adapter_interpolates_complex_values_on_target_grid():
    module = _load_module()
    source_z = np.asarray([-0.5, 0.0, 0.5])
    source_vpar = np.asarray([-2.0, 0.0, 2.0])
    source_mu = np.asarray([0.0, 1.0])
    z, vpar, mu = np.meshgrid(source_z, source_vpar, source_mu, indexing="ij")
    values = (z + 2.0 * vpar + 3.0 * mu) + 1j * (4.0 * z - vpar + mu)

    result = module.interpolate_phase_space_to_grid(
        values,
        source_z=source_z,
        source_vpar=source_vpar,
        source_mu=source_mu,
        target_z=np.asarray([-0.25, 0.25]),
        target_vpar=np.asarray([-1.0, 1.0]),
        target_mu=np.asarray([0.25, 0.75]),
    )

    target = np.meshgrid(
        [-0.25, 0.25], [-1.0, 1.0], [0.25, 0.75], indexing="ij"
    )
    expected = (target[0] + 2.0 * target[1] + 3.0 * target[2]) + 1j * (
        4.0 * target[0] - target[1] + target[2]
    )
    np.testing.assert_allclose(result, expected)


def test_phase_space_adapter_rejects_extrapolation():
    module = _load_module()

    with np.testing.assert_raises_regex(ValueError, "target vpar.*extrapolation"):
        module.interpolate_phase_space_to_grid(
            np.zeros((2, 2, 2)),
            source_z=[0.0, 1.0],
            source_vpar=[-1.0, 1.0],
            source_mu=[0.0, 1.0],
            target_z=[0.0],
            target_vpar=[-2.0],
            target_mu=[0.5],
        )


def test_weighted_complex_metrics_remove_global_complex_scale():
    module = _load_module()
    reference = np.arange(1, 9).reshape(2, 2, 2) * (1.0 + 0.5j)
    candidate = reference / (2.0j)

    metrics = module.weighted_complex_metrics(
        reference,
        candidate,
        w_z=[0.25, 0.75],
        w_vpar=[1.0, 2.0],
        w_mu=[0.4, 0.6],
    )

    assert metrics["raw_relative_l2_error"] > 0.0
    assert metrics["aligned_relative_l2_error"] < 1.0e-14
    assert abs(metrics["alignment_scale_real"]) < 1.0e-14
    assert abs(metrics["alignment_scale_imag"] - 2.0) < 1.0e-14


def test_weighted_complex_metrics_accept_z_dependent_mu_weights():
    reference = np.ones((2, 2, 2), dtype=complex)
    candidate = reference.copy()
    metrics = _load_module().weighted_complex_metrics(
        reference,
        candidate,
        w_z=[0.5, 0.5],
        w_vpar=[1.0, 2.0],
        w_mu=[[0.1, 0.2], [0.3, 0.4]],
    )

    assert metrics["raw_relative_l2_error"] == 0.0


def test_stella_array_loader_infers_calls_drops_endpoint_and_converts_rhs(tmp_path):
    module = _load_module()
    path = tmp_path / "trace.dat"
    header = (
        "record step term iky ikx iz it ivmu iv imu is vpa mu "
        "wgts_vpa wgts_mu code_time code_dt real imag"
    )
    lines = [header]
    records = (
        ("pdf_g", "input_pdf"),
        ("phi", "field_phi"),
        ("rhs_delta", "mirror_force"),
        ("rhs_delta", "magnetic_drift_y"),
        ("rhs_delta", "magnetic_drift_x"),
        ("rhs_delta", "equilibrium_drive_wstar"),
        ("rhs_delta", "parallel_streaming"),
        ("rhs_total", "total"),
    )
    for call in range(2):
        for record, term in records:
            for iz in (-1, 0, 1):
                is_field = record == "phi"
                value = 100.0 * call + 10.0 + iz
                lines.append(
                    f"{record} 20 {term} 4 1 {iz} 1 0 "
                    f"{0 if is_field else 1} {0 if is_field else 1} "
                    f"{0 if is_field else 1} "
                    f"{0.0 if is_field else -1.0} {0.0 if is_field else 0.5} "
                    f"{0.0 if is_field else 2.0} {0.0 if is_field else 3.0 + iz} "
                    f"1.9 0.1 {value} {-value}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    common = {
        "iz_range": [-1, 1],
        "iv_range": [1, 1],
        "imu_range": [1, 1],
        "vpa_range": [-1.0, -1.0],
        "mu_range": [0.5, 0.5],
    }
    summary = {
        "code_dts": [0.1],
        "term_summaries": [
            {"record": record, "term": term, **common} for record, term in records
        ],
    }

    trace = module.load_stella_array_trace(path, summary)

    assert trace["rhs_call_count"] == 2
    assert trace["distribution"].shape == (2, 2, 1, 1)
    np.testing.assert_array_equal(trace["z"], [-0.5, 0.0])
    np.testing.assert_array_equal(trace["w_mu"], [[2.0], [3.0]])
    assert trace["distribution"][1, 0, 0, 0] == 109.0 - 109.0j
    assert trace["mirror_force"][0, 0, 0, 0] == 90.0 - 90.0j


def test_committed_stella_rhs_trace_comparison_contract():
    status = json.loads((COMPARISON / "stella_solver_rhs_trace_comparison_status.json").read_text())
    contract = json.loads((COMPARISON / "array_contract.json").read_text())
    rows = tuple(csv.DictReader((COMPARISON / "term_norm_comparison.csv").open()))
    by_group = {row["comparison_group"]: row for row in rows}

    weighted_rows = tuple(csv.DictReader((COMPARISON / "weighted_array_comparison.csv").open()))

    assert status["status"] == "weighted_array_parity_failed"
    assert status["raw_trace_used"] is True
    assert status["stella_required_record_terms_present"] is True
    assert status["direct_array_parity_ready"] is True
    assert contract["stella_n_z_raw"] == 257
    assert contract["stella_n_z_after_endpoint_drop"] == 256
    assert contract["stella_endpoint_drop_applied"] is True
    assert contract["stella_n_vpar"] == 32
    assert contract["solver_case"]["n_vpar"] == 16
    assert contract["inferred_stella_rhs_calls"] == 3
    assert contract["missing_array_records"] == []
    assert contract["rhs_calls_explicitly_labeled"] is True
    assert len(weighted_rows) == 30
    assert {row["quantity"] for row in weighted_rows} == {
        "distribution",
        "parallel_streaming",
        "mirror_force",
        "magnetic_drift",
        "equilibrium_drive",
        "total_rhs",
        "phi",
        "quasineutrality_numerator",
        "quasineutrality_denominator",
        "normalization",
    }
    denominator_rows = [
        row for row in weighted_rows if row["quantity"] == "quasineutrality_denominator"
    ]
    assert max(float(row["aligned_relative_l2_error"]) for row in denominator_rows) < 5.0e-4
    assert status["max_aligned_array_relative_l2_error"] > 0.99
    assert float(by_group["parallel_streaming_bundle"]["stella_rhs_fraction_of_total_l2"]) > 0.0
    assert float(by_group["mirror_force"]["stella_rhs_fraction_of_total_l2"]) > 0.0
    assert float(by_group["total_rhs"]["stella_rhs_fraction_of_total_l2"]) == 1.0
