from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts/summarize_stella_w7x_rhs_trace.py"
    spec = importlib.util.spec_from_file_location("summarize_stella_w7x_rhs_trace", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_trace_counts_terms_and_norms(tmp_path: Path):
    module = _load_module()
    trace = tmp_path / "trace.dat"
    trace.write_text(
        "\n".join(
            (
                "record step term iky ikx iz it ivmu iv imu is vpa mu code_time code_dt real imag",
                "pdf_g 20 input_pdf 4 1 -1 1 0 1 1 1 -3.0 0.1 2.0 0.1 3.0 4.0",
                "rhs_delta 20 mirror_force 4 1 -1 1 0 1 1 1 -3.0 0.1 2.0 0.1 0.0 2.0",
                "rhs_delta 20 mirror_force 4 1 1 1 1 2 1 1 -2.0 0.1 2.0 0.1 0.0 -2.0",
                "",
            )
        ),
        encoding="utf-8",
    )

    summary = module.summarize_trace(
        trace,
        required_record_terms=(("pdf_g", "input_pdf"), ("rhs_delta", "mirror_force")),
        provenance={"stella_revision": "test-revision"},
    )
    terms = {(entry["record"], entry["term"]): entry for entry in summary["term_summaries"]}

    assert summary["total_rows"] == 3
    assert summary["steps"] == [20]
    assert summary["iky_values"] == [4]
    assert summary["ikx_values"] == [1]
    assert summary["required_record_terms_present"] is True
    assert summary["trace_format"] == "stellarator_gk_stella_rhs_trace_v1"
    assert summary["velocity_weight_columns_present"] is False
    assert summary["provenance"] == {"stella_revision": "test-revision"}
    assert terms[("pdf_g", "input_pdf")]["rows"] == 1
    assert terms[("pdf_g", "input_pdf")]["l2_norm"] == pytest.approx(5.0)
    assert terms[("pdf_g", "input_pdf")]["velocity_weight_columns_present"] is False
    assert terms[("rhs_delta", "mirror_force")]["rows"] == 2
    assert terms[("rhs_delta", "mirror_force")]["l2_norm"] == pytest.approx(2.0**1.5)
    assert terms[("rhs_delta", "mirror_force")]["iz_range"] == [-1, 1]
    assert terms[("rhs_delta", "mirror_force")]["ivmu_range"] == [0, 1]


def test_summarize_trace_accepts_v2_velocity_weight_columns(tmp_path: Path):
    module = _load_module()
    trace = tmp_path / "trace_v2.dat"
    trace.write_text(
        "\n".join(
            (
                "record step term iky ikx iz it ivmu iv imu is vpa mu "
                "wgts_vpa wgts_mu code_time code_dt real imag",
                "pdf_g 20 input_pdf 4 1 -1 1 0 1 1 1 -3.0 0.1 "
                "0.25 2.0 2.0 0.1 3.0 4.0",
                "pdf_g 20 input_pdf 4 1 0 1 1 2 1 1 -2.0 0.1 "
                "0.5 3.0 2.0 0.1 0.0 2.0",
                "",
            )
        ),
        encoding="utf-8",
    )

    summary = module.summarize_trace(trace, required_record_terms=(("pdf_g", "input_pdf"),))
    term = summary["term_summaries"][0]

    assert summary["trace_format"] == "stellarator_gk_stella_rhs_trace_v2"
    assert summary["rhs_calls"] == [0]
    assert summary["v3_required_record_terms_present"] is False
    assert summary["velocity_weight_columns_present"] is True
    assert term["velocity_weight_columns_present"] is True
    assert term["wgts_vpa_range"] == [0.25, 0.5]
    assert term["wgts_mu_range"] == [2.0, 3.0]
    assert term["l2_norm"] == pytest.approx((25.0 + 4.0) ** 0.5)
    assert term["weighted_velocity_l2_norm"] == pytest.approx((0.25 * 2.0 * 25.0 + 0.5 * 3.0 * 4.0) ** 0.5)


def test_summarize_trace_accepts_v3_rhs_call_column(tmp_path: Path):
    module = _load_module()
    trace = tmp_path / "trace_v3.dat"
    trace.write_text(
        "\n".join(
            (
                "record step rhs_call term iky ikx iz it ivmu iv imu is vpa mu "
                "wgts_vpa wgts_mu code_time code_dt real imag",
                "pdf_g 20 1 input_pdf 4 1 -1 1 0 1 1 1 -1.0 0.5 0.2 0.3 1.9 0.1 1.0 2.0",
                "pdf_g 20 2 input_pdf 4 1 -1 1 0 1 1 1 -1.0 0.5 0.2 0.3 1.9 0.1 3.0 4.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_trace(
        trace, required_record_terms=(("pdf_g", "input_pdf"),)
    )

    assert summary["trace_format"] == "stellarator_gk_stella_rhs_trace_v3"
    assert summary["rhs_calls"] == [1, 2]
    assert summary["term_summaries"][0]["rhs_call_range"] == [1, 2]


def test_summarize_trace_reports_missing_required_terms(tmp_path: Path):
    module = _load_module()
    trace = tmp_path / "trace.dat"
    trace.write_text(
        "\n".join(
            (
                "record step term iky ikx iz it ivmu iv imu is vpa mu code_time code_dt real imag",
                "pdf_g 20 input_pdf 4 1 -1 1 0 1 1 1 -3.0 0.1 2.0 0.1 1.0 0.0",
                "",
            )
        ),
        encoding="utf-8",
    )

    summary = module.summarize_trace(
        trace,
        required_record_terms=(("pdf_g", "input_pdf"), ("rhs_delta", "parallel_streaming")),
    )

    assert summary["required_record_terms_present"] is False
    assert summary["missing_record_terms"] == [{"record": "rhs_delta", "term": "parallel_streaming"}]


def test_committed_w7x_rhs_trace_summary_contract():
    summary_path = ROOT / "fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json"
    summary = json.loads(summary_path.read_text())
    terms = {(entry["record"], entry["term"]): entry for entry in summary["term_summaries"]}

    assert summary["required_record_terms_present"] is True
    assert summary["steps"] == [2000]
    assert summary["iky_values"] == [4]
    assert summary["ikx_values"] == [1]
    assert summary["trace_format"] == "stellarator_gk_stella_rhs_trace_v3"
    assert summary["rhs_calls"] == [1, 2, 3]
    assert summary["v3_required_record_terms_present"] is True
    assert summary["velocity_weight_columns_present"] is True
    assert summary["provenance"]["stella_revision"] == (
        "564ca09b89904c231421c17c00068a9362061278"
    )
    assert summary["rhs_units"] == "stella_native_rhs_times_code_dt"
    assert terms[("rhs_delta", "mirror_force")]["l2_norm"] > 0.0
    assert terms[("rhs_delta", "parallel_streaming")]["l2_norm"] > 0.0
    assert terms[("rhs_delta", "magnetic_drift_y")]["l2_norm"] > 0.0
    assert terms[("pdf_g", "input_pdf")]["wgts_vpa_range"][0] > 0.0
    assert terms[("pdf_g", "input_pdf")]["wgts_mu_range"][0] > 0.0
    assert terms[("rhs_delta", "equilibrium_drive_wstar")]["l2_norm"] > 0.0
    assert terms[("rhs_delta", "magnetic_drift_x")]["rows"] > 0
    assert terms[("quasineutrality", "numerator")]["rows"] == 771
    assert terms[("quasineutrality", "denominator")]["rows"] == 771
    assert terms[("normalization", "native_state_scale")]["rows"] == 3
