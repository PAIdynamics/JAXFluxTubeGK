from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts/audit_w7x_stella_solver_parity.py"
    spec = importlib.util.spec_from_file_location("audit_w7x_stella_solver_parity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_committed_w7x_stella_solver_parity_audit_stops_at_field_line_length(
    tmp_path,
):
    module = _load_module()

    report = module.run_w7x_stella_solver_parity_audit(output=tmp_path / "audit.json")

    written = json.loads((tmp_path / "audit.json").read_text())
    assert report["first_failed_check"] == written["first_failed_check"]
    report = written
    assert report["status"] == "open"
    assert report["first_failed_check"] == "field_line_length"
    checks = {item["name"]: item for item in report["ordered_checks"]}
    assert checks["eik_z_coordinate_convention"]["passed"]
    assert checks["ky_normalization"]["passed"]
    assert not checks["field_line_length"]["passed"]
    assert checks["field_line_length"]["solver_field_line_periods"] == 1.0
    assert checks["field_line_length"]["stella_zeta_turns_from_metadata"] > 6.9
    assert not checks["twist_and_shift_linking"]["passed"]
    assert checks["twist_and_shift_linking"]["solver_n_kx"] == 3
    assert checks["twist_and_shift_linking"]["stella_nakx"] == 1
    assert not checks["growth_window_time_normalization"]["passed"]
    assert checks["growth_window_time_normalization"]["solver_total_time"] == 0.012
    assert checks["growth_window_time_normalization"]["stella_total_time"] == 200.0
    assert "stella-exported geometry" in report["next_action"]


def test_stella_matched_observed_fixture_stops_at_growth_window_time(
    tmp_path,
):
    module = _load_module()
    observed_dir = ROOT / "fixtures/w7x_itg_stella_matched_observed"

    report = module.run_w7x_stella_solver_parity_audit(
        solver_config=observed_dir / "run_config.json",
        solver_metadata=observed_dir / "convergence_metadata.json",
        solver_fixture=observed_dir / "mode_structures.csv",
        gate_status=observed_dir / "mode_structure_gate/gate_status.json",
        output=tmp_path / "audit.json",
    )

    assert report["status"] == "open"
    assert report["first_failed_check"] == "growth_window_time_normalization"
    checks = {item["name"]: item for item in report["ordered_checks"]}
    assert checks["eik_z_coordinate_convention"]["passed"]
    assert checks["field_line_length"]["passed"]
    assert checks["field_line_length"]["absolute_turn_error"] == 0.0
    assert checks["ky_normalization"]["passed"]
    assert checks["twist_and_shift_linking"]["passed"]
    assert checks["twist_and_shift_linking"]["solver_n_kx"] == 1
    assert not checks["growth_window_time_normalization"]["passed"]
    assert checks["growth_window_time_normalization"]["solver_total_time"] == 0.006
    assert report["gate"]["max_growth_error"] > 0.6
    assert "late-time window" in report["next_action"]
