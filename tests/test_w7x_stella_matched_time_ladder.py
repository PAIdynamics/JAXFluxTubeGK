from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "fixtures/w7x_itg_stella_matched_time_ladder"


def _load_module():
    path = ROOT / "scripts/run_w7x_stella_matched_time_ladder.py"
    spec = importlib.util.spec_from_file_location("run_w7x_stella_matched_time_ladder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_stella_matched_time_cases_reach_stella_window():
    module = _load_module()

    cases = module.default_time_cases()
    names = [case.name for case in cases]
    time_200 = next(case for case in cases if case.name == "time_200")

    assert names == ["smoke_0p006", "time_1", "time_5", "time_20", "time_100", "time_200"]
    assert cases[0].total_time == 0.006
    assert time_200.dt == 0.02
    assert time_200.steps_per_window == 5
    assert time_200.n_windows == 2000
    assert time_200.total_time == 200.0


def test_case_from_total_time_rounds_to_whole_windows():
    module = _load_module()

    case = module.case_from_total_time(1.2, dt=0.05, steps_per_window=4)

    assert case.name == "time_1p2"
    assert case.n_windows == 6
    assert abs(case.total_time - 1.2) < 1.0e-12


def test_stella_matched_scan_args_keep_geometry_and_mode_controls(tmp_path):
    module = _load_module()
    case = module.case_from_total_time(5.0, dt=0.02, steps_per_window=5)

    scan_args = module._scan_args(
        case,
        tmp_path / "run",
        module.DEFAULT_STELLA_GEOMETRY,
        "0.1,0.2,0.3",
    )

    assert scan_args[scan_args.index("--geometry-source") + 1] == "stella-geometry"
    assert scan_args[scan_args.index("--stella-geometry") + 1].endswith(
        "stella_w7x_adiabatic_electrons.geometry"
    )
    assert scan_args[scan_args.index("--n-kx") + 1] == "1"
    assert scan_args[scan_args.index("--kx-max") + 1] == "0.0"
    assert scan_args[scan_args.index("--ikxspace") + 1] == "1"
    assert scan_args[scan_args.index("--ky-values") + 1] == "0.1,0.2,0.3"
    assert scan_args[scan_args.index("--growth-window-fraction") + 1] == "0.5"
    assert scan_args[scan_args.index("--n-windows") + 1] == "50"


def test_committed_stella_matched_time_ladder_reaches_time_window():
    status = json.loads((LADDER / "time_ladder_status.json").read_text())
    rows = tuple(csv.DictReader((LADDER / "time_ladder_summary.csv").open()))
    latest = status["latest_case"]
    first_failures = status["first_failed_check_by_case"]

    assert status["status"] == "time_window_reached_gate_open"
    assert status["stella_comparable_time_window_reached"]
    assert status["reached_stella_tend"]
    assert status["finite_outputs"]
    assert latest["case"] == "time_200"
    assert latest["actual_total_time"] == 200.0
    assert latest["time_window_passed"]
    assert latest["first_failed_check"] == "velocity_rhs_terms"
    assert latest["max_growth_error"] < 1.0e-2
    assert latest["max_frequency_error"] > 1.0e-1
    assert latest["max_profile_error"] > 1.0e-1
    assert first_failures["time_100"] == "velocity_rhs_terms"
    assert first_failures["time_20"] == "growth_window_time_normalization"
    assert len(rows) == 6 * 3

    by_case_ky = {(row["case"], float(row["ky"])): row for row in rows}
    assert float(by_case_ky[("time_200", 0.3)]["growth_error"]) < 1.0e-2
    assert by_case_ky[("time_200", 0.3)]["time_window_passed"] == "True"
