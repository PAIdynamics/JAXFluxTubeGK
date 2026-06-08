from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "fixtures/w7x_itg_convergence_study"


def _load_module():
    path = ROOT / "scripts/run_w7x_reduced_convergence_study.py"
    spec = importlib.util.spec_from_file_location("run_w7x_reduced_convergence_study", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_w7x_convergence_cases_cover_item_5_axes():
    module = _load_module()

    cases = module.default_convergence_cases()
    names = [case.name for case in cases]
    axes = {case.axis for case in cases}
    baseline = cases[0]
    dt_half = next(case for case in cases if case.name == "dt_half")
    two_periods = next(case for case in cases if case.name == "two_periods")

    assert names[0] == "baseline"
    assert len(names) == len(set(names))
    assert {
        "reference",
        "parallel_resolution",
        "velocity_resolution",
        "kx_grid",
        "time_step",
        "field_line_length",
        "growth_window",
    }.issubset(axes)
    assert dt_half.dt * dt_half.steps_per_window == baseline.dt * baseline.steps_per_window
    assert two_periods.field_line_periods == 2
    assert two_periods.n_z > baseline.n_z


def test_w7x_convergence_rows_attach_baseline_deltas():
    module = _load_module()
    rows = [
        {"case": "baseline", "ky": 0.1, "growth_rate": 1.0, "frequency": 2.0},
        {"case": "variant", "ky": 0.1, "growth_rate": 1.25, "frequency": 1.5},
        {"case": "variant", "ky": 0.2, "growth_rate": 0.0, "frequency": 0.0},
    ]

    with_deltas = module.rows_with_baseline_deltas(rows)

    assert with_deltas[0]["growth_delta_from_baseline"] == 0.0
    assert with_deltas[1]["growth_delta_from_baseline"] == 0.25
    assert with_deltas[1]["frequency_delta_from_baseline"] == -0.5
    assert with_deltas[2]["growth_delta_from_baseline"] == ""
    assert with_deltas[2]["frequency_delta_from_baseline"] == ""


def test_committed_w7x_convergence_study_artifacts_record_reduced_status():
    metadata = json.loads((STUDY / "study_metadata.json").read_text())
    timing = json.loads((STUDY / "timing_summary.json").read_text())
    readiness = json.loads((STUDY / "optimization_readiness.json").read_text())
    rows = tuple(csv.DictReader((STUDY / "convergence_summary.csv").open()))

    assert metadata["case_count"] == 9
    assert metadata["geometry_source"] == "GX/GIST W7-X eik"
    assert timing["status"] == "reduced_solver_regression_not_external_parity"
    assert not timing["external_parity_ready"]
    assert timing["production_timing_claim"] == (
        "not_claimed_pending_external_parity_and_production_run"
    )
    assert readiness["reduced_fixture_ready"]
    assert readiness["reduced_convergence_study_ready"]
    assert not readiness["external_w7x_parity_ready"]
    assert readiness["production_cpu_timing_contract_ready"]
    assert readiness["production_cpu_timing_artifact"].endswith("production_cpu_timing.json")
    assert not readiness["production_cpu_timing_ready"]
    assert len(rows) == 9 * 4

    growth = np.asarray([float(row["growth_rate"]) for row in rows])
    frequency = np.asarray([float(row["frequency"]) for row in rows])
    assert np.all(np.isfinite(growth))
    assert np.all(np.isfinite(frequency))

    by_case_ky = {(row["case"], float(row["ky"])): row for row in rows}
    assert abs(float(by_case_ky[("dt_half", 0.3)]["growth_delta_from_baseline"])) < 1.0e-8
    assert abs(float(by_case_ky[("late_mean_window", 0.3)]["growth_delta_from_baseline"])) < 1.0e-4
    assert float(by_case_ky[("nz_49", 0.3)]["growth_delta_from_baseline"]) > 1.0

    memory = timing["production_gx_control_memory_estimate"]
    assert memory["dimensions"] == {
        "n_kx": 1,
        "n_ky": 28,
        "n_mu": 8,
        "n_species": 1,
        "n_vpar": 16,
        "n_z": 256,
    }
    assert memory["total_bytes"] > memory["state_bytes"] > 0
    assert memory["total_bytes_human"].endswith("MiB")
    assert "/Users/" not in (STUDY / "study_metadata.json").read_text()
