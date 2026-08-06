from __future__ import annotations

import importlib.util
import json
import sys
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISCRIMINATOR = ROOT / "fixtures/w7x_itg_stella_velocity_discriminator"


def _load_module():
    path = ROOT / "scripts/run_w7x_stella_velocity_discriminator.py"
    spec = importlib.util.spec_from_file_location("run_w7x_stella_velocity_discriminator", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_velocity_cases_include_spectral_and_gkw_grid_controls():
    module = _load_module()

    cases = module.default_velocity_cases()
    by_name = {case.name: case for case in cases}

    assert tuple(by_name) == (
        "cheb_4x4",
        "cheb_6x6",
        "cheb_8x8",
        "gkw_fd_16x8",
        "native_32x8",
    )
    assert by_name["cheb_4x4"].velocity_backend == "chebyshev"
    assert by_name["cheb_8x8"].n_vpar == 8
    assert by_name["cheb_8x8"].n_mu == 8
    assert by_name["gkw_fd_16x8"].velocity_backend == "finite_difference"
    assert by_name["gkw_fd_16x8"].n_vpar == 16
    assert by_name["gkw_fd_16x8"].n_mu == 8
    assert by_name["gkw_fd_16x8"].total_time == 200.0
    assert by_name["native_32x8"].velocity_backend == "midpoint_gauss_laguerre"
    assert by_name["native_32x8"].velocity_measure_normalization == "full_gyroangle"
    assert by_name["native_32x8"].mirror_advance == "semi_lagrangian"
    assert by_name["native_32x8"].mirror_interpolation == "cubic"
    assert by_name["native_32x8"].parallel_advance == "stella_implicit"
    assert by_name["native_32x8"].dt == 0.1
    assert by_name["native_32x8"].steps_per_window == 1
    assert by_name["native_32x8"].total_time == 200.0
    assert by_name["native_32x8"].n_vpar == 32
    assert by_name["native_32x8"].n_mu == 8
    assert by_name["native_32x8"].vpar_max == 3.0


def test_velocity_discriminator_scan_args_hold_w7x_stella_controls_fixed(tmp_path):
    module = _load_module()
    case = module.StellaVelocityCase(
        name="cheb_6x6",
        n_vpar=6,
        n_mu=6,
        velocity_backend="chebyshev",
    )

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
    assert scan_args[scan_args.index("--n-vpar") + 1] == "6"
    assert scan_args[scan_args.index("--n-mu") + 1] == "6"
    assert scan_args[scan_args.index("--velocity-backend") + 1] == "chebyshev"
    assert scan_args[scan_args.index("--dt") + 1] == "0.02"
    assert scan_args[scan_args.index("--steps-per-window") + 1] == "5"
    assert scan_args[scan_args.index("--n-windows") + 1] == "2000"
    assert scan_args[scan_args.index("--growth-window-fraction") + 1] == "0.5"


def test_native_velocity_case_selects_zero_free_gauss_laguerre_backend(tmp_path):
    module = _load_module()
    case = next(case for case in module.default_velocity_cases() if case.name == "native_32x8")

    scan_args = module._scan_args(
        case,
        tmp_path / "run",
        module.DEFAULT_STELLA_GEOMETRY,
        "0.3",
    )

    assert scan_args[scan_args.index("--n-vpar") + 1] == "32"
    assert scan_args[scan_args.index("--n-mu") + 1] == "8"
    assert scan_args[scan_args.index("--velocity-backend") + 1] == (
        "midpoint_gauss_laguerre"
    )
    assert scan_args[scan_args.index("--velocity-measure-normalization") + 1] == (
        "full_gyroangle"
    )
    assert scan_args[scan_args.index("--mirror-advance") + 1] == "semi_lagrangian"
    assert scan_args[scan_args.index("--mirror-interpolation") + 1] == "cubic"
    assert scan_args[scan_args.index("--parallel-advance") + 1] == "stella_implicit"
    assert scan_args[scan_args.index("--dt") + 1] == "0.1"
    assert scan_args[scan_args.index("--steps-per-window") + 1] == "1"
    assert scan_args[scan_args.index("--vpar-max") + 1] == "3.0"
    assert float(scan_args[scan_args.index("--mu-max") + 1]) > 4.9


def test_velocity_baseline_deltas_are_per_ky():
    module = _load_module()
    rows = [
        {
            "case": "cheb_4x4",
            "ky": 0.1,
            "growth_rate": 1.0,
            "frequency": -2.0,
            "frequency_error": -0.5,
            "phi_phase_aligned_error": 0.2,
        },
        {
            "case": "cheb_4x4",
            "ky": 0.3,
            "growth_rate": 3.0,
            "frequency": -4.0,
            "frequency_error": -1.5,
            "phi_phase_aligned_error": 0.6,
        },
        {
            "case": "cheb_8x8",
            "ky": 0.1,
            "growth_rate": 1.25,
            "frequency": -2.1,
            "frequency_error": -0.25,
            "phi_phase_aligned_error": 0.15,
        },
        {
            "case": "cheb_8x8",
            "ky": 0.3,
            "growth_rate": 2.5,
            "frequency": -3.75,
            "frequency_error": -1.0,
            "phi_phase_aligned_error": 0.3,
        },
    ]

    observed = module.rows_with_velocity_baseline_deltas(rows)
    by_case_ky = {(row["case"], row["ky"]): row for row in observed}

    assert by_case_ky[("cheb_4x4", 0.1)]["growth_delta_from_baseline"] == 0.0
    assert by_case_ky[("cheb_8x8", 0.1)]["growth_delta_from_baseline"] == 0.25
    assert abs(by_case_ky[("cheb_8x8", 0.1)]["frequency_delta_from_baseline"] + 0.1) < 1.0e-12
    assert by_case_ky[("cheb_8x8", 0.1)]["abs_frequency_error_delta_from_baseline"] == -0.25
    assert by_case_ky[("cheb_8x8", 0.3)]["profile_error_delta_from_baseline"] == -0.3


def test_time_ladder_baseline_source_is_detected():
    module = _load_module()
    case = module.default_velocity_cases()[0]

    source = module._baseline_source_for_case(case, module.DEFAULT_TIME_LADDER, True)

    assert source is not None
    assert source.name == "time_200"


def test_committed_velocity_discriminator_moves_to_rhs_terms():
    status = json.loads((DISCRIMINATOR / "velocity_discriminator_status.json").read_text())
    rows = tuple(csv.DictReader((DISCRIMINATOR / "velocity_discriminator_summary.csv").open()))
    by_case_ky = {(row["case"], float(row["ky"])): row for row in rows}

    assert status["status"] == "open_rhs_terms_after_velocity_discriminator"
    assert status["stella_velocity_case_present"]
    assert not status["focus_gate_closed"]
    assert status["best_focus_row"]["case"] == "gkw_fd_16x8"
    assert abs(float(by_case_ky[("gkw_fd_16x8", 0.3)]["frequency_error"])) > 0.16
    assert float(by_case_ky[("gkw_fd_16x8", 0.3)]["phi_phase_aligned_error"]) > 0.15
    assert by_case_ky[("gkw_fd_16x8", 0.3)]["first_failed_check"] == "velocity_rhs_terms"
    assert "RHS/model term balance" in status["next_action"]
