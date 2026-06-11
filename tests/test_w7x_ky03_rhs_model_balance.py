from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from stellarator_gk import FourierGridSpec, build_fourier_grid
from stellarator_gk.physics import FLRFactors


ROOT = Path(__file__).resolve().parents[1]
BALANCE = ROOT / "fixtures/w7x_ky03_rhs_model_balance"


def _load_module():
    path = ROOT / "scripts/audit_w7x_ky03_rhs_model_balance.py"
    spec = importlib.util.spec_from_file_location("audit_w7x_ky03_rhs_model_balance", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_balance_cases_include_focus_velocity_case():
    module = _load_module()

    cases = module.default_balance_cases()
    by_name = {case.name: case for case in cases}

    assert tuple(by_name) == ("cheb_4x4", "gkw_fd_16x8")
    assert by_name["cheb_4x4"].total_time == 200.0
    assert by_name["gkw_fd_16x8"].velocity_backend == "finite_difference"
    assert by_name["gkw_fd_16x8"].n_vpar == 16
    assert by_name["gkw_fd_16x8"].n_mu == 8
    assert by_name["gkw_fd_16x8"].total_time == 200.0


def test_scan_args_hold_w7x_ky03_controls_fixed():
    module = _load_module()
    case = module.default_balance_cases()[1]

    scan_args = module._scan_args(case, module.DEFAULT_STELLA_GEOMETRY)

    assert scan_args[scan_args.index("--geometry-source") + 1] == "stella-geometry"
    assert scan_args[scan_args.index("--stella-geometry") + 1].endswith(
        "stella_w7x_adiabatic_electrons.geometry"
    )
    assert scan_args[scan_args.index("--n-kx") + 1] == "1"
    assert scan_args[scan_args.index("--kx-max") + 1] == "0.0"
    assert scan_args[scan_args.index("--ikxspace") + 1] == "1"
    assert scan_args[scan_args.index("--ky-values") + 1] == "0.3"
    assert scan_args[scan_args.index("--n-vpar") + 1] == "16"
    assert scan_args[scan_args.index("--n-mu") + 1] == "8"
    assert scan_args[scan_args.index("--velocity-backend") + 1] == "finite_difference"
    assert scan_args[scan_args.index("--dt") + 1] == "0.02"
    assert scan_args[scan_args.index("--steps-per-window") + 1] == "5"
    assert scan_args[scan_args.index("--n-windows") + 1] == "2000"


def test_selected_term_balance_projections_sum_to_total():
    module = _load_module()
    term_a = jnp.ones((2, 1, 3, 1, 1), dtype=jnp.complex128)
    term_b = 2j * term_a
    total = term_a + term_b
    field = SimpleNamespace(
        phi_weight=jnp.ones((1, 2, 1, 3, 1, 1), dtype=jnp.float64),
        n_species=1,
    )

    rows = module.selected_term_balance_rows(
        ("a", "b"),
        (term_a, term_b),
        total,
        field,
        ix=0,
        iy=0,
        case="synthetic",
        ky=0.3,
    )

    assert [row["term"] for row in rows] == ["a", "b"]
    assert abs(sum(row["projection_real"] for row in rows) - 1.0) < 1.0e-12
    assert abs(sum(row["projection_imag"] for row in rows)) < 1.0e-12
    assert abs(rows[0]["rhs_fraction_of_total_l2"] - 1.0 / np.sqrt(5.0)) < 1.0e-12
    assert abs(rows[1]["rhs_fraction_of_total_l2"] - 2.0 / np.sqrt(5.0)) < 1.0e-12
    assert rows[0]["density_moment_rms"] == 2.0
    assert rows[1]["density_moment_rms"] == 4.0


def test_selected_geometry_model_rows_capture_ky03_inputs():
    module = _load_module()
    z = jnp.asarray([-0.5, 0.0, 0.5])
    geometry = SimpleNamespace(
        z=z,
        theta=2.0 * jnp.pi * z,
        phi=0.25 + z,
        B=jnp.asarray([1.0, 1.2, 1.1]),
        F=jnp.asarray([0.5, 0.6, 0.7]),
        G=jnp.asarray([0.1, 0.2, 0.3]),
        E_y=jnp.asarray([0.4, 0.5, 0.6]),
        D_x=jnp.asarray([0.0, 0.0, 0.0]),
        D_y=jnp.asarray([1.0, 1.5, 2.0]),
        g_xx=jnp.ones(3),
        g_xy=jnp.zeros(3),
        g_yy=2.0 * jnp.ones(3),
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=1, kx_max=0.0, ky_values=(0.3,), ikxspace=1)
    )
    precompute = SimpleNamespace(
        rhs=SimpleNamespace(
            flr_factors=FLRFactors(
                bessel_argument=jnp.zeros((1, 2, 3, 1, 1)),
                bessel_j0=jnp.ones((1, 2, 3, 1, 1)),
                polarization_argument=jnp.zeros((1, 3, 1, 1)),
                gamma0=0.9 * jnp.ones((1, 3, 1, 1)),
            ),
            magnetic_drift_frequency=jnp.ones((1, 2, 2, 3, 1, 1)),
            maxwellian=jnp.ones((1, 2, 2, 3)),
            drive_factor=2.0 * jnp.ones((1, 2, 2, 3)),
        ),
        field=SimpleNamespace(
            denominator=-jnp.ones((3, 1, 1)),
            ion_polarization=-0.1 * jnp.ones((3, 1, 1)),
            electron_response=jnp.asarray(1.0),
        ),
    )

    rows = module.selected_geometry_model_rows(
        geometry,
        fourier,
        precompute,
        ix=0,
        iy=0,
        case="synthetic",
        ky=0.3,
    )

    assert len(rows) == 3
    assert rows[0]["ky"] == 0.3
    assert rows[0]["kperp2"] == 0.18
    assert rows[1]["B"] == 1.2
    assert rows[2]["bessel_j0_rms_over_mu"] == 1.0
    assert rows[2]["drive_factor_mean_over_velocity"] == 2.0


def test_committed_rhs_balance_fixture_is_solver_side_next_step():
    status = json.loads((BALANCE / "rhs_model_balance_status.json").read_text())
    rows = tuple(csv.DictReader((BALANCE / "rhs_term_balance.csv").open()))
    focus_rows = [row for row in rows if row["case"] == "gkw_fd_16x8"]

    assert status["status"] == "solver_side_rhs_balance_ready"
    assert status["focus_ky"] == 0.3
    assert status["max_rhs_reconstruction_abs_error"] < 1.0e-10
    assert focus_rows
    assert any(row["term"] == "magnetic_drift" for row in focus_rows)
    assert "stella-exported RHS" in status["next_action"]
