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


def test_selected_mode_phase_space_uses_z_vpar_mu_axis_order():
    module = _load_module()
    values = np.arange(2 * 3 * 4 * 2 * 2).reshape(2, 3, 4, 2, 2)

    selected = module._selected_mode_phase_space(values, ix=1, iy=0)

    assert selected.shape == (4, 2, 3)
    np.testing.assert_array_equal(selected, np.transpose(values[..., 1, 0], (2, 0, 1)))


def test_selected_mode_array_trace_records_arrays_weights_and_metadata(tmp_path):
    module = _load_module()
    shape = (4, 2, 3)
    distribution = np.arange(np.prod(shape)).reshape(shape) * (1.0 + 2.0j)
    output = tmp_path / "selected_mode.npz"

    module.write_selected_mode_array_trace(
        output,
        z=np.linspace(-0.5, 0.25, shape[0]),
        vpar=np.asarray([-1.0, 1.0]),
        mu=np.asarray([0.1, 0.5, 0.9]),
        w_z=np.full(shape[0], 0.25),
        w_vpar=np.asarray([1.0, 1.0]),
        w_mu=np.asarray([0.2, 0.3, 0.5]),
        distribution=distribution,
        phi=np.ones(shape[0], dtype=complex),
        rhs_terms={"parallel_streaming": 2.0 * distribution},
        total_rhs=2.0 * distribution,
        quasineutrality_numerator=np.arange(shape[0], dtype=complex),
        quasineutrality_denominator=-np.ones(shape[0]),
        log_normalization=3.5,
        metadata={"schema": "test", "axis_order": ["z", "vpar", "mu"]},
    )

    with np.load(output) as trace:
        metadata = json.loads(str(trace["metadata_json"]))
        np.testing.assert_array_equal(trace["distribution"], distribution)
        np.testing.assert_array_equal(trace["rhs_parallel_streaming"], 2.0 * distribution)
        np.testing.assert_array_equal(trace["w_mu"], [0.2, 0.3, 0.5])
        assert trace["distribution"].shape == shape
        assert float(trace["log_normalization"]) == 3.5
        assert metadata["axis_order"] == ["z", "vpar", "mu"]
        assert metadata["rhs_terms"] == ["parallel_streaming"]


def test_array_output_must_be_external_and_use_npz(tmp_path):
    module = _load_module()

    with np.testing.assert_raises_regex(ValueError, "outside the repository"):
        module._validate_external_array_output(ROOT / "trace.npz")
    with np.testing.assert_raises_regex(ValueError, "\\.npz suffix"):
        module.write_selected_mode_array_trace(
            tmp_path / "trace.dat",
            z=np.zeros(1),
            vpar=np.zeros(1),
            mu=np.zeros(1),
            w_z=np.ones(1),
            w_vpar=np.ones(1),
            w_mu=np.ones(1),
            distribution=np.zeros((1, 1, 1)),
            phi=np.zeros(1),
            rhs_terms={},
            total_rhs=np.zeros((1, 1, 1)),
            quasineutrality_numerator=np.zeros(1),
            quasineutrality_denominator=np.zeros(1),
            log_normalization=0.0,
            metadata={},
        )


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
