import dataclasses
import json

import numpy as np
import pytest

from stellarator_gk.tem_validation import (
    TemCaseSpec,
    _build_tem_system,
    _initial_tem_state,
    compare_tem_external_reference,
    gyaradax_tem_case_spec,
    load_tem_external_reference,
    run_reduced_tem_linear_smoke,
    run_tem_physics_preflight,
    tem_species,
)
from scripts.run_gyaradax_tem_reference import _parse_args


def test_tem_species_are_charge_neutral_and_electron_driven():
    ion, electron = tem_species()

    assert ion.charge == 1.0
    assert electron.charge == -1.0
    assert ion.charge * ion.density + electron.charge * electron.density == 0.0
    assert ion.temperature_gradient == 0.0
    assert electron.temperature_gradient == pytest.approx(6.9)
    assert electron.mass == pytest.approx(0.01)


def test_tem_preflight_exercises_coupled_kinetic_electron_path():
    report = run_tem_physics_preflight()

    assert report.passed
    assert report.n_species == 2
    assert report.field_model == "kinetic"
    assert report.field_residual_max_abs < 1.0e-10
    assert report.rhs_max_abs > 0.0
    assert np.isfinite(report.estimated_cfl_dt)
    assert report.estimated_cfl_dt > 0.0
    assert report.electron_to_ion_streaming_ratio == pytest.approx(10.0)
    assert report.background_charge_density == pytest.approx(0.0)
    assert not report.external_growth_frequency_validated
    assert "external_tem_parity_open" in report.status


def test_tem_preflight_responds_to_electron_mass_ratio():
    spec = dataclasses.replace(TemCaseSpec(), electron_mass=0.04)
    report = run_tem_physics_preflight(spec)

    assert report.passed
    assert report.expected_streaming_ratio == pytest.approx(5.0)
    assert report.electron_to_ion_streaming_ratio == pytest.approx(5.0)


def test_reduced_tem_time_advance_stays_a_nonexternal_smoke_result():
    result = run_reduced_tem_linear_smoke(steps_per_window=2, n_windows=3)

    assert result.finite
    assert np.isfinite(result.growth_rate)
    assert np.isfinite(result.frequency)
    assert result.dt <= result.estimated_cfl_dt
    assert not result.externally_validated
    assert result.status == "reduced_tem_time_advance_not_external_validation"
    assert len(result.mode_structure) == TemCaseSpec().n_z


def test_reduced_tem_time_advance_rejects_timestep_above_cfl():
    with pytest.raises(ValueError, match="no larger than estimated CFL"):
        run_reduced_tem_linear_smoke(dt=1.0, steps_per_window=1, n_windows=3)


def test_gyaradax_tem_profile_matches_pinned_producer_discretization():
    spec = gyaradax_tem_case_spec()
    _velocity, parallel, _fourier, _geometry, _species, precompute = _build_tem_system(spec)
    state = _initial_tem_state(precompute, parallel, spec)

    assert spec.n_z == 32
    assert spec.n_vpar == 32
    assert spec.n_mu == 16
    assert spec.velocity_backend == "finite_difference"
    assert spec.parallel_backend == "finite_difference"
    assert spec.parallel_derivative_model == "gkw_upwind"
    assert spec.ky == pytest.approx(0.7 / (spec.q / (2.0 * np.pi * spec.eps)))
    np.testing.assert_allclose(np.diff(parallel.z), 3.0 / 32.0)
    np.testing.assert_allclose(parallel.z[0], -1.5 + 1.5 / 32.0)
    np.testing.assert_allclose(state[:, 0, 0], state[:, -1, -1])


def test_gyaradax_tem_runner_requires_explicit_external_root_and_output(tmp_path):
    args = _parse_args(
        [
            "--gyaradax-root",
            str(tmp_path / "gyaradax"),
            "--output",
            str(tmp_path / "tem.json"),
        ]
    )

    assert args.gyaradax_root == tmp_path / "gyaradax"
    assert args.output == tmp_path / "tem.json"
    assert args.n_windows == 200
    assert args.steps_per_window == 20


def test_tem_external_reference_loader_and_quantitative_gate(tmp_path):
    result = dataclasses.replace(
        run_reduced_tem_linear_smoke(steps_per_window=2, n_windows=3),
        late_window_growth_delta=0.0,
    )
    mode = np.asarray(result.mode_structure)
    path = tmp_path / "tem.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "gyaradax",
                "revision": "abc123",
                "case": {"ky": 0.7},
                "growth_rate": result.growth_rate,
                "frequency": result.frequency,
                "final_time": result.final_time,
                "mode_structure_real": mode.real.tolist(),
                "mode_structure_imag": mode.imag.tolist(),
            }
        )
    )
    reference = load_tem_external_reference(path)
    report = compare_tem_external_reference(result, reference)

    assert reference.revision == "abc123"
    assert report.passed
    assert report.growth_relative_error == pytest.approx(0.0)
    assert report.frequency_relative_error == pytest.approx(0.0)
    assert report.mode_structure_relative_l2_error < 1.0e-14

    failed = compare_tem_external_reference(
        dataclasses.replace(result, growth_rate=2.0 * result.growth_rate), reference
    )
    assert not failed.passed


@pytest.mark.parametrize(
    "updates",
    (
        {"electron_mass": 0.0},
        {"electron_mass": 1.0},
        {"ky": 0.0},
        {"n_mu": 1},
    ),
)
def test_tem_case_rejects_invalid_controls(updates):
    with pytest.raises(ValueError):
        TemCaseSpec(**updates)
