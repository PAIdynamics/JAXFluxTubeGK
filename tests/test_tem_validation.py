import dataclasses

import numpy as np
import pytest

from stellarator_gk.tem_validation import (
    TemCaseSpec,
    run_reduced_tem_linear_smoke,
    run_tem_physics_preflight,
    tem_species,
)


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


def test_reduced_tem_time_advance_rejects_timestep_above_cfl():
    with pytest.raises(ValueError, match="no larger than estimated CFL"):
        run_reduced_tem_linear_smoke(dt=1.0, steps_per_window=1, n_windows=3)


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
