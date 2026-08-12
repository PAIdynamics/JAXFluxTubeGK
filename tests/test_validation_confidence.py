import json

import pytest

from jax_fluxtube_gk.validation_confidence import (
    ScientificClaimNotReadyError,
    priority5_confidence_gaps,
    require_validation_claim_ready,
    validation_claim_readiness,
    write_validation_confidence_report,
)


def test_priority5_ledger_keeps_known_gaps_explicit():
    by_id = {gap.identifier: gap for gap in priority5_confidence_gaps()}

    assert set(by_id) == {
        "gkw_selected_mode_state_history",
        "gkw_multi_time_velocity_slice",
        "cyclone_low_ky_branch_shape",
        "kinetic_electron_tem_external_parity",
        "production_collisions_electromagnetic_parity",
        "nonlinear_stationary_heat_flux_parity",
        "full_equilibrium_shape_optimization",
    }
    assert by_id["kinetic_electron_tem_external_parity"].status == "passed"
    assert all(
        gap.status == "open"
        for identifier, gap in by_id.items()
        if identifier != "kinetic_electron_tem_external_parity"
    )
    velocity_metrics = {
        metric.name: metric.value for metric in by_id["gkw_multi_time_velocity_slice"].metrics
    }
    assert velocity_metrics["step_20_complex_max_error"] == pytest.approx(3.99e-3)
    assert velocity_metrics["step_800_complex_max_error"] == pytest.approx(3.67e-2)
    assert velocity_metrics["acceptance_tolerance"] == pytest.approx(2.0e-2)
    em_metrics = {
        metric.name: metric.value
        for metric in by_id["production_collisions_electromagnetic_parity"].metrics
    }
    assert em_metrics["em_local_growth_finest_change"] == pytest.approx(2.19627e-2)
    assert em_metrics["em_finest_growth_parity_error"] == pytest.approx(1.71945e-5)
    nonlinear_metrics = {
        metric.name: metric.value
        for metric in by_id["nonlinear_stationary_heat_flux_parity"].metrics
    }
    assert nonlinear_metrics["regenerated_merged_total_energy_mean"] == pytest.approx(-24.2322731)
    assert nonlinear_metrics["regenerated_merged_relative_drift"] == pytest.approx(-0.69131945)


def test_w7x_claim_records_narrow_independent_supersession():
    readiness = validation_claim_readiness("w7x_linear_stellarator_branch")

    assert readiness.ready
    assert set(readiness.superseded_gap_ids) == {
        "gkw_selected_mode_state_history",
        "gkw_multi_time_velocity_slice",
    }
    assert readiness.blocking_gap_ids == ()


@pytest.mark.parametrize(
    ("claim", "gap_id"),
    (
        ("gkw_cyclone_full_state_history_parity", "gkw_selected_mode_state_history"),
        (
            "gkw_cyclone_full_velocity_space_history_parity",
            "gkw_multi_time_velocity_slice",
        ),
        ("cyclone_multi_ky_mode_structure_validation", "cyclone_low_ky_branch_shape"),
        (
            "collisional_electromagnetic_production_physics",
            "production_collisions_electromagnetic_parity",
        ),
        ("nonlinear_turbulence_validation", "nonlinear_stationary_heat_flux_parity"),
        (
            "production_equilibrium_shape_optimization",
            "full_equilibrium_shape_optimization",
        ),
    ),
)
def test_broad_claims_remain_blocked(claim, gap_id):
    readiness = validation_claim_readiness(claim)

    assert not readiness.ready
    assert readiness.blocking_gap_ids == (gap_id,)
    with pytest.raises(ScientificClaimNotReadyError, match=gap_id):
        require_validation_claim_ready(claim)


def test_confidence_report_is_compact_and_schema_versioned(tmp_path):
    output = write_validation_confidence_report(tmp_path / "confidence.json")
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 1
    assert len(payload["gaps"]) == 7
    assert "state" not in payload["gaps"][0]
    with pytest.raises(FileExistsError):
        write_validation_confidence_report(output)


def test_kinetic_electron_tem_claim_is_ready_after_external_gate():
    readiness = validation_claim_readiness("kinetic_electron_tem_validation")

    assert readiness.ready
    assert readiness.blocking_gap_ids == ()
