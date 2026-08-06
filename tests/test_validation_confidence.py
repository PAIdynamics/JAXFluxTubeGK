import json

import pytest

from stellarator_gk.validation_confidence import (
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
        "cyclone_gx_low_ky_branch_shape",
        "kinetic_electron_tem_external_parity",
    }
    assert all(gap.status == "open" for gap in by_id.values())
    velocity_metrics = {
        metric.name: metric.value
        for metric in by_id["gkw_multi_time_velocity_slice"].metrics
    }
    assert velocity_metrics["step_20_complex_max_error"] == pytest.approx(3.99e-3)
    assert velocity_metrics["step_800_complex_max_error"] == pytest.approx(3.67e-2)
    assert velocity_metrics["acceptance_tolerance"] == pytest.approx(2.0e-2)


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
        ("cyclone_gx_multi_ky_mode_structure_parity", "cyclone_gx_low_ky_branch_shape"),
        ("kinetic_electron_tem_validation", "kinetic_electron_tem_external_parity"),
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
    assert len(payload["gaps"]) == 4
    assert "state" not in payload["gaps"][0]
    with pytest.raises(FileExistsError):
        write_validation_confidence_report(output)
