import pytest

from jax_fluxtube_gk import (
    PRODUCTION_VELOCITY_REPRESENTATION,
    VelocityBackendNotReadyError,
    require_velocity_backend_for_claim,
    velocity_backend_decision,
    velocity_backend_decisions,
)


def test_production_cpu_decision_keeps_collocation_as_primary_representation():
    assert PRODUCTION_VELOCITY_REPRESENTATION == "collocation"
    decisions = {item.backend: item for item in velocity_backend_decisions()}

    assert decisions["chebyshev"].maturity == "production"
    assert decisions["midpoint_gauss_laguerre"].maturity == "production"
    assert decisions["finite_difference"].maturity == "validated_specialist"
    assert decisions["hermite_laguerre"].maturity == "experimental"


def test_w7x_claim_requires_source_matched_velocity_recipe():
    decision = require_velocity_backend_for_claim(
        "midpoint_gauss_laguerre", "w7x_linear_cpu"
    )

    assert decision.representation == "collocation"
    with pytest.raises(VelocityBackendNotReadyError, match="source-matched"):
        require_velocity_backend_for_claim("chebyshev", "w7x_linear_cpu")
    with pytest.raises(VelocityBackendNotReadyError, match="not integrated"):
        require_velocity_backend_for_claim("hermite_laguerre", "w7x_linear_cpu")


def test_gkw_finite_difference_scope_does_not_overclaim_history_parity():
    decision = require_velocity_backend_for_claim(
        "finite_difference", "gkw_term_operator_parity"
    )

    assert "history" in decision.limitations
    with pytest.raises(VelocityBackendNotReadyError):
        require_velocity_backend_for_claim(
            "finite_difference", "gkw_cyclone_full_state_history_parity"
        )


def test_arbitrary_native_grid_is_plumbing_until_named_gate_passes():
    decision = velocity_backend_decision("native")

    assert decision.supported_claims == ("custom_grid_plumbing",)
    assert decision.maturity == "experimental"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown velocity backend"):
        velocity_backend_decision("magic")
