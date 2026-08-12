"""Explicit production-readiness policy for velocity-space backends."""

from __future__ import annotations

from dataclasses import dataclass


PRODUCTION_VELOCITY_REPRESENTATION = "collocation"


@dataclass(frozen=True)
class VelocityBackendDecision:
    """Validated role and claim scope for one velocity representation."""

    backend: str
    representation: str
    maturity: str
    supported_claims: tuple[str, ...]
    rationale: str
    limitations: str

    def __post_init__(self) -> None:
        if self.maturity not in ("production", "validated_specialist", "experimental"):
            raise ValueError("unsupported velocity-backend maturity")


class VelocityBackendNotReadyError(RuntimeError):
    """Raised when a backend is used outside its validated scientific scope."""


_DECISIONS = {
    "chebyshev": VelocityBackendDecision(
        backend="chebyshev",
        representation="collocation",
        maturity="production",
        supported_claims=("reduced_linear_cpu", "differentiable_linear_cpu"),
        rationale=(
            "Default finite-box collocation with complete integration into the "
            "kinetic residual, field solve, time advance, and design objective."
        ),
        limitations=(
            "It is not the source-matched W7-X/stella velocity recipe and does not "
            "by itself establish external code parity."
        ),
    ),
    "finite_difference": VelocityBackendDecision(
        backend="finite_difference",
        representation="collocation",
        maturity="validated_specialist",
        supported_claims=("reduced_linear_cpu", "gkw_term_operator_parity"),
        rationale=(
            "GKW-style finite-box nodes and derivative stencils retain the strongest "
            "term/operator convention checks against GKW."
        ),
        limitations=(
            "Full GKW state-history and long-time velocity-slice parity remain open."
        ),
    ),
    "midpoint_gauss_laguerre": VelocityBackendDecision(
        backend="midpoint_gauss_laguerre",
        representation="collocation",
        maturity="production",
        supported_claims=(
            "reduced_linear_cpu",
            "differentiable_linear_cpu",
            "w7x_linear_cpu",
        ),
        rationale=(
            "The zero-free midpoint v-parallel and Gauss-Laguerre mu recipe is used "
            "by the converged source-matched W7-X/stella branch."
        ),
        limitations=(
            "The W7-X claim also requires its validated measure, split ordering, "
            "resolution, timestep, initialization, and convergence controls."
        ),
    ),
    "native": VelocityBackendDecision(
        backend="native",
        representation="collocation",
        maturity="experimental",
        supported_claims=("custom_grid_plumbing",),
        rationale="Provider-supplied nodes permit explicit independent quadrature tests.",
        limitations=(
            "Arbitrary native nodes receive no production claim until their exact "
            "quadrature and derivative contract passes a named validation gate."
        ),
    ),
    "hermite_laguerre": VelocityBackendDecision(
        backend="hermite_laguerre",
        representation="moments",
        maturity="experimental",
        supported_claims=("experimental_moment_discriminator",),
        rationale=(
            "Moment transforms, recurrences, hypercollision, and a reduced linear "
            "RHS are useful branch-shape discriminators."
        ),
        limitations=(
            "It is not integrated with the production stellarator geometry/residual, "
            "kinetic field solve, design objective, convergence, or timing gates."
        ),
    ),
}


def velocity_backend_decisions() -> tuple[VelocityBackendDecision, ...]:
    """Return every maintained backend decision in deterministic order."""

    return tuple(_DECISIONS[name] for name in sorted(_DECISIONS))


def velocity_backend_decision(backend: str) -> VelocityBackendDecision:
    """Return the readiness decision for an exact backend identifier."""

    try:
        return _DECISIONS[str(backend)]
    except KeyError as error:
        known = ", ".join(sorted(_DECISIONS))
        raise ValueError(f"unknown velocity backend {backend!r}; expected one of {known}") from error


def require_velocity_backend_for_claim(
    backend: str,
    claim: str,
) -> VelocityBackendDecision:
    """Require that a backend's recorded validation scope includes a claim."""

    decision = velocity_backend_decision(backend)
    if claim not in decision.supported_claims:
        raise VelocityBackendNotReadyError(
            f"velocity backend {backend!r} is not ready for claim {claim!r}: "
            f"{decision.limitations}"
        )
    return decision


__all__ = [
    "PRODUCTION_VELOCITY_REPRESENTATION",
    "VelocityBackendDecision",
    "VelocityBackendNotReadyError",
    "require_velocity_backend_for_claim",
    "velocity_backend_decision",
    "velocity_backend_decisions",
]
