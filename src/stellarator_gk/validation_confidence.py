"""Machine-readable confidence gaps and the scientific claims they block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


VALIDATION_CONFIDENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConfidenceMetric:
    """One compact numerical observation attached to a confidence gap."""

    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True)
class ValidationConfidenceGap:
    """An unresolved validation gap with an explicit claim boundary."""

    identifier: str
    status: str
    summary: str
    blocks_claims: tuple[str, ...]
    superseded_for_claims: tuple[str, ...]
    evidence: str
    next_action: str
    metrics: tuple[ConfidenceMetric, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("open", "passed", "superseded"):
            raise ValueError("confidence-gap status must be open, passed, or superseded")
        if not self.identifier or not self.summary or not self.evidence:
            raise ValueError("confidence gaps require identifier, summary, and evidence")
        overlap = set(self.blocks_claims) & set(self.superseded_for_claims)
        if overlap:
            raise ValueError(f"claims cannot be both blocked and superseded: {sorted(overlap)}")


@dataclass(frozen=True)
class ClaimReadiness:
    """Readiness decision for one named scientific claim."""

    claim: str
    ready: bool
    blocking_gap_ids: tuple[str, ...]
    superseded_gap_ids: tuple[str, ...]


class ScientificClaimNotReadyError(RuntimeError):
    """Raised when a workflow requests a claim with unresolved blocking gaps."""


def priority5_confidence_gaps() -> tuple[ValidationConfidenceGap, ...]:
    """Return the maintained Priority 5 confidence ledger."""

    return (
        ValidationConfidenceGap(
            identifier="gkw_selected_mode_state_history",
            status="open",
            summary="Full selected-mode GKW state-history parity is not established.",
            blocks_claims=("gkw_cyclone_full_state_history_parity",),
            superseded_for_claims=("w7x_linear_stellarator_branch",),
            evidence=(
                "Selected-mode RHS/action, imported-state one-window replay, "
                "initial normalization, and row-normalized phi profiles pass, "
                "but no accepted full time-history contract exists. Independent "
                "stella validation covers the converged W7-X ky=0.3 branch only."
            ),
            next_action=(
                "Generate a compact revision-pinned GKW selected-mode history in "
                "external scratch storage and compare matched complex states at "
                "multiple accepted windows."
            ),
        ),
        ValidationConfidenceGap(
            identifier="gkw_multi_time_velocity_slice",
            status="open",
            summary="The GKW velocity-slice discrepancy grows over long time windows.",
            blocks_claims=("gkw_cyclone_full_velocity_space_history_parity",),
            superseded_for_claims=("w7x_linear_stellarator_branch",),
            evidence=(
                "The direct complex maximum error grows from 3.99e-3 at step 20 "
                "to 3.67e-2 at step 800; the existing 2e-2 multi-time tolerance "
                "is therefore not satisfied."
            ),
            next_action=(
                "Use a fresh pinned GKW producer to discriminate accumulated phase, "
                "time integration, and velocity-boundary effects before changing "
                "the tolerance."
            ),
            metrics=(
                ConfidenceMetric("step_20_complex_max_error", 3.99e-3),
                ConfidenceMetric("step_800_complex_max_error", 3.67e-2),
                ConfidenceMetric("acceptance_tolerance", 2.0e-2),
            ),
        ),
        ValidationConfidenceGap(
            identifier="cyclone_gx_low_ky_branch_shape",
            status="open",
            summary="The low-ky Cyclone/GX complex branch shape is not closed.",
            blocks_claims=("cyclone_gx_multi_ky_mode_structure_parity",),
            superseded_for_claims=(),
            evidence=(
                "The selected calibrated branch is useful as a scalar guardrail, "
                "but the portable multi-ky mode-structure comparison does not "
                "establish the low-ky complex eigenfunction branch."
            ),
            next_action=(
                "Run the revision-pinned GX producer with big diagnostics and compare "
                "phase-aligned per-ky structures only after matching branch windows."
            ),
        ),
        ValidationConfidenceGap(
            identifier="kinetic_electron_tem_external_parity",
            status="passed",
            summary="Kinetic-electron TEM growth, frequency, and mode structure pass.",
            blocks_claims=("kinetic_electron_tem_validation",),
            superseded_for_claims=(),
            evidence=(
                "At GKW kthrho=0.7 (internal krho=0.56548668) the converged local "
                "result gamma=0.63379954 and omega=-1.01859892 agrees with pinned "
                "Gyaradax gamma=0.66370834 and omega=-1.02976757 within declared "
                "10%/20% tolerances. The "
                "phase-aligned complex phi(z) relative L2 error is 0.0285 against "
                "a 0.25 tolerance, and late growth drift is 4.8e-15."
            ),
            next_action=(
                "Retain the revision-pinned producer and repeat the gate when the "
                "kinetic field, velocity grid, recurrence, or time advance changes."
            ),
        ),
        ValidationConfidenceGap(
            identifier="production_collisions_electromagnetic_parity",
            status="open",
            summary="Production collisions and electromagnetic fields lack parity.",
            blocks_claims=("collisional_electromagnetic_production_physics",),
            superseded_for_claims=(),
            evidence=(
                "A differentiable species-local conserving BGK model is integrated, "
                "but no Landau/Fokker--Planck inter-species operator is validated. "
                "A_parallel/B_parallel fields, the mixed-state transform, and the "
                "isolated electromagnetic RHS increment match pinned Gyaradax. The "
                "EM timestep bound dominates a small exact operator row sum; the "
                "production trajectory gate remains open."
            ),
            next_action=(
                "Close the electrostatic stencil baseline, then compare dispersion, "
                "growth, and mode structure against revision-pinned independent runs."
            ),
        ),
        ValidationConfidenceGap(
            identifier="nonlinear_stationary_heat_flux_parity",
            status="open",
            summary="Nonlinear stationary heat-flux parity is not established.",
            blocks_claims=("nonlinear_turbulence_validation",),
            superseded_for_claims=(),
            evidence=(
                "The dealiased ExB bracket, combined CFL driver, and saturation "
                "statistics pass unit tests, but no stationary resolution/domain "
                "ladder or independent heat-flux comparison has been accepted."
            ),
            next_action=(
                "Run a revision-pinned nonlinear benchmark through transient, "
                "stationarity, resolution, box-size, and independent parity gates."
            ),
        ),
        ValidationConfidenceGap(
            identifier="full_equilibrium_shape_optimization",
            status="open",
            summary="Unrestricted equilibrium-shape optimization is not validated.",
            blocks_claims=("production_equilibrium_shape_optimization",),
            superseded_for_claims=(),
            evidence=(
                "Reduced fixed-topology W7-X optimization passes, while end-to-end "
                "MHD autodiff, remeshing transitions, and production transport "
                "objectives are not established."
            ),
            next_action=(
                "Close the remaining geometry, gradient, timing, and physics gates "
                "before extending the optimizer to unrestricted boundary parameters."
            ),
        ),
    )


def validation_claim_readiness(
    claim: str,
    gaps: tuple[ValidationConfidenceGap, ...] | None = None,
) -> ClaimReadiness:
    """Return whether all ledger entries permit one named claim."""

    ledger = priority5_confidence_gaps() if gaps is None else tuple(gaps)
    blockers = tuple(
        gap.identifier
        for gap in ledger
        if gap.status == "open" and claim in gap.blocks_claims
    )
    superseded = tuple(
        gap.identifier
        for gap in ledger
        if gap.status == "open" and claim in gap.superseded_for_claims
    )
    return ClaimReadiness(
        claim=claim,
        ready=not blockers,
        blocking_gap_ids=blockers,
        superseded_gap_ids=superseded,
    )


def require_validation_claim_ready(
    claim: str,
    gaps: tuple[ValidationConfidenceGap, ...] | None = None,
) -> ClaimReadiness:
    """Raise with the exact blocking gaps unless a scientific claim is ready."""

    readiness = validation_claim_readiness(claim, gaps)
    if not readiness.ready:
        joined = ", ".join(readiness.blocking_gap_ids)
        raise ScientificClaimNotReadyError(f"claim {claim!r} is blocked by: {joined}")
    return readiness


def write_validation_confidence_report(
    path: str | Path,
    gaps: tuple[ValidationConfidenceGap, ...] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the current compact ledger without embedding external run data."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"validation confidence report already exists: {target}")
    ledger = priority5_confidence_gaps() if gaps is None else tuple(gaps)
    payload = {
        "schema_version": VALIDATION_CONFIDENCE_SCHEMA_VERSION,
        "gaps": [asdict(gap) for gap in ledger],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


__all__ = [
    "VALIDATION_CONFIDENCE_SCHEMA_VERSION",
    "ClaimReadiness",
    "ConfidenceMetric",
    "ScientificClaimNotReadyError",
    "ValidationConfidenceGap",
    "priority5_confidence_gaps",
    "require_validation_claim_ready",
    "validation_claim_readiness",
    "write_validation_confidence_report",
]
