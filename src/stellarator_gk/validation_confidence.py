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
                "result gamma=0.6637083371 and omega=-1.0297675735 agrees with "
                "pinned Gyaradax gamma=0.6637083356 and omega=-1.0297675725. "
                "Relative growth, frequency, and phase-aligned complex phi(z) "
                "errors are 2.23e-9, 1.02e-9, and 8.84e-10; late growth drift is "
                "2.35e-14."
            ),
            next_action=(
                "Retain the revision-pinned producer and repeat the gate when the "
                "kinetic field, velocity grid, recurrence, or time advance changes."
            ),
        ),
        ValidationConfidenceGap(
            identifier="production_collisions_electromagnetic_parity",
            status="open",
            summary="Inter-species production collision parity remains open.",
            blocks_claims=("collisional_electromagnetic_production_physics",),
            superseded_for_claims=(),
            evidence=(
                "A differentiable species-local conserving BGK model is integrated, "
                "and a standalone nine-point test-particle Fokker--Planck stencil/action "
                "matches pinned Gyaradax to 2e-12. The latter is solver-integrated with "
                "an exact row-sum CFL bound. A distinct species-local Xu correction "
                "matches pinned Gyaradax/GKW factors, weights, and corrected action to "
                "2e-12. The separate global completion preserves per-species density "
                "and combined momentum/energy to roundoff, but its exchange coefficients "
                "lack independent inter-species Landau field-particle parity. An "
                "experimental pairwise path now retains ordered target/background "
                "stencils, couples both distributions, and conserves density plus "
                "combined momentum/energy separately for every collision pair. GKW "
                "cannot close this gate because its conservation operator is not "
                "global; stella coefficient/action parity is still absent. "
                "A_parallel/B_parallel fields, the mixed-state transform, and the "
                "complete one-state electromagnetic RHS, one RK4 step, and a five-step "
                "trajectory match pinned Gyaradax when both codes use its separable "
                "upwind backend. A reduced final-time-10 finite-beta run also passes "
                "growth, frequency, mode-shape, and late-drift gates. The default EM "
                "ladder passes per-rung parity and 5% finest-pair convergence at "
                "12x12x6 -> 16x16x8. The production 16x16x8 -> 24x24x12 -> "
                "32x32x16 ladder also passes: finest local growth/frequency changes "
                "are 2.196%/1.206%, independent changes are 2.203%/1.206%, and "
                "finest growth/frequency/mode parity errors are 1.72e-5/1.08e-9/"
                "7.58e-10. The EM timestep bound dominates a small exact operator "
                "row sum, closing the production-grid electromagnetic sub-gap."
            ),
            next_action=(
                "Build an executable compact stella Laguerre--Legendre field-particle "
                "reference and compare reciprocal inter-species coefficients and action."
            ),
            metrics=(
                ConfidenceMetric("em_local_growth_finest_change", 2.1962696972943164e-2),
                ConfidenceMetric("em_local_frequency_finest_change", 1.2058357404686887e-2),
                ConfidenceMetric("em_reference_growth_finest_change", 2.2032779226274737e-2),
                ConfidenceMetric("em_reference_frequency_finest_change", 1.205835737686247e-2),
                ConfidenceMetric("em_finest_growth_parity_error", 1.719448606648505e-5),
                ConfidenceMetric("em_finest_mode_parity_error", 7.576553792617767e-10),
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
                "statistics pass unit tests, and the collocation heat response matches "
                "direct quadrature. A corrected twist-and-shift 12x12x6, 9x5 trajectory "
                "produced the first accepted local window at t=220--300: mean native flux "
                "-25.7841, relative standard error 0.462%, relative drift 0.0572, and "
                "nonzonal-potential growth -1.93e-3. The following t=300--400 segment has "
                "a consistent mean (-26.256, 1.8% change) but correctly fails its drift "
                "gate. Contract-checked restarts and a segment merger support windows "
                "independent of checkpoint boundaries. The GX summarizer and local producer "
                "now emit explicit stationarity decisions, and the shared loader preserves "
                "those decisions instead of accepting rejected windows from drift alone. "
                "A fail-closed campaign command evaluates separate resolution and domain "
                "ladders plus independent parity. No second stationary local rung or GX "
                "comparison has yet passed. Source tracing against pinned GX established "
                "its total-energy moment, Fourier factor, and s-alpha flux weight; the local "
                "producer now exposes the matched gx_Q_over_Q_GB diagnostic separately from "
                "its historical non-advective heat moment."
            ),
            next_action=(
                "Run independently initialized resolved and wider-domain stationary rungs, "
                "run the pinned GX case with the source-matched total-energy diagnostic, "
                "then execute the combined campaign gate."
            ),
            metrics=(
                ConfidenceMetric("coarse_stationary_mean_native_flux", -25.7841),
                ConfidenceMetric("coarse_stationary_relative_standard_error", 4.62e-3),
                ConfidenceMetric("coarse_stationary_relative_drift", 5.72e-2),
                ConfidenceMetric("coarse_stationary_nonzonal_growth", -1.93e-3),
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
        gap.identifier for gap in ledger if gap.status == "open" and claim in gap.blocks_claims
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
