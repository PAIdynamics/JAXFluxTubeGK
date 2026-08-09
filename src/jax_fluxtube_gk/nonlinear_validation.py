"""Acceptance contracts for stationary nonlinear heat-flux validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class NonlinearHeatFluxRecord:
    producer: str
    normalization: str
    mean: float
    standard_error: float
    relative_window_drift: float
    n_samples: int
    stationary: bool = True


@dataclass(frozen=True)
class NonlinearHeatFluxParityReport:
    passed: bool
    local_stationary: bool
    reference_stationary: bool
    mean_relative_error: float
    local_relative_standard_error: float
    reference_relative_standard_error: float
    mean_tolerance: float
    drift_tolerance: float
    relative_standard_error_tolerance: float
    local_to_reference_factor: float


@dataclass(frozen=True)
class NonlinearHeatFluxConvergenceReport:
    passed: bool
    finest_relative_change: float
    tolerance: float
    all_stationary: bool


@dataclass(frozen=True)
class NonlinearHeatFluxEnsembleReport:
    passed: bool
    n_lineages: int
    n_stationary: int
    stationary_fraction: float
    all_lineages_unique: bool
    ensemble_mean: float
    between_lineage_standard_error: float
    maximum_relative_mean_deviation: float
    mean_spread_tolerance: float
    minimum_stationary_fraction: float


def load_nonlinear_heat_flux_record(path: str | Path) -> NonlinearHeatFluxRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("nonlinear heat-flux report must use schema version 1")
    producer = payload.get("producer")
    normalization = payload.get("normalization")
    statistics = payload.get("statistics")
    stationary = payload.get("stationary")
    if (
        not isinstance(producer, str)
        or not producer.strip()
        or not isinstance(normalization, str)
        or not normalization.strip()
    ):
        raise ValueError("nonlinear heat-flux report lacks producer or normalization")
    if not isinstance(statistics, dict):
        raise ValueError("nonlinear heat-flux report lacks statistics")
    if not isinstance(stationary, bool):
        raise ValueError("nonlinear heat-flux report lacks an explicit stationary decision")
    try:
        record = NonlinearHeatFluxRecord(
            producer=producer,
            normalization=normalization,
            mean=float(statistics["mean"]),
            standard_error=float(statistics["standard_error"]),
            relative_window_drift=float(statistics["relative_window_drift"]),
            n_samples=int(statistics["n_samples"]),
            stationary=stationary,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("nonlinear heat-flux statistics are incomplete") from exc
    _validate_heat_flux_record(record)
    raw_n_samples = statistics.get("n_samples")
    if isinstance(raw_n_samples, bool) or not float(raw_n_samples).is_integer():
        raise ValueError("nonlinear heat-flux sample count must be an integer")
    return record


def compare_nonlinear_heat_flux(
    local: NonlinearHeatFluxRecord,
    reference: NonlinearHeatFluxRecord,
    *,
    local_to_reference_factor: float | None = None,
    mean_tolerance: float = 0.20,
    drift_tolerance: float = 0.20,
    relative_standard_error_tolerance: float = 0.10,
) -> NonlinearHeatFluxParityReport:
    """Require stationary local/reference windows and normalized mean parity."""

    _validate_heat_flux_record(local)
    _validate_heat_flux_record(reference)
    tolerances = (mean_tolerance, drift_tolerance, relative_standard_error_tolerance)
    if not all(math.isfinite(value) and value > 0.0 for value in tolerances):
        raise ValueError("nonlinear heat-flux tolerances must be positive")
    if local_to_reference_factor is None:
        if local.normalization != reference.normalization:
            raise ValueError("heat-flux normalizations differ; supply local_to_reference_factor")
        factor = 1.0
    else:
        factor = float(local_to_reference_factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("local_to_reference_factor must be positive")
    local_mean = factor * local.mean
    local_error = factor * local.standard_error
    local_rse = abs(local_error) / max(abs(local_mean), 1.0e-14)
    reference_rse = abs(reference.standard_error) / max(abs(reference.mean), 1.0e-14)
    local_stationary = (
        local.stationary
        and abs(local.relative_window_drift) <= drift_tolerance
        and local_rse <= relative_standard_error_tolerance
    )
    reference_stationary = (
        reference.stationary
        and abs(reference.relative_window_drift) <= drift_tolerance
        and reference_rse <= relative_standard_error_tolerance
    )
    mean_error = abs(local_mean - reference.mean) / max(abs(reference.mean), 1.0e-14)
    return NonlinearHeatFluxParityReport(
        passed=local_stationary and reference_stationary and mean_error <= mean_tolerance,
        local_stationary=local_stationary,
        reference_stationary=reference_stationary,
        mean_relative_error=mean_error,
        local_relative_standard_error=local_rse,
        reference_relative_standard_error=reference_rse,
        mean_tolerance=mean_tolerance,
        drift_tolerance=drift_tolerance,
        relative_standard_error_tolerance=relative_standard_error_tolerance,
        local_to_reference_factor=factor,
    )


def compare_nonlinear_heat_flux_convergence(
    records: tuple[NonlinearHeatFluxRecord, ...],
    *,
    tolerance: float = 0.15,
    drift_tolerance: float = 0.20,
    relative_standard_error_tolerance: float = 0.10,
) -> NonlinearHeatFluxConvergenceReport:
    """Require stationary rungs and convergence between the finest two means."""

    if len(records) < 2 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("convergence requires at least two rungs and positive tolerance")
    for record in records:
        _validate_heat_flux_record(record)
    if len({record.normalization for record in records}) != 1:
        raise ValueError("nonlinear convergence normalizations differ")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (drift_tolerance, relative_standard_error_tolerance)
    ):
        raise ValueError("nonlinear convergence tolerances must be positive")
    stationary = tuple(
        record.stationary
        and abs(record.relative_window_drift) <= drift_tolerance
        and abs(record.standard_error) / max(abs(record.mean), 1.0e-14)
        <= relative_standard_error_tolerance
        for record in records
    )
    change = abs(records[-1].mean - records[-2].mean) / max(abs(records[-1].mean), 1.0e-14)
    return NonlinearHeatFluxConvergenceReport(
        passed=all(stationary) and change <= tolerance,
        finest_relative_change=change,
        tolerance=tolerance,
        all_stationary=all(stationary),
    )


def compare_nonlinear_heat_flux_ensemble(
    records: tuple[NonlinearHeatFluxRecord, ...],
    lineage_ids: tuple[str, ...],
    *,
    minimum_lineages: int = 3,
    mean_spread_tolerance: float = 0.15,
    minimum_stationary_fraction: float = 1.0,
    drift_tolerance: float = 0.20,
    relative_standard_error_tolerance: float = 0.10,
) -> NonlinearHeatFluxEnsembleReport:
    """Reject cherry-picked chaotic trajectories using independent lineages."""

    if len(records) != len(lineage_ids):
        raise ValueError("records and lineage_ids must have equal length")
    if len(records) < minimum_lineages or minimum_lineages < 2:
        raise ValueError("nonlinear ensemble requires the declared minimum lineages")
    for record in records:
        _validate_heat_flux_record(record)
    if (
        not math.isfinite(mean_spread_tolerance)
        or mean_spread_tolerance <= 0.0
        or not math.isfinite(minimum_stationary_fraction)
        or not 0.0 < minimum_stationary_fraction <= 1.0
        or not math.isfinite(drift_tolerance)
        or drift_tolerance <= 0.0
        or not math.isfinite(relative_standard_error_tolerance)
        or relative_standard_error_tolerance <= 0.0
    ):
        raise ValueError("ensemble tolerances must be positive and fraction at most one")
    if len({record.normalization for record in records}) != 1:
        raise ValueError("nonlinear ensemble normalizations differ")

    stationary = tuple(
        record.stationary
        and abs(record.relative_window_drift) <= drift_tolerance
        and abs(record.standard_error) / max(abs(record.mean), 1.0e-14)
        <= relative_standard_error_tolerance
        for record in records
    )
    means = tuple(record.mean for record in records)
    ensemble_mean = sum(means) / len(means)
    maximum_deviation = max(abs(value - ensemble_mean) for value in means) / max(
        abs(ensemble_mean), 1.0e-14
    )
    variance = sum((value - ensemble_mean) ** 2 for value in means) / (len(means) - 1)
    between_standard_error = (variance / len(means)) ** 0.5
    n_stationary = sum(stationary)
    stationary_fraction = n_stationary / len(records)
    all_unique = len(set(lineage_ids)) == len(lineage_ids)
    return NonlinearHeatFluxEnsembleReport(
        passed=(
            all_unique
            and stationary_fraction >= minimum_stationary_fraction
            and maximum_deviation <= mean_spread_tolerance
        ),
        n_lineages=len(records),
        n_stationary=n_stationary,
        stationary_fraction=stationary_fraction,
        all_lineages_unique=all_unique,
        ensemble_mean=ensemble_mean,
        between_lineage_standard_error=between_standard_error,
        maximum_relative_mean_deviation=maximum_deviation,
        mean_spread_tolerance=mean_spread_tolerance,
        minimum_stationary_fraction=minimum_stationary_fraction,
    )


def _validate_heat_flux_record(record: NonlinearHeatFluxRecord) -> None:
    if not isinstance(record.producer, str) or not record.producer.strip():
        raise ValueError("nonlinear heat-flux record requires a producer")
    if not isinstance(record.normalization, str) or not record.normalization.strip():
        raise ValueError("nonlinear heat-flux record requires a normalization")
    if not all(
        math.isfinite(value)
        for value in (record.mean, record.standard_error, record.relative_window_drift)
    ):
        raise ValueError("nonlinear heat-flux statistics must be finite")
    if record.standard_error < 0.0:
        raise ValueError("nonlinear heat-flux standard error must be nonnegative")
    if isinstance(record.n_samples, bool) or record.n_samples < 2:
        raise ValueError("nonlinear heat-flux window requires at least two samples")


__all__ = [
    "NonlinearHeatFluxConvergenceReport",
    "NonlinearHeatFluxEnsembleReport",
    "NonlinearHeatFluxParityReport",
    "NonlinearHeatFluxRecord",
    "compare_nonlinear_heat_flux",
    "compare_nonlinear_heat_flux_convergence",
    "compare_nonlinear_heat_flux_ensemble",
    "load_nonlinear_heat_flux_record",
]
