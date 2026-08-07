"""Acceptance contracts for stationary nonlinear heat-flux validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
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


def load_nonlinear_heat_flux_record(path: str | Path) -> NonlinearHeatFluxRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("nonlinear heat-flux report must use schema version 1")
    producer = payload.get("producer")
    normalization = payload.get("normalization")
    statistics = payload.get("statistics")
    stationary = payload.get("stationary")
    if not isinstance(producer, str) or not isinstance(normalization, str):
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
    if record.n_samples < 2:
        raise ValueError("nonlinear heat-flux window requires at least two samples")
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

    if min(mean_tolerance, drift_tolerance, relative_standard_error_tolerance) <= 0.0:
        raise ValueError("nonlinear heat-flux tolerances must be positive")
    if local_to_reference_factor is None:
        if local.normalization != reference.normalization:
            raise ValueError("heat-flux normalizations differ; supply local_to_reference_factor")
        factor = 1.0
    else:
        factor = float(local_to_reference_factor)
        if factor <= 0.0:
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

    if len(records) < 2 or tolerance <= 0.0:
        raise ValueError("convergence requires at least two rungs and positive tolerance")
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


__all__ = [
    "NonlinearHeatFluxConvergenceReport",
    "NonlinearHeatFluxParityReport",
    "NonlinearHeatFluxRecord",
    "compare_nonlinear_heat_flux",
    "compare_nonlinear_heat_flux_convergence",
    "load_nonlinear_heat_flux_record",
]
