#!/usr/bin/env python3
"""Evaluate nonlinear resolution, domain, and independent-reference gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from stellarator_gk import (
    compare_nonlinear_heat_flux,
    compare_nonlinear_heat_flux_convergence,
    compare_nonlinear_heat_flux_ensemble,
    load_nonlinear_heat_flux_record,
)


def evaluate_campaign(
    resolution_paths: tuple[Path, ...],
    domain_paths: tuple[Path, ...],
    reference_path: Path,
    *,
    lineage_paths: tuple[Path, ...],
    local_to_reference_factor: float | None = None,
    convergence_tolerance: float = 0.15,
    mean_tolerance: float = 0.20,
    drift_tolerance: float = 0.20,
    relative_standard_error_tolerance: float = 0.10,
    lineage_mean_spread_tolerance: float = 0.15,
) -> dict:
    """Load caller-owned reports and evaluate all nonlinear acceptance gates."""

    if len(resolution_paths) < 2 or len(domain_paths) < 2:
        raise ValueError("resolution and domain ladders each require at least two reports")
    resolution_records = tuple(map(load_nonlinear_heat_flux_record, resolution_paths))
    domain_records = tuple(map(load_nonlinear_heat_flux_record, domain_paths))
    reference = load_nonlinear_heat_flux_record(reference_path)
    lineage_records = tuple(map(load_nonlinear_heat_flux_record, lineage_paths))
    lineage_ids = tuple(_lineage_id(path) for path in lineage_paths)
    ensemble = compare_nonlinear_heat_flux_ensemble(
        lineage_records,
        lineage_ids,
        mean_spread_tolerance=lineage_mean_spread_tolerance,
        drift_tolerance=drift_tolerance,
        relative_standard_error_tolerance=relative_standard_error_tolerance,
    )
    resolution = compare_nonlinear_heat_flux_convergence(
        resolution_records,
        tolerance=convergence_tolerance,
        drift_tolerance=drift_tolerance,
        relative_standard_error_tolerance=relative_standard_error_tolerance,
    )
    domain = compare_nonlinear_heat_flux_convergence(
        domain_records,
        tolerance=convergence_tolerance,
        drift_tolerance=drift_tolerance,
        relative_standard_error_tolerance=relative_standard_error_tolerance,
    )
    parity = compare_nonlinear_heat_flux(
        resolution_records[-1],
        reference,
        local_to_reference_factor=local_to_reference_factor,
        mean_tolerance=mean_tolerance,
        drift_tolerance=drift_tolerance,
        relative_standard_error_tolerance=relative_standard_error_tolerance,
    )
    return {
        "passed": resolution.passed and domain.passed and parity.passed and ensemble.passed,
        "lineage_ensemble": asdict(ensemble),
        "resolution_convergence": asdict(resolution),
        "domain_convergence": asdict(domain),
        "independent_parity": asdict(parity),
    }


def _lineage_id(path: Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lineage = payload.get("trajectory_lineage")
    if not isinstance(lineage, dict) or lineage.get("schema_version") != 1:
        raise ValueError("lineage ensemble report lacks schema-v1 trajectory lineage")
    keys = ("seed", "initial_amplitude", "initial_zonal_fraction")
    if any(key not in lineage for key in keys):
        raise ValueError("trajectory lineage lacks initialization controls")
    return json.dumps({key: lineage[key] for key in keys}, sort_keys=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-report", type=Path, action="append", required=True)
    parser.add_argument("--domain-report", type=Path, action="append", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--lineage-report", type=Path, action="append", required=True)
    parser.add_argument("--local-to-reference-factor", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--convergence-tolerance", type=float, default=0.15)
    parser.add_argument("--mean-tolerance", type=float, default=0.20)
    parser.add_argument("--drift-tolerance", type=float, default=0.20)
    parser.add_argument("--relative-standard-error-tolerance", type=float, default=0.10)
    parser.add_argument("--lineage-mean-spread-tolerance", type=float, default=0.15)
    args = parser.parse_args(argv)
    if len(args.resolution_report) < 2 or len(args.domain_report) < 2:
        parser.error("repeat both ladder report options at least twice")
    if len(args.lineage_report) < 3:
        parser.error("repeat --lineage-report at least three times")
    tolerances = (
        args.convergence_tolerance,
        args.mean_tolerance,
        args.drift_tolerance,
        args.relative_standard_error_tolerance,
        args.lineage_mean_spread_tolerance,
    )
    if min(tolerances) <= 0.0 or (
        args.local_to_reference_factor is not None and args.local_to_reference_factor <= 0.0
    ):
        parser.error("normalization factor and tolerances must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = evaluate_campaign(
        tuple(args.resolution_report),
        tuple(args.domain_report),
        args.reference_report,
        lineage_paths=tuple(args.lineage_report),
        local_to_reference_factor=args.local_to_reference_factor,
        convergence_tolerance=args.convergence_tolerance,
        mean_tolerance=args.mean_tolerance,
        drift_tolerance=args.drift_tolerance,
        relative_standard_error_tolerance=args.relative_standard_error_tolerance,
        lineage_mean_spread_tolerance=args.lineage_mean_spread_tolerance,
    )
    payload = {
        "schema_version": 1,
        "producer": "optimal-fusion/nonlinear-heat-flux-campaign",
        "inputs": {
            "resolution_reports": [str(path.resolve()) for path in args.resolution_report],
            "domain_reports": [str(path.resolve()) for path in args.domain_report],
            "reference_report": str(args.reference_report.resolve()),
            "lineage_reports": [str(path.resolve()) for path in args.lineage_report],
        },
        "report": report,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"passed={report['passed']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
