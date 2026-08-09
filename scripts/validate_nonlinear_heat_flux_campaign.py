#!/usr/bin/env python3
"""Evaluate nonlinear resolution, domain, and independent-reference gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from jax_fluxtube_gk import (
    compare_nonlinear_heat_flux,
    compare_nonlinear_heat_flux_convergence,
    compare_nonlinear_heat_flux_ensemble,
    load_nonlinear_heat_flux_record,
)


_LOCAL_PRODUCERS = frozenset(
    {
        "jax-fluxtube-gk/nonlinear-heat-flux",
        "jax-fluxtube-gk/nonlinear-heat-flux-merged",
    }
)
_REFERENCE_PRODUCER = "gx-nonlinear-heat-flux"
_MIN_STATIONARY_SAMPLES = 100
_MIN_STATIONARY_DURATION = 10.0
_MIN_STATIONARY_BLOCKS = 6
_MIN_NONZONAL_RMS_RATIO = 0.8
_MAX_ABSOLUTE_FIELD_GROWTH = 0.02


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
    resolution_contract = _validate_resolution_ladder(resolution_paths)
    domain_contract = _validate_domain_ladder(domain_paths)
    cross_ladder_contract = _validate_cross_ladder_anchor(
        resolution_paths[-1], domain_paths[0]
    )
    resolution_records = tuple(map(load_nonlinear_heat_flux_record, resolution_paths))
    domain_records = tuple(map(load_nonlinear_heat_flux_record, domain_paths))
    reference = load_nonlinear_heat_flux_record(reference_path)
    parity_contract = _validate_reference_case(domain_paths[-1], reference_path)
    lineage_records = tuple(map(load_nonlinear_heat_flux_record, lineage_paths))
    lineage_producers_valid = all(record.producer in _LOCAL_PRODUCERS for record in lineage_records)
    lineage_cases_valid = _validate_lineage_cases(lineage_paths, domain_paths[0])
    local_evidence_paths = tuple(dict.fromkeys((*resolution_paths, *domain_paths, *lineage_paths)))
    stationarity_evidence = {
        "local": [
            _validate_stationarity_evidence(
                path,
                drift_tolerance=drift_tolerance,
                relative_standard_error_tolerance=relative_standard_error_tolerance,
                require_field_evidence=True,
            )
            for path in local_evidence_paths
        ],
        "reference": _validate_stationarity_evidence(
            reference_path,
            drift_tolerance=drift_tolerance,
            relative_standard_error_tolerance=relative_standard_error_tolerance,
            require_field_evidence=False,
        ),
    }
    stationarity_evidence["passed"] = all(
        item["passed"] for item in stationarity_evidence["local"]
    ) and stationarity_evidence["reference"]["passed"]
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
        domain_records[-1],
        reference,
        local_to_reference_factor=local_to_reference_factor,
        mean_tolerance=mean_tolerance,
        drift_tolerance=drift_tolerance,
        relative_standard_error_tolerance=relative_standard_error_tolerance,
    )
    return {
        "passed": (
            resolution_contract["passed"]
            and domain_contract["passed"]
            and cross_ladder_contract["passed"]
            and resolution.passed
            and domain.passed
            and parity.passed
            and parity_contract["passed"]
            and ensemble.passed
            and lineage_producers_valid
            and lineage_cases_valid
            and stationarity_evidence["passed"]
        ),
        "resolution_ladder_contract": resolution_contract,
        "domain_ladder_contract": domain_contract,
        "cross_ladder_contract": cross_ladder_contract,
        "lineage_ensemble": asdict(ensemble),
        "lineage_producer_contract": {
            "passed": lineage_producers_valid and lineage_cases_valid,
            "allowed": sorted(_LOCAL_PRODUCERS),
            "fixed_case": lineage_cases_valid,
        },
        "stationarity_evidence_contract": stationarity_evidence,
        "resolution_convergence": asdict(resolution),
        "domain_convergence": asdict(domain),
        "independent_parity": asdict(parity),
        "independent_parity_contract": parity_contract,
    }


_PHASE_RESOLUTION_KEYS = ("n_z", "n_vpar", "n_mu")
_PHYSICS_KEYS = (
    "parallel_boundary_model",
    "parallel_recurrence_rate",
    "rmaj_over_lref",
    "gx_fprim",
    "gx_tprim",
    "density_gradient_R_over_Ln",
    "temperature_gradient_R_over_LT",
    "hyperdiffusion",
    "collision_frequency",
    "flux_moment",
    "ikxspace",
)


def _case_contract(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    case = payload.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"nonlinear ladder report {path} lacks a case contract")
    required = (*_PHASE_RESOLUTION_KEYS, "n_kx", "n_ky", "kx", "ky", *_PHYSICS_KEYS)
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"nonlinear ladder report {path} lacks case keys: {missing}")
    kx = np.asarray(case["kx"], dtype=float)
    ky = np.asarray(case["ky"], dtype=float)
    kx_spacing = np.diff(kx)
    ky_spacing = np.diff(ky)
    if (
        kx.shape != (int(case["n_kx"]),)
        or ky.shape != (int(case["n_ky"]),)
        or kx.size < 3
        or kx.size % 2 == 0
        or ky.size < 2
        or not np.all(np.isfinite(kx))
        or not np.all(np.isfinite(ky))
        or np.any(kx_spacing <= 0.0)
        or np.any(ky_spacing <= 0.0)
        or not np.allclose(kx_spacing, kx_spacing[0], rtol=1.0e-12, atol=1.0e-14)
        or not np.allclose(ky_spacing, ky_spacing[0], rtol=1.0e-12, atol=1.0e-14)
        or not np.allclose(kx, -kx[::-1], rtol=1.0e-12, atol=1.0e-14)
        or not np.isclose(ky[0], 0.0, rtol=0.0, atol=1.0e-14)
    ):
        raise ValueError(f"nonlinear ladder report {path} has an invalid Fourier grid")
    return case


def _same_values(left: dict, right: dict, keys: tuple[str, ...]) -> bool:
    return all(left[key] == right[key] for key in keys)


def _validate_resolution_ladder(paths: tuple[Path, ...]) -> dict:
    cases = tuple(_case_contract(path) for path in paths)
    local_producers = _local_producers_valid(paths)
    base = cases[0]
    fixed_fourier = all(
        _same_values(base, case, ("n_kx", "n_ky", "kx", "ky")) for case in cases[1:]
    )
    fixed_physics = all(_same_values(base, case, _PHYSICS_KEYS) for case in cases[1:])
    resolutions = tuple(tuple(int(case[key]) for key in _PHASE_RESOLUTION_KEYS) for case in cases)
    strictly_refined = all(
        all(right >= left for left, right in zip(previous, current, strict=True))
        and any(right > left for left, right in zip(previous, current, strict=True))
        for previous, current in zip(resolutions, resolutions[1:], strict=False)
    )
    return {
        "passed": local_producers and fixed_fourier and fixed_physics and strictly_refined,
        "local_producers": local_producers,
        "fixed_fourier_grid": fixed_fourier,
        "fixed_physics": fixed_physics,
        "strictly_refined": strictly_refined,
        "phase_space_resolutions": [list(values) for values in resolutions],
    }


def _fourier_extent(case: dict) -> dict:
    kx = np.asarray(case["kx"], dtype=float)
    ky = np.asarray(case["ky"], dtype=float)
    positive_kx_spacing = np.diff(kx)
    positive_ky = ky[ky > 0.0]
    if positive_kx_spacing.size == 0 or positive_ky.size == 0:
        raise ValueError("domain ladder needs multiple kx modes and a positive ky mode")
    return {
        "delta_kx": float(np.min(positive_kx_spacing)),
        "delta_ky": float(positive_ky[0]),
        "max_abs_kx": float(np.max(np.abs(kx))),
        "max_ky": float(ky[-1]),
    }


def _validate_domain_ladder(paths: tuple[Path, ...]) -> dict:
    cases = tuple(_case_contract(path) for path in paths)
    local_producers = _local_producers_valid(paths)
    base = cases[0]
    fixed_phase_resolution = all(
        _same_values(base, case, _PHASE_RESOLUTION_KEYS) for case in cases[1:]
    )
    fixed_physics = all(_same_values(base, case, _PHYSICS_KEYS) for case in cases[1:])
    extents = tuple(_fourier_extent(case) for case in cases)
    domain_expanded = all(
        current["delta_kx"] <= previous["delta_kx"] * (1.0 + 1.0e-12)
        and current["delta_ky"] <= previous["delta_ky"] * (1.0 + 1.0e-12)
        and (
            current["delta_kx"] < previous["delta_kx"] * (1.0 - 1.0e-12)
            or current["delta_ky"] < previous["delta_ky"] * (1.0 - 1.0e-12)
        )
        and current["max_abs_kx"] >= previous["max_abs_kx"] * (1.0 - 1.0e-12)
        and current["max_ky"] >= previous["max_ky"] * (1.0 - 1.0e-12)
        for previous, current in zip(extents, extents[1:], strict=False)
    )
    return {
        "passed": local_producers and fixed_phase_resolution and fixed_physics and domain_expanded,
        "local_producers": local_producers,
        "fixed_phase_space_resolution": fixed_phase_resolution,
        "fixed_physics": fixed_physics,
        "domain_expanded_without_lost_bandwidth": domain_expanded,
        "fourier_extents": list(extents),
    }


def _validate_cross_ladder_anchor(resolution_path: Path, domain_path: Path) -> dict:
    resolution = _case_contract(resolution_path)
    domain = _case_contract(domain_path)
    fixed_fourier = _same_values(resolution, domain, ("n_kx", "n_ky", "kx", "ky"))
    fixed_physics = _same_values(resolution, domain, _PHYSICS_KEYS)
    return {
        "passed": fixed_fourier and fixed_physics,
        "shared_base_fourier_grid": fixed_fourier,
        "shared_physics": fixed_physics,
    }


def _validate_lineage_cases(paths: tuple[Path, ...], anchor_path: Path) -> bool:
    anchor = _case_contract(anchor_path)
    keys = (*_PHASE_RESOLUTION_KEYS, "n_kx", "n_ky", "kx", "ky", *_PHYSICS_KEYS)
    return all(_same_values(anchor, _case_contract(path), keys) for path in paths)


def _validate_reference_case(local_path: Path, reference_path: Path) -> dict:
    reference_payload = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if reference_payload.get("producer") != _REFERENCE_PRODUCER:
        return {
            "passed": False,
            "required": True,
            "checks": {"producer": False},
            "expected_producer": _REFERENCE_PRODUCER,
        }
    local = _case_contract(local_path)
    reference = reference_payload.get("case")
    if not isinstance(reference, dict):
        return {"passed": False, "required": True, "checks": {"case_contract_present": False}}
    ky = np.asarray(local["ky"], dtype=float)
    positive_ky = ky[ky > 0.0]
    expected = {
        "geometry": "s-alpha",
        "q": 1.4,
        "shat": 0.8,
        "eps": 0.18,
        "rmaj_over_lref": float(local["rmaj_over_lref"]),
        "fprim": float(local["gx_fprim"]),
        "tprim": float(local["gx_tprim"]),
        "ky_min": float(positive_ky[0]),
        "boundary": "linked" if local["parallel_boundary_model"] == "twist_shift" else "periodic",
        "electrostatic": True,
        "hyperdiffusion": float(local["hyperdiffusion"]),
        "hyperdiffusion_order": 4,
        "collision_frequency": float(local["collision_frequency"]),
    }
    checks = {
        key: (
            bool(np.isclose(float(reference.get(key, np.nan)), value, rtol=1.0e-12, atol=1.0e-12))
            if isinstance(value, float)
            else reference.get(key) == value
        )
        for key, value in expected.items()
    }
    return {"passed": all(checks.values()), "required": True, "checks": checks}


def _local_producers_valid(paths: tuple[Path, ...]) -> bool:
    return all(
        json.loads(Path(path).read_text(encoding="utf-8")).get("producer")
        in _LOCAL_PRODUCERS
        for path in paths
    )


def _validate_stationarity_evidence(
    path: Path,
    *,
    drift_tolerance: float,
    relative_standard_error_tolerance: float,
    require_field_evidence: bool,
) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        return {"path": str(path), "passed": False, "checks": {"statistics": False}}
    case = payload.get("case")
    controls = case if require_field_evidence else payload.get("stationarity_controls")
    if not isinstance(controls, dict):
        return {"path": str(path), "passed": False, "checks": {"controls": False}}

    if require_field_evidence:
        duration = payload.get("stationary_window_duration", np.nan)
        declared = {
            "max_relative_drift": controls.get("max_relative_drift", np.nan),
            "max_relative_standard_error": controls.get(
                "max_relative_standard_error", np.nan
            ),
            "min_samples": controls.get("min_stationary_samples", np.nan),
            "min_duration": controls.get("min_stationary_window_duration", np.nan),
            "min_blocks": controls.get("min_stationary_blocks", np.nan),
        }
    else:
        duration = float(statistics.get("window_end_time", np.nan)) - float(
            statistics.get("window_start_time", np.nan)
        )
        declared = {
            "max_relative_drift": controls.get("max_relative_drift", np.nan),
            "max_relative_standard_error": controls.get(
                "max_relative_standard_error", np.nan
            ),
            "min_samples": controls.get("min_stationary_samples", np.nan),
            "min_duration": controls.get("min_stationary_window_duration", np.nan),
            "min_blocks": controls.get("min_stationary_blocks", np.nan),
        }

    checks = {
        "producer_stationary": payload.get("stationary") is True,
        "sample_count": _finite_at_least(
            statistics.get("n_samples"), _MIN_STATIONARY_SAMPLES
        ),
        "window_duration": _finite_at_least(duration, _MIN_STATIONARY_DURATION),
        "physical_time_blocks": _finite_at_least(
            statistics.get("n_blocks"), _MIN_STATIONARY_BLOCKS
        ),
        "declared_sample_minimum": _finite_at_least(
            declared["min_samples"], _MIN_STATIONARY_SAMPLES
        ),
        "declared_duration_minimum": _finite_at_least(
            declared["min_duration"], _MIN_STATIONARY_DURATION
        ),
        "declared_block_minimum": _finite_at_least(
            declared["min_blocks"], _MIN_STATIONARY_BLOCKS
        ),
        "declared_drift_limit": _finite_at_most(
            declared["max_relative_drift"], drift_tolerance
        ),
        "declared_error_limit": _finite_at_most(
            declared["max_relative_standard_error"],
            relative_standard_error_tolerance,
        ),
    }
    if require_field_evidence:
        checks |= {
            "declared_amplitude_minimum": _finite_at_least(
                controls.get("min_phi_rms_ratio"), _MIN_NONZONAL_RMS_RATIO
            ),
            "declared_growth_limit": _finite_at_most(
                controls.get("max_absolute_phi_growth_rate"),
                _MAX_ABSOLUTE_FIELD_GROWTH,
            ),
            "nonzonal_amplitude_ratio": _finite_at_least(
                payload.get("nonzonal_phi_rms_ratio"), _MIN_NONZONAL_RMS_RATIO
            ),
            "nonzonal_growth_rate": _finite_absolute_at_most(
                payload.get("candidate_nonzonal_phi_growth_rate"),
                _MAX_ABSOLUTE_FIELD_GROWTH,
            ),
        }
    return {"path": str(path), "passed": all(checks.values()), "checks": checks}


def _finite_at_least(value, limit: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and numeric >= limit)


def _finite_at_most(value, limit: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and 0.0 < numeric <= limit)


def _finite_absolute_at_most(value, limit: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and abs(numeric) <= limit)


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
        "producer": "jax-fluxtube-gk/nonlinear-heat-flux-campaign",
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
