#!/usr/bin/env python3
"""Merge contiguous local nonlinear reports into one stationarity window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jax_fluxtube_gk import correlated_flux_statistics


_CONTRACT_KEYS = (
    "n_z",
    "n_vpar",
    "n_mu",
    "n_kx",
    "n_ky",
    "kx",
    "ky",
    "ikxspace",
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
)
_INPUT_PRODUCERS = frozenset(
    {
        "jax-fluxtube-gk/nonlinear-heat-flux",
        "jax-fluxtube-gk/nonlinear-heat-flux-merged",
    }
)


def _contract(payload: dict) -> dict:
    case = payload.get("case")
    if not isinstance(case, dict):
        raise ValueError("nonlinear segment lacks a case contract")
    missing = [key for key in _CONTRACT_KEYS if key not in case]
    if missing:
        raise ValueError(f"nonlinear segment case lacks contract keys: {missing}")
    return {key: case[key] for key in _CONTRACT_KEYS}


def _trace(payload: dict):
    arrays = tuple(
        np.asarray(payload.get(key), dtype=float)
        for key in ("times", "heat_flux", "nonzonal_phi_rms")
    )
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("segment times, heat flux, and nonzonal RMS must be one-dimensional")
    if len({values.size for values in arrays}) != 1 or arrays[0].size < 2:
        raise ValueError("segment traces must share at least two samples")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("segment traces must be finite")
    if np.any(np.diff(arrays[0]) <= 0.0) or np.any(arrays[2] <= 0.0):
        raise ValueError("segment times must increase and nonzonal RMS must be positive")
    start_time = float(payload.get("start_time", np.nan))
    end_time = float(payload.get("end_time", np.nan))
    if (
        not np.isfinite(start_time)
        or not np.isfinite(end_time)
        or not np.isclose(arrays[0][0], start_time, rtol=1.0e-10, atol=1.0e-12)
        or not np.isclose(arrays[0][-1], end_time, rtol=1.0e-10, atol=1.0e-12)
    ):
        raise ValueError("segment trace endpoints do not match report start/end times")
    return arrays


def _lineage_root(payload: dict) -> dict:
    lineage = payload.get("trajectory_lineage")
    if not isinstance(lineage, dict) or lineage.get("schema_version") != 1:
        raise ValueError("nonlinear segment lacks schema-v1 trajectory lineage")
    end_times = lineage.get("segment_end_times")
    if not isinstance(end_times, list) or not end_times:
        raise ValueError("nonlinear segment trajectory lineage lacks segment endpoints")
    end_time = float(payload.get("end_time", np.nan))
    if not np.isfinite(end_time) or not np.isclose(float(end_times[-1]), end_time):
        raise ValueError("nonlinear segment lineage does not terminate at its report time")
    keys = ("seed", "initial_amplitude", "initial_zonal_fraction")
    if any(key not in lineage for key in keys):
        raise ValueError("nonlinear segment trajectory lineage lacks initialization controls")
    return {key: lineage[key] for key in keys}


def _lineage_schedule(payload: dict) -> tuple[float, ...]:
    lineage = payload["trajectory_lineage"]
    schedule = tuple(float(value) for value in lineage["segment_end_times"])
    if not all(np.isfinite(value) and value > 0.0 for value in schedule) or any(
        right <= left for left, right in zip(schedule, schedule[1:], strict=False)
    ):
        raise ValueError("nonlinear segment trajectory endpoints must strictly increase")
    return schedule


def merge_nonlinear_heat_flux_segments(
    paths,
    *,
    start_fraction: float = 0.5,
    max_relative_drift: float = 0.2,
    max_relative_standard_error: float = 0.1,
    min_phi_rms_ratio: float = 0.8,
    max_absolute_phi_growth_rate: float = 0.02,
    min_samples: int = 100,
    min_window_duration: float = 10.0,
    block_duration: float = 5.0,
    min_blocks: int = 6,
) -> dict:
    """Return a schema-v1 report for contiguous, contract-identical segments."""

    paths = tuple(Path(path).expanduser().resolve() for path in paths)
    if not paths:
        raise ValueError("at least one nonlinear segment is required")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1)")
    stationarity_limits = (
        max_relative_drift,
        max_relative_standard_error,
        min_phi_rms_ratio,
        max_absolute_phi_growth_rate,
        min_window_duration,
        block_duration,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in stationarity_limits):
        raise ValueError("stationarity controls must be finite and positive")
    if min_samples < 2 or min_window_duration <= 0.0 or block_duration <= 0.0:
        raise ValueError("stationarity requires at least two samples and positive duration")
    if min_blocks < 2:
        raise ValueError("stationarity requires at least two physical-time blocks")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(payload.get("schema_version") != 1 for payload in payloads):
        raise ValueError("nonlinear segments must use schema version 1")
    if any(payload.get("producer") not in _INPUT_PRODUCERS for payload in payloads):
        raise ValueError("nonlinear segments must come from a local producer")
    normalization = payloads[0].get("normalization")
    if not isinstance(normalization, str) or not normalization.strip():
        raise ValueError("nonlinear segments require a normalization")
    if any(
        payload.get("normalization") != payloads[0].get("normalization") for payload in payloads
    ):
        raise ValueError("nonlinear segment normalizations do not match")
    contract = _contract(payloads[0])
    if any(_contract(payload) != contract for payload in payloads[1:]):
        raise ValueError("nonlinear segment contracts do not match")
    lineage_root = _lineage_root(payloads[0])
    if any(_lineage_root(payload) != lineage_root for payload in payloads[1:]):
        raise ValueError("nonlinear segments do not share one trajectory initialization")
    lineage_schedules = tuple(_lineage_schedule(payload) for payload in payloads)
    if any(
        len(current) != len(previous) + 1 or current[:-1] != previous
        for previous, current in zip(lineage_schedules, lineage_schedules[1:], strict=False)
    ):
        raise ValueError("nonlinear segment lineage schedules do not extend in supplied order")

    merged = [[], [], []]
    previous_end = None
    for payload in payloads:
        trace = _trace(payload)
        start = 0
        if previous_end is not None:
            tolerance = 1.0e-10 * max(1.0, abs(previous_end))
            if abs(float(trace[0][0]) - previous_end) > tolerance:
                raise ValueError("nonlinear segments must be contiguous in the supplied order")
            if not np.isclose(merged[1][-1][-1], trace[1][0], rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("nonlinear segment heat flux is discontinuous at restart")
            if not np.isclose(merged[2][-1][-1], trace[2][0], rtol=1.0e-10, atol=1.0e-12):
                raise ValueError("nonlinear segment nonzonal RMS is discontinuous at restart")
            start = 1
        for target, values in zip(merged, trace, strict=True):
            target.append(values[start:])
        previous_end = float(trace[0][-1])
    times, flux, amplitude = (np.concatenate(items) for items in merged)

    start = min(int(times.size * start_fraction), times.size - 2)
    window_times = times[start:]
    window_flux = flux[start:]
    window_amplitude = amplitude[start:]
    statistics = correlated_flux_statistics(
        times,
        flux,
        start_fraction=start_fraction,
        block_duration=block_duration,
    )
    mean = float(statistics.mean)
    standard_deviation = float(statistics.standard_deviation)
    standard_error = float(statistics.standard_error)
    relative_standard_error = standard_error / max(abs(mean), 1.0e-14)
    centered_time = window_times - np.mean(window_times)
    duration = float(window_times[-1] - window_times[0])
    relative_drift = float(statistics.relative_window_drift)
    log_amplitude = np.log(np.maximum(window_amplitude, 1.0e-14))
    growth = float(
        centered_time @ (log_amplitude - np.mean(log_amplitude)) / (centered_time @ centered_time)
    )
    amplitude_ratio = float(amplitude[-1] / max(amplitude[0], 1.0e-14))
    stationary = bool(
        window_flux.size >= min_samples
        and statistics.n_blocks >= min_blocks
        and duration >= min_window_duration
        and abs(relative_drift) <= max_relative_drift
        and relative_standard_error <= max_relative_standard_error
        and float(window_amplitude[-1] / window_amplitude[0]) >= min_phi_rms_ratio
        and abs(growth) <= max_absolute_phi_growth_rate
    )
    return {
        "schema_version": 1,
        "producer": "jax-fluxtube-gk/nonlinear-heat-flux-merged",
        "normalization": payloads[0]["normalization"],
        "case": contract
        | {
            "segments": [str(path) for path in paths],
            "start_fraction": start_fraction,
            "max_relative_drift": max_relative_drift,
            "max_relative_standard_error": max_relative_standard_error,
            "min_phi_rms_ratio": min_phi_rms_ratio,
            "max_absolute_phi_growth_rate": max_absolute_phi_growth_rate,
            "min_stationary_samples": min_samples,
            "min_stationary_window_duration": min_window_duration,
            "stationary_block_duration": block_duration,
            "min_stationary_blocks": min_blocks,
        },
        "start_time": float(times[0]),
        "end_time": float(times[-1]),
        "stationary": stationary,
        "trajectory_lineage": {
            "schema_version": 1,
            **lineage_root,
            "segment_end_times": list(lineage_schedules[-1]),
            "source_segment_end_times": [
                payload["trajectory_lineage"]["segment_end_times"] for payload in payloads
            ],
        },
        "stationary_window_duration": duration,
        "relative_standard_error": relative_standard_error,
        "nonzonal_phi_rms_initial": float(amplitude[0]),
        "nonzonal_phi_rms_final": float(amplitude[-1]),
        "nonzonal_phi_rms_ratio": amplitude_ratio,
        "candidate_nonzonal_phi_rms_initial": float(window_amplitude[0]),
        "candidate_nonzonal_phi_rms_final": float(window_amplitude[-1]),
        "candidate_nonzonal_phi_rms_ratio": float(window_amplitude[-1] / window_amplitude[0]),
        "candidate_nonzonal_phi_growth_rate": growth,
        "statistics": {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "standard_error": standard_error,
            "relative_window_drift": relative_drift,
            "n_samples": int(window_flux.size),
            "n_blocks": statistics.n_blocks,
            "block_duration": statistics.block_duration,
        },
        "times": times.tolist(),
        "heat_flux": flux.tolist(),
        "nonzonal_phi_rms": amplitude.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    report = merge_nonlinear_heat_flux_segments(
        args.segments,
        start_fraction=args.start_fraction,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}; stationary={report['stationary']}")


if __name__ == "__main__":
    main()
