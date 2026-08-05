"""Compare the W7-X stella ``ky=0.3`` RHS trace with solver balance artifacts.

The patched stella trace contains complex selected-mode arrays, while the
committed solver-side fixture intentionally contains only compact scalar
summaries. This comparator supports both that standalone fallback and an
external solver array archive:

* ingest the raw stella trace when it is available, otherwise use the committed
  compact summary;
* convert stella RHS deltas from native ``rhs*dt`` to continuous-time RHS
  norms;
* compare scale-free term/total norm ratios against the solver-side balance;
* optionally interpolate stella arrays onto the solver overlap grid and report
  weighted complex errors for every inferred RHS call;
* report the remaining blockers for full array parity.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = Path(
    "/tmp/stellarator_gk_stella_w7x_rhs_trace/run/stellarator_gk_w7x_ky03_rhs_trace.dat"
)
DEFAULT_STELLA_SUMMARY = (
    ROOT / "fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json"
)
DEFAULT_SOLVER_BALANCE = ROOT / "fixtures/w7x_ky03_rhs_model_balance"
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_ky03_stella_rhs_trace_comparison"
DEFAULT_ARRAY_COMPARISON = Path("/tmp/stellarator_gk_w7x_ky03_array_comparison.csv")

TERM_GROUPS = (
    {
        "group": "parallel_streaming_bundle",
        "stella_terms": (("rhs_delta", "parallel_streaming"),),
        "solver_terms": ("parallel_streaming", "parallel_field_drive"),
        "note": "stella explicit streaming includes the distribution and phi-gradient pieces",
    },
    {
        "group": "mirror_force",
        "stella_terms": (("rhs_delta", "mirror_force"),),
        "solver_terms": ("mirror_force",),
        "note": "direct mirror-force term",
    },
    {
        "group": "magnetic_drift_bundle",
        "stella_terms": (("rhs_delta", "magnetic_drift_y"), ("rhs_delta", "magnetic_drift_x")),
        "solver_terms": ("magnetic_drift", "drift_field_drive"),
        "note": "stella drift call includes magnetic-drift field pieces; kx=0 makes x drift zero",
    },
    {
        "group": "equilibrium_drive",
        "stella_terms": (("rhs_delta", "equilibrium_drive_wstar"),),
        "solver_terms": ("equilibrium_drive",),
        "note": "diamagnetic/equilibrium-gradient drive",
    },
)


def drop_stella_periodic_endpoint(z_indices, *arrays, axis: int = 0):
    """Drop stella's duplicated upper z endpoint from trace arrays."""

    z_indices = np.asarray(z_indices, dtype=int)
    if z_indices.ndim != 1 or z_indices.size < 2:
        raise ValueError("stella z indices must be a one-dimensional periodic grid")
    if not np.array_equal(np.diff(z_indices), np.ones(z_indices.size - 1, dtype=int)):
        raise ValueError("stella z indices must be contiguous and ordered")
    trimmed = []
    for values in arrays:
        array = np.asarray(values)
        if array.shape[axis] != z_indices.size:
            raise ValueError("trace array z axis does not match stella z indices")
        trimmed.append(np.take(array, np.arange(z_indices.size - 1), axis=axis))
    return (z_indices[:-1], *trimmed)


def interpolate_phase_space_to_grid(
    values,
    *,
    source_z,
    source_vpar,
    source_mu,
    target_z,
    target_vpar,
    target_mu,
):
    """Linearly map a complex ``(z,vpar,mu)`` array without extrapolation.

    The target velocity grid must lie inside the source grid.  This makes the
    scientific loss explicit: callers comparing unlike stella/solver grids
    should map the finer/ wider trace onto their chosen common domain and use
    that target grid's quadrature weights.
    """

    array = np.asarray(values)
    source_axes = tuple(
        np.asarray(axis, dtype=float) for axis in (source_z, source_vpar, source_mu)
    )
    target_axes = tuple(
        np.asarray(axis, dtype=float) for axis in (target_z, target_vpar, target_mu)
    )
    expected = tuple(axis.size for axis in source_axes)
    if array.shape != expected:
        raise ValueError(f"phase-space array has shape {array.shape}; expected {expected}")
    for name, source, target in zip(
        ("z", "vpar", "mu"), source_axes, target_axes, strict=True
    ):
        if source.ndim != 1 or target.ndim != 1 or np.any(np.diff(source) <= 0.0):
            raise ValueError(f"{name} coordinates must be one-dimensional and increasing")
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(source)))
        if target.size and (
            np.min(target) < source[0] - tolerance or np.max(target) > source[-1] + tolerance
        ):
            raise ValueError(f"target {name} coordinates require extrapolation")

    result = array
    for axis, (source, target) in enumerate(zip(source_axes, target_axes, strict=True)):
        result = _interpolate_complex_axis(result, source, target, axis=axis)
    return result


def weighted_complex_metrics(
    reference, candidate, *, w_z, w_vpar, w_mu, alignment_scale=None
):
    """Return target-grid weighted complex error with optimal phase/scale fit."""

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate arrays must have matching shapes")
    expected = (len(w_z), len(w_vpar), len(w_mu))
    if reference.shape != expected:
        raise ValueError(f"weighted array shape is {reference.shape}; expected {expected}")
    weights = (
        np.asarray(w_z, dtype=float)[:, None, None]
        * np.asarray(w_vpar, dtype=float)[None, :, None]
        * np.asarray(w_mu, dtype=float)[None, None, :]
    )
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("quadrature weights must be finite and nonnegative")
    reference_norm = _weighted_l2(reference, weights)
    candidate_norm = _weighted_l2(candidate, weights)
    denominator = np.sum(weights * np.abs(candidate) ** 2)
    fitted_scale = (
        0.0j
        if denominator == 0.0
        else np.sum(weights * np.conj(candidate) * reference) / denominator
    )
    scale = fitted_scale if alignment_scale is None else complex(alignment_scale)
    aligned_error = _weighted_l2(reference - scale * candidate, weights)
    raw_error = _weighted_l2(reference - candidate, weights)
    return {
        "reference_weighted_l2": reference_norm,
        "candidate_weighted_l2": candidate_norm,
        "raw_relative_l2_error": _safe_ratio(raw_error, reference_norm),
        "aligned_relative_l2_error": _safe_ratio(aligned_error, reference_norm),
        "alignment_scale_real": float(np.real(scale)),
        "alignment_scale_imag": float(np.imag(scale)),
        "alignment_scale_source": "fitted" if alignment_scale is None else "provided",
    }


def _interpolate_complex_axis(values, source, target, *, axis: int):
    moved = np.moveaxis(np.asarray(values), axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    interpolated = np.empty((len(target), flat.shape[1]), dtype=np.result_type(values, complex))
    for column in range(flat.shape[1]):
        interpolated[:, column] = np.interp(target, source, flat[:, column].real) + 1j * np.interp(
            target, source, flat[:, column].imag
        )
    shaped = interpolated.reshape((len(target), *moved.shape[1:]))
    return np.moveaxis(shaped, 0, axis)


def _weighted_l2(values, weights) -> float:
    return float(np.sqrt(np.sum(weights * np.abs(values) ** 2)))


def load_stella_array_trace(trace_path: Path, summary: dict[str, Any]):
    """Load all inferred RHS-call occurrences from the external stella v2 trace."""

    state_summary = _stella_terms(summary)[("pdf_g", "input_pdf")]
    iz_min, iz_max = (int(value) for value in state_summary["iz_range"])
    n_z_raw = iz_max - iz_min + 1
    n_vpar = int(state_summary["iv_range"][1] - state_summary["iv_range"][0] + 1)
    n_mu = int(state_summary["imu_range"][1] - state_summary["imu_range"][0] + 1)
    phase_shape = (n_z_raw, n_vpar, n_mu)
    phase_records: dict[tuple[str, str], list[np.ndarray]] = {}
    field_records: dict[tuple[str, str], list[np.ndarray]] = {}
    scalar_records: dict[tuple[str, str], list[np.ndarray]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    vpar = np.full(n_vpar, np.nan)
    mu = np.full(n_mu, np.nan)
    w_vpar = np.full(n_vpar, np.nan)
    w_mu = np.full((n_z_raw, n_mu), np.nan)
    last_key = None
    stage = -1
    header = None

    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            if header is None:
                header = {name: index for index, name in enumerate(fields)}
                continue
            key = (fields[header["record"]], fields[header["term"]])
            if key != last_key:
                inferred_stage = occurrences.get(key, 0)
                if "rhs_call" in header:
                    stage = int(fields[header["rhs_call"]]) - 1
                    if stage != inferred_stage:
                        raise ValueError(
                            f"stella rhs_call is not contiguous for {key}: {stage + 1}"
                        )
                else:
                    stage = inferred_stage
                occurrences[key] = stage + 1
                if key[0] == "normalization":
                    target = scalar_records
                    shape = (1,)
                elif key[0] in ("phi", "quasineutrality"):
                    target = field_records
                    shape = (n_z_raw,)
                else:
                    target = phase_records
                    shape = phase_shape
                target.setdefault(key, []).append(np.zeros(shape, dtype=np.complex128))
                last_key = key
            iz = int(fields[header["iz"]]) - iz_min
            value = float(fields[header["real"]]) + 1j * float(fields[header["imag"]])
            if key[0] == "normalization":
                scalar_records[key][stage][0] = value
                continue
            if key[0] in ("phi", "quasineutrality"):
                field_records[key][stage][iz] = value
                continue
            iv = int(fields[header["iv"]]) - 1
            imu = int(fields[header["imu"]]) - 1
            phase_records[key][stage][iz, iv, imu] = value
            vpar[iv] = float(fields[header["vpa"]])
            mu[imu] = float(fields[header["mu"]])
            w_vpar[iv] = float(fields[header["wgts_vpa"]])
            w_mu[iz, imu] = float(fields[header["wgts_mu"]])

    required = (
        ("pdf_g", "input_pdf"),
        ("phi", "field_phi"),
        ("rhs_delta", "mirror_force"),
        ("rhs_delta", "magnetic_drift_y"),
        ("rhs_delta", "magnetic_drift_x"),
        ("rhs_delta", "equilibrium_drive_wstar"),
        ("rhs_delta", "parallel_streaming"),
        ("rhs_total", "total"),
    )
    if summary.get("trace_format") == "stellarator_gk_stella_rhs_trace_v3":
        required += (
            ("quasineutrality", "numerator"),
            ("quasineutrality", "denominator"),
            ("normalization", "native_state_scale"),
        )
    missing = [
        key
        for key in required
        if key not in phase_records and key not in field_records and key not in scalar_records
    ]
    if missing:
        raise ValueError(f"raw stella trace is missing array records: {missing}")
    call_counts = {occurrences[key] for key in required}
    if len(call_counts) != 1:
        raise ValueError(f"stella record call counts differ: {occurrences}")
    if not all(np.all(np.isfinite(values)) for values in (vpar, mu, w_vpar, w_mu)):
        raise ValueError("stella coordinates or quadrature weights are incomplete")

    code_dt = _single_float(summary["code_dts"])
    z_indices = np.arange(iz_min, iz_max + 1)
    z = z_indices[:-1] / float(n_z_raw - 1)

    def phase(key, *, rhs=False):
        values = np.stack(phase_records[key], axis=0)[:, :-1]
        return values / code_dt if rhs else values

    return {
        "z": z,
        "vpar": vpar,
        "mu": mu,
        "w_vpar": w_vpar,
        "w_mu": w_mu[:-1],
        "distribution": phase(("pdf_g", "input_pdf")),
        "phi": np.stack(field_records[("phi", "field_phi")], axis=0)[:, :-1],
        "mirror_force": phase(("rhs_delta", "mirror_force"), rhs=True),
        "magnetic_drift": phase(("rhs_delta", "magnetic_drift_y"), rhs=True)
        + phase(("rhs_delta", "magnetic_drift_x"), rhs=True),
        "equilibrium_drive": phase(("rhs_delta", "equilibrium_drive_wstar"), rhs=True),
        "parallel_streaming": phase(("rhs_delta", "parallel_streaming"), rhs=True),
        "total_rhs": phase(("rhs_total", "total"), rhs=True),
        "quasineutrality_numerator": (
            np.stack(field_records[("quasineutrality", "numerator")], axis=0)[:, :-1]
            if ("quasineutrality", "numerator") in field_records
            else None
        ),
        "quasineutrality_denominator": (
            np.stack(field_records[("quasineutrality", "denominator")], axis=0)[:, :-1]
            if ("quasineutrality", "denominator") in field_records
            else None
        ),
        "native_state_scale": (
            np.stack(scalar_records[("normalization", "native_state_scale")], axis=0)[:, 0]
            if ("normalization", "native_state_scale") in scalar_records
            else None
        ),
        "rhs_call_count": call_counts.pop(),
    }


def compare_stella_solver_arrays(stella: dict[str, Any], solver_array: Path):
    """Compare trace arrays on the solver grid restricted to stella's domain."""

    with np.load(solver_array) as archive:
        solver = {name: np.asarray(archive[name]) for name in archive.files}
    solver_z = np.asarray(solver["z"], dtype=float)
    solver_vpar = np.asarray(solver["vpar"], dtype=float)
    solver_mu = np.asarray(solver["mu"], dtype=float)
    vmask = (solver_vpar >= stella["vpar"][0]) & (solver_vpar <= stella["vpar"][-1])
    mumask = (solver_mu >= stella["mu"][0]) & (solver_mu <= stella["mu"][-1])
    if not np.any(vmask) or not np.any(mumask):
        raise ValueError("stella and solver velocity grids have no common target nodes")
    target_vpar = solver_vpar[vmask]
    target_mu = solver_mu[mumask]
    weights = {
        "w_z": solver["w_z"],
        "w_vpar": solver["w_vpar"][vmask],
        "w_mu": solver["w_mu"][mumask],
    }
    if "rhs_parallel_streaming" in solver:
        solver_streaming = solver["rhs_parallel_streaming"] + solver["rhs_parallel_field_drive"]
    else:
        solver_streaming = (
            solver["rhs_gkw_parallel_streaming_recurrence"]
            + solver["rhs_gkw_parallel_field_drive"]
        )
    solver_arrays = {
        "distribution": solver["distribution"][:, vmask][:, :, mumask],
        "parallel_streaming": solver_streaming[:, vmask][:, :, mumask],
        "mirror_force": solver["rhs_mirror_force"][:, vmask][:, :, mumask],
        "magnetic_drift": (
            solver["rhs_magnetic_drift"] + solver["rhs_drift_field_drive"]
        )[:, vmask][:, :, mumask],
        "equilibrium_drive": solver["rhs_equilibrium_drive"][:, vmask][:, :, mumask],
        "total_rhs": solver["total_rhs"][:, vmask][:, :, mumask],
    }
    rows = []
    for rhs_call in range(int(stella["rhs_call_count"])):
        mapped = {
            name: interpolate_phase_space_to_grid(
                stella[name][rhs_call],
                source_z=stella["z"],
                source_vpar=stella["vpar"],
                source_mu=stella["mu"],
                target_z=solver_z,
                target_vpar=target_vpar,
                target_mu=target_mu,
            )
            for name in solver_arrays
        }
        distribution_metrics = weighted_complex_metrics(
            mapped["distribution"], solver_arrays["distribution"], **weights
        )
        alignment_scale = complex(
            distribution_metrics["alignment_scale_real"],
            distribution_metrics["alignment_scale_imag"],
        )
        for name, candidate in solver_arrays.items():
            metrics = (
                distribution_metrics
                if name == "distribution"
                else weighted_complex_metrics(
                    mapped[name], candidate, alignment_scale=alignment_scale, **weights
                )
            )
            rows.append(
                {
                    "rhs_call": rhs_call + 1,
                    "quantity": name,
                    "target_n_z": solver_z.size,
                    "target_n_vpar": target_vpar.size,
                    "target_n_mu": target_mu.size,
                    "solver_vpar_nodes_excluded": int(np.count_nonzero(~vmask)),
                    "solver_mu_nodes_excluded": int(np.count_nonzero(~mumask)),
                    **metrics,
                }
            )
        field_pairs = {"phi": solver["phi"]}
        if stella["quasineutrality_numerator"] is not None:
            field_pairs["quasineutrality_numerator"] = solver["quasineutrality_numerator"]
            field_pairs["quasineutrality_denominator"] = solver[
                "quasineutrality_denominator"
            ]
        for name, candidate in field_pairs.items():
            reference = _interpolate_complex_axis(
                stella[name][rhs_call], stella["z"], solver_z, axis=0
            )
            field_scale = -1.0 if name == "quasineutrality_denominator" else alignment_scale
            metrics = weighted_complex_metrics(
                reference[:, None, None],
                candidate[:, None, None],
                w_z=weights["w_z"],
                w_vpar=[1.0],
                w_mu=[1.0],
                alignment_scale=field_scale,
            )
            rows.append(
                {
                    "rhs_call": rhs_call + 1,
                    "quantity": name,
                    "target_n_z": solver_z.size,
                    "target_n_vpar": 1,
                    "target_n_mu": 1,
                    "solver_vpar_nodes_excluded": 0,
                    "solver_mu_nodes_excluded": 0,
                    **metrics,
                }
            )
        if stella["native_state_scale"] is not None:
            reference_scale = float(np.real(stella["native_state_scale"][rhs_call]))
            candidate_scale = float(np.exp(solver["log_normalization"]))
            aligned_scale = abs(reference_scale - alignment_scale * candidate_scale)
            rows.append(
                {
                    "rhs_call": rhs_call + 1,
                    "quantity": "normalization",
                    "target_n_z": 1,
                    "target_n_vpar": 1,
                    "target_n_mu": 1,
                    "solver_vpar_nodes_excluded": 0,
                    "solver_mu_nodes_excluded": 0,
                    "reference_weighted_l2": abs(reference_scale),
                    "candidate_weighted_l2": abs(candidate_scale),
                    "raw_relative_l2_error": _safe_ratio(
                        abs(reference_scale - candidate_scale), abs(reference_scale)
                    ),
                    "aligned_relative_l2_error": _safe_ratio(
                        aligned_scale, abs(reference_scale)
                    ),
                    "alignment_scale_real": float(np.real(alignment_scale)),
                    "alignment_scale_imag": float(np.imag(alignment_scale)),
                    "alignment_scale_source": "distribution_fitted",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compare_w7x_stella_rhs_trace_to_solver_balance(
        trace_path=args.trace,
        stella_summary=args.stella_summary,
        solver_balance_dir=args.solver_balance_dir,
        output_dir=args.output_dir,
        require_raw_trace=args.require_raw_trace,
        solver_array=args.solver_array,
        array_comparison_output=args.array_comparison_output,
    )
    print(result["status_json"])
    print(result["term_comparison_csv"])
    if result.get("array_comparison_csv"):
        print(result["array_comparison_csv"])
    return 0


def compare_w7x_stella_rhs_trace_to_solver_balance(
    *,
    trace_path: Path = DEFAULT_TRACE,
    stella_summary: Path = DEFAULT_STELLA_SUMMARY,
    solver_balance_dir: Path = DEFAULT_SOLVER_BALANCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    require_raw_trace: bool = False,
    solver_array: Path | None = None,
    array_comparison_output: Path = DEFAULT_ARRAY_COMPARISON,
) -> dict[str, str]:
    """Write comparison artifacts and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = Path(trace_path)
    raw_trace_used = trace_path.is_file()
    if require_raw_trace and not raw_trace_used:
        raise FileNotFoundError(trace_path)
    if raw_trace_used:
        stella = _summarize_raw_trace(trace_path)
    else:
        stella = _read_json(stella_summary)

    solver_rows = _read_solver_term_rows(solver_balance_dir / "rhs_term_balance.csv")
    solver_status = _read_json(solver_balance_dir / "rhs_model_balance_status.json")
    solver_metadata = _read_json(solver_balance_dir / "rhs_model_balance_metadata.json")
    focus_case = _focus_solver_case(solver_status)
    solver_velocity = _solver_velocity_contract(str(focus_case["case"]))

    comparison_rows = _term_comparison_rows(stella, solver_rows)
    contract = _array_contract_payload(
        stella,
        solver_rows=solver_rows,
        solver_status=solver_status,
        solver_metadata=solver_metadata,
        solver_velocity=solver_velocity,
        raw_trace_used=raw_trace_used,
        trace_path=trace_path,
    )
    status = _status_payload(
        stella,
        solver_status=solver_status,
        comparison_rows=comparison_rows,
        contract=contract,
        raw_trace_used=raw_trace_used,
        trace_path=trace_path,
    )

    array_rows = None
    if solver_array is not None:
        if not raw_trace_used:
            raise ValueError("--solver-array requires the raw stella --trace")
        stella_arrays = load_stella_array_trace(trace_path, stella)
        array_rows = compare_stella_solver_arrays(stella_arrays, Path(solver_array))
        _write_csv(array_comparison_output, array_rows)
        contract["array_comparisons_available"] = sorted(
            {str(row["quantity"]) for row in array_rows}
        )
        contract["inferred_stella_rhs_calls"] = len(stella_arrays["distribution"])
        contract["missing_array_records"] = [
            name
            for name, available in (
                ("stella_quasineutrality_numerator", stella_arrays["quasineutrality_numerator"]),
                (
                    "stella_quasineutrality_denominator",
                    stella_arrays["quasineutrality_denominator"],
                ),
                ("stella_native_state_scale", stella_arrays["native_state_scale"]),
            )
            if available is None
        ]
        has_explicit_calls = stella.get("trace_format") == "stellarator_gk_stella_rhs_trace_v3"
        contract["rhs_calls_explicitly_labeled"] = has_explicit_calls
        contract["direct_array_parity_ready"] = has_explicit_calls and not contract[
            "missing_array_records"
        ]
        contract["array_parity_blockers"] = []
        if not has_explicit_calls:
            contract["array_parity_blockers"].append(
                "stella trace does not label the inferred RHS calls/stages"
            )
        if contract["missing_array_records"]:
            contract["array_parity_blockers"].append(
                "stella trace lacks required quasineutrality or normalization arrays"
            )
        status["array_parity_blockers"] = contract["array_parity_blockers"]
        status["direct_array_parity_ready"] = contract["direct_array_parity_ready"]
        max_array_error = max(
            float(row["aligned_relative_l2_error"])
            for row in array_rows
            if row["quantity"] != "normalization"
        )
        array_tolerance = 0.1
        status["max_aligned_array_relative_l2_error"] = max_array_error
        status["array_relative_l2_tolerance"] = array_tolerance
        status["status"] = (
            "weighted_array_parity_passed"
            if contract["direct_array_parity_ready"] and max_array_error <= array_tolerance
            else "weighted_array_parity_failed"
        )
        status["comparison_kind"] = "weighted_complex_arrays_on_solver_overlap_grid"
        status["array_comparison_csv"] = str(array_comparison_output)
        status["stella_rhs_call_selection"] = (
            "all explicitly labeled calls reported separately"
            if has_explicit_calls
            else "all inferred calls reported separately"
        )
        status["interpretation"] = (
            "The external solver archive enables weighted complex comparisons for "
            "the distribution, phi, RHS bundles, quasineutrality, and normalization. "
            "One distribution-fitted complex scale is reused for all state-dependent "
            "terms; the quasineutrality denominator uses the documented opposite-sign "
            "solver convention. Contract completeness and numerical parity are "
            "reported separately."
        )
        status["passed"] = status["status"] == "weighted_array_parity_passed"
        status["next_action"] = (
            "resolve the distribution/mode-structure convention mismatch, then rerun "
            "this complete weighted array gate"
        )

    term_csv = output_dir / "term_norm_comparison.csv"
    status_json = output_dir / "stella_solver_rhs_trace_comparison_status.json"
    contract_json = output_dir / "array_contract.json"
    _write_csv(term_csv, comparison_rows)
    _write_json(status_json, status)
    _write_json(contract_json, contract)
    _write_readme(output_dir)
    result = {
        "status_json": str(status_json),
        "term_comparison_csv": str(term_csv),
        "array_contract_json": str(contract_json),
    }
    if array_rows is not None:
        result["array_comparison_csv"] = str(array_comparison_output)
    return result


def _term_comparison_rows(
    stella_summary: dict[str, Any],
    solver_rows: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    stella_terms = _stella_terms(stella_summary)
    stella_dt = _single_float(stella_summary.get("code_dts", ()))
    stella_total = stella_terms[("rhs_total", "total")]
    stella_total_l2 = _rhs_l2(stella_total, stella_dt)
    solver_total_l2 = _first_solver_total_l2(solver_rows)
    rows: list[dict[str, object]] = []

    for group in TERM_GROUPS:
        stella_l2_sum = sum(_rhs_l2(stella_terms[key], stella_dt) for key in group["stella_terms"])
        stella_max_abs = max(_rhs_max_abs(stella_terms[key], stella_dt) for key in group["stella_terms"])
        solver_l2_sum = sum(solver_rows[name]["rhs_l2"] for name in group["solver_terms"])
        solver_fraction_sum = sum(
            solver_rows[name]["rhs_fraction_of_total_l2"] for name in group["solver_terms"]
        )
        rows.append(
            {
                "comparison_group": group["group"],
                "stella_terms": "+".join(f"{record}:{term}" for record, term in group["stella_terms"]),
                "solver_terms": "+".join(group["solver_terms"]),
                "stella_rhs_l2_continuous": stella_l2_sum,
                "stella_rhs_fraction_of_total_l2": _safe_ratio(stella_l2_sum, stella_total_l2),
                "stella_rhs_max_abs_continuous": stella_max_abs,
                "solver_rhs_l2_sum": solver_l2_sum,
                "solver_rhs_fraction_sum_of_total_l2": solver_fraction_sum,
                "solver_total_rhs_l2": solver_total_l2,
                "absolute_l2_scale_ratio_stella_over_solver": _safe_ratio(stella_l2_sum, solver_l2_sum),
                "fraction_difference_stella_minus_solver": (
                    _safe_ratio(stella_l2_sum, stella_total_l2) - solver_fraction_sum
                ),
                "comparison_kind": "scale_free_l2_norm_ratio",
                "note": group["note"],
            }
        )

    rows.append(
        {
            "comparison_group": "total_rhs",
            "stella_terms": "rhs_total:total",
            "solver_terms": "total_rhs",
            "stella_rhs_l2_continuous": stella_total_l2,
            "stella_rhs_fraction_of_total_l2": 1.0,
            "stella_rhs_max_abs_continuous": _rhs_max_abs(stella_total, stella_dt),
            "solver_rhs_l2_sum": solver_total_l2,
            "solver_rhs_fraction_sum_of_total_l2": 1.0,
            "solver_total_rhs_l2": solver_total_l2,
            "absolute_l2_scale_ratio_stella_over_solver": _safe_ratio(stella_total_l2, solver_total_l2),
            "fraction_difference_stella_minus_solver": 0.0,
            "comparison_kind": "absolute_scale_only_not_a_parity_gate",
            "note": "absolute scale differs because stella and solver states are independently normalized",
        }
    )
    return rows


def _array_contract_payload(
    stella_summary: dict[str, Any],
    *,
    solver_rows: dict[str, dict[str, float]],
    solver_status: dict[str, Any],
    solver_metadata: dict[str, Any],
    solver_velocity: dict[str, object],
    raw_trace_used: bool,
    trace_path: Path,
) -> dict[str, object]:
    stella_terms = {
        (str(entry["record"]), str(entry["term"])): entry
        for entry in stella_summary.get("term_summaries", ())
    }
    state = stella_terms[("pdf_g", "input_pdf")]
    total = stella_terms[("rhs_total", "total")]
    stella_n_z_raw = int(state["iz_range"][1] - state["iz_range"][0] + 1)
    stella_n_vpar = int(state["iv_range"][1] - state["iv_range"][0] + 1)
    stella_n_mu = int(state["imu_range"][1] - state["imu_range"][0] + 1)
    solver_case = _focus_solver_case(solver_status)
    solver_n_z = _solver_n_z(solver_metadata)
    blockers = []
    endpoint_drop_applied = False
    if stella_n_z_raw != solver_n_z:
        if stella_n_z_raw == solver_n_z + 1:
            endpoint_drop_applied = True
        else:
            blockers.append("stella and solver z dimensions do not match")
    if stella_n_vpar != int(solver_case["n_vpar"]):
        blockers.append("stella and committed solver balance use different n_vpar")
    if stella_n_mu != int(solver_case["n_mu"]):
        blockers.append("stella and committed solver balance use different n_mu")
    if tuple(state["vpa_range"]) != tuple(solver_velocity["vpar_range"]):
        blockers.append("stella and committed solver balance use different vpar ranges")
    if tuple(state["mu_range"]) != tuple(solver_velocity["mu_range"]):
        blockers.append("stella and committed solver balance use different mu grids/ranges")
    blockers.append("committed solver balance fixture stores scalar term summaries, not full term arrays")
    velocity_weights_present = bool(
        stella_summary.get(
            "velocity_weight_columns_present",
            state.get("velocity_weight_columns_present", False),
        )
    )
    if not velocity_weights_present:
        blockers.append("stella trace currently stores vpa/mu coordinates but not velocity quadrature weights")
    blockers.append("trace forces stella mirror/streaming explicit, unlike the production stella growth run")
    return {
        "raw_trace_used": raw_trace_used,
        "raw_trace_path": str(trace_path),
        "stella_trace_format": stella_summary.get("trace_format"),
        "stella_rhs_units": stella_summary.get("rhs_units"),
        "stella_total_rows": stella_summary.get("total_rows"),
        "stella_step": _single_int(stella_summary.get("steps", ())),
        "stella_code_dt": _single_float(stella_summary.get("code_dts", ())),
        "stella_n_z_raw": stella_n_z_raw,
        "stella_n_z_after_endpoint_drop": (
            stella_n_z_raw - 1 if endpoint_drop_applied else stella_n_z_raw
        ),
        "stella_endpoint_policy": "exclude_upper_periodic_endpoint",
        "stella_endpoint_drop_applied": endpoint_drop_applied,
        "stella_n_vpar": stella_n_vpar,
        "stella_n_mu": stella_n_mu,
        "stella_vpa_range": state["vpa_range"],
        "stella_mu_range": state["mu_range"],
        "stella_velocity_weight_columns_present": velocity_weights_present,
        "stella_wgts_vpa_range": state.get("wgts_vpa_range"),
        "stella_wgts_mu_range": state.get("wgts_mu_range"),
        "stella_total_rhs_rows": total["rows"],
        "solver_case": solver_case,
        "solver_n_z": solver_n_z,
        "solver_velocity_contract": solver_velocity,
        "solver_terms_available": sorted(solver_rows),
        "interpolation_adapter": {
            "available": True,
            "method": "separable_linear_complex_interpolation",
            "extrapolation": "forbidden",
            "weighting": "use_quadrature_of_the_chosen_target_grid",
            "axis_order": ["z", "vpar", "mu"],
        },
        "direct_array_parity_ready": False,
        "array_parity_blockers": blockers,
    }


def _status_payload(
    stella_summary: dict[str, Any],
    *,
    solver_status: dict[str, Any],
    comparison_rows: list[dict[str, object]],
    contract: dict[str, object],
    raw_trace_used: bool,
    trace_path: Path,
) -> dict[str, object]:
    max_fraction_delta = max(
        abs(float(row["fraction_difference_stella_minus_solver"]))
        for row in comparison_rows
        if row["comparison_group"] != "total_rhs"
    )
    return {
        "benchmark_name": "w7x_ky03_stella_rhs_trace_comparison",
        "status": "blocked_array_contract_mismatch",
        "passed": False,
        "raw_trace_used": raw_trace_used,
        "raw_trace_path": str(trace_path),
        "stella_required_record_terms_present": bool(
            stella_summary.get("required_record_terms_present", False)
        ),
        "solver_balance_status": solver_status.get("status"),
        "comparison_kind": "stella_trace_vs_solver_scalar_term_balance",
        "rhs_units_after_conversion": "continuous_time_rhs_l2_for_stella_rhs_records",
        "max_scale_free_fraction_delta": max_fraction_delta,
        "direct_array_parity_ready": contract["direct_array_parity_ready"],
        "array_parity_blockers": contract["array_parity_blockers"],
        "interpretation": (
            "The raw stella term trace is now ingested and its rhs*dt records are "
            "converted to continuous-time RHS norms.  The current committed solver "
            "fixture only supports scale-free term-norm comparison; direct array "
            "parity requires a solver-side full-array trace on the stella velocity "
            "grid or a documented interpolation/weighting contract."
        ),
        "next_action": (
            "emit a solver-side selected-mode full-array trace on a stella-compatible "
            "z/vpa/mu grid, or add a stella velocity-grid adapter plus velocity weights "
            "to make direct term-array comparison meaningful"
        ),
    }


def _summarize_raw_trace(trace_path: Path) -> dict[str, Any]:
    module_path = ROOT / "scripts/summarize_stella_w7x_rhs_trace.py"
    spec = importlib.util.spec_from_file_location("summarize_stella_w7x_rhs_trace", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.summarize_trace(trace_path)


def _stella_terms(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    terms = {
        (str(entry["record"]), str(entry["term"])): entry
        for entry in summary.get("term_summaries", ())
    }
    missing = [key for group in TERM_GROUPS for key in group["stella_terms"] if key not in terms]
    if ("rhs_total", "total") not in terms:
        missing.append(("rhs_total", "total"))
    if ("pdf_g", "input_pdf") not in terms:
        missing.append(("pdf_g", "input_pdf"))
    if missing:
        raise ValueError(f"missing stella term summaries: {missing}")
    return terms


def _read_solver_term_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        out[str(row["term"])] = {
            "rhs_l2": float(row["rhs_l2"]),
            "rhs_fraction_of_total_l2": float(row["rhs_fraction_of_total_l2"]),
            "total_rhs_l2": float(row["total_rhs_l2"]),
        }
    aliases = {
        "gkw_parallel_streaming_recurrence": "parallel_streaming",
        "gkw_parallel_field_drive": "parallel_field_drive",
    }
    for source, target in aliases.items():
        if source in out and target not in out:
            out[target] = out[source]
    return out


def _focus_solver_case(status: dict[str, Any]) -> dict[str, object]:
    summaries = status.get("case_summaries", ())
    if not summaries:
        raise ValueError("solver status has no case_summaries")
    return dict(summaries[0])


def _solver_velocity_contract(case_name: str) -> dict[str, object]:
    audit = _load_rhs_balance_module()
    cases = {case.name: case for case in audit.default_balance_cases()}
    case = cases[case_name]
    if case.velocity_backend == "finite_difference":
        dv = 2.0 * case.vpar_max / case.n_vpar
        vpar = [-case.vpar_max + 0.5 * dv, case.vpar_max - 0.5 * dv]
        vperp_max = (2.0 * case.mu_max) ** 0.5
        dvperp = vperp_max / case.n_mu
        mu = [0.5 * (0.5 * dvperp) ** 2, 0.5 * ((case.n_mu - 0.5) * dvperp) ** 2]
    else:
        vpar = [-case.vpar_max, case.vpar_max]
        mu = [0.0, case.mu_max]
    return {
        "backend": case.velocity_backend,
        "n_vpar": case.n_vpar,
        "n_mu": case.n_mu,
        "vpar_max_parameter": case.vpar_max,
        "mu_max_parameter": case.mu_max,
        "vpar_range": vpar,
        "mu_range": mu,
    }


def _load_rhs_balance_module():
    module_path = ROOT / "scripts/audit_w7x_ky03_rhs_model_balance.py"
    spec = importlib.util.spec_from_file_location("audit_w7x_ky03_rhs_model_balance", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _solver_n_z(metadata: dict[str, Any]) -> int:
    geometry_csv = ROOT / str(metadata["geometry_balance_csv"])
    with geometry_csv.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _first_solver_total_l2(rows: dict[str, dict[str, float]]) -> float:
    for row in rows.values():
        return row["total_rhs_l2"]
    raise ValueError("no solver rows")


def _rhs_l2(term: dict[str, Any], code_dt: float) -> float:
    value = float(term["l2_norm"])
    if str(term["record"]).startswith("rhs"):
        return value / code_dt
    return value


def _rhs_max_abs(term: dict[str, Any], code_dt: float) -> float:
    value = float(term["max_abs"])
    if str(term["record"]).startswith("rhs"):
        return value / code_dt
    return value


def _single_float(values) -> float:
    values = tuple(values)
    if len(values) != 1:
        raise ValueError(f"expected one value, got {values}")
    return float(values[0])


def _single_int(values) -> int:
    values = tuple(values)
    if len(values) != 1:
        raise ValueError(f"expected one value, got {values}")
    return int(values[0])


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else float(numerator / denominator)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# W7-X ky=0.3 stella RHS Trace Comparison",
                "",
                "This fixture compares the patched stella selected-mode RHS trace",
                "with the current solver-side RHS/model balance fixture.",
                "",
                "The comparison converts stella `rhs*dt` records to continuous-time",
                "RHS norms and compares scale-free term/total norm ratios. It does",
                "not claim direct array parity yet, because the committed solver",
                "fixture stores scalar term summaries and uses a different velocity",
                "grid from the stella trace. The array adapter uses separable linear",
                "complex interpolation, forbids extrapolation, and evaluates weighted",
                "errors with the chosen target grid's `w_z*w_vpar*w_mu` quadrature.",
                "",
                "When `--solver-array` is supplied, `weighted_array_comparison.csv`",
                "retains compact metrics for every labeled stella RHS call. Raw",
                "stella and solver arrays remain external. The v3 contract includes",
                "quasineutrality and normalization; numerical parity currently fails",
                "the declared relative-L2 tolerance.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--stella-summary", type=Path, default=DEFAULT_STELLA_SUMMARY)
    parser.add_argument("--solver-balance-dir", type=Path, default=DEFAULT_SOLVER_BALANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-raw-trace", action="store_true")
    parser.add_argument(
        "--solver-array",
        type=Path,
        help="external solver selected-mode .npz archive for weighted array comparison",
    )
    parser.add_argument(
        "--array-comparison-output",
        type=Path,
        default=DEFAULT_ARRAY_COMPARISON,
        help="external CSV path for weighted array metrics",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
