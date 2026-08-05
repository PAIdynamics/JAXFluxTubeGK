"""Compare the W7-X stella ``ky=0.3`` RHS trace with solver balance artifacts.

The patched stella trace contains complex selected-mode arrays, but the current
committed solver-side fixture contains scalar term-balance summaries rather
than full solver arrays.  This comparator therefore does the strongest honest
check currently possible from committed artifacts:

* ingest the raw stella trace when it is available, otherwise use the committed
  compact summary;
* convert stella RHS deltas from native ``rhs*dt`` to continuous-time RHS
  norms;
* compare scale-free term/total norm ratios against the solver-side balance;
* report the remaining blockers for direct array parity.
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = compare_w7x_stella_rhs_trace_to_solver_balance(
        trace_path=args.trace,
        stella_summary=args.stella_summary,
        solver_balance_dir=args.solver_balance_dir,
        output_dir=args.output_dir,
        require_raw_trace=args.require_raw_trace,
    )
    print(result["status_json"])
    print(result["term_comparison_csv"])
    return 0


def compare_w7x_stella_rhs_trace_to_solver_balance(
    *,
    trace_path: Path = DEFAULT_TRACE,
    stella_summary: Path = DEFAULT_STELLA_SUMMARY,
    solver_balance_dir: Path = DEFAULT_SOLVER_BALANCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    require_raw_trace: bool = False,
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

    term_csv = output_dir / "term_norm_comparison.csv"
    status_json = output_dir / "stella_solver_rhs_trace_comparison_status.json"
    contract_json = output_dir / "array_contract.json"
    _write_csv(term_csv, comparison_rows)
    _write_json(status_json, status)
    _write_json(contract_json, contract)
    _write_readme(output_dir)
    return {
        "status_json": str(status_json),
        "term_comparison_csv": str(term_csv),
        "array_contract_json": str(contract_json),
    }


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
                "grid from the stella trace.",
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
