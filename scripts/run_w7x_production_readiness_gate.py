"""Run the W7-X production-readiness gate for convergence, timing, and optimization.

The gate intentionally separates reduced-regression readiness from production
readiness.  It audits the committed reduced convergence/timing artifacts,
runs the external W7-X mode-structure gate, records the production-control
memory estimate, and keeps DESC optimization labeled reduced until the missing
external parity and production CPU timing artifacts exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERGENCE_DIR = ROOT / "fixtures/w7x_itg_convergence_study"
DEFAULT_OBSERVED_FIXTURE = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"
DEFAULT_REFERENCE_FIXTURE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_EXTERNAL_GATE_DIR = DEFAULT_CONVERGENCE_DIR / "external_mode_structure_gate"
DEFAULT_OUTPUT = DEFAULT_CONVERGENCE_DIR / "production_readiness_gate.json"
DEFAULT_PRODUCTION_TIMING = DEFAULT_CONVERGENCE_DIR / "production_cpu_timing.json"


@dataclass(frozen=True)
class W7XReadinessThresholds:
    """Reduced-regression tolerances used before external production parity exists."""

    dt_growth_tolerance: float = 1.0e-8
    late_mean_growth_tolerance: float = 1.0e-4
    required_case_count: int = 9
    required_ky_count: int = 4


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    thresholds = W7XReadinessThresholds(
        dt_growth_tolerance=args.dt_growth_tolerance,
        late_mean_growth_tolerance=args.late_mean_growth_tolerance,
        required_case_count=args.required_case_count,
        required_ky_count=args.required_ky_count,
    )
    report = run_w7x_production_readiness_gate(
        convergence_dir=args.convergence_dir,
        observed_fixture=args.observed_fixture,
        reference_fixture=args.reference_fixture,
        external_gate_dir=args.external_gate_dir,
        output_path=args.output,
        production_timing_path=args.production_timing,
        ky_values=args.ky_values,
        thresholds=thresholds,
    )
    status = report["status"]
    passed = "PASS" if report["passed"] else "OPEN"
    print(f"{passed}: W7-X production readiness {status}")
    print(args.output)
    return 0 if report["passed"] or not args.require_pass else 2


def run_w7x_production_readiness_gate(
    *,
    convergence_dir: Path = DEFAULT_CONVERGENCE_DIR,
    observed_fixture: Path = DEFAULT_OBSERVED_FIXTURE,
    reference_fixture: Path = DEFAULT_REFERENCE_FIXTURE,
    external_gate_dir: Path = DEFAULT_EXTERNAL_GATE_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    production_timing_path: Path = DEFAULT_PRODUCTION_TIMING,
    ky_values: str = "0.1,0.2,0.3",
    thresholds: W7XReadinessThresholds = W7XReadinessThresholds(),
) -> dict[str, object]:
    """Evaluate and write the W7-X production-readiness ledger."""

    reduced = evaluate_reduced_convergence_artifacts(convergence_dir, thresholds)
    external = run_external_mode_structure_gate(
        observed_fixture=observed_fixture,
        reference_fixture=reference_fixture,
        output_dir=external_gate_dir,
        ky_values=ky_values,
    )
    production_timing = evaluate_production_timing_artifact(
        production_timing_path,
        external_parity_ready=bool(external["passed"]),
    )

    actions = required_actions(
        reduced=reduced,
        external=external,
        production_timing=production_timing,
        reference_fixture=reference_fixture,
        production_timing_path=production_timing_path,
    )
    passed = (
        bool(reduced["passed"])
        and bool(external["passed"])
        and bool(production_timing["passed"])
    )
    status = "pass" if passed else _blocked_status(reduced, external, production_timing)
    report = {
        "benchmark_name": "w7x_itg_production_readiness_gate",
        "status": status,
        "passed": passed,
        "reduced_convergence_regression": reduced,
        "external_mode_structure_gate": external,
        "production_cpu_timing": production_timing,
        "desc_optimization_status": (
            "production_ready"
            if passed
            else "keep_reduced_until_w7x_external_parity_and_production_timing_pass"
        ),
        "required_actions": actions,
    }
    _write_json(output_path, report)
    return report


def evaluate_reduced_convergence_artifacts(
    convergence_dir: Path,
    thresholds: W7XReadinessThresholds = W7XReadinessThresholds(),
) -> dict[str, object]:
    """Check the committed reduced W7-X convergence/timing artifacts."""

    summary_csv = convergence_dir / "convergence_summary.csv"
    timing_json = convergence_dir / "timing_summary.json"
    readiness_json = convergence_dir / "optimization_readiness.json"
    metadata_json = convergence_dir / "study_metadata.json"
    rows = _read_csv_rows(summary_csv)
    timing = _load_json(timing_json)
    readiness = _load_json(readiness_json)
    metadata = _load_json(metadata_json)

    finite_growth = all(_is_float(row["growth_rate"]) for row in rows)
    finite_frequency = all(_is_float(row["frequency"]) for row in rows)
    cases = sorted({row["case"] for row in rows})
    ky_values = sorted({float(row["ky"]) for row in rows})
    expected_rows = thresholds.required_case_count * thresholds.required_ky_count
    dt_delta = _max_abs_growth_delta(rows, "dt_half")
    late_delta = _max_abs_growth_delta(rows, "late_mean_window")
    reduced_status_ok = timing.get("status") == "reduced_solver_regression_not_external_parity"
    readiness_status_ok = bool(readiness.get("reduced_convergence_study_ready"))

    checks = {
        "finite_growth": finite_growth,
        "finite_frequency": finite_frequency,
        "row_count": len(rows) == expected_rows,
        "case_count": len(cases) == thresholds.required_case_count,
        "ky_count": len(ky_values) == thresholds.required_ky_count,
        "dt_half_growth_delta": dt_delta <= thresholds.dt_growth_tolerance,
        "late_mean_growth_delta": late_delta <= thresholds.late_mean_growth_tolerance,
        "reduced_timing_status": reduced_status_ok,
        "reduced_readiness_status": readiness_status_ok,
    }
    memory = timing.get("production_gx_control_memory_estimate", {})
    return {
        "passed": all(checks.values()),
        "status": "pass" if all(checks.values()) else "open",
        "checks": checks,
        "summary_csv": _display_path(summary_csv),
        "timing_json": _display_path(timing_json),
        "readiness_json": _display_path(readiness_json),
        "metadata_json": _display_path(metadata_json),
        "metadata_case_count": metadata.get("case_count"),
        "row_count": len(rows),
        "case_count": len(cases),
        "ky_values": ky_values,
        "max_abs_dt_half_growth_delta": dt_delta,
        "max_abs_late_mean_growth_delta": late_delta,
        "production_memory_estimate": memory,
        "thresholds": thresholds.__dict__,
    }


def run_external_mode_structure_gate(
    *,
    observed_fixture: Path,
    reference_fixture: Path,
    output_dir: Path,
    ky_values: str,
) -> dict[str, object]:
    """Run the external W7-X mode-structure gate and return its status JSON."""

    from examples.run_w7x_mode_structure_gate import main as run_mode_gate

    run_mode_gate(
        [
            "--observed-fixture",
            str(observed_fixture),
            "--reference-fixture",
            str(reference_fixture),
            "--output-dir",
            str(output_dir),
            "--ky-values",
            ky_values,
        ]
    )
    status_path = output_dir / "gate_status.json"
    status = _load_json(status_path)
    return {
        "passed": bool(status.get("passed")),
        "status": status.get("status", "missing_status"),
        "gate_status_json": _display_path(status_path),
        "reference_fixture": status.get("reference_fixture", _display_path(reference_fixture)),
        "observed_fixture": status.get("observed_fixture", _display_path(observed_fixture)),
        "ky_values": status.get("ky_values"),
        "max_growth_error": status.get("max_growth_error"),
        "max_frequency_error": status.get("max_frequency_error"),
        "max_profile_error": status.get("max_profile_error"),
    }


def evaluate_production_timing_artifact(
    production_timing_path: Path,
    *,
    external_parity_ready: bool,
) -> dict[str, object]:
    """Check a future true production CPU timing artifact."""

    if not external_parity_ready:
        return {
            "passed": False,
            "status": "blocked_until_external_parity_passes",
            "artifact": _display_path(production_timing_path),
        }
    if not production_timing_path.exists():
        return {
            "passed": False,
            "status": "missing_production_cpu_timing_artifact",
            "artifact": _display_path(production_timing_path),
        }
    payload = _load_json(production_timing_path)
    return {
        "passed": bool(payload.get("passed")),
        "status": payload.get("status", "missing_status"),
        "artifact": _display_path(production_timing_path),
        "payload": payload,
    }


def required_actions(
    *,
    reduced: dict[str, object],
    external: dict[str, object],
    production_timing: dict[str, object],
    reference_fixture: Path,
    production_timing_path: Path,
) -> list[str]:
    """Return human-readable next actions for failed readiness components."""

    actions: list[str] = []
    if not reduced["passed"]:
        actions.append(
            "regenerate or inspect fixtures/w7x_itg_convergence_study reduced "
            "convergence/timing artifacts"
        )
    if not external["passed"]:
        actions.append(
            "export a matched external W7-X mode-structure fixture at "
            f"{_display_path(reference_fixture)}"
        )
    if not production_timing["passed"]:
        actions.append(
            "run true production-control CPU timing and store "
            f"{_display_path(production_timing_path)}"
        )
    return actions


def _blocked_status(
    reduced: dict[str, object],
    external: dict[str, object],
    production_timing: dict[str, object],
) -> str:
    if not reduced["passed"]:
        return "blocked_reduced_convergence_regression"
    if not external["passed"]:
        return "blocked_external_reference"
    if not production_timing["passed"]:
        return str(production_timing["status"])
    return "open"


def _max_abs_growth_delta(rows: list[dict[str, str]], case: str) -> float:
    values = [
        abs(float(row["growth_delta_from_baseline"]))
        for row in rows
        if row["case"] == case and row["growth_delta_from_baseline"] != ""
    ]
    return max(values) if values else float("inf")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _is_float(value: object) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convergence-dir", type=Path, default=DEFAULT_CONVERGENCE_DIR)
    parser.add_argument("--observed-fixture", type=Path, default=DEFAULT_OBSERVED_FIXTURE)
    parser.add_argument("--reference-fixture", type=Path, default=DEFAULT_REFERENCE_FIXTURE)
    parser.add_argument("--external-gate-dir", type=Path, default=DEFAULT_EXTERNAL_GATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--production-timing", type=Path, default=DEFAULT_PRODUCTION_TIMING)
    parser.add_argument("--ky-values", default="0.1,0.2,0.3")
    parser.add_argument("--dt-growth-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--late-mean-growth-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--required-case-count", type=int, default=9)
    parser.add_argument("--required-ky-count", type=int, default=4)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
