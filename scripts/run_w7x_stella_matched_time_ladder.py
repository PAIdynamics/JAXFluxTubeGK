"""Run a stella-matched W7-X solver time-window ladder.

This driver keeps the stella-imported geometry and mode controls fixed while
only extending the solver trace length.  It is meant to close the ordered
``growth_window_time_normalization`` blocker before any RHS or velocity-space
physics changes are interpreted against stella.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_itg_stella_matched_time_ladder"
DEFAULT_STELLA_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)
DEFAULT_REFERENCE_FIXTURE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_KY_VALUES = "0.1,0.2,0.3"
STELLA_TEND = 200.0


@dataclass(frozen=True)
class StellaMatchedTimeCase:
    """One fixed-control stella-matched W7-X time-horizon case."""

    name: str
    target_total_time: float
    dt: float
    steps_per_window: int
    n_windows: int
    n_vpar: int = 4
    n_mu: int = 4
    vpar_max: float = 2.0
    mu_max: float = 1.5

    @property
    def total_time(self) -> float:
        return self.dt * self.steps_per_window * self.n_windows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = default_time_cases(
        end_times=_parse_float_tuple(args.end_times),
        dt=args.dt,
        steps_per_window=args.steps_per_window,
        include_smoke=not args.no_smoke,
    )
    if args.max_total_time is not None:
        cases = tuple(case for case in cases if case.total_time <= args.max_total_time)
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case.name in selected)
        missing = selected.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")
    summary = run_w7x_stella_matched_time_ladder(
        output_dir=args.output_dir,
        cases=cases,
        stella_geometry=args.stella_geometry,
        reference_fixture=args.reference_fixture,
        ky_values=args.ky_values,
    )
    print(summary["summary_csv"])
    print(summary["status_json"])
    return 0


def default_time_cases(
    *,
    end_times: tuple[float, ...] = (1.0, 5.0, 20.0, 100.0, 200.0),
    dt: float = 0.02,
    steps_per_window: int = 5,
    include_smoke: bool = True,
) -> tuple[StellaMatchedTimeCase, ...]:
    """Return the staged time ladder used for the stella-matched W7-X audit."""

    cases = []
    if include_smoke:
        cases.append(
            StellaMatchedTimeCase(
                name="smoke_0p006",
                target_total_time=0.006,
                dt=0.001,
                steps_per_window=1,
                n_windows=6,
            )
        )
    for end_time in end_times:
        cases.append(
            case_from_total_time(
                end_time,
                dt=dt,
                steps_per_window=steps_per_window,
            )
        )
    return tuple(cases)


def case_from_total_time(
    total_time: float,
    *,
    dt: float,
    steps_per_window: int,
) -> StellaMatchedTimeCase:
    """Build a ladder case whose actual total time is rounded to whole windows."""

    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if steps_per_window < 1:
        raise ValueError("steps_per_window must be at least one")
    window_time = dt * steps_per_window
    n_windows = max(1, int(round(total_time / window_time)))
    return StellaMatchedTimeCase(
        name=f"time_{_format_case_time(total_time)}",
        target_total_time=float(total_time),
        dt=float(dt),
        steps_per_window=int(steps_per_window),
        n_windows=n_windows,
    )


def run_w7x_stella_matched_time_ladder(
    *,
    output_dir: Path,
    cases: tuple[StellaMatchedTimeCase, ...],
    stella_geometry: Path = DEFAULT_STELLA_GEOMETRY,
    reference_fixture: Path = DEFAULT_REFERENCE_FIXTURE,
    ky_values: str = DEFAULT_KY_VALUES,
) -> dict[str, object]:
    """Run all cases, write per-case gates/audits, and summarize the ladder."""

    if not cases:
        raise ValueError("at least one time-ladder case is required")

    from examples.run_stellarator_linear_scan import main as run_scan
    from examples.run_w7x_mode_structure_gate import main as run_mode_gate
    from scripts.audit_w7x_stella_solver_parity import run_w7x_stella_solver_parity_audit

    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = output_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for case in cases:
        case_output = run_root / case.name
        gate_output = case_output / "mode_structure_gate"
        start = perf_counter()
        run_scan(_scan_args(case, case_output, stella_geometry, ky_values))
        wall_seconds = perf_counter() - start
        run_mode_gate(
            [
                "--observed-fixture",
                str(case_output / "mode_structures.csv"),
                "--reference-fixture",
                str(reference_fixture),
                "--ky-values",
                ky_values,
                "--resample-reference-to-observed-z",
                "--output-dir",
                str(gate_output),
            ]
        )
        audit = run_w7x_stella_solver_parity_audit(
            solver_config=case_output / "run_config.json",
            solver_metadata=case_output / "convergence_metadata.json",
            solver_fixture=case_output / "mode_structures.csv",
            gate_status=gate_output / "gate_status.json",
            output=case_output / "stella_solver_parity_audit.json",
        )
        case_rows, run_summary = _read_case_outputs(
            case,
            case_output,
            wall_seconds,
            audit,
        )
        rows.extend(case_rows)
        run_summaries.append(run_summary)

    rows = rows_with_time_baseline_deltas(rows)
    status_payload = _status_payload(
        output_dir=output_dir,
        rows=rows,
        run_summaries=run_summaries,
        cases=cases,
        stella_geometry=stella_geometry,
        reference_fixture=reference_fixture,
        ky_values=ky_values,
    )
    summary_csv = output_dir / "time_ladder_summary.csv"
    status_json = output_dir / "time_ladder_status.json"
    metadata_json = output_dir / "time_ladder_metadata.json"
    _write_summary_csv(summary_csv, rows)
    _write_json(status_json, status_payload)
    _write_json(
        metadata_json,
        {
            "benchmark_name": "w7x_stella_matched_time_ladder",
            "purpose": "extend time horizon before changing RHS or velocity-space physics",
            "geometry_source": "stella-geometry",
            "stella_geometry": _display_path(stella_geometry),
            "reference_fixture": _display_path(reference_fixture),
            "ky_values": _parse_float_tuple(ky_values),
            "stella_tend": STELLA_TEND,
            "cases": [case.__dict__ | {"actual_total_time": case.total_time} for case in cases],
            "summary_csv": _display_path(summary_csv),
            "status_json": _display_path(status_json),
        },
    )
    _write_readme(output_dir)
    return {
        "summary_csv": str(summary_csv),
        "status_json": str(status_json),
        "metadata_json": str(metadata_json),
        "rows": rows,
        "run_summaries": run_summaries,
    }


def rows_with_time_baseline_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach deltas relative to the first time case for each ky."""

    if not rows:
        return rows
    first_case = str(rows[0]["case"])
    baseline = {float(row["ky"]): row for row in rows if row["case"] == first_case}
    with_deltas = []
    for row in rows:
        copied = dict(row)
        reference = baseline.get(float(row["ky"]))
        if reference is None:
            copied["growth_delta_from_first_case"] = ""
            copied["frequency_delta_from_first_case"] = ""
        else:
            copied["growth_delta_from_first_case"] = float(row["growth_rate"]) - float(
                reference["growth_rate"]
            )
            copied["frequency_delta_from_first_case"] = float(row["frequency"]) - float(
                reference["frequency"]
            )
        with_deltas.append(copied)
    return with_deltas


def _scan_args(
    case: StellaMatchedTimeCase,
    output_dir: Path,
    stella_geometry: Path,
    ky_values: str,
) -> list[str]:
    return [
        "--geometry-source",
        "stella-geometry",
        "--stella-geometry",
        str(stella_geometry),
        "--output-dir",
        str(output_dir),
        "--n-kx",
        "1",
        "--kx-max",
        "0.0",
        "--ikxspace",
        "1",
        "--ky-values",
        ky_values,
        "--n-vpar",
        str(case.n_vpar),
        "--n-mu",
        str(case.n_mu),
        "--vpar-max",
        str(case.vpar_max),
        "--mu-max",
        str(case.mu_max),
        "--density",
        "1.0",
        "--temperature",
        "1.0",
        "--density-gradient",
        "1.0",
        "--temperature-gradient",
        "3.0",
        "--electron-density",
        "1.0",
        "--electron-temperature",
        "1.0",
        "--dt",
        str(case.dt),
        "--steps-per-window",
        str(case.steps_per_window),
        "--n-windows",
        str(case.n_windows),
        "--growth-diagnostic",
        "late_fit",
        "--growth-window-fraction",
        "0.5",
    ]


def _read_case_outputs(
    case: StellaMatchedTimeCase,
    case_output: Path,
    wall_seconds: float,
    audit: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    convergence = _load_json(case_output / "convergence_metadata.json")
    gate = _load_json(case_output / "mode_structure_gate/gate_status.json")
    gate_rows = _read_gate_rows(case_output / "mode_structure_gate/mode_structure_gate.csv")
    audit_checks = {item["name"]: item for item in audit["ordered_checks"]}
    time_check = audit_checks["growth_window_time_normalization"]
    rows = []
    with (case_output / "ky_growth.csv").open(newline="") as handle:
        for growth_row in csv.DictReader(handle):
            ky = float(growth_row["ky"])
            gate_row = gate_rows.get(ky, {})
            rows.append(
                {
                    "case": case.name,
                    "target_total_time": case.target_total_time,
                    "actual_total_time": case.total_time,
                    "stella_tend": STELLA_TEND,
                    "solver_to_stella_total_time_ratio": float(
                        time_check["solver_to_stella_total_time_ratio"]
                    ),
                    "time_window_passed": bool(time_check["passed"]),
                    "first_failed_check": audit["first_failed_check"],
                    "ky": ky,
                    "growth_rate": float(growth_row["growth_rate"]),
                    "reference_growth": _optional_float(gate_row.get("reference_growth")),
                    "growth_error": _optional_float(gate_row.get("growth_error")),
                    "frequency": float(growth_row["frequency"]),
                    "reference_frequency": _optional_float(gate_row.get("reference_frequency")),
                    "frequency_error": _optional_float(gate_row.get("frequency_error")),
                    "phi_phase_aligned_error": _optional_float(
                        gate_row.get("phi_phase_aligned_error")
                    ),
                    "kperp2_average": float(growth_row["kperp2_average"]),
                    "quasilinear_contribution": float(
                        growth_row["quasilinear_contribution"]
                    ),
                    "raw_final_amplitude": float(growth_row["raw_final_amplitude"]),
                    "max_growth_error": float(gate["max_growth_error"]),
                    "max_frequency_error": float(gate["max_frequency_error"]),
                    "max_profile_error": float(gate["max_profile_error"]),
                    "late_window_growth_delta": float(
                        convergence["max_abs_late_window_growth_delta"]
                    ),
                    "estimated_cfl_dt": float(convergence["estimated_cfl_dt"]),
                    "finite_growth": bool(convergence["finite_growth"]),
                    "finite_frequency": bool(convergence["finite_frequency"]),
                    "dt": case.dt,
                    "steps_per_window": case.steps_per_window,
                    "n_windows": case.n_windows,
                    "wall_seconds": float(wall_seconds),
                }
            )
    return rows, {
        "case": case.name,
        "target_total_time": case.target_total_time,
        "actual_total_time": case.total_time,
        "solver_to_stella_total_time_ratio": float(
            time_check["solver_to_stella_total_time_ratio"]
        ),
        "time_window_passed": bool(time_check["passed"]),
        "first_failed_check": audit["first_failed_check"],
        "gate_passed": bool(gate["passed"]),
        "max_growth_error": float(gate["max_growth_error"]),
        "max_frequency_error": float(gate["max_frequency_error"]),
        "max_profile_error": float(gate["max_profile_error"]),
        "late_window_growth_delta": float(convergence["max_abs_late_window_growth_delta"]),
        "finite_growth": bool(convergence["finite_growth"]),
        "finite_frequency": bool(convergence["finite_frequency"]),
        "estimated_cfl_dt": float(convergence["estimated_cfl_dt"]),
        "wall_seconds": float(wall_seconds),
        "run_dir": _display_path(case_output),
    }


def _status_payload(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    run_summaries: list[dict[str, object]],
    cases: tuple[StellaMatchedTimeCase, ...],
    stella_geometry: Path,
    reference_fixture: Path,
    ky_values: str,
) -> dict[str, object]:
    finite = all(bool(item["finite_growth"]) and bool(item["finite_frequency"]) for item in rows)
    time_window_cases = [item for item in run_summaries if item["time_window_passed"]]
    reached_stella_tend = any(float(item["actual_total_time"]) >= STELLA_TEND for item in run_summaries)
    first_failures = {str(item["case"]): item["first_failed_check"] for item in run_summaries}
    best_profile = min(run_summaries, key=lambda item: float(item["max_profile_error"]))
    latest = max(run_summaries, key=lambda item: float(item["actual_total_time"]))
    passed = finite and bool(time_window_cases) and bool(latest["gate_passed"])
    status = (
        "pass"
        if passed
        else (
            "time_window_reached_gate_open"
            if time_window_cases
            else "open_growth_window_time_normalization"
        )
    )
    return {
        "benchmark_name": "w7x_stella_matched_time_ladder",
        "status": status,
        "passed": passed,
        "finite_outputs": finite,
        "stella_comparable_time_window_reached": bool(time_window_cases),
        "reached_stella_tend": reached_stella_tend,
        "case_count": len(cases),
        "ky_values": _parse_float_tuple(ky_values),
        "stella_tend": STELLA_TEND,
        "stella_geometry": _display_path(stella_geometry),
        "reference_fixture": _display_path(reference_fixture),
        "summary_csv": _display_path(output_dir / "time_ladder_summary.csv"),
        "first_failed_check_by_case": first_failures,
        "latest_case": latest,
        "best_profile_case": best_profile,
        "run_summaries": run_summaries,
        "next_action": (
            "inspect RHS/velocity-space parity at the longest matched-time case"
            if time_window_cases
            else "extend the longest case until the ordered audit passes growth-window time"
        ),
    }


def _read_gate_rows(path: Path) -> dict[float, dict[str, str]]:
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[float(row["ky"])] = row
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "case",
        "target_total_time",
        "actual_total_time",
        "stella_tend",
        "solver_to_stella_total_time_ratio",
        "time_window_passed",
        "first_failed_check",
        "ky",
        "growth_rate",
        "reference_growth",
        "growth_error",
        "growth_delta_from_first_case",
        "frequency",
        "reference_frequency",
        "frequency_error",
        "frequency_delta_from_first_case",
        "phi_phase_aligned_error",
        "kperp2_average",
        "quasilinear_contribution",
        "raw_final_amplitude",
        "max_growth_error",
        "max_frequency_error",
        "max_profile_error",
        "late_window_growth_delta",
        "estimated_cfl_dt",
        "finite_growth",
        "finite_frequency",
        "dt",
        "steps_per_window",
        "n_windows",
        "wall_seconds",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# W7-X stella-Matched Time Ladder",
                "",
                "Regenerate from the repository root with:",
                "",
                "```bash",
                "uv run python scripts/run_w7x_stella_matched_time_ladder.py",
                "```",
                "",
                "The ladder holds the stella-imported geometry, selected ky set,",
                "and kx=0/n_kx=1 controls fixed. It only extends the solver time",
                "horizon so the ordered stella parity audit can move beyond",
                "`growth_window_time_normalization` before any RHS or velocity",
                "space terms are changed.",
                "",
            )
        )
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _optional_float(value: str | None) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)


def _format_case_time(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stella-geometry", type=Path, default=DEFAULT_STELLA_GEOMETRY)
    parser.add_argument("--reference-fixture", type=Path, default=DEFAULT_REFERENCE_FIXTURE)
    parser.add_argument("--ky-values", default=DEFAULT_KY_VALUES)
    parser.add_argument("--end-times", default="1,5,20,100,200")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps-per-window", type=int, default=5)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--max-total-time", type=float)
    parser.add_argument("--case", action="append", help="run only the named case")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
