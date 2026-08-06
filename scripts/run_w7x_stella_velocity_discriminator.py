"""Run a stella-matched W7-X velocity-space discriminator.

The time/geometry/mode controls are held fixed at the matched ``t=200`` W7-X
stella comparison.  Only the velocity-space grid/backend changes.  This keeps
the remaining ``ky=0.3`` frequency/profile mismatch focused on
velocity-space/RHS physics instead of re-opening coordinate, field-line-length,
mode-linking, or growth-window questions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_itg_stella_velocity_discriminator"
DEFAULT_TIME_LADDER = ROOT / "fixtures/w7x_itg_stella_matched_time_ladder"
DEFAULT_STELLA_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)
DEFAULT_REFERENCE_FIXTURE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_KY_VALUES = "0.1,0.2,0.3"
FOCUS_KY = 0.3
STELLA_TEND = 200.0


@dataclass(frozen=True)
class StellaVelocityCase:
    """One fixed-control W7-X velocity-space case."""

    name: str
    n_vpar: int
    n_mu: int
    velocity_backend: str = "chebyshev"
    velocity_measure_normalization: str = "legacy"
    mirror_advance: str = "explicit"
    vpar_max: float = 2.0
    mu_max: float = 1.5
    dt: float = 0.02
    steps_per_window: int = 5
    n_windows: int = 2000

    @property
    def total_time(self) -> float:
        return self.dt * self.steps_per_window * self.n_windows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = default_velocity_cases()
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case.name in selected)
        missing = selected.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")
    if not cases:
        raise ValueError("at least one velocity-discriminator case is required")
    summary = run_w7x_stella_velocity_discriminator(
        output_dir=args.output_dir,
        cases=cases,
        stella_geometry=args.stella_geometry,
        reference_fixture=args.reference_fixture,
        ky_values=args.ky_values,
        reuse_time_ladder_baseline=not args.no_reuse_time_ladder_baseline,
        time_ladder=args.time_ladder,
        reuse_existing=args.reuse_existing,
    )
    print(summary["summary_csv"])
    print(summary["status_json"])
    return 0


def default_velocity_cases() -> tuple[StellaVelocityCase, ...]:
    """Return the staged velocity-space cases for the W7-X stella comparison."""

    return (
        StellaVelocityCase(name="cheb_4x4", n_vpar=4, n_mu=4),
        StellaVelocityCase(name="cheb_6x6", n_vpar=6, n_mu=6),
        StellaVelocityCase(name="cheb_8x8", n_vpar=8, n_mu=8),
        StellaVelocityCase(
            name="gkw_fd_16x8",
            n_vpar=16,
            n_mu=8,
            velocity_backend="finite_difference",
        ),
        StellaVelocityCase(
            name="native_32x8",
            n_vpar=32,
            n_mu=8,
            velocity_backend="midpoint_gauss_laguerre",
            velocity_measure_normalization="full_gyroangle",
            mirror_advance="semi_lagrangian",
            vpar_max=3.0,
            mu_max=4.916958697837631,
        ),
    )


def run_w7x_stella_velocity_discriminator(
    *,
    output_dir: Path,
    cases: tuple[StellaVelocityCase, ...],
    stella_geometry: Path = DEFAULT_STELLA_GEOMETRY,
    reference_fixture: Path = DEFAULT_REFERENCE_FIXTURE,
    ky_values: str = DEFAULT_KY_VALUES,
    reuse_time_ladder_baseline: bool = True,
    time_ladder: Path = DEFAULT_TIME_LADDER,
    reuse_existing: bool = False,
) -> dict[str, object]:
    """Run velocity cases, write per-case gates/audits, and summarize results."""

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
        baseline_source = _baseline_source_for_case(
            case,
            time_ladder,
            reuse_time_ladder_baseline,
        )
        if baseline_source is not None:
            case_output = baseline_source
            wall_seconds = 0.0
            reused = True
        else:
            reused = _case_outputs_exist(case_output) and reuse_existing
            if reused:
                wall_seconds = 0.0
            else:
                start = perf_counter()
                run_scan(_scan_args(case, case_output, stella_geometry, ky_values))
                wall_seconds = perf_counter() - start
                run_mode_gate(_gate_args(case_output, reference_fixture, ky_values))
                run_w7x_stella_solver_parity_audit(
                    solver_config=case_output / "run_config.json",
                    solver_metadata=case_output / "convergence_metadata.json",
                    solver_fixture=case_output / "mode_structures.csv",
                    gate_status=case_output / "mode_structure_gate/gate_status.json",
                    output=case_output / "stella_solver_parity_audit.json",
                )
        case_rows, run_summary = _read_case_outputs(
            case,
            case_output,
            wall_seconds,
            reused=reused,
            baseline_source=baseline_source,
        )
        rows.extend(case_rows)
        run_summaries.append(run_summary)

    rows = rows_with_velocity_baseline_deltas(rows)
    status_payload = _status_payload(
        output_dir=output_dir,
        rows=rows,
        run_summaries=run_summaries,
        cases=cases,
        stella_geometry=stella_geometry,
        reference_fixture=reference_fixture,
        ky_values=ky_values,
    )
    summary_csv = output_dir / "velocity_discriminator_summary.csv"
    status_json = output_dir / "velocity_discriminator_status.json"
    metadata_json = output_dir / "velocity_discriminator_metadata.json"
    _write_summary_csv(summary_csv, rows)
    _write_json(status_json, status_payload)
    _write_json(
        metadata_json,
        {
            "benchmark_name": "w7x_stella_velocity_discriminator",
            "purpose": (
                "vary velocity-space resolution/backend at fixed stella-matched "
                "W7-X geometry, modes, and t=200 growth window"
            ),
            "geometry_source": "stella-geometry",
            "stella_geometry": _display_path(stella_geometry),
            "reference_fixture": _display_path(reference_fixture),
            "ky_values": _parse_float_tuple(ky_values),
            "focus_ky": FOCUS_KY,
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


def rows_with_velocity_baseline_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach deltas relative to the first case for each ky."""

    if not rows:
        return rows
    first_case = str(rows[0]["case"])
    baseline = {float(row["ky"]): row for row in rows if row["case"] == first_case}
    with_deltas = []
    for row in rows:
        copied = dict(row)
        reference = baseline.get(float(row["ky"]))
        if reference is None:
            copied["growth_delta_from_baseline"] = ""
            copied["frequency_delta_from_baseline"] = ""
            copied["profile_error_delta_from_baseline"] = ""
            copied["abs_frequency_error_delta_from_baseline"] = ""
        else:
            copied["growth_delta_from_baseline"] = float(row["growth_rate"]) - float(
                reference["growth_rate"]
            )
            copied["frequency_delta_from_baseline"] = float(row["frequency"]) - float(
                reference["frequency"]
            )
            copied["profile_error_delta_from_baseline"] = _optional_delta(
                row.get("phi_phase_aligned_error"),
                reference.get("phi_phase_aligned_error"),
            )
            copied["abs_frequency_error_delta_from_baseline"] = _optional_delta(
                _abs_or_blank(row.get("frequency_error")),
                _abs_or_blank(reference.get("frequency_error")),
            )
        with_deltas.append(copied)
    return with_deltas


def _scan_args(
    case: StellaVelocityCase,
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
        "--velocity-backend",
        case.velocity_backend,
        "--velocity-measure-normalization",
        case.velocity_measure_normalization,
        "--mirror-advance",
        case.mirror_advance,
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


def _gate_args(case_output: Path, reference_fixture: Path, ky_values: str) -> list[str]:
    return [
        "--observed-fixture",
        str(case_output / "mode_structures.csv"),
        "--reference-fixture",
        str(reference_fixture),
        "--ky-values",
        ky_values,
        "--resample-reference-to-observed-z",
        "--output-dir",
        str(case_output / "mode_structure_gate"),
    ]


def _read_case_outputs(
    case: StellaVelocityCase,
    case_output: Path,
    wall_seconds: float,
    *,
    reused: bool,
    baseline_source: Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    convergence = _load_json(case_output / "convergence_metadata.json")
    gate = _load_json(case_output / "mode_structure_gate/gate_status.json")
    audit = _load_json(case_output / "stella_solver_parity_audit.json")
    gate_rows = _read_gate_rows(case_output / "mode_structure_gate/mode_structure_gate.csv")
    checks = {item["name"]: item for item in audit["ordered_checks"]}
    time_check = checks["growth_window_time_normalization"]
    rows = []
    with (case_output / "ky_growth.csv").open(newline="") as handle:
        for growth_row in csv.DictReader(handle):
            ky = float(growth_row["ky"])
            gate_row = gate_rows.get(ky, {})
            rows.append(
                {
                    "case": case.name,
                    "n_vpar": case.n_vpar,
                    "n_mu": case.n_mu,
                    "velocity_backend": case.velocity_backend,
                    "actual_total_time": case.total_time,
                    "stella_tend": STELLA_TEND,
                    "time_window_passed": bool(time_check["passed"]),
                    "first_failed_check": audit["first_failed_check"],
                    "ky": ky,
                    "focus_ky": bool(abs(ky - FOCUS_KY) <= 1.0e-12),
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
                    "reused_existing_output": bool(reused),
                    "baseline_source": "" if baseline_source is None else _display_path(baseline_source),
                }
            )
    run_summary = {
        "case": case.name,
        "n_vpar": case.n_vpar,
        "n_mu": case.n_mu,
        "velocity_backend": case.velocity_backend,
        "actual_total_time": case.total_time,
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
        "reused_existing_output": bool(reused),
        "baseline_source": "" if baseline_source is None else _display_path(baseline_source),
        "run_dir": _display_path(case_output),
    }
    return rows, run_summary


def _status_payload(
    *,
    output_dir: Path,
    rows: list[dict[str, object]],
    run_summaries: list[dict[str, object]],
    cases: tuple[StellaVelocityCase, ...],
    stella_geometry: Path,
    reference_fixture: Path,
    ky_values: str,
) -> dict[str, object]:
    finite = all(bool(item["finite_growth"]) and bool(item["finite_frequency"]) for item in rows)
    time_window = all(bool(item["time_window_passed"]) for item in run_summaries)
    focus_rows = [row for row in rows if bool(row["focus_ky"])]
    best_focus = min(
        focus_rows,
        key=lambda row: (
            abs(_float_or_large(row.get("frequency_error"))),
            _float_or_large(row.get("phi_phase_aligned_error")),
        ),
    )
    baseline_focus = focus_rows[0]
    abs_frequency_errors = [abs(float(row["frequency_error"])) for row in focus_rows]
    profile_errors = [float(row["phi_phase_aligned_error"]) for row in focus_rows]
    focus_frequency_error_span = max(abs_frequency_errors) - min(abs_frequency_errors)
    focus_profile_error_span = max(profile_errors) - min(profile_errors)
    velocity_sensitive = focus_frequency_error_span > 2.0e-2 or focus_profile_error_span > 2.0e-2
    stella_velocity_case_present = any(
        str(item["velocity_backend"]) in ("finite_difference", "midpoint_gauss_laguerre")
        and int(item["n_vpar"]) >= 16
        and int(item["n_mu"]) >= 8
        for item in run_summaries
    )
    native_mirror_case_present = any(
        str(item["velocity_backend"]) == "midpoint_gauss_laguerre"
        and int(item["n_vpar"]) >= 32
        and int(item["n_mu"]) >= 8
        for item in run_summaries
    )
    focus_closed = (
        abs(_float_or_large(best_focus.get("growth_error"))) <= 2.0e-2
        and abs(_float_or_large(best_focus.get("frequency_error"))) <= 2.0e-2
        and _float_or_large(best_focus.get("phi_phase_aligned_error")) <= 2.0e-2
    )
    passed = finite and time_window and any(bool(item["gate_passed"]) for item in run_summaries)
    if passed:
        status = "pass"
        next_action = "promote the passing velocity/backend controls into the W7-X readiness gate"
    elif native_mirror_case_present:
        status = "open_profile_after_native_mirror"
        next_action = (
            "extend the native 32x8 solver and pinned stella runs to converged "
            "per-ky windows, then add stella-equivalent implicit parallel "
            "streaming and cubic mirror interpolation if profile parity remains open"
        )
    elif stella_velocity_case_present and not focus_closed:
        status = "open_rhs_terms_after_velocity_discriminator"
        next_action = (
            "inspect W7-X ky=0.3 RHS/model term balance and normalization; "
            "the stella-level finite-difference velocity grid did not close "
            "the frequency/profile gap"
        )
    elif velocity_sensitive:
        status = "open_velocity_sensitive"
        next_action = (
            "run the stella-level 16x8 and intermediate spectral cases, then "
            "inspect the ky=0.3 velocity/RHS term balance if the profile or "
            "frequency errors do not continue toward tolerance"
        )
    else:
        status = "open_velocity_insensitive"
        next_action = (
            "move to term-level W7-X RHS parity for ky=0.3; simple velocity "
            "resolution/backend changes are not moving the dominant mismatch"
        )
    return {
        "benchmark_name": "w7x_stella_velocity_discriminator",
        "status": status,
        "passed": passed,
        "finite_outputs": finite,
        "time_window_passed": time_window,
        "case_count": len(cases),
        "ky_values": _parse_float_tuple(ky_values),
        "focus_ky": FOCUS_KY,
        "stella_tend": STELLA_TEND,
        "stella_geometry": _display_path(stella_geometry),
        "reference_fixture": _display_path(reference_fixture),
        "summary_csv": _display_path(output_dir / "velocity_discriminator_summary.csv"),
        "baseline_focus_row": baseline_focus,
        "best_focus_row": best_focus,
        "focus_abs_frequency_error_span": focus_frequency_error_span,
        "focus_profile_error_span": focus_profile_error_span,
        "velocity_sensitive": velocity_sensitive,
        "stella_velocity_case_present": stella_velocity_case_present,
        "native_mirror_case_present": native_mirror_case_present,
        "focus_gate_closed": focus_closed,
        "run_summaries": run_summaries,
        "next_action": next_action,
    }


def _baseline_source_for_case(
    case: StellaVelocityCase,
    time_ladder: Path,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    if case.name != "cheb_4x4":
        return None
    if case.velocity_backend != "chebyshev" or case.n_vpar != 4 or case.n_mu != 4:
        return None
    source = time_ladder / "runs/time_200"
    return source if _case_outputs_exist(source) else None


def _case_outputs_exist(case_output: Path) -> bool:
    required = (
        "ky_growth.csv",
        "mode_structures.csv",
        "convergence_metadata.json",
        "mode_structure_gate/gate_status.json",
        "mode_structure_gate/mode_structure_gate.csv",
        "stella_solver_parity_audit.json",
    )
    return all((case_output / name).exists() for name in required)


def _read_gate_rows(path: Path) -> dict[float, dict[str, str]]:
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[float(row["ky"])] = row
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "case",
        "n_vpar",
        "n_mu",
        "velocity_backend",
        "actual_total_time",
        "stella_tend",
        "time_window_passed",
        "first_failed_check",
        "ky",
        "focus_ky",
        "growth_rate",
        "reference_growth",
        "growth_error",
        "growth_delta_from_baseline",
        "frequency",
        "reference_frequency",
        "frequency_error",
        "frequency_delta_from_baseline",
        "abs_frequency_error_delta_from_baseline",
        "phi_phase_aligned_error",
        "profile_error_delta_from_baseline",
        "kperp2_average",
        "quasilinear_contribution",
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
        "reused_existing_output",
        "baseline_source",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# W7-X stella Velocity Discriminator",
                "",
                "Regenerate from the repository root with:",
                "",
                "```bash",
                "uv run python scripts/run_w7x_stella_velocity_discriminator.py",
                "```",
                "",
                "The discriminator holds the stella geometry, kx=0/n_kx=1,",
                "`ky=(0.1,0.2,0.3)`, species gradients, and t=200 late-half",
                "growth window fixed. It varies only velocity-space resolution",
                "and backend, then reports whether the remaining ky=0.3",
                "frequency/profile mismatch is velocity-sensitive.",
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


def _optional_delta(left, right) -> float | str:
    if left in (None, "") or right in (None, ""):
        return ""
    return float(left) - float(right)


def _abs_or_blank(value) -> float | str:
    if value in (None, ""):
        return ""
    return abs(float(value))


def _float_or_large(value) -> float:
    if value in (None, ""):
        return 1.0e300
    return float(value)


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stella-geometry", type=Path, default=DEFAULT_STELLA_GEOMETRY)
    parser.add_argument("--reference-fixture", type=Path, default=DEFAULT_REFERENCE_FIXTURE)
    parser.add_argument("--ky-values", default=DEFAULT_KY_VALUES)
    parser.add_argument("--time-ladder", type=Path, default=DEFAULT_TIME_LADDER)
    parser.add_argument("--no-reuse-time-ladder-baseline", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse complete per-case outputs already present under --output-dir",
    )
    parser.add_argument("--case", action="append", help="run only the named velocity case")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
