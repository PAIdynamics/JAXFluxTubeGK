"""Run the reduced W7-X convergence/timing study for item 5.

The study is deliberately small enough for the default test/development
environment.  It uses the real GX/GIST W7-X eik geometry but keeps the
diagnostics labeled as reduced solver-regression data until an external W7-X
``.big.nc``/``.out.nc`` reference is exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_itg_convergence_study"
DEFAULT_EIK_REFERENCE = (
    ROOT
    / "relevant-codes/gx/geometry_modules/vmec/tests/"
    "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
)
DEFAULT_KY_VALUES = "0.0,0.1,0.2,0.3"


@dataclass(frozen=True)
class W7XConvergenceCase:
    """One reduced W7-X scan in the convergence matrix."""

    name: str
    axis: str
    n_z: int = 33
    field_line_periods: int = 1
    ky_values: str = DEFAULT_KY_VALUES
    n_kx: int = 3
    kx_max: float = 0.3
    ikxspace: int = 2
    n_vpar: int = 4
    n_mu: int = 4
    vpar_max: float = 2.0
    mu_max: float = 1.5
    dt: float = 0.002
    steps_per_window: int = 1
    n_windows: int = 6
    growth_diagnostic: str = "late_fit"
    growth_window_fraction: float = 0.5


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    cases = default_convergence_cases()
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case.name in selected)
        missing = selected.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")
    summary = run_w7x_reduced_convergence_study(
        output_dir,
        cases=cases,
        eik_reference=args.eik_reference,
        keep_runs=args.keep_runs,
    )
    print(summary["summary_csv"])
    print(summary["timing_json"])
    return 0


def default_convergence_cases() -> tuple[W7XConvergenceCase, ...]:
    """Return the reduced W7-X convergence matrix used for committed artifacts."""

    baseline = W7XConvergenceCase(name="baseline", axis="reference")
    return (
        baseline,
        replace(baseline, name="nz_17", axis="parallel_resolution", n_z=17),
        replace(baseline, name="nz_49", axis="parallel_resolution", n_z=49),
        replace(baseline, name="velocity_3x3", axis="velocity_resolution", n_vpar=3, n_mu=3),
        replace(baseline, name="velocity_5x5", axis="velocity_resolution", n_vpar=5, n_mu=5),
        replace(baseline, name="kx5", axis="kx_grid", n_kx=5, kx_max=0.45),
        replace(
            baseline,
            name="dt_half",
            axis="time_step",
            dt=0.001,
            steps_per_window=2,
        ),
        replace(
            baseline,
            name="two_periods",
            axis="field_line_length",
            n_z=65,
            field_line_periods=2,
        ),
        replace(
            baseline,
            name="late_mean_window",
            axis="growth_window",
            growth_diagnostic="late_mean_window",
        ),
    )


def run_w7x_reduced_convergence_study(
    output_dir: Path,
    *,
    cases: tuple[W7XConvergenceCase, ...] | None = None,
    eik_reference: Path = DEFAULT_EIK_REFERENCE,
    keep_runs: bool = False,
) -> dict[str, object]:
    """Run reduced scans and write convergence/timing artifacts."""

    from examples.run_stellarator_linear_scan import main as run_scan
    from stellarator_gk import estimate_linear_memory_from_dimensions, format_bytes

    cases = cases or default_convergence_cases()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = (
        output_dir / "runs"
        if keep_runs
        else Path(tempfile.mkdtemp(prefix="w7x_convergence_"))
    )
    if keep_runs:
        run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for case in cases:
        case_output = run_root / case.name
        start = perf_counter()
        run_scan(_scan_args(case, case_output, eik_reference))
        wall_seconds = perf_counter() - start
        case_rows, run_summary = _read_case_outputs(case, case_output, wall_seconds)
        rows.extend(case_rows)
        run_summaries.append(run_summary)

    rows = rows_with_baseline_deltas(rows)
    memory = estimate_linear_memory_from_dimensions(
        n_vpar=16,
        n_mu=8,
        n_z=256,
        n_kx=1,
        n_ky=28,
        n_steps=0,
        store_history=False,
    )
    timing_payload = {
        "benchmark_name": "w7x_itg_reduced_convergence_timing",
        "status": "reduced_solver_regression_not_external_parity",
        "external_parity_ready": False,
        "production_timing_claim": "not_claimed_pending_external_parity_and_production_run",
        "run_summaries": run_summaries,
        "production_gx_control_memory_estimate": {
            "dimensions": {
                "n_z": 256,
                "n_kx": 1,
                "n_ky": 28,
                "n_vpar": 16,
                "n_mu": 8,
                "n_species": 1,
            },
            "state_shape": memory.state_shape,
            "field_shape": memory.field_shape,
            "state_bytes": memory.state_bytes,
            "field_bytes": memory.field_bytes,
            "coefficient_bytes": memory.coefficient_bytes,
            "history_bytes": memory.history_bytes,
            "total_bytes": memory.total_bytes,
            "total_bytes_human": format_bytes(memory.total_bytes),
            "store_history": memory.store_history,
        },
    }
    readiness_payload = {
        "benchmark_name": "w7x_itg_optimization_readiness",
        "reduced_fixture_ready": True,
        "reduced_convergence_study_ready": True,
        "external_w7x_parity_ready": False,
        "production_cpu_timing_ready": False,
        "desc_optimization_status": (
            "keep_reduced_until_w7x_external_parity_and_production_timing_pass"
        ),
        "next_required_artifact": "fixtures/w7x_itg_external_mode_structure_fixture.csv",
    }

    summary_csv = output_dir / "convergence_summary.csv"
    timing_json = output_dir / "timing_summary.json"
    readiness_json = output_dir / "optimization_readiness.json"
    metadata_json = output_dir / "study_metadata.json"
    _write_convergence_csv(summary_csv, rows)
    _write_json(timing_json, timing_payload)
    _write_json(readiness_json, readiness_payload)
    _write_json(
        metadata_json,
        {
            "benchmark_name": "w7x_itg_reduced_convergence_study",
            "geometry_source": "GX/GIST W7-X eik",
            "eik_reference": _display_path(eik_reference),
            "case_count": len(cases),
            "cases": [case.__dict__ for case in cases],
            "summary_csv": _display_path(summary_csv),
            "timing_json": _display_path(timing_json),
            "readiness_json": _display_path(readiness_json),
            "tolerance_ladder": {
                "reduced_regression": "finite growth/frequency and reproducible artifacts",
                "near_grid_delta": "inspect absolute delta against baseline by ky",
                "external_parity": "pending matched GX/GKW/GS2/stella W7-X fixture",
            },
        },
    )
    _write_readme(output_dir)
    return {
        "summary_csv": str(summary_csv),
        "timing_json": str(timing_json),
        "readiness_json": str(readiness_json),
        "metadata_json": str(metadata_json),
        "rows": rows,
        "run_summaries": run_summaries,
    }


def rows_with_baseline_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Attach per-ky deltas relative to the baseline case."""

    baseline = {
        float(row["ky"]): row for row in rows if row["case"] == "baseline"
    }
    with_deltas: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        reference = baseline.get(float(row["ky"]))
        if reference is None:
            copied["growth_delta_from_baseline"] = ""
            copied["frequency_delta_from_baseline"] = ""
        else:
            copied["growth_delta_from_baseline"] = float(row["growth_rate"]) - float(
                reference["growth_rate"]
            )
            copied["frequency_delta_from_baseline"] = float(row["frequency"]) - float(
                reference["frequency"]
            )
        with_deltas.append(copied)
    return with_deltas


def _scan_args(case: W7XConvergenceCase, output_dir: Path, eik_reference: Path) -> list[str]:
    return [
        "--geometry-source",
        "eik",
        "--eik-reference",
        str(eik_reference),
        "--output-dir",
        str(output_dir),
        "--n-z",
        str(case.n_z),
        "--field-line-periods",
        str(case.field_line_periods),
        "--ky-values",
        case.ky_values,
        "--n-kx",
        str(case.n_kx),
        "--kx-max",
        str(case.kx_max),
        "--ikxspace",
        str(case.ikxspace),
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
        case.growth_diagnostic,
        "--growth-window-fraction",
        str(case.growth_window_fraction),
    ]


def _read_case_outputs(
    case: W7XConvergenceCase,
    case_output: Path,
    wall_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    convergence = json.loads((case_output / "convergence_metadata.json").read_text())
    run_config = json.loads((case_output / "run_config.json").read_text())
    rows: list[dict[str, object]] = []
    with (case_output / "ky_growth.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "case": case.name,
                    "axis": case.axis,
                    "ky": float(row["ky"]),
                    "growth_rate": float(row["growth_rate"]),
                    "frequency": float(row["frequency"]),
                    "kperp2_average": float(row["kperp2_average"]),
                    "quasilinear_contribution": float(row["quasilinear_contribution"]),
                    "wall_seconds": float(wall_seconds),
                    "estimated_cfl_dt": float(convergence["estimated_cfl_dt"]),
                    "late_window_growth_delta": float(
                        convergence["max_abs_late_window_growth_delta"]
                    ),
                    "n_z": case.n_z,
                    "field_line_periods": case.field_line_periods,
                    "n_vpar": case.n_vpar,
                    "n_mu": case.n_mu,
                    "n_kx": case.n_kx,
                    "kx_max": case.kx_max,
                    "dt": case.dt,
                    "steps_per_window": case.steps_per_window,
                    "n_windows": case.n_windows,
                    "growth_diagnostic": case.growth_diagnostic,
                }
            )
    return rows, {
        "case": case.name,
        "axis": case.axis,
        "wall_seconds": float(wall_seconds),
        "finite_growth": bool(convergence["finite_growth"]),
        "finite_frequency": bool(convergence["finite_frequency"]),
        "estimated_cfl_dt": float(convergence["estimated_cfl_dt"]),
        "late_window_growth_delta": float(convergence["max_abs_late_window_growth_delta"]),
        "ky_values": run_config["ky_values"],
    }


def _write_convergence_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "case",
        "axis",
        "ky",
        "growth_rate",
        "frequency",
        "growth_delta_from_baseline",
        "frequency_delta_from_baseline",
        "kperp2_average",
        "quasilinear_contribution",
        "wall_seconds",
        "estimated_cfl_dt",
        "late_window_growth_delta",
        "n_z",
        "field_line_periods",
        "n_vpar",
        "n_mu",
        "n_kx",
        "kx_max",
        "dt",
        "steps_per_window",
        "n_windows",
        "growth_diagnostic",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# W7-X Reduced Convergence and Timing Study",
                "",
                "Regenerate from the repository root with:",
                "",
                "```bash",
                "uv run python scripts/run_w7x_reduced_convergence_study.py",
                "```",
                "",
                "This is a reduced solver-regression convergence matrix using the",
                "real GX/GIST W7-X eik table. It is not an external-code parity",
                "claim. The production W7-X comparison remains pending until",
                "`fixtures/w7x_itg_external_mode_structure_fixture.csv` is exported",
                "from a matched GX/GKW/GS2/stella run.",
                "",
            )
        )
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eik-reference", type=Path, default=DEFAULT_EIK_REFERENCE)
    parser.add_argument("--case", action="append", help="run only the named case")
    parser.add_argument("--keep-runs", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
