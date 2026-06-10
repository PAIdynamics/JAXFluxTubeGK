"""Audit W7-X solver-vs-stella parity blockers in a fixed debugging order."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STELLA_RUN_DIR = ROOT / "fixtures/stella_w7x_mode_structure_run"
DEFAULT_STELLA_METADATA = DEFAULT_STELLA_RUN_DIR / "mode_structure_run_metadata.json"
DEFAULT_STELLA_GEOMETRY = DEFAULT_STELLA_RUN_DIR / "stella_w7x_adiabatic_electrons.geometry"
DEFAULT_SOLVER_CONFIG = ROOT / "fixtures/w7x_itg_reduced_benchmark/run_config.json"
DEFAULT_SOLVER_METADATA = ROOT / "fixtures/w7x_itg_reduced_benchmark/benchmark_metadata.json"
DEFAULT_SOLVER_FIXTURE = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"
DEFAULT_STELLA_FIXTURE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_GATE_STATUS = (
    ROOT / "fixtures/w7x_itg_convergence_study/external_mode_structure_gate/gate_status.json"
)
DEFAULT_OUTPUT = ROOT / "fixtures/w7x_itg_convergence_study/stella_solver_parity_audit.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_w7x_stella_solver_parity_audit(
        stella_metadata=args.stella_metadata,
        stella_geometry=args.stella_geometry,
        solver_config=args.solver_config,
        solver_metadata=args.solver_metadata,
        solver_fixture=args.solver_fixture,
        stella_fixture=args.stella_fixture,
        gate_status=args.gate_status,
        output=args.output,
    )
    status = "PASS" if report["passed"] else "OPEN"
    print(
        f"{status}: first_failed_check={report['first_failed_check']} "
        f"max_growth_error={report['gate'].get('max_growth_error')}"
    )
    print(_display_path(args.output))
    return 0


def run_w7x_stella_solver_parity_audit(
    *,
    stella_metadata: Path = DEFAULT_STELLA_METADATA,
    stella_geometry: Path = DEFAULT_STELLA_GEOMETRY,
    solver_config: Path = DEFAULT_SOLVER_CONFIG,
    solver_metadata: Path = DEFAULT_SOLVER_METADATA,
    solver_fixture: Path = DEFAULT_SOLVER_FIXTURE,
    stella_fixture: Path = DEFAULT_STELLA_FIXTURE,
    gate_status: Path = DEFAULT_GATE_STATUS,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Write and return the ordered solver-vs-stella parity audit report."""

    metadata = _load_json(stella_metadata)
    config = _load_json(solver_config)
    solver_metadata_payload = _load_json(solver_metadata) if solver_metadata.exists() else {}
    gate = _load_json(gate_status) if gate_status.exists() else {}
    solver_rows = _load_fixture_summary(solver_fixture)
    stella_rows = _load_fixture_summary(stella_fixture)
    stella_geom = _load_stella_geometry_summary(stella_geometry)

    solver_geometry = config.get("geometry", {})
    solver_benchmark_controls = solver_metadata_payload.get(
        "reduced_solver_controls",
        solver_metadata_payload,
    )
    solver_controls = {
        "n_z": solver_geometry.get("n_z"),
        "field_line_periods": solver_geometry.get("field_line_periods"),
        "n_kx": config.get("n_kx"),
        "ikxspace": config.get("ikxspace"),
        "dt": _first_optional_float(config, solver_benchmark_controls, "dt"),
        "steps_per_window": _first_optional_int(config, solver_benchmark_controls, "steps_per_window"),
        "n_windows": _first_optional_int(config, solver_benchmark_controls, "n_windows"),
        "growth_window_fraction": _first_optional_float(
            config,
            solver_benchmark_controls,
            "growth_window_fraction",
        ),
    }
    stella_grid = metadata.get("grid", {})
    stella_time = metadata.get("time", {})
    stella_nfield_periods = float(metadata.get("nfield_periods", math.nan))
    stella_nfp = float(metadata.get("nfp", math.nan))
    stella_zeta_turns_from_metadata = stella_nfield_periods / stella_nfp
    solver_total_time = _total_time_from_controls(solver_controls)
    stella_total_time = float(stella_time.get("tend", math.nan))

    checks = [
        _check_z_coordinate_convention(solver_rows, stella_rows, stella_geom),
        _check_field_line_length(
            solver_controls,
            stella_geom,
            stella_zeta_turns_from_metadata,
        ),
        _check_ky_normalization(solver_rows, stella_rows, stella_grid),
        _check_twist_and_shift_linking(solver_controls, stella_grid),
        _check_growth_window_time_normalization(
            solver_controls,
            solver_total_time,
            stella_time,
            stella_total_time,
        ),
        _check_field_normalization(solver_rows, stella_rows, gate),
        _check_velocity_rhs_terms(),
    ]
    first_failed = next((item["name"] for item in checks if not item["passed"]), None)
    report = {
        "benchmark_name": "w7x_stella_solver_parity_audit",
        "status": "pass" if first_failed is None else "open",
        "passed": first_failed is None,
        "first_failed_check": first_failed,
        "ordered_checks": checks,
        "solver": {
            "config": _display_path(solver_config),
            "metadata": _display_path(solver_metadata),
            "fixture": _display_path(solver_fixture),
            "controls": solver_controls,
            "total_time": solver_total_time,
            "z": solver_rows["z"],
            "ky": solver_rows["ky"],
            "normalization": solver_rows["normalization"],
        },
        "stella": {
            "metadata": _display_path(stella_metadata),
            "geometry": _display_path(stella_geometry),
            "fixture": _display_path(stella_fixture),
            "nfield_periods": stella_nfield_periods,
            "nfp": stella_nfp,
            "zeta_turns_from_metadata": stella_zeta_turns_from_metadata,
            "zeta_turns_from_geometry": stella_geom["zeta_turns"],
            "grid": stella_grid,
            "time": stella_time,
            "z": stella_rows["z"],
            "ky": stella_rows["ky"],
            "normalization": stella_rows["normalization"],
        },
        "gate": gate,
        "next_action": _next_action(first_failed),
    }
    _write_json(output, report)
    return report


def _check_z_coordinate_convention(
    solver_fixture: dict[str, Any],
    stella_fixture: dict[str, Any],
    stella_geometry: dict[str, Any],
) -> dict[str, Any]:
    solver_z = solver_fixture["z"]
    stella_z = stella_fixture["z"]
    solver_span = solver_z["max"] - solver_z["min"]
    normalized = (
        abs(stella_z["min"] + 0.5) <= 1.0e-12
        and abs(stella_z["max"] - 0.5) <= 1.0e-12
        and abs(solver_span - 1.0) <= 1.0 / max(1, solver_z["count"])
    )
    return {
        "name": "eik_z_coordinate_convention",
        "passed": bool(normalized),
        "status": "pass" if normalized else "open",
        "solver_z_min": solver_z["min"],
        "solver_z_max": solver_z["max"],
        "stella_fixture_z_min": stella_z["min"],
        "stella_fixture_z_max": stella_z["max"],
        "stella_geometry_zed_min": stella_geometry["zed"]["min"],
        "stella_geometry_zed_max": stella_geometry["zed"]["max"],
        "notes": (
            "The stella fixture must be exported with zed_over_2pi so both "
            "fixtures use normalized z. Endpoint policy may still differ: "
            "the solver eik grid excludes the upper endpoint while stella "
            "writes both -pi and +pi."
        ),
    }


def _check_field_line_length(
    solver_controls: dict[str, Any],
    stella_geometry: dict[str, Any],
    stella_zeta_turns: float,
) -> dict[str, Any]:
    solver_periods = float(solver_controls.get("field_line_periods") or math.nan)
    geometry_turns = float(stella_geometry["zeta_turns"])
    target_turns = geometry_turns if math.isfinite(geometry_turns) else stella_zeta_turns
    error = abs(solver_periods - target_turns)
    passed = error <= 5.0e-2
    return {
        "name": "field_line_length",
        "passed": bool(passed),
        "status": "pass" if passed else "open",
        "solver_field_line_periods": solver_periods,
        "stella_zeta_turns_from_geometry": geometry_turns,
        "stella_zeta_turns_from_metadata": stella_zeta_turns,
        "absolute_turn_error": error,
        "notes": (
            "The committed reduced solver fixture uses a short eik segment. "
            "The matched stella VMEC run spans nfield_periods/nfp toroidal "
            "turns along zeta; compare against a solver geometry sampled on "
            "the same field-line length before changing RHS physics."
        ),
    }


def _check_ky_normalization(
    solver_fixture: dict[str, Any],
    stella_fixture: dict[str, Any],
    stella_grid: dict[str, Any],
) -> dict[str, Any]:
    solver_ky = tuple(float(value) for value in solver_fixture["ky"])
    stella_ky = tuple(float(value) for value in stella_fixture["ky"])
    export_ky = tuple(float(value) for value in stella_grid.get("export_ky_values", ()))
    solver_selected = solver_ky[-len(stella_ky) :]
    passed = _tuple_allclose(solver_selected, stella_ky) and _tuple_allclose(stella_ky, export_ky)
    return {
        "name": "ky_normalization",
        "passed": bool(passed),
        "status": "pass" if passed else "open",
        "solver_ky": solver_ky,
        "stella_fixture_ky": stella_ky,
        "stella_export_ky": export_ky,
        "notes": (
            "The selected ky labels match numerically. This does not prove "
            "the same rho_i/a or flux-surface normalization, but it removes "
            "a simple wrong-mode comparison."
        ),
    }


def _check_twist_and_shift_linking(
    solver_controls: dict[str, Any],
    stella_grid: dict[str, Any],
) -> dict[str, Any]:
    solver_n_kx = int(solver_controls.get("n_kx") or -1)
    stella_nakx = int(stella_grid.get("nakx") or -1)
    passed = solver_n_kx == stella_nakx == 1
    return {
        "name": "twist_and_shift_linking",
        "passed": bool(passed),
        "status": "pass" if passed else "open",
        "solver_n_kx": solver_n_kx,
        "solver_ikxspace": solver_controls.get("ikxspace"),
        "stella_nakx": stella_nakx,
        "notes": (
            "The stella reference is a kx=0, nakx=1 run. The committed reduced "
            "solver fixture uses a linked three-kx chain, so the current "
            "profile gate mixes boundary/connectivity effects into the "
            "stella comparison."
        ),
    }


def _check_growth_window_time_normalization(
    solver_controls: dict[str, Any],
    solver_total_time: float,
    stella_time: dict[str, Any],
    stella_total_time: float,
) -> dict[str, Any]:
    fraction = float(stella_time.get("growth_average_fraction", math.nan))
    solver_fraction = float(solver_controls.get("growth_window_fraction") or math.nan)
    time_ratio = solver_total_time / stella_total_time
    passed = (
        math.isfinite(time_ratio)
        and time_ratio >= 0.5
        and abs(solver_fraction - fraction) <= 1.0e-12
    )
    return {
        "name": "growth_window_time_normalization",
        "passed": bool(passed),
        "status": "pass" if passed else "open",
        "solver_total_time": solver_total_time,
        "stella_total_time": stella_total_time,
        "solver_growth_window_fraction": solver_fraction,
        "stella_growth_average_fraction": fraction,
        "solver_to_stella_total_time_ratio": time_ratio,
        "notes": (
            "The reduced solver fixture is a very short regression run. "
            "Growth/frequency parity requires a solver trace long enough to "
            "use the same late-half window as stella."
        ),
    }


def _check_field_normalization(
    solver_fixture: dict[str, Any],
    stella_fixture: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    phase_aligned_profile_gate = bool(gate.get("require_profile", False))
    return {
        "name": "field_normalization",
        "passed": bool(phase_aligned_profile_gate),
        "status": "pass" if phase_aligned_profile_gate else "open",
        "solver_normalization": solver_fixture["normalization"],
        "stella_normalization": stella_fixture["normalization"],
        "max_profile_error": gate.get("max_profile_error"),
        "notes": (
            "The scalar amplitude normalizations differ by construction, but "
            "the gate uses row-normalized, phase-aligned complex phi(z). "
            "Raw amplitude parity should wait until coordinate, length, kx, "
            "and time-window controls are matched."
        ),
    }


def _check_velocity_rhs_terms() -> dict[str, Any]:
    return {
        "name": "velocity_rhs_terms",
        "passed": False,
        "status": "not_evaluated",
        "notes": (
            "Not evaluated in this ordered audit. Do not tune velocity-space "
            "or RHS terms until coordinate, field-line length, ky, kx/linking, "
            "growth-window, and field-normalization checks pass."
        ),
    }


def _load_fixture_summary(path: Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no fixture rows")
    ky = sorted({float(row["ky"]) for row in rows})
    z = [float(row["z"]) for row in rows if float(row["ky"]) == ky[0]]
    return {
        "path": _display_path(path),
        "ky": ky,
        "z": {
            "count": len(z),
            "min": min(z),
            "max": max(z),
            "span": max(z) - min(z),
        },
        "normalization": rows[0].get("normalization", ""),
        "source": rows[0].get("source", ""),
    }


def _load_stella_geometry_summary(path: Path) -> dict[str, Any]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = [float(item) for item in stripped.split()]
            if len(values) >= 12:
                rows.append(values)
    if not rows:
        raise ValueError(f"{path} contains no stella geometry rows")
    zed = [row[1] for row in rows]
    zeta = [row[2] for row in rows]
    return {
        "path": _display_path(path),
        "row_count": len(rows),
        "zed": {
            "min": min(zed),
            "max": max(zed),
            "span": max(zed) - min(zed),
        },
        "zeta": {
            "min": min(zeta),
            "max": max(zeta),
            "span": max(zeta) - min(zeta),
        },
        "zeta_turns": (max(zeta) - min(zeta)) / (2.0 * math.pi),
    }


def _total_time_from_controls(controls: dict[str, Any]) -> float:
    dt = controls.get("dt")
    steps = controls.get("steps_per_window")
    windows = controls.get("n_windows")
    if dt is None or steps is None or windows is None:
        return math.nan
    return float(dt) * int(steps) * int(windows)


def _load_optional_float(mapping: dict[str, Any], name: str) -> float | None:
    value = mapping.get(name)
    return None if value is None else float(value)


def _load_optional_int(mapping: dict[str, Any], name: str) -> int | None:
    value = mapping.get(name)
    return None if value is None else int(value)


def _first_optional_float(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    name: str,
) -> float | None:
    value = primary.get(name, fallback.get(name))
    return None if value is None else float(value)


def _first_optional_int(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    name: str,
) -> int | None:
    value = primary.get(name, fallback.get(name))
    return None if value is None else int(value)


def _tuple_allclose(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= 1.0e-10 for a, b in zip(left, right, strict=True))


def _next_action(first_failed: str | None) -> str:
    if first_failed is None:
        return "run the production W7-X solver-vs-stella parity gate"
    if first_failed == "field_line_length":
        return (
            "build a solver observed fixture from stella-exported geometry or "
            "from an eik table sampled on the same field-line length before "
            "changing RHS physics"
        )
    if first_failed == "twist_and_shift_linking":
        return "rerun the observed solver fixture with n_kx=1, kx_max=0, and no linked kx chain"
    if first_failed == "growth_window_time_normalization":
        return "rerun the observed solver trace to a stella-comparable late-time window"
    return f"resolve {first_failed}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella-metadata", type=Path, default=DEFAULT_STELLA_METADATA)
    parser.add_argument("--stella-geometry", type=Path, default=DEFAULT_STELLA_GEOMETRY)
    parser.add_argument("--solver-config", type=Path, default=DEFAULT_SOLVER_CONFIG)
    parser.add_argument("--solver-metadata", type=Path, default=DEFAULT_SOLVER_METADATA)
    parser.add_argument("--solver-fixture", type=Path, default=DEFAULT_SOLVER_FIXTURE)
    parser.add_argument("--stella-fixture", type=Path, default=DEFAULT_STELLA_FIXTURE)
    parser.add_argument("--gate-status", type=Path, default=DEFAULT_GATE_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
