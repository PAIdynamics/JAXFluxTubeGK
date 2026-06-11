"""Audit the stella data needed for W7-X ``ky=0.3`` RHS-term parity.

The existing matched stella run writes complex ``phi`` and geometry arrays, but
standard stella diagnostics do not write the complex distribution or per-term
RHS/source arrays needed for streaming/mirror parity.  This script compares the
available stella geometry/field-contract data against the solver-side
``ky=0.3`` RHS-balance fixture and writes a machine-readable contract for the
missing trace.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STELLA_OUTPUT = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.out.nc"
)
DEFAULT_SOLVER_BALANCE = ROOT / "fixtures/w7x_ky03_rhs_model_balance"
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_ky03_stella_trace_contract"
FOCUS_KY = 0.3
Z_SCALE = 1.0 / (2.0 * np.pi)
GEOMETRY_TOLERANCE = 5.0e-10
STELLA_GEOMETRY_FILE_ROUNDING_TOLERANCE = 7.0e-4


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = audit_w7x_stella_ky03_trace_contract(
        stella_output=args.stella_output,
        solver_balance_dir=args.solver_balance_dir,
        output_dir=args.output_dir,
        ky=args.ky,
        z_scale=Z_SCALE if args.stella_z_coordinate == "zed_over_2pi" else 1.0,
    )
    print(result["status_json"])
    print(result["geometry_comparison_csv"])
    return 0


def audit_w7x_stella_ky03_trace_contract(
    *,
    stella_output: Path,
    solver_balance_dir: Path,
    output_dir: Path,
    ky: float = FOCUS_KY,
    z_scale: float = Z_SCALE,
) -> dict[str, object]:
    """Compare available stella data to solver balance and write trace contract."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stella = load_standard_stella_trace_summary(stella_output, ky=ky, z_scale=z_scale)
    solver_geometry = _read_csv_dicts(solver_balance_dir / "geometry_model_balance.csv")
    solver_terms = _read_csv_dicts(solver_balance_dir / "rhs_term_balance.csv")
    solver_status = _read_json(solver_balance_dir / "rhs_model_balance_status.json")
    comparison_rows = compare_stella_geometry_to_solver(stella, solver_geometry)
    geometry_summary = summarize_geometry_comparison(comparison_rows)
    availability = build_trace_availability(stella)
    dominant_terms = _dominant_solver_terms(solver_terms)
    status = build_status_payload(
        stella_output=stella_output,
        solver_balance_dir=solver_balance_dir,
        comparison_rows=comparison_rows,
        geometry_summary=geometry_summary,
        availability=availability,
        dominant_terms=dominant_terms,
        solver_status=solver_status,
        output_dir=output_dir,
        ky=ky,
    )

    geometry_csv = output_dir / "stella_solver_geometry_comparison.csv"
    availability_json = output_dir / "stella_standard_trace_availability.json"
    status_json = output_dir / "stella_ky03_trace_contract_status.json"
    patch_plan = output_dir / "stella_rhs_trace_patch_plan.md"
    _write_csv(geometry_csv, comparison_rows)
    _write_json(availability_json, availability)
    _write_json(status_json, status)
    patch_plan.write_text(_patch_plan_text(status), encoding="utf-8")
    _write_readme(output_dir)
    return {
        "status_json": str(status_json),
        "availability_json": str(availability_json),
        "geometry_comparison_csv": str(geometry_csv),
        "patch_plan": str(patch_plan),
        "status": status,
    }


def load_standard_stella_trace_summary(path: Path, *, ky: float, z_scale: float) -> dict[str, object]:
    """Load available standard stella ``.out.nc`` arrays for the focused mode."""

    dataset_cls = _import_netcdf_dataset()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with dataset_cls(path, mode="r") as data:
        variables = data.variables
        variable_names = tuple(sorted(variables))
        required = ("ky", "kx", "zed", "bmag", "b_dot_gradz", "kperp2", "phi_vs_t")
        missing = tuple(name for name in required if name not in variables)
        if missing:
            raise ValueError(f"stella output missing required variables: {missing}")
        ky_values = np.asarray(variables["ky"][:], dtype=float)
        kx_values = np.asarray(variables["kx"][:], dtype=float)
        ky_index = int(np.argmin(np.abs(ky_values - ky)))
        if abs(float(ky_values[ky_index]) - ky) > 1.0e-8:
            raise ValueError(f"requested ky={ky} was not found in stella output")
        kx_index = int(np.argmin(np.abs(kx_values)))
        zed = np.asarray(variables["zed"][:], dtype=float)
        z = z_scale * zed
        arrays = {
            "z": z,
            "zed": zed,
            "bmag": _alpha0(variables["bmag"]),
            "b_dot_gradz": _alpha0(variables["b_dot_gradz"]),
            "kperp2": np.asarray(variables["kperp2"][:, 0, kx_index, ky_index], dtype=float),
            "phi": _complex_phi(variables["phi_vs_t"], kx_index=kx_index, ky_index=ky_index),
        }
        optional_1d = (
            "B_times_gradB_dot_grady",
            "B_times_kappa_dot_grady",
            "B_times_gradB_dot_gradx",
            "B_times_kappa_dot_gradx",
            "grady_dot_grady",
            "gradx_dot_grady",
            "gradx_dot_gradx",
        )
        for name in optional_1d:
            if name in variables:
                arrays[name] = _alpha0(variables[name])
        optional_shapes = {
            name: tuple(int(size) for size in variables[name].shape)
            for name in variable_names
            if name.startswith(("g2", "h2", "f2")) or name in ("density", "upar", "temperature")
        }
        time = np.asarray(variables["t"][:], dtype=float) if "t" in variables else np.asarray([])
    arrays = _drop_duplicate_endpoint(arrays)
    return {
        "path": str(path),
        "ky": float(ky_values[ky_index]),
        "ky_index_python": ky_index,
        "ky_index_fortran": ky_index + 1,
        "kx": float(kx_values[kx_index]),
        "kx_index_python": kx_index,
        "kx_index_fortran": kx_index + 1,
        "time_final": None if time.size == 0 else float(time[-1]),
        "variable_names": variable_names,
        "optional_trace_shapes": optional_shapes,
        "arrays": arrays,
    }


def compare_stella_geometry_to_solver(
    stella: dict[str, object],
    solver_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Compare focused stella geometry arrays with solver-balance rows."""

    arrays = stella["arrays"]
    z = np.asarray(arrays["z"], dtype=float)
    solver = [_float_row(row) for row in solver_rows if abs(float(row["ky"]) - stella["ky"]) <= 1.0e-12]
    if len(solver) != z.shape[0]:
        raise ValueError(
            f"solver/stella z-size mismatch after endpoint handling: {len(solver)} vs {z.shape[0]}"
        )
    rows = []
    for index, row in enumerate(solver):
        stella_f = float(arrays["b_dot_gradz"][index]) / (2.0 * np.pi)
        stella_gradb_grady = _optional_array_value(arrays, "B_times_gradB_dot_grady", index)
        stella_kappa_grady = _optional_array_value(arrays, "B_times_kappa_dot_grady", index)
        stella_drift_y_sum = (
            ""
            if stella_gradb_grady == "" or stella_kappa_grady == ""
            else stella_gradb_grady + stella_kappa_grady
        )
        rows.append(
            {
                "z_index": index,
                "solver_z": row["z"],
                "stella_z": float(z[index]),
                "z_error": float(z[index]) - row["z"],
                "solver_B": row["B"],
                "stella_bmag": float(arrays["bmag"][index]),
                "B_error": float(arrays["bmag"][index]) - row["B"],
                "solver_F": row["F"],
                "stella_b_dot_gradz_over_2pi": stella_f,
                "F_error": stella_f - row["F"],
                "solver_kperp2": row["kperp2"],
                "stella_kperp2": float(arrays["kperp2"][index]),
                "kperp2_error": float(arrays["kperp2"][index]) - row["kperp2"],
                "solver_D_y": row["D_y"],
                "stella_B_times_gradB_dot_grady": stella_gradb_grady,
                "stella_B_times_kappa_dot_grady": stella_kappa_grady,
                "stella_B_times_drift_y_sum": stella_drift_y_sum,
                "phi_abs": float(abs(arrays["phi"][index])),
            }
        )
    return rows


def summarize_geometry_comparison(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return max absolute errors for direct geometry comparisons."""

    summary = {
        "max_abs_z_error": _max_abs_column(rows, "z_error"),
        "max_abs_B_error": _max_abs_column(rows, "B_error"),
        "max_abs_F_error": _max_abs_column(rows, "F_error"),
        "max_abs_kperp2_error": _max_abs_column(rows, "kperp2_error"),
    }
    summary["strict_out_nc_precision_contract_passed"] = all(
        summary[key] <= GEOMETRY_TOLERANCE
        for key in (
            "max_abs_z_error",
            "max_abs_B_error",
            "max_abs_F_error",
            "max_abs_kperp2_error",
        )
    )
    summary["stella_geometry_file_rounding_contract_passed"] = all(
        summary[key] <= STELLA_GEOMETRY_FILE_ROUNDING_TOLERANCE
        for key in (
            "max_abs_z_error",
            "max_abs_B_error",
            "max_abs_F_error",
            "max_abs_kperp2_error",
        )
    )
    summary["direct_geometry_contract_passed"] = summary[
        "stella_geometry_file_rounding_contract_passed"
    ]
    summary["strict_out_nc_precision_tolerance"] = GEOMETRY_TOLERANCE
    summary["stella_geometry_file_rounding_tolerance"] = STELLA_GEOMETRY_FILE_ROUNDING_TOLERANCE
    return summary


def build_trace_availability(stella: dict[str, object]) -> dict[str, object]:
    """Describe standard stella trace availability and true parity blockers."""

    variables = set(stella["variable_names"])
    distribution_energy = sorted(name for name in variables if name.startswith(("g2", "h2", "f2")))
    missing = (
        "complex_g(vpa,mu,z,kx,ky,species)",
        "rhs_total(vpa,mu,z,kx,ky,species,ri)",
        "rhs_parallel_streaming(vpa,mu,z,kx,ky,species,ri)",
        "rhs_mirror_force(vpa,mu,z,kx,ky,species,ri)",
        "rhs_magnetic_drift(vpa,mu,z,kx,ky,species,ri)",
        "rhs_field_drive(vpa,mu,z,kx,ky,species,ri)",
    )
    return {
        "standard_output_has_complex_phi": "phi_vs_t" in variables,
        "standard_output_has_geometry": all(
            name in variables for name in ("bmag", "b_dot_gradz", "kperp2")
        ),
        "standard_output_distribution_energy_variables": distribution_energy,
        "standard_output_optional_trace_shapes": stella["optional_trace_shapes"],
        "standard_output_has_complex_distribution": False,
        "standard_output_has_per_term_rhs": False,
        "missing_for_true_term_parity": missing,
        "usable_now": (
            "geometry/streaming coefficient convention audit",
            "complex phi phase/profile comparison",
            "phase-free distribution-energy profile checks if needed",
        ),
    }


def build_status_payload(
    *,
    stella_output: Path,
    solver_balance_dir: Path,
    comparison_rows: list[dict[str, object]],
    geometry_summary: dict[str, object],
    availability: dict[str, object],
    dominant_terms: list[dict[str, object]],
    solver_status: dict[str, object],
    output_dir: Path,
    ky: float,
) -> dict[str, object]:
    """Build the machine-readable trace-contract status."""

    missing_term_trace = not availability["standard_output_has_complex_distribution"] or not availability[
        "standard_output_has_per_term_rhs"
    ]
    return {
        "benchmark_name": "w7x_ky03_stella_trace_contract",
        "status": (
            "blocked_missing_complex_stella_rhs_trace"
            if missing_term_trace
            else "ready_for_term_parity"
        ),
        "passed": (not missing_term_trace) and bool(geometry_summary["direct_geometry_contract_passed"]),
        "focus_ky": float(ky),
        "stella_output": _display_path(stella_output),
        "solver_balance_dir": _display_path(solver_balance_dir),
        "geometry_summary": geometry_summary,
        "availability": availability,
        "dominant_solver_rhs_terms": dominant_terms,
        "solver_rhs_reconstruction_max_abs_error": solver_status.get(
            "max_rhs_reconstruction_abs_error"
        ),
        "comparison_row_count": len(comparison_rows),
        "patch_plan": _display_path(output_dir / "stella_rhs_trace_patch_plan.md"),
        "next_action": (
            "rerun stella with a targeted complex distribution/RHS trace or apply the "
            "patch plan around add_explicit_gyrokinetic_terms, then compare the "
            "streaming-dominated ky=0.3 term balance against the solver fixture"
        ),
    }


def _patch_plan_text(status: dict[str, object]) -> str:
    missing = "\n".join(f"- `{name}`" for name in status["availability"]["missing_for_true_term_parity"])
    return f"""# stella W7-X ky=0.3 RHS Trace Patch Plan

The standard matched stella `.out.nc` file was audited against
`fixtures/w7x_ky03_rhs_model_balance/`.  The solver fixture was built from
stella's rounded ASCII `.geometry` file, while this audit reads full-precision
arrays from `.out.nc`.  The practical geometry-file precision contract passes,
while strict `.out.nc` precision does not:

- max `z` error: `{status["geometry_summary"]["max_abs_z_error"]}`
- max `B` error: `{status["geometry_summary"]["max_abs_B_error"]}`
- max `F=b_dot_gradz/(2*pi)` error: `{status["geometry_summary"]["max_abs_F_error"]}`
- max `kperp2` error: `{status["geometry_summary"]["max_abs_kperp2_error"]}`
- stella `.geometry` rounding tolerance:
  `{status["geometry_summary"]["stella_geometry_file_rounding_tolerance"]}`
- strict `.out.nc` tolerance:
  `{status["geometry_summary"]["strict_out_nc_precision_tolerance"]}`

The standard file does not contain the complex arrays required for true term
parity:

{missing}

Minimal stella-side insertion points:

1. In `STELLA_CODE/gyrokinetic_equation/gyrokinetic_equation_explicit.f90`,
   inside `add_explicit_gyrokinetic_terms`, snapshot `rhs` immediately before
   and after these calls:
   - `advance_mirror_explicit(pdf, rhs)`
   - `advance_wdrifty_explicit(pdf, phi, bpar, rhs)`
   - `advance_wdriftx_explicit(pdf, phi, bpar, rhs)`
   - `advance_wstar_explicit(phi, rhs)`
   - `advance_parallel_streaming_explicit(pdf, phi, bpar, rhs)`
2. Write the selected serial-run arrays for Fortran indices `iky=4`, `ikx=1`,
   all `z`, all `vpa`, all `mu`, species 1, at the final or requested trace
   step.  Store each delta as real/imag columns with term names.
3. Also write the input `pdf` state used by the RHS call and the solved `phi`.
4. Keep the trace units as stella's native `rhs*dt`; the Python comparator
   should divide by `delt` only if comparing against a continuous-time RHS.

This patch plan intentionally does not reinterpret stella's `|g|^2` diagnostics
as a complex distribution.  The solver-side balance is streaming dominated, so
the first true comparison should prioritize the streaming and mirror deltas.
"""


def _alpha0(variable):
    values = np.asarray(variable[:], dtype=float)
    if values.ndim == 2:
        return values[:, 0]
    if values.ndim == 1:
        return values
    raise ValueError(f"expected 1D or z/alpha variable, got shape {values.shape}")


def _complex_phi(variable, *, kx_index: int, ky_index: int):
    values = np.asarray(variable[:], dtype=float)
    if values.ndim != 6 or values.shape[-1] != 2:
        raise ValueError("stella phi_vs_t must have shape (t,tube,zed,kx,ky,ri)")
    return values[-1, 0, :, kx_index, ky_index, 0] + 1j * values[-1, 0, :, kx_index, ky_index, 1]


def _drop_duplicate_endpoint(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    z = np.asarray(arrays["z"], dtype=float)
    if z.shape[0] < 3:
        return arrays
    span = abs((z[-1] - z[0]) - 1.0)
    contract_names = ("bmag", "b_dot_gradz", "kperp2")
    periodic_like = all(
        np.isclose(np.asarray(values)[0], np.asarray(values)[-1], rtol=1.0e-8, atol=1.0e-10)
        for name, values in arrays.items()
        if name in contract_names and np.asarray(values).ndim == 1
    )
    if span <= 1.0e-8 and periodic_like:
        return {name: np.asarray(values)[:-1] for name, values in arrays.items()}
    return arrays


def _optional_array_value(arrays: dict[str, np.ndarray], name: str, index: int):
    if name not in arrays:
        return ""
    return float(np.asarray(arrays[name])[index])


def _float_row(row: dict[str, str]) -> dict[str, float]:
    converted = {}
    for key, value in row.items():
        try:
            converted[key] = float(value)
        except ValueError:
            pass
    return converted


def _dominant_solver_terms(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: float(row["rhs_fraction_of_total_l2"]),
        reverse=True,
    )
    return [
        {
            "term": row["term"],
            "rhs_fraction_of_total_l2": float(row["rhs_fraction_of_total_l2"]),
            "projection_real": float(row["projection_real"]),
            "projection_imag": float(row["projection_imag"]),
        }
        for row in sorted_rows[:4]
    ]


def _max_abs_column(rows: list[dict[str, object]], name: str) -> float:
    return float(max(abs(float(row[name])) for row in rows)) if rows else float("nan")


def _import_netcdf_dataset():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("netCDF4 is required to read stella .out.nc files") from exc
    return Dataset


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            (
                "# W7-X ky=0.3 stella trace contract",
                "",
                "This fixture audits what the standard matched stella `.out.nc`",
                "output can compare against the solver-side RHS balance.",
                "",
                "The direct geometry, streaming multiplier, and `kperp2` contract",
                "is compared in `stella_solver_geometry_comparison.csv`.",
                "The status JSON records that standard stella diagnostics do not",
                "contain the complex distribution or per-term RHS/source arrays",
                "needed for true streaming/mirror parity.",
                "",
                "Use `stella_rhs_trace_patch_plan.md` as the next stella-side",
                "diagnostic target.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella-output", type=Path, default=DEFAULT_STELLA_OUTPUT)
    parser.add_argument("--solver-balance-dir", type=Path, default=DEFAULT_SOLVER_BALANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ky", type=float, default=FOCUS_KY)
    parser.add_argument(
        "--stella-z-coordinate",
        choices=("zed", "zed_over_2pi"),
        default="zed_over_2pi",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
