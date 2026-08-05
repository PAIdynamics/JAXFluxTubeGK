"""Replay traced stella W7-X states through the local solver RHS.

Unlike the evolved-state comparison, this diagnostic applies both operators to
the same interpolated distribution and potential.  It therefore isolates RHS
and quasineutrality conventions from eigenmode, phase, and growth-history
differences.  The large stella trace remains an explicit external input; only
compact comparison rows are written to the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from scripts.audit_w7x_ky03_rhs_model_balance import (
    DEFAULT_STELLA_GEOMETRY,
    RHSBalanceCase,
    _build_w7x_setup,
    split_rhs_terms,
)
from scripts.compare_w7x_stella_rhs_trace_to_solver_balance import (
    DEFAULT_STELLA_SUMMARY,
    _interpolate_complex_axis,
    interpolate_phase_space_to_grid,
    load_stella_array_trace,
    weighted_complex_metrics,
)
from stellarator_gk import linear_residual
from stellarator_gk.physics import adiabatic_density_numerator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_ky03_stella_state_replay"
DEFAULT_TOLERANCE = 0.1


def replay_cases() -> tuple[RHSBalanceCase, ...]:
    """Return grids fully contained in the traced stella velocity domain."""

    common = {
        "n_vpar": 16,
        "n_mu": 4,
        "velocity_backend": "finite_difference",
        "vpar_max": 3.2,
        "mu_max": 3.0,
    }
    return (
        RHSBalanceCase(name="replay_periodic_16x4", **common),
        RHSBalanceCase(
            name="replay_open_16x4",
            parallel_derivative_model="gkw_upwind",
            **common,
        ),
    )


def phase_space_to_solver(values: np.ndarray) -> jnp.ndarray:
    """Convert canonical ``(z,vpar,mu)`` data to solver phase-space order."""

    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError("phase-space replay input must have (z,vpar,mu) order")
    return jnp.asarray(np.transpose(array, (1, 2, 0))[..., None, None])


def selected_mode_from_solver(values: Any) -> np.ndarray:
    """Convert a one-mode solver array back to canonical trace order."""

    array = np.asarray(values)
    if array.ndim != 5 or array.shape[-2:] != (1, 1):
        raise ValueError("solver replay output must contain exactly one Fourier mode")
    return np.transpose(array[..., 0, 0], (2, 0, 1))


def bundled_solver_rhs(split: Any, total_rhs: Any) -> dict[str, np.ndarray]:
    """Map solver implementation terms to the semantic stella trace bundles."""

    terms = dict(zip(split.names, split.terms, strict=True))
    streaming_name = (
        "gkw_parallel_streaming_recurrence"
        if "gkw_parallel_streaming_recurrence" in terms
        else "parallel_streaming"
    )
    field_name = (
        "gkw_parallel_field_drive"
        if "gkw_parallel_field_drive" in terms
        else "parallel_field_drive"
    )
    return {
        "parallel_streaming": selected_mode_from_solver(
            terms[streaming_name] + terms[field_name]
        ),
        "mirror_force": selected_mode_from_solver(terms["mirror_force"]),
        "magnetic_drift": selected_mode_from_solver(
            terms["magnetic_drift"] + terms["drift_field_drive"]
        ),
        "equilibrium_drive": selected_mode_from_solver(terms["equilibrium_drive"]),
        "total_rhs": selected_mode_from_solver(total_rhs),
    }


def run_same_state_replay(
    *,
    trace_path: Path,
    stella_summary: Path = DEFAULT_STELLA_SUMMARY,
    stella_geometry: Path = DEFAULT_STELLA_GEOMETRY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, str]:
    """Replay every labeled stella state and write compact parity artifacts."""

    trace_path = Path(trace_path)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    summary = json.loads(Path(stella_summary).read_text(encoding="utf-8"))
    if summary.get("trace_format") != "stellarator_gk_stella_rhs_trace_v3":
        raise ValueError("same-state replay requires an explicitly labeled v3 trace")
    stella = load_stella_array_trace(trace_path, summary)

    rows: list[dict[str, Any]] = []
    for case in replay_cases():
        setup = _build_w7x_setup(case, Path(stella_geometry))
        target_z = np.asarray(setup["geometry"].z)
        target_vpar = np.asarray(setup["velocity"].vpar)
        target_mu = np.asarray(setup["velocity"].mu)
        weights = {
            "w_z": np.asarray(setup["geometry"].w_z),
            "w_vpar": np.asarray(setup["velocity"].w_vpar),
            "w_mu": np.asarray(setup["velocity"].w_mu),
        }

        for call in range(int(stella["rhs_call_count"])):
            mapped_state = interpolate_phase_space_to_grid(
                stella["distribution"][call],
                source_z=stella["z"],
                source_vpar=stella["vpar"],
                source_mu=stella["mu"],
                target_z=target_z,
                target_vpar=target_vpar,
                target_mu=target_mu,
            )
            state = phase_space_to_solver(mapped_state)
            phi_values = _interpolate_complex_axis(
                stella["phi"][call], stella["z"], target_z, axis=0
            )
            phi = jnp.asarray(phi_values[:, None, None])
            split = split_rhs_terms(state, phi, setup["precompute"].rhs)
            total_rhs = linear_residual(state, precomputed=setup["precompute"], phi=phi)
            candidate_arrays = bundled_solver_rhs(split, total_rhs)

            for quantity, candidate in candidate_arrays.items():
                reference = interpolate_phase_space_to_grid(
                    stella[quantity][call],
                    source_z=stella["z"],
                    source_vpar=stella["vpar"],
                    source_mu=stella["mu"],
                    target_z=target_z,
                    target_vpar=target_vpar,
                    target_mu=target_mu,
                )
                rows.append(
                    _comparison_row(
                        case=case,
                        rhs_call=call + 1,
                        quantity=quantity,
                        metrics=_fixed_and_best_fit_metrics(
                            reference, candidate, alignment_scale=1.0, **weights
                        ),
                    )
                )

            numerator = np.asarray(
                adiabatic_density_numerator(state, setup["precompute"].field)
            )[:, 0, 0]
            denominator = np.asarray(setup["precompute"].field.denominator)[:, 0, 0]
            for quantity, candidate, scale in (
                ("quasineutrality_numerator", numerator, 1.0),
                ("quasineutrality_denominator", denominator, -1.0),
            ):
                reference = _interpolate_complex_axis(
                    stella[quantity][call], stella["z"], target_z, axis=0
                )
                metrics = _fixed_and_best_fit_metrics(
                    reference[:, None, None],
                    candidate[:, None, None],
                    w_z=weights["w_z"],
                    w_vpar=[1.0],
                    w_mu=[1.0],
                    alignment_scale=scale,
                )
                rows.append(
                    _comparison_row(
                        case=case,
                        rhs_call=call + 1,
                        quantity=quantity,
                        metrics=metrics,
                        n_vpar=1,
                        n_mu=1,
                    )
                )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "same_state_rhs_comparison.csv"
    status_path = output_dir / "same_state_rhs_replay_status.json"
    readme_path = output_dir / "README.md"
    _write_csv(csv_path, rows)
    status = _status(rows, trace_path, Path(stella_geometry), tolerance)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(_fixture_readme(status), encoding="utf-8")
    return {"comparison_csv": str(csv_path), "status_json": str(status_path)}


def _comparison_row(
    *, case: RHSBalanceCase, rhs_call: int, quantity: str, metrics: dict[str, Any],
    n_vpar: int | None = None, n_mu: int | None = None,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "parallel_derivative_model": case.parallel_derivative_model,
        "rhs_call": rhs_call,
        "quantity": quantity,
        "target_n_z": 256,
        "target_n_vpar": case.n_vpar if n_vpar is None else n_vpar,
        "target_n_mu": case.n_mu if n_mu is None else n_mu,
        **metrics,
    }


def _fixed_and_best_fit_metrics(reference, candidate, *, alignment_scale, **weights):
    fixed = weighted_complex_metrics(
        reference, candidate, alignment_scale=alignment_scale, **weights
    )
    fitted = weighted_complex_metrics(reference, candidate, **weights)
    return {
        **fixed,
        "best_fit_relative_l2_error": fitted["aligned_relative_l2_error"],
        "best_fit_scale_real": fitted["alignment_scale_real"],
        "best_fit_scale_imag": fitted["alignment_scale_imag"],
    }


def _status(
    rows: list[dict[str, Any]], trace_path: Path, geometry_path: Path, tolerance: float
) -> dict[str, Any]:
    rhs_rows = [row for row in rows if not row["quantity"].startswith("quasineutrality")]
    best_by_quantity = {}
    for quantity in sorted({row["quantity"] for row in rows}):
        selected = min(
            (row for row in rows if row["quantity"] == quantity),
            key=lambda row: float(row["aligned_relative_l2_error"]),
        )
        best_by_quantity[quantity] = {
            "case": selected["case"],
            "rhs_call": selected["rhs_call"],
            "relative_l2_error": selected["aligned_relative_l2_error"],
            "best_fit_relative_l2_error": selected["best_fit_relative_l2_error"],
            "best_fit_scale_real": selected["best_fit_scale_real"],
            "best_fit_scale_imag": selected["best_fit_scale_imag"],
        }
    maximum = max(float(row["aligned_relative_l2_error"]) for row in rhs_rows)
    passed = maximum <= tolerance
    return {
        "schema": "stellarator_gk_w7x_same_state_rhs_replay_v1",
        "status": "same_state_rhs_parity_passed" if passed else "same_state_rhs_parity_failed",
        "passed": passed,
        "relative_l2_tolerance": tolerance,
        "max_rhs_relative_l2_error": maximum,
        "trace_source": str(trace_path),
        "geometry_source": str(geometry_path),
        "raw_trace_committed": False,
        "rhs_calls": sorted({int(row["rhs_call"]) for row in rows}),
        "cases": sorted({str(row["case"]) for row in rows}),
        "best_by_quantity": best_by_quantity,
        "interpretation": (
            "Each solver RHS is evaluated on the same stella distribution and phi "
            "after interpolation onto a contained 16x4 finite-difference velocity grid. "
            "No fitted amplitude or phase is used; the quasineutrality denominator alone "
            "uses the documented opposite-sign convention."
        ),
        "next_action": (
            "isolate the largest same-state term discrepancy before rerunning the "
            "mode-structure and production gates"
            if not passed
            else "rerun the matched mode-structure gate"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_readme(status: dict[str, Any]) -> str:
    return f"""# W7-X stella same-state RHS replay

This compact fixture records solver RHS operators applied directly to traced
stella distribution and potential arrays. The external raw trace is not stored
in the repository. Both periodic spectral and open GKW-upwind parallel models
are reported on a 16×4 velocity grid contained inside the stella domain.

Status: `{status['status']}`. Maximum RHS relative L2 error:
`{status['max_rhs_relative_l2_error']:.8g}` (tolerance `{status['relative_l2_tolerance']}`).
"""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--stella-summary", type=Path, default=DEFAULT_STELLA_SUMMARY)
    parser.add_argument("--stella-geometry", type=Path, default=DEFAULT_STELLA_GEOMETRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_same_state_replay(
        trace_path=args.trace,
        stella_summary=args.stella_summary,
        stella_geometry=args.stella_geometry,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
    )
    print(result["status_json"])
    print(result["comparison_csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
