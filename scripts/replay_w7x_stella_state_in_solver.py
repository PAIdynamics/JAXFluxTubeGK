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
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from scripts.audit_w7x_ky03_rhs_model_balance import (
    DEFAULT_STELLA_GEOMETRY,
    FOCUS_KY,
    RHSBalanceCase,
    RHSTermSplit,
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
from stellarator_gk import (
    AdiabaticElectronParams,
    SpeciesParams,
    build_linear_residual_precompute,
    build_velocity_grid_from_nodes,
    linear_residual,
)
from stellarator_gk.geometry import load_stella_geometry_data
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
        RHSBalanceCase(
            name="replay_stella_coefficients_16x4",
            parallel_derivative_model="gkw_upwind",
            **common,
        ),
        RHSBalanceCase(
            name="replay_stella_coefficients_32x4",
            n_vpar=32,
            n_mu=4,
            velocity_backend="finite_difference",
            vpar_max=96.0 / 31.0,
            mu_max=3.0,
            parallel_derivative_model="gkw_upwind",
        ),
        RHSBalanceCase(
            name="replay_stella_native_32x8",
            n_vpar=32,
            n_mu=8,
            velocity_backend="finite_difference",
            vpar_max=96.0 / 31.0,
            mu_max=5.0,
            parallel_derivative_model="gkw_upwind",
        ),
    )


def apply_stella_coefficient_contract(case, precompute, stella_geometry: Path):
    """Apply source-derived stella coefficient conventions for discrimination."""

    if not case.name.startswith(("replay_stella_coefficients_", "replay_stella_native_")):
        return precompute
    geometry_data = load_stella_geometry_data(stella_geometry)
    flux_fac = geometry_data.global_value("flux_fac")
    rhs = replace(
        precompute.rhs,
        E_y=jnp.full_like(precompute.rhs.E_y, flux_fac),
        # stella puts normalization in its velocity quadrature and stores the
        # Maxwellian factor as exp(-v^2); the solver stores pi^(-3/2) in F_M.
        maxwellian=np.pi**1.5 * precompute.rhs.maxwellian,
    )
    return replace(precompute, rhs=rhs)


def build_native_stella_setup(case, stella_geometry: Path, stella: dict[str, Any]):
    """Build the standard W7-X setup on traced native velocity nodes and measure."""

    setup = _build_w7x_setup(case, stella_geometry)
    velocity = build_velocity_grid_from_nodes(
        vpar=stella["vpar"],
        mu=stella["mu"],
        w_vpar=stella["w_vpar"],
        w_mu=np.mean(stella["w_mu"], axis=0),
    )
    species = SpeciesParams(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        density_gradient=1.0,
        temperature_gradient=3.0,
    )
    electrons = AdiabaticElectronParams(
        density=1.0,
        temperature=1.0,
        zonal_correction=False,
    )
    measure = (
        np.asarray(stella["w_vpar"])[None, :, None]
        * np.asarray(stella["w_mu"])[:, None, :]
    )
    precompute = build_linear_residual_precompute(
        velocity,
        setup["parallel"],
        setup["fourier"],
        setup["geometry"],
        species,
        electron_params=electrons,
        mode_connectivity=setup["connectivity"],
        parallel_derivative_model=case.parallel_derivative_model,
        phase_space_measure=measure,
    )
    return {**setup, "velocity": velocity, "precompute": precompute}


def stella_third_order_upwind_matrix(n: int, spacing: float, sign: int) -> np.ndarray:
    """Reproduce stella's explicit zero-BC third-order upwind derivative."""

    if n < 4:
        raise ValueError("stella third-order upwind derivative requires at least four nodes")
    if spacing <= 0.0:
        raise ValueError("velocity spacing must be positive")
    if sign not in (-1, 1):
        raise ValueError("upwind sign must be -1 or 1")
    matrix = np.zeros((n, n), dtype=float)
    start, end = (0, n - 1) if sign == -1 else (n - 1, 0)
    matrix[start, start] = -sign / spacing
    adjacent = start - sign
    matrix[adjacent, adjacent - sign] = -sign * 2.0 / (6.0 * spacing)
    matrix[adjacent, adjacent] = -sign * 3.0 / (6.0 * spacing)
    matrix[adjacent, adjacent + sign] = sign * 6.0 / (6.0 * spacing)
    matrix[end, end + sign] = sign / spacing
    matrix[end, end] = -sign / spacing
    for row in range(start - 2 * sign, end + sign, -sign):
        matrix[row, row - sign] = -sign * 2.0 / (6.0 * spacing)
        matrix[row, row] = -sign * 3.0 / (6.0 * spacing)
        matrix[row, row + sign] = sign * 6.0 / (6.0 * spacing)
        matrix[row, row + 2 * sign] = -sign / (6.0 * spacing)
    return matrix


def stella_second_order_centered_matrix(n: int, spacing: float, sign: int) -> np.ndarray:
    """Reproduce stella's centered z derivative with open zero ghost values."""

    if n < 3:
        raise ValueError("stella centered derivative requires at least three nodes")
    if spacing <= 0.0:
        raise ValueError("grid spacing must be positive")
    if sign not in (-1, 1):
        raise ValueError("upwind sign must be -1 or 1")
    matrix = np.zeros((n, n), dtype=float)
    for row in range(1, n - 1):
        matrix[row, row - 1] = -0.5 / spacing
        matrix[row, row + 1] = 0.5 / spacing
    if sign > 0:
        matrix[0, 0] = -1.0 / spacing
        matrix[0, 1] = 1.0 / spacing
        matrix[-1, -2] = -0.5 / spacing
    else:
        matrix[0, 1] = 0.5 / spacing
        matrix[-1, -2] = -1.0 / spacing
        matrix[-1, -1] = 1.0 / spacing
    return matrix


def apply_stella_mirror_stencil(
    case, state, split, rhs_precompute, *, native_coefficient=None
) -> RHSTermSplit:
    """Replace the generic mirror derivative with stella's explicit stencil."""

    if not case.name.startswith(("replay_stella_coefficients_", "replay_stella_native_")):
        return split
    state_array = jnp.asarray(state)
    if native_coefficient is None:
        coefficient = jnp.asarray(rhs_precompute.mirror_force_coeff)
        if coefficient.ndim == 3:
            if coefficient.shape[0] != 1:
                raise ValueError("same-state stella replay requires one kinetic species")
            coefficient = coefficient[0]
        if coefficient.ndim != 2:
            raise ValueError("mirror coefficient must have (mu,z) order")
        coefficient_full = coefficient[None, :, :]
    else:
        coefficient_trace = np.asarray(native_coefficient)
        if coefficient_trace.shape != (
            state_array.shape[2],
            state_array.shape[0],
            state_array.shape[1],
        ):
            raise ValueError("native mirror coefficient must have (z,vpar,mu) order")
        coefficient_full = jnp.asarray(np.transpose(coefficient_trace, (1, 2, 0)))
    n_vpar = state_array.shape[0]
    # Coefficient shape is (mu,z); its sign is independent of positive mu.
    spacing = float(case.vpar_max * 2.0 / case.n_vpar)
    matrices = {
        sign: jnp.asarray(stella_third_order_upwind_matrix(n_vpar, spacing, sign))
        for sign in (-1, 1)
    }
    derivatives = []
    for iz in range(state_array.shape[2]):
        sign = 1 if float(jnp.real(coefficient_full[0, 0, iz])) >= 0.0 else -1
        derivatives.append(jnp.einsum("ij,jmxy->imxy", matrices[sign], state_array[:, :, iz]))
    derivative = jnp.stack(derivatives, axis=2)
    mirror = coefficient_full[..., None, None] * derivative
    index = split.names.index("mirror_force")
    terms = list(split.terms)
    terms[index] = mirror
    return RHSTermSplit(names=split.names, terms=tuple(terms))


def apply_stella_native_drift_algebra(
    case,
    state,
    phi,
    split,
    native_coefficients: dict[str, np.ndarray],
) -> RHSTermSplit:
    """Replace drift pieces with stella's traced coefficient algebra."""

    if not case.name.startswith("replay_stella_native_"):
        return split
    state_array = jnp.asarray(state)
    phi_array = jnp.asarray(phi)

    def coefficient(name):
        values = np.asarray(native_coefficients[name])
        return jnp.asarray(np.transpose(values, (1, 2, 0)))[..., None, None]

    gyroaverage = coefficient("gyroaverage_j0")
    derivative_y = 1j * FOCUS_KY
    drift_g = derivative_y * coefficient("magnetic_drift_g_y") * state_array
    drift_phi = (
        derivative_y
        * coefficient("magnetic_drift_phi_y")
        * gyroaverage
        * phi_array[None, None, :, :, :]
    )
    terms = list(split.terms)
    terms[split.names.index("magnetic_drift")] = drift_g
    terms[split.names.index("drift_field_drive")] = drift_phi
    return RHSTermSplit(names=split.names, terms=tuple(terms))


def apply_stella_native_streaming_algebra(
    case,
    state,
    phi,
    split,
    rhs_precompute,
    stella: dict[str, Any],
    call: int,
) -> RHSTermSplit:
    """Replace streaming pieces using stella's 257-point open-chain stencil."""

    if not case.name.startswith("replay_stella_native_"):
        return split
    state_array = np.asarray(state)[..., 0, 0]
    phi_array = np.asarray(phi)[:, 0, 0]
    state_full = np.concatenate(
        (state_array, stella["upper_endpoint"]["distribution"][call][:, :, None]),
        axis=2,
    )
    phi_full = np.concatenate(
        (phi_array, [stella["upper_endpoint"]["phi"][call]])
    )
    coefficient = np.transpose(
        np.asarray(stella["native_coefficients"]["parallel_streaming"]),
        (1, 2, 0),
    )
    coefficient_full = np.concatenate(
        (
            coefficient,
            stella["upper_endpoint"]["native_coefficients"]["parallel_streaming"][
                :, :, None
            ],
        ),
        axis=2,
    )
    gyroaverage = np.transpose(
        np.asarray(stella["native_coefficients"]["gyroaverage_j0"]),
        (1, 2, 0),
    )
    gyroaverage_full = np.concatenate(
        (
            gyroaverage,
            stella["upper_endpoint"]["native_coefficients"]["gyroaverage_j0"][
                :, :, None
            ],
        ),
        axis=2,
    )
    spacing = 2.0 * np.pi / (state_full.shape[2] - 1)
    distribution_derivative = np.empty_like(state_full)
    field_derivative = np.empty_like(state_full)
    for iv in range(state_full.shape[0]):
        sign = 1 if coefficient_full[iv, 0, 0].real >= 0.0 else -1
        upwind = stella_third_order_upwind_matrix(state_full.shape[2], spacing, sign)
        centered = stella_second_order_centered_matrix(state_full.shape[2], spacing, sign)
        distribution_derivative[iv] = np.einsum("ij,mj->mi", upwind, state_full[iv])
        field_values = gyroaverage_full[iv] * phi_full[None, :]
        field_derivative[iv] = np.einsum("ij,mj->mi", centered, field_values)
    maxwellian = np.asarray(rhs_precompute.maxwellian)
    if maxwellian.ndim == 4:
        maxwellian = maxwellian[0]
    streaming = coefficient * (
        distribution_derivative[:, :, :-1]
        + maxwellian * field_derivative[:, :, :-1]
    )
    terms = list(split.terms)
    terms[split.names.index("gkw_parallel_streaming_recurrence")] = jnp.asarray(
        streaming[..., None, None]
    )
    terms[split.names.index("gkw_parallel_field_drive")] = jnp.zeros_like(
        terms[split.names.index("gkw_parallel_field_drive")]
    )
    return RHSTermSplit(names=split.names, terms=tuple(terms))


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


def native_stella_mirror_reconstruction(stella: dict[str, Any]) -> dict[str, float]:
    """Check the traced mirror coefficient/stencil before any grid interpolation."""

    coefficient = np.asarray(stella["native_coefficients"]["mirror_force"])
    spacing = float(stella["vpar"][1] - stella["vpar"][0])
    matrices = {
        sign: stella_third_order_upwind_matrix(len(stella["vpar"]), spacing, sign)
        for sign in (-1, 1)
    }
    weights = (
        np.asarray(stella["w_vpar"])[None, :, None]
        * np.asarray(stella["w_mu"])[:, None, :]
    )
    errors = []
    for state, reference in zip(
        stella["distribution"], stella["mirror_force"], strict=True
    ):
        candidate = np.empty_like(state)
        for iz in range(state.shape[0]):
            sign = 1 if coefficient[iz, 0, 0].real >= 0.0 else -1
            candidate[iz] = coefficient[iz] * (matrices[sign] @ state[iz])
        reference_norm = np.sqrt(np.sum(weights * np.abs(reference) ** 2))
        error_norm = np.sqrt(np.sum(weights * np.abs(reference - candidate) ** 2))
        errors.append(float(error_norm / reference_norm))
    return {
        "max_relative_l2_error": max(errors),
        "rhs_calls_checked": len(errors),
    }


def native_stella_quasineutrality_reconstruction(
    stella: dict[str, Any],
) -> dict[str, float]:
    """Reconstruct stella's density numerator with its native J0 and weights."""

    gyroaverage = np.asarray(stella["native_coefficients"]["gyroaverage_j0"])
    weights = (
        np.asarray(stella["w_vpar"])[None, :, None]
        * np.asarray(stella["w_mu"])[:, None, :]
    )
    errors = []
    for state, reference in zip(
        stella["distribution"],
        stella["quasineutrality_numerator"],
        strict=True,
    ):
        candidate = np.sum(weights * gyroaverage * state, axis=(1, 2))
        errors.append(float(np.linalg.norm(reference - candidate) / np.linalg.norm(reference)))
    return {
        "max_relative_l2_error": max(errors),
        "rhs_calls_checked": len(errors),
    }


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
    if summary.get("trace_format") not in {
        "stellarator_gk_stella_rhs_trace_v3",
        "stellarator_gk_stella_rhs_trace_v4",
        "stellarator_gk_stella_rhs_trace_v5",
    }:
        raise ValueError("same-state replay requires an explicitly labeled v3/v4/v5 trace")
    stella = load_stella_array_trace(trace_path, summary)

    rows: list[dict[str, Any]] = []
    for case in replay_cases():
        setup = (
            build_native_stella_setup(case, Path(stella_geometry), stella)
            if case.name.startswith("replay_stella_native_")
            else _build_w7x_setup(case, Path(stella_geometry))
        )
        precompute = apply_stella_coefficient_contract(
            case, setup["precompute"], Path(stella_geometry)
        )
        target_z = np.asarray(setup["geometry"].z)
        target_vpar = np.asarray(setup["velocity"].vpar)
        target_mu = np.asarray(setup["velocity"].mu)
        weights = {
            "w_z": np.asarray(setup["geometry"].w_z),
            "w_vpar": np.asarray(setup["velocity"].w_vpar),
            "w_mu": (
                np.asarray(stella["w_mu"])
                if case.name.startswith("replay_stella_native_")
                else np.asarray(setup["velocity"].w_mu)
            ),
        }
        native_mirror = None
        if "mirror_force" in stella["native_coefficients"]:
            native_mirror = interpolate_phase_space_to_grid(
                stella["native_coefficients"]["mirror_force"],
                source_z=stella["z"],
                source_vpar=stella["vpar"],
                source_mu=stella["mu"],
                target_z=target_z,
                target_vpar=target_vpar,
                target_mu=target_mu,
            )

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
            split = split_rhs_terms(state, phi, precompute.rhs)
            split = apply_stella_mirror_stencil(
                case,
                state,
                split,
                precompute.rhs,
                native_coefficient=native_mirror,
            )
            split = apply_stella_native_drift_algebra(
                case,
                state,
                phi,
                split,
                stella["native_coefficients"],
            )
            split = apply_stella_native_streaming_algebra(
                case,
                state,
                phi,
                split,
                precompute.rhs,
                stella,
                call,
            )
            if case.name.startswith(("replay_stella_coefficients_", "replay_stella_native_")):
                total_rhs = sum(split.terms, jnp.zeros_like(state))
            else:
                total_rhs = linear_residual(state, precomputed=precompute, phi=phi)
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
                adiabatic_density_numerator(state, precompute.field)
            )[:, 0, 0]
            denominator = np.asarray(precompute.field.denominator)[:, 0, 0]
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
    status = _status(
        rows,
        trace_path,
        Path(stella_geometry),
        tolerance,
        native_mirror=native_stella_mirror_reconstruction(stella),
        native_quasineutrality=native_stella_quasineutrality_reconstruction(stella),
    )
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
    rows: list[dict[str, Any]],
    trace_path: Path,
    geometry_path: Path,
    tolerance: float,
    *,
    native_mirror: dict[str, float],
    native_quasineutrality: dict[str, float],
) -> dict[str, Any]:
    rhs_rows = [row for row in rows if not row["quantity"].startswith("quasineutrality")]
    acceptance_case = "replay_stella_native_32x8"
    acceptance_rows = [row for row in rhs_rows if row["case"] == acceptance_case]
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
    maximum = max(float(row["aligned_relative_l2_error"]) for row in acceptance_rows)
    passed = maximum <= tolerance
    return {
        "schema": "stellarator_gk_w7x_same_state_rhs_replay_v1",
        "status": "same_state_rhs_parity_passed" if passed else "same_state_rhs_parity_failed",
        "passed": passed,
        "relative_l2_tolerance": tolerance,
        "max_rhs_relative_l2_error": maximum,
        "acceptance_case": acceptance_case,
        "max_all_discriminators_rhs_relative_l2_error": max(
            float(row["aligned_relative_l2_error"]) for row in rhs_rows
        ),
        "trace_source": str(trace_path),
        "geometry_source": str(geometry_path),
        "raw_trace_committed": False,
        "native_grid_mirror_reconstruction": native_mirror,
        "native_grid_quasineutrality_reconstruction": native_quasineutrality,
        "rhs_calls": sorted({int(row["rhs_call"]) for row in rows}),
        "cases": sorted({str(row["case"]) for row in rows}),
        "best_by_quantity": best_by_quantity,
        "interpretation": (
            "Each solver RHS is evaluated on the same stella distribution and phi "
            "on contained 16x4 and exact-vpar-node 32x4 finite-difference grids, "
            "plus a provider-native 32x8 grid with its full z-dependent phase-space "
            "measure. "
            "No fitted amplitude or phase is used; the quasineutrality denominator alone "
            "uses the documented opposite-sign convention. The native acceptance case "
            "tests traced gyroaverage and drift coefficients, source mirror and parallel "
            "stencils, geometry-header flux_fac, and Maxwellian normalization. Provider "
            "geometry now preserves separate grad-B and curvature drives and the verified "
            "mirror orientation; traced coefficient replacement remains diagnostic-only."
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fixture_readme(status: dict[str, Any]) -> str:
    return f"""# W7-X stella same-state RHS replay

This compact fixture records solver RHS operators applied directly to traced
stella distribution and potential arrays. The external raw trace is not stored
in the repository. Both periodic spectral and open GKW-upwind parallel models
are reported on a 16×4 velocity grid contained inside the stella domain. Two
additional discriminators apply source-derived stella mirror, drift, and
drive coefficients at 16×4 and an exact-node 32×4 velocity resolution. The
acceptance case runs on stella's native 32×8 nodes and z-dependent phase-space
measure, with traced coefficient arrays and source stencils used only for the
same-state diagnostic.

Provider-neutral production geometry now preserves separate grad-B and
curvature drift components and uses the verified mirror-force orientation.

The v4 trace also verifies the mirror operator directly on stella's native
256×32×8 grid before interpolation. Its maximum reconstruction error is
`{status['native_grid_mirror_reconstruction']['max_relative_l2_error']:.8g}`.

The v5 trace similarly reconstructs stella's quasineutrality numerator with
its native gyroaverage and z-dependent velocity weights. Its maximum relative
L2 error is
`{status['native_grid_quasineutrality_reconstruction']['max_relative_l2_error']:.8g}`.

Status: `{status['status']}`. Acceptance-case maximum RHS relative L2 error:
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
