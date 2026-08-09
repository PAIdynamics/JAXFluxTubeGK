"""Audit the W7-X ``ky=0.3`` RHS/model balance on stella-imported geometry.

The velocity discriminator showed that simple velocity-grid refinement does not
close the W7-X/stella ``ky=0.3`` frequency/profile gap.  This script freezes the
matched stella geometry, ``kx=0`` topology, species parameters, and late-time
normalization convention, then decomposes the solver RHS into model terms at the
final normalized state.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_mode_connectivity,
    build_velocity_grid,
    estimate_linear_cfl_dt,
    integrate_fixed_step,
    k_perp_squared,
    linear_residual,
    mode_chain_amplitude,
    normalize_by_ky_amplitude,
    solve_field_from_state,
)
from jax_fluxtube_gk.physics import (
    adiabatic_density_numerator,
    adiabatic_quasineutrality_residual,
    dissipation,
    drift_field_drive,
    equilibrium_drive,
    gkw_igh_streaming_mirror,
    gkw_parallel_field_drive,
    gkw_parallel_streaming,
    magnetic_drift_advection,
    mirror_force,
    parallel_field_drive,
    parallel_recurrence_control,
    parallel_streaming,
    velocity_recurrence_control,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_ky03_rhs_model_balance"
DEFAULT_STELLA_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)
FOCUS_KY = 0.3
STELLA_TEND = 200.0


@dataclass(frozen=True)
class RHSBalanceCase:
    """One W7-X ``ky=0.3`` RHS-balance case."""

    name: str
    n_vpar: int
    n_mu: int
    velocity_backend: str = "chebyshev"
    vpar_max: float = 2.0
    mu_max: float = 1.5
    dt: float = 0.02
    steps_per_window: int = 5
    n_windows: int = 2000
    parallel_derivative_model: str = "matrix"

    @property
    def total_time(self) -> float:
        return self.dt * self.steps_per_window * self.n_windows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_names = tuple(args.case) if args.case else ("gkw_fd_16x8",)
    cases = tuple(case for case in default_balance_cases() if case.name in selected_names)
    missing = set(selected_names).difference(case.name for case in cases)
    if missing:
        raise ValueError(f"unknown case(s): {', '.join(sorted(missing))}")
    if args.n_windows is not None:
        cases = tuple(replace(case, n_windows=args.n_windows) for case in cases)
    if args.steps_per_window is not None:
        cases = tuple(replace(case, steps_per_window=args.steps_per_window) for case in cases)
    if args.dt is not None:
        cases = tuple(replace(case, dt=args.dt) for case in cases)

    summary = run_w7x_ky03_rhs_model_balance(
        output_dir=args.output_dir,
        cases=cases,
        stella_geometry=args.stella_geometry,
        array_output=args.array_output,
    )
    print(summary["status_json"])
    print(summary["term_balance_csv"])
    return 0


def default_balance_cases() -> tuple[RHSBalanceCase, ...]:
    """Return fixed-control cases useful for W7-X ``ky=0.3`` RHS audits."""

    return (
        RHSBalanceCase(name="cheb_4x4", n_vpar=4, n_mu=4),
        RHSBalanceCase(
            name="gkw_fd_16x8",
            n_vpar=16,
            n_mu=8,
            velocity_backend="finite_difference",
        ),
        RHSBalanceCase(
            name="stella_open_16x8",
            n_vpar=16,
            n_mu=8,
            velocity_backend="finite_difference",
            parallel_derivative_model="gkw_upwind",
        ),
    )


def run_w7x_ky03_rhs_model_balance(
    *,
    output_dir: Path,
    cases: tuple[RHSBalanceCase, ...],
    stella_geometry: Path = DEFAULT_STELLA_GEOMETRY,
    array_output: Path | None = None,
) -> dict[str, object]:
    """Run cases, write RHS-balance artifacts, and return a compact summary."""

    if array_output is not None:
        if len(cases) != 1:
            raise ValueError("--array-output requires exactly one selected case")
        _validate_external_array_output(array_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_summaries = []
    term_rows = []
    density_rows = []
    geometry_rows = []

    for case in cases:
        start = perf_counter()
        setup = _build_w7x_setup(case, stella_geometry)
        evolved = _evolve_case(setup)
        split = split_rhs_terms(evolved["state"], evolved["phi"], setup["precompute"].rhs)
        total_rhs = linear_residual(
            evolved["state"],
            precomputed=setup["precompute"],
            phi=evolved["phi"],
        )
        reconstruction = sum(split.terms, jnp.zeros_like(total_rhs))
        reconstruction_error = _max_abs(reconstruction - total_rhs)
        case_term_rows = selected_term_balance_rows(
            split.names,
            split.terms,
            total_rhs,
            setup["precompute"].field,
            ix=setup["fourier"].ixzero,
            iy=0,
            case=case.name,
            ky=FOCUS_KY,
        )
        case_density_rows = selected_density_profile_rows(
            split.names,
            split.terms,
            setup["precompute"].field,
            setup["geometry"].z,
            ix=setup["fourier"].ixzero,
            iy=0,
            case=case.name,
            ky=FOCUS_KY,
        )
        case_geometry_rows = selected_geometry_model_rows(
            setup["geometry"],
            setup["fourier"],
            setup["precompute"],
            ix=setup["fourier"].ixzero,
            iy=0,
            case=case.name,
            ky=FOCUS_KY,
        )
        field_residual = adiabatic_quasineutrality_residual(
            evolved["phi"],
            evolved["state"],
            setup["precompute"].field,
        )
        if array_output is not None:
            ix = setup["fourier"].ixzero
            field = setup["precompute"].field
            write_selected_mode_array_trace(
                array_output,
                z=setup["geometry"].z,
                vpar=setup["velocity"].vpar,
                mu=setup["velocity"].mu,
                w_z=setup["geometry"].w_z,
                w_vpar=setup["velocity"].w_vpar,
                w_mu=setup["velocity"].w_mu,
                distribution=_selected_mode_phase_space(evolved["state"], ix=ix, iy=0),
                phi=np.asarray(evolved["phi"])[:, ix, 0],
                rhs_terms={
                    name: _selected_mode_phase_space(term, ix=ix, iy=0)
                    for name, term in zip(split.names, split.terms, strict=True)
                },
                total_rhs=_selected_mode_phase_space(total_rhs, ix=ix, iy=0),
                quasineutrality_numerator=np.asarray(
                    adiabatic_density_numerator(evolved["state"], field)
                )[:, ix, 0],
                quasineutrality_denominator=np.asarray(field.denominator)[:, ix, 0],
                log_normalization=evolved["log_normalization"][0],
                metadata={
                    "schema": "jax_fluxtube_gk_selected_mode_array_trace_v1",
                    "case": case.name,
                    "ky": FOCUS_KY,
                    "kx": 0.0,
                    "time": case.total_time,
                    "dt": case.dt,
                    "steps_per_window": case.steps_per_window,
                    "n_windows": case.n_windows,
                    "axis_order": ["z", "vpar", "mu"],
                    "geometry_source": setup["geometry_metadata"]["geometry_source"],
                    "stella_geometry": _display_path(stella_geometry),
                    "normalization": (
                        "stored arrays are the final renormalized solver state; "
                        "log_normalization restores its accumulated amplitude"
                    ),
                },
            )
        wall_seconds = perf_counter() - start
        case_summary = {
            "case": case.name,
            "n_vpar": case.n_vpar,
            "n_mu": case.n_mu,
            "velocity_backend": case.velocity_backend,
            "parallel_derivative_model": case.parallel_derivative_model,
            "dt": case.dt,
            "steps_per_window": case.steps_per_window,
            "n_windows": case.n_windows,
            "total_time": case.total_time,
            "stella_tend": STELLA_TEND,
            "estimated_cfl_dt": float(estimate_linear_cfl_dt(setup["precompute"])),
            "final_growth_rate": float(evolved["growth_rate"][0]),
            "final_frequency": float(evolved["frequency"][0]),
            "final_log_normalization": float(evolved["log_normalization"][0]),
            "final_raw_amplitude": float(evolved["raw_amplitude"][-1, 0]),
            "rhs_reconstruction_max_abs_error": reconstruction_error,
            "quasineutrality_residual_rms": _rms(field_residual[:, setup["fourier"].ixzero, 0]),
            "total_rhs_rms": _rms(np.asarray(total_rhs)[..., setup["fourier"].ixzero, 0]),
            "dominant_rhs_terms": _dominant_terms(case_term_rows),
            "wall_seconds": wall_seconds,
            "geometry_source": setup["geometry_metadata"]["geometry_source"],
            "stella_geometry": _display_path(stella_geometry),
        }
        case_summaries.append(case_summary)
        term_rows.extend(case_term_rows)
        density_rows.extend(case_density_rows)
        geometry_rows.extend(case_geometry_rows)

    term_balance_csv = output_dir / "rhs_term_balance.csv"
    density_balance_csv = output_dir / "rhs_density_balance.csv"
    geometry_balance_csv = output_dir / "geometry_model_balance.csv"
    status_json = output_dir / "rhs_model_balance_status.json"
    metadata_json = output_dir / "rhs_model_balance_metadata.json"
    _write_csv(term_balance_csv, term_rows)
    _write_csv(density_balance_csv, density_rows)
    _write_csv(geometry_balance_csv, geometry_rows)
    _write_json(status_json, _status_payload(case_summaries, term_balance_csv))
    _write_json(
        metadata_json,
        {
            "benchmark_name": "w7x_ky03_rhs_model_balance",
            "purpose": (
                "solver-side RHS/model term balance for the W7-X stella-matched "
                "ky=0.3 discrepancy after velocity-grid discrimination"
            ),
            "focus_ky": FOCUS_KY,
            "stella_tend": STELLA_TEND,
            "stella_geometry": _display_path(stella_geometry),
            "case_summaries": case_summaries,
            "term_balance_csv": _display_path(term_balance_csv),
            "density_balance_csv": _display_path(density_balance_csv),
            "geometry_balance_csv": _display_path(geometry_balance_csv),
        },
    )
    _write_readme(output_dir)
    return {
        "status_json": str(status_json),
        "metadata_json": str(metadata_json),
        "term_balance_csv": str(term_balance_csv),
        "case_summaries": case_summaries,
        "array_output": str(array_output) if array_output is not None else None,
    }


def write_selected_mode_array_trace(
    path: Path,
    *,
    z,
    vpar,
    mu,
    w_z,
    w_vpar,
    w_mu,
    distribution,
    phi,
    rhs_terms: dict[str, object],
    total_rhs,
    quasineutrality_numerator,
    quasineutrality_denominator,
    log_normalization,
    metadata: dict[str, object],
) -> None:
    """Write a compact, opt-in selected-mode trace with explicit coordinates."""

    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("selected-mode array output must use the .npz suffix")
    shape = (len(z), len(vpar), len(mu))
    arrays = {
        "z": np.asarray(z, dtype=float),
        "vpar": np.asarray(vpar, dtype=float),
        "mu": np.asarray(mu, dtype=float),
        "w_z": np.asarray(w_z, dtype=float),
        "w_vpar": np.asarray(w_vpar, dtype=float),
        "w_mu": np.asarray(w_mu, dtype=float),
        "distribution": _require_shape(distribution, shape, "distribution"),
        "phi": _require_shape(phi, (shape[0],), "phi"),
        "total_rhs": _require_shape(total_rhs, shape, "total_rhs"),
        "quasineutrality_numerator": _require_shape(
            quasineutrality_numerator, (shape[0],), "quasineutrality_numerator"
        ),
        "quasineutrality_denominator": _require_shape(
            quasineutrality_denominator, (shape[0],), "quasineutrality_denominator"
        ),
        "log_normalization": np.asarray(log_normalization, dtype=float),
    }
    for name, values in rhs_terms.items():
        arrays[f"rhs_{name}"] = _require_shape(values, shape, f"rhs_{name}")
    trace_metadata = dict(metadata)
    trace_metadata["rhs_terms"] = list(rhs_terms)
    trace_metadata["array_keys"] = sorted(arrays)
    arrays["metadata_json"] = np.asarray(json.dumps(trace_metadata, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _selected_mode_phase_space(values, *, ix: int, iy: int, species: int = 0):
    """Select one species/mode and return the canonical ``(z,vpar,mu)`` order."""

    array = np.asarray(values)
    if array.ndim == 6:
        array = array[species]
    if array.ndim != 5:
        raise ValueError("phase-space array must have 5 or 6 dimensions")
    return np.transpose(array[..., ix, iy], (2, 0, 1))


def _require_shape(values, expected: tuple[int, ...], name: str):
    array = np.asarray(values)
    if array.shape != expected:
        raise ValueError(f"{name} has shape {array.shape}; expected {expected}")
    return array


def _validate_external_array_output(path: Path) -> None:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(ROOT.resolve()):
        raise ValueError("--array-output must be outside the repository; use /tmp or a cache path")


@dataclass(frozen=True)
class RHSTermSplit:
    names: tuple[str, ...]
    terms: tuple[object, ...]


def split_rhs_terms(distribution, phi, rhs_precompute) -> RHSTermSplit:
    """Return named RHS term arrays in the same sum order as the solver RHS."""

    magnetic_drift = magnetic_drift_advection(
        distribution,
        rhs_precompute.magnetic_drift_frequency,
    )
    equilibrium = equilibrium_drive(phi, rhs_precompute)
    drift_field = drift_field_drive(phi, rhs_precompute)
    damping = dissipation(distribution, rhs_precompute.perpendicular_damping)

    if rhs_precompute.parallel_derivative_model == "gkw_igh":
        return RHSTermSplit(
            names=(
                "gkw_igh_streaming_mirror_recurrence",
                "magnetic_drift",
                "equilibrium_drive",
                "gkw_parallel_field_drive",
                "drift_field_drive",
                "dissipation",
            ),
            terms=(
                gkw_igh_streaming_mirror(distribution, rhs_precompute),
                magnetic_drift,
                equilibrium,
                gkw_parallel_field_drive(phi, rhs_precompute),
                drift_field,
                damping,
            ),
        )

    if rhs_precompute.parallel_derivative_model == "gkw_upwind":
        return RHSTermSplit(
            names=(
                "gkw_parallel_streaming_recurrence",
                "magnetic_drift",
                "mirror_force",
                "equilibrium_drive",
                "gkw_parallel_field_drive",
                "drift_field_drive",
                "dissipation",
                "velocity_recurrence",
            ),
            terms=(
                gkw_parallel_streaming(distribution, rhs_precompute),
                magnetic_drift,
                mirror_force(
                    distribution,
                    rhs_precompute.D_vpar,
                    rhs_precompute.mirror_force_coeff,
                ),
                equilibrium,
                gkw_parallel_field_drive(phi, rhs_precompute),
                drift_field,
                damping,
                velocity_recurrence_control(
                    distribution,
                    rhs_precompute.velocity_recurrence_operator,
                    rhs_precompute.velocity_recurrence_coeff,
                ),
            ),
        )

    return RHSTermSplit(
        names=(
            "parallel_streaming",
            "magnetic_drift",
            "mirror_force",
            "equilibrium_drive",
            "parallel_field_drive",
            "drift_field_drive",
            "dissipation",
            "parallel_recurrence",
            "velocity_recurrence",
        ),
        terms=(
            parallel_streaming(
                distribution,
                rhs_precompute.D_z,
                rhs_precompute.parallel_streaming_coeff,
            ),
            magnetic_drift,
            mirror_force(
                distribution,
                rhs_precompute.D_vpar,
                rhs_precompute.mirror_force_coeff,
            ),
            equilibrium,
            parallel_field_drive(phi, rhs_precompute.D_z, rhs_precompute),
            drift_field,
            damping,
            parallel_recurrence_control(
                distribution,
                rhs_precompute.parallel_recurrence_operator,
                rhs_precompute.parallel_recurrence_coeff,
            ),
            velocity_recurrence_control(
                distribution,
                rhs_precompute.velocity_recurrence_operator,
                rhs_precompute.velocity_recurrence_coeff,
            ),
        ),
    )


def selected_term_balance_rows(
    term_names: tuple[str, ...],
    terms: tuple[object, ...],
    total_rhs,
    field_precompute,
    *,
    ix: int,
    iy: int,
    case: str,
    ky: float,
) -> list[dict[str, object]]:
    """Summarize selected-mode RHS pieces by norm and projection."""

    selected_total = np.asarray(total_rhs)[..., ix, iy]
    total_rms = _rms(selected_total)
    total_l2 = _l2(selected_total)
    denominator = np.vdot(selected_total.ravel(), selected_total.ravel())
    rows: list[dict[str, object]] = []
    for name, term in zip(term_names, terms, strict=True):
        selected = np.asarray(term)[..., ix, iy]
        projection = _projection(selected_total, selected, denominator)
        density = np.asarray(adiabatic_density_numerator(term, field_precompute))[:, ix, iy]
        rows.append(
            {
                "case": case,
                "ky": ky,
                "term": name,
                "rhs_rms": _rms(selected),
                "rhs_l2": _l2(selected),
                "rhs_max_abs": _max_abs(selected),
                "rhs_fraction_of_total_l2": _safe_ratio(_l2(selected), total_l2),
                "projection_real": float(np.real(projection)),
                "projection_imag": float(np.imag(projection)),
                "projection_abs": float(abs(projection)),
                "density_moment_rms": _rms(density),
                "density_moment_max_abs": _max_abs(density),
                "total_rhs_rms": total_rms,
                "total_rhs_l2": total_l2,
            }
        )
    return rows


def selected_density_profile_rows(
    term_names: tuple[str, ...],
    terms: tuple[object, ...],
    field_precompute,
    z,
    *,
    ix: int,
    iy: int,
    case: str,
    ky: float,
) -> list[dict[str, object]]:
    """Return per-z quasineutrality numerator contributions from RHS terms."""

    z_values = np.asarray(z, dtype=float)
    rows: list[dict[str, object]] = []
    for name, term in zip(term_names, terms, strict=True):
        density = np.asarray(adiabatic_density_numerator(term, field_precompute))[:, ix, iy]
        for index, value in enumerate(density):
            rows.append(
                {
                    "case": case,
                    "ky": ky,
                    "term": name,
                    "z_index": index,
                    "z": float(z_values[index]),
                    "density_real": float(np.real(value)),
                    "density_imag": float(np.imag(value)),
                    "density_abs": float(abs(value)),
                }
            )
    return rows


def selected_geometry_model_rows(geometry, fourier, precompute, *, ix: int, iy: int, case: str, ky: float):
    """Return the z-local geometry, FLR, and field-denominator model inputs."""

    kperp2 = np.asarray(k_perp_squared(geometry, fourier), dtype=float)[:, ix, iy]
    bessel_j0 = np.asarray(precompute.rhs.flr_factors.bessel_j0)[..., ix, iy]
    gamma0 = np.asarray(precompute.rhs.flr_factors.gamma0)[..., ix, iy]
    drift_frequency = np.asarray(precompute.rhs.magnetic_drift_frequency)[..., ix, iy]
    maxwellian = np.asarray(precompute.rhs.maxwellian)
    drive_factor = np.asarray(precompute.rhs.drive_factor)
    denominator = np.asarray(precompute.field.denominator)[:, ix, iy]
    ion_polarization = np.asarray(precompute.field.ion_polarization)[:, ix, iy]
    rows = []
    for index, z_value in enumerate(np.asarray(geometry.z, dtype=float)):
        rows.append(
            {
                "case": case,
                "ky": ky,
                "z_index": index,
                "z": float(z_value),
                "theta": float(np.asarray(geometry.theta)[index]),
                "phi": float(np.asarray(geometry.phi)[index]),
                "B": float(np.asarray(geometry.B)[index]),
                "F": float(np.asarray(geometry.F)[index]),
                "G": float(np.asarray(geometry.G)[index]),
                "E_y": float(np.asarray(geometry.E_y)[index]),
                "D_x": float(np.asarray(geometry.D_x)[index]),
                "D_y": float(np.asarray(geometry.D_y)[index]),
                "g_xx": float(np.asarray(geometry.g_xx)[index]),
                "g_xy": float(np.asarray(geometry.g_xy)[index]),
                "g_yy": float(np.asarray(geometry.g_yy)[index]),
                "kperp2": float(kperp2[index]),
                "bessel_j0_rms_over_mu": _rms(bessel_j0[..., index]),
                "gamma0_mean_over_species": float(np.mean(np.real(gamma0[..., index]))),
                "magnetic_drift_frequency_rms_over_velocity": _rms(drift_frequency[..., index]),
                "maxwellian_mean_over_velocity": float(np.mean(maxwellian[..., index])),
                "drive_factor_mean_over_velocity": float(np.mean(drive_factor[..., index])),
                "field_denominator_real": float(np.real(denominator[index])),
                "field_denominator_imag": float(np.imag(denominator[index])),
                "ion_polarization_real": float(np.real(ion_polarization[index])),
                "electron_response": float(precompute.field.electron_response),
            }
        )
    return rows


def _build_w7x_setup(case: RHSBalanceCase, stella_geometry: Path) -> dict[str, object]:
    from examples.run_stellarator_linear_scan import _load_geometry, _parse_args, _parse_float_tuple

    scan_args = _parse_args(_scan_args(case, stella_geometry))
    ky_values = _parse_float_tuple(scan_args.ky_values)
    geometry, parallel, geometry_metadata = _load_geometry(scan_args)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=case.n_vpar,
            n_mu=case.n_mu,
            vpar_max=case.vpar_max,
            mu_max=case.mu_max,
            backend=case.velocity_backend,
        )
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=1,
            n_ky=1,
            kx_max=0.0,
            ky_values=ky_values,
            ikxspace=1,
        )
    )
    connectivity = build_mode_connectivity(fourier)
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
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        electron_params=electrons,
        mode_connectivity=connectivity,
        parallel_derivative_model=case.parallel_derivative_model,
    )
    return {
        "case": case,
        "geometry": geometry,
        "parallel": parallel,
        "velocity": velocity,
        "fourier": fourier,
        "connectivity": connectivity,
        "precompute": precompute,
        "geometry_metadata": geometry_metadata,
    }


def _evolve_case(setup: dict[str, object]) -> dict[str, object]:
    from examples.run_stellarator_linear_scan import (
        _frequency_from_phi_samples,
        _growth_from_log_amplitudes,
        _initial_state,
        _late_start_index,
    )

    case = setup["case"]
    geometry = setup["geometry"]
    velocity = setup["velocity"]
    parallel = setup["parallel"]
    fourier = setup["fourier"]
    connectivity = setup["connectivity"]
    precompute = setup["precompute"]
    state = _initial_state(velocity, parallel, fourier, 1.0e-2)

    solve_phi = jax.jit(lambda state_value: solve_field_from_state(state_value, precompute))
    advance_window = jax.jit(
        lambda state_value: integrate_fixed_step(
            state_value,
            case.dt,
            case.steps_per_window,
            linear_residual,
            precompute,
            store_history=False,
        ).state
    )
    times = []
    log_amplitudes = []
    raw_amplitudes = []
    phi_samples = []
    log_normalization = jnp.zeros((fourier.ky.shape[0],), dtype=jnp.float64)

    def snapshot(time_value, state_value):
        phi_value = solve_phi(state_value)
        amplitude = mode_chain_amplitude(phi_value, w_z=geometry.w_z, connectivity=connectivity)
        floor = jnp.asarray(1.0e-300, dtype=amplitude.dtype)
        times.append(float(time_value))
        raw_amplitudes.append(np.asarray(amplitude, dtype=float))
        log_amplitudes.append(
            np.asarray(jnp.log(jnp.maximum(amplitude, floor)) + log_normalization)
        )
        phi_samples.append(phi_value)
        return amplitude

    snapshot(0.0, state)
    for window in range(case.n_windows):
        state = advance_window(state)
        time_value = (window + 1) * case.steps_per_window * case.dt
        amplitude = snapshot(time_value, state)
        normalized = normalize_by_ky_amplitude(
            state,
            amplitude,
            log_normalization=log_normalization,
        )
        state = normalized.state
        log_normalization = normalized.log_normalization

    times_array = np.asarray(times, dtype=float)
    log_amplitude_array = np.asarray(log_amplitudes, dtype=float)
    late_start = _late_start_index(len(times_array), 0.5)
    phi = solve_phi(state)
    return {
        "state": state,
        "phi": phi,
        "times": times_array,
        "log_amplitude": log_amplitude_array,
        "raw_amplitude": np.asarray(raw_amplitudes, dtype=float),
        "growth_rate": _growth_from_log_amplitudes(
            times_array,
            log_amplitude_array,
            "late_fit",
            late_start,
        ),
        "frequency": _frequency_from_phi_samples(
            times_array,
            phi_samples,
            late_start,
            w_z=geometry.w_z,
            connectivity=connectivity,
        ),
        "log_normalization": np.asarray(log_normalization, dtype=float),
    }


def _scan_args(case: RHSBalanceCase, stella_geometry: Path) -> list[str]:
    return [
        "--geometry-source",
        "stella-geometry",
        "--stella-geometry",
        str(stella_geometry),
        "--output-dir",
        str(DEFAULT_OUTPUT_DIR / "unused_scan_output"),
        "--n-kx",
        "1",
        "--kx-max",
        "0.0",
        "--ikxspace",
        "1",
        "--ky-values",
        str(FOCUS_KY),
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


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stella-geometry", type=Path, default=DEFAULT_STELLA_GEOMETRY)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in default_balance_cases()),
        help="case to run; repeat for multiple cases; default: gkw_fd_16x8",
    )
    parser.add_argument("--n-windows", type=int, help="override case window count")
    parser.add_argument("--steps-per-window", type=int, help="override case steps per window")
    parser.add_argument("--dt", type=float, help="override case timestep")
    parser.add_argument(
        "--array-output",
        type=Path,
        help=(
            "write the selected-mode full-array trace to an external .npz path; "
            "requires exactly one case and refuses repository-local paths"
        ),
    )
    return parser.parse_args(argv)


def _status_payload(case_summaries: list[dict[str, object]], term_balance_csv: Path):
    max_reconstruction_error = max(
        float(item["rhs_reconstruction_max_abs_error"]) for item in case_summaries
    )
    production_time_cases = [
        item for item in case_summaries if abs(float(item["total_time"]) - STELLA_TEND) <= 1.0e-12
    ]
    passed = max_reconstruction_error <= 1.0e-10 and bool(production_time_cases)
    return {
        "benchmark_name": "w7x_ky03_rhs_model_balance",
        "status": "solver_side_rhs_balance_ready",
        "passed": passed,
        "focus_ky": FOCUS_KY,
        "stella_tend": STELLA_TEND,
        "term_balance_csv": _display_path(term_balance_csv),
        "max_rhs_reconstruction_abs_error": max_reconstruction_error,
        "case_summaries": case_summaries,
        "interpretation": (
            "This is a solver-side balance on stella-imported W7-X geometry. "
            "It identifies which model terms dominate the ky=0.3 branch, but "
            "it is not a stella parity proof until a matched stella source-term "
            "or distribution trace is exported and compared term by term."
        ),
        "next_action": (
            "compare these ky=0.3 streaming, mirror, drift, field-drive, FLR, "
            "and quasineutrality contributions against a stella-exported RHS "
            "or source-term trace before changing collocation physics"
        ),
    }


def _dominant_terms(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row["rhs_fraction_of_total_l2"]), reverse=True)
    return [
        {
            "term": str(row["term"]),
            "rhs_fraction_of_total_l2": float(row["rhs_fraction_of_total_l2"]),
            "projection_real": float(row["projection_real"]),
            "projection_imag": float(row["projection_imag"]),
        }
        for row in ordered[:4]
    ]


def _projection(selected_total, selected_term, denominator):
    if abs(denominator) <= 0.0:
        return 0.0 + 0.0j
    return np.vdot(selected_total.ravel(), selected_term.ravel()) / denominator


def _rms(values) -> float:
    array = np.asarray(values)
    return float(np.sqrt(np.mean(np.abs(array) ** 2)))


def _l2(values) -> float:
    array = np.asarray(values)
    return float(np.sqrt(np.sum(np.abs(array) ** 2)))


def _max_abs(values) -> float:
    return float(np.max(np.abs(np.asarray(values))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
                "# W7-X ky=0.3 RHS/model balance",
                "",
                "This fixture freezes the stella-imported W7-X geometry, `kx=0`, ",
                "`n_kx=1`, species gradients, and late-time normalization controls ",
                "used by the W7-X/stella comparison, then decomposes the solver RHS ",
                "for the discrepant `ky=0.3` branch.",
                "",
                "Files:",
                "",
                "- `rhs_term_balance.csv`: scalar selected-mode RHS norms and projections.",
                "- `rhs_density_balance.csv`: z profiles of quasineutrality numerator rates.",
                "- `geometry_model_balance.csv`: z-local geometry, FLR, drift, and field inputs.",
                "- `rhs_model_balance_status.json`: diagnostic status and next action.",
                "",
                "For direct parity, pass `--array-output` with an external `.npz` ",
                "path. The opt-in archive stores `(z, vpar, mu)` complex arrays, ",
                "coordinates, quadrature weights, quasineutrality data, and ",
                "normalization; it is deliberately not a committed fixture.",
                "",
                "This is a solver-side diagnostic.  A production parity claim still ",
                "requires a matched stella source-term or distribution/RHS trace.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
