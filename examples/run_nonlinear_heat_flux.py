#!/usr/bin/env python3
"""Run a reduced nonlinear ITG heat-flux trajectory and write compact JSON."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_exb_pseudospectral_precompute,
    build_fourier_grid,
    build_gkw_parallel_grid,
    build_linear_residual_precompute,
    build_mode_connectivity,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    correlated_flux_statistics,
    gyrokinetic_energy_response,
    gyrokinetic_heat_response,
    integrate_nonlinear_adaptive,
    radial_flux_spectrum,
    solve_adiabatic_electron_phi,
)


def _hyperdiffusion(fourier, coefficient: float):
    kx_scale = jnp.max(jnp.abs(fourier.kx))
    ky_scale = jnp.max(jnp.abs(fourier.ky))
    return coefficient * (
        (jnp.abs(fourier.kx)[:, None] / kx_scale) ** 4
        + (jnp.abs(fourier.ky)[None, :] / ky_scale) ** 4
    )


def _require_x64(enabled: bool) -> None:
    if not enabled:
        raise RuntimeError(
            "nonlinear acceptance runs require x64; rerun with JAX_ENABLE_X64=1"
        )


def _initial_state(precompute, amplitude: float, seed: int, zonal_fraction: float = 0.0):
    # maxwellian is (species,vpar,mu,z); Fourier topology comes from FLR J0.
    fourier_shape = precompute.rhs.flr_factors.bessel_j0.shape[-2:]
    shape = precompute.rhs.maxwellian.shape + fourier_shape
    noise = jax.random.normal(jax.random.key(seed), shape)
    state = amplitude * precompute.rhs.maxwellian[..., None, None] * noise
    if precompute.n_species == 1:
        state = state[0]
    state = state.at[..., :, 0].multiply(zonal_fraction)
    # Enforce the real-field Hermitian constraint on the ky=0 line.
    center = fourier_shape[0] // 2
    ky_zero = state[..., :, 0]
    for offset in range(1, center + 1):
        ky_zero = ky_zero.at[..., center - offset].set(jnp.conj(ky_zero[..., center + offset]))
    ky_zero = ky_zero.at[..., center].set(jnp.real(ky_zero[..., center]))
    return state.at[..., :, 0].set(ky_zero).astype(jnp.complex128)


def _phi_rms_diagnostics(phi_history):
    """Return total, nonzonal, and per-ky initial/final potential amplitudes."""

    phi_history = jnp.asarray(phi_history)
    if phi_history.ndim != 4 or phi_history.shape[-1] < 2:
        raise ValueError("phi history must have shape (time,z,kx,ky) with at least two ky modes")

    def rms(values, axes):
        return jnp.sqrt(jnp.mean(jnp.abs(values) ** 2, axis=axes))

    total = rms(phi_history, (1, 2, 3))
    nonzonal = rms(phi_history[..., 1:], (1, 2, 3))
    by_ky = rms(phi_history, (1, 2))
    floor = jnp.asarray(1.0e-14, dtype=total.dtype)
    return {
        "phi_rms_initial": total[0],
        "phi_rms_final": total[-1],
        "phi_rms_ratio": total[-1] / jnp.maximum(total[0], floor),
        "nonzonal_phi_rms_initial": nonzonal[0],
        "nonzonal_phi_rms_final": nonzonal[-1],
        "nonzonal_phi_rms_ratio": nonzonal[-1] / jnp.maximum(nonzonal[0], floor),
        "phi_rms_by_ky_initial": by_ky[0],
        "phi_rms_by_ky_final": by_ky[-1],
        "phi_rms_ratio_by_ky": by_ky[-1] / jnp.maximum(by_ky[0], floor),
    }


def _nonzonal_phi_rms_history(phi_history):
    """Return the compact nonzonal amplitude trace used by merged windows."""

    phi_history = jnp.asarray(phi_history)
    if phi_history.ndim != 4 or phi_history.shape[-1] < 2:
        raise ValueError("phi history must contain at least two ky modes")
    return jnp.sqrt(jnp.mean(jnp.abs(phi_history[..., 1:]) ** 2, axis=(1, 2, 3)))


def _candidate_window_phi_growth(phi_history, times, start_fraction: float):
    """Fit nonzonal potential growth over the candidate stationary window."""

    phi_history = jnp.asarray(phi_history)
    times = jnp.asarray(times)
    if phi_history.ndim != 4 or times.shape != (phi_history.shape[0],):
        raise ValueError("phi history and times must share a time dimension")
    if phi_history.shape[-1] < 2 or not 0.0 <= start_fraction < 1.0:
        raise ValueError("candidate growth requires nonzonal modes and valid start fraction")
    start = min(int(phi_history.shape[0] * start_fraction), phi_history.shape[0] - 2)
    amplitude = _nonzonal_phi_rms_history(phi_history)[start:]
    return _candidate_window_amplitude_growth(amplitude, times[start:])


def _candidate_window_amplitude_growth(amplitude, times):
    """Fit logarithmic growth for an already reduced positive amplitude trace."""

    amplitude = jnp.asarray(amplitude)
    window_times = jnp.asarray(times)
    if amplitude.ndim != 1 or window_times.shape != amplitude.shape or amplitude.size < 2:
        raise ValueError("amplitude and time traces must contain at least two matching samples")
    if bool(jnp.any(amplitude <= 0.0)):
        raise ValueError("amplitude trace must be positive")
    log_amplitude = jnp.log(jnp.maximum(amplitude, 1.0e-14))
    centered_time = window_times - jnp.mean(window_times)
    growth_rate = jnp.sum(centered_time * (log_amplitude - jnp.mean(log_amplitude))) / jnp.sum(
        centered_time**2
    )
    return {
        "candidate_nonzonal_phi_rms_initial": amplitude[0],
        "candidate_nonzonal_phi_rms_final": amplitude[-1],
        "candidate_nonzonal_phi_rms_ratio": amplitude[-1] / jnp.maximum(amplitude[0], 1.0e-14),
        "candidate_nonzonal_phi_growth_rate": growth_rate,
    }


def _checkpoint_contract(args, fourier, state_shape, state_dtype="complex128") -> dict:
    """Return the immutable numerical contract required for a safe restart."""

    return {
        "schema_version": 1,
        "state_shape": list(state_shape),
        "state_dtype": str(np.dtype(state_dtype)),
        "n_z": args.n_z,
        "n_vpar": args.n_vpar,
        "n_mu": args.n_mu,
        "n_kx": args.n_kx,
        "n_ky": args.n_ky,
        "kx": np.asarray(fourier.kx).tolist(),
        "ky": np.asarray(fourier.ky).tolist(),
        "ikxspace": args.ikxspace,
        "parallel_boundary_model": args.parallel_boundary_model,
        "parallel_recurrence_rate": args.parallel_recurrence_rate,
        "rmaj_over_lref": args.rmaj_over_lref,
        "gx_fprim": args.gx_fprim,
        "gx_tprim": args.gx_tprim,
        "hyperdiffusion": args.hyperdiffusion,
        "collision_frequency": args.collision_frequency,
    }


def _write_checkpoint(path: Path, state, time: float, contract: dict, lineage: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            state=np.asarray(state),
            time=np.asarray(time, dtype=np.float64),
            contract=np.asarray(json.dumps(contract, sort_keys=True)),
            lineage=np.asarray(json.dumps(lineage, sort_keys=True)),
        )


def _load_checkpoint(path: Path, expected_contract: dict, expected_lineage_root: dict | None = None):
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as checkpoint:
        required = {"state", "time", "contract", "lineage"}
        if not required.issubset(checkpoint.files):
            raise ValueError("nonlinear checkpoint is missing state, time, contract, or lineage")
        state = np.asarray(checkpoint["state"])
        time = float(np.asarray(checkpoint["time"]))
        contract = json.loads(str(np.asarray(checkpoint["contract"])))
        lineage = json.loads(str(np.asarray(checkpoint["lineage"])))
    if "state_dtype" not in contract:
        contract["state_dtype"] = str(state.dtype)
    if contract != expected_contract or str(state.dtype) != expected_contract["state_dtype"]:
        raise ValueError("nonlinear checkpoint contract does not match requested run")
    if state.shape != tuple(expected_contract["state_shape"]):
        raise ValueError("nonlinear checkpoint state shape does not match its contract")
    if not np.isfinite(state).all() or not np.isfinite(time) or time < 0.0:
        raise ValueError("nonlinear checkpoint state and time must be finite and nonnegative")
    if (
        not isinstance(lineage, dict)
        or lineage.get("schema_version") != 1
        or not isinstance(lineage.get("segment_end_times"), list)
        or not lineage["segment_end_times"]
        or abs(float(lineage["segment_end_times"][-1]) - time) > 1.0e-10 * max(1.0, abs(time))
    ):
        raise ValueError("nonlinear checkpoint trajectory lineage is invalid")
    if expected_lineage_root is not None:
        keys = ("seed", "initial_amplitude", "initial_zonal_fraction")
        if any(lineage.get(key) != expected_lineage_root.get(key) for key in keys):
            raise ValueError(
                "nonlinear checkpoint initialization controls do not match requested run"
            )
    return jnp.asarray(state), time, lineage


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--n-z", type=int, default=12)
    parser.add_argument("--n-vpar", type=int, default=12)
    parser.add_argument("--n-mu", type=int, default=6)
    parser.add_argument("--n-kx", type=int, default=9)
    parser.add_argument("--n-ky", type=int, default=5)
    parser.add_argument("--kx-max", type=float, default=0.8)
    parser.add_argument("--ky-min", type=float, default=0.1)
    parser.add_argument("--ikxspace", type=int, default=1)
    parser.add_argument(
        "--parallel-boundary-model",
        choices=("twist_shift", "periodic_chains"),
        default="twist_shift",
    )
    parser.add_argument("--parallel-recurrence-rate", type=float, default=1.0)
    parser.add_argument("--rmaj-over-lref", type=float, default=2.77778)
    parser.add_argument("--gx-fprim", type=float, default=0.8)
    parser.add_argument("--gx-tprim", type=float, default=2.49)
    parser.add_argument("--hyperdiffusion", type=float, default=0.05)
    parser.add_argument("--collision-frequency", type=float, default=0.0)
    parser.add_argument("--initial-amplitude", type=float, default=1.0e-3)
    parser.add_argument("--initial-zonal-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    parser.add_argument(
        "--flux-moment",
        choices=("nonadvective_heat", "gx_total_energy"),
        default="nonadvective_heat",
    )
    parser.add_argument("--max-relative-drift", type=float, default=0.2)
    parser.add_argument("--max-relative-standard-error", type=float, default=0.1)
    parser.add_argument("--min-phi-rms-ratio", type=float, default=0.8)
    parser.add_argument("--min-stationary-samples", type=int, default=100)
    parser.add_argument("--min-stationary-window-duration", type=float, default=10.0)
    parser.add_argument("--stationary-block-duration", type=float, default=5.0)
    parser.add_argument("--min-stationary-blocks", type=int, default=6)
    parser.add_argument("--max-absolute-phi-growth-rate", type=float, default=0.02)
    parser.add_argument("--diagnostic-stride", type=int, default=1)
    parser.add_argument("--require-stationary", action="store_true")
    args = parser.parse_args(argv)
    if args.n_kx < 3 or args.n_kx % 2 == 0 or args.n_ky < 2:
        parser.error("n-kx must be odd and at least 3; n-ky must be at least 2")
    if min(args.n_z, args.n_vpar, args.n_mu) < 2 or args.final_time <= 0.0:
        parser.error("phase-space sizes must be at least 2 and final-time positive")
    if args.min_phi_rms_ratio <= 0.0:
        parser.error("min-phi-rms-ratio must be positive")
    if not 0.0 <= args.initial_zonal_fraction <= 1.0:
        parser.error("initial-zonal-fraction must lie in [0, 1]")
    if min(args.rmaj_over_lref, args.gx_fprim, args.gx_tprim) <= 0.0:
        parser.error("rmaj-over-lref, gx-fprim, and gx-tprim must be positive")
    if args.collision_frequency < 0.0:
        parser.error("collision-frequency must be nonnegative")
    if args.ikxspace < 1 or args.parallel_recurrence_rate < 0.0:
        parser.error("ikxspace must be positive and parallel-recurrence-rate nonnegative")
    if args.min_stationary_samples < 2 or args.min_stationary_window_duration <= 0.0:
        parser.error("stationarity requires at least two samples and positive window duration")
    if args.stationary_block_duration <= 0.0 or args.min_stationary_blocks < 2:
        parser.error("stationarity requires positive blocks and at least two block means")
    if args.max_absolute_phi_growth_rate <= 0.0:
        parser.error("max-absolute-phi-growth-rate must be positive")
    if args.diagnostic_stride < 1:
        parser.error("diagnostic-stride must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _require_x64(jax.config.x64_enabled)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=args.n_vpar,
            n_mu=args.n_mu,
            vpar_max=3.5,
            mu_max=6.0,
            backend="finite_difference",
        )
    )
    twist_shift = args.parallel_boundary_model == "twist_shift"
    parallel = (
        build_gkw_parallel_grid(args.n_z)
        if twist_shift
        else build_parallel_grid(
            ParallelGridSpec(n_z=args.n_z, z_min=-0.5, z_max=0.5, topology="periodic")
        )
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=args.n_kx,
            n_ky=args.n_ky,
            kx_max=args.kx_max,
            ky_values=tuple(args.ky_min * index for index in range(args.n_ky)),
            ikxspace=args.ikxspace,
            q=1.4,
            shat=0.8,
            eps=0.18,
            use_gkw_shear_spacing=twist_shift,
        )
    )
    connectivity = build_mode_connectivity(fourier, scale_shift_by_ky=True) if twist_shift else None
    geometry = build_s_alpha_geometry(parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18))
    density_gradient = args.gx_fprim * args.rmaj_over_lref
    temperature_gradient = args.gx_tprim * args.rmaj_over_lref
    ion = SpeciesParams(
        1.0,
        1.0,
        1.0,
        1.0,
        density_gradient,
        temperature_gradient,
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        ion,
        electron_params=AdiabaticElectronParams(1.0, 1.0),
        perpendicular_damping=_hyperdiffusion(fourier, args.hyperdiffusion),
        parallel_recurrence_rate=(args.parallel_recurrence_rate if twist_shift else 0.0),
        mode_connectivity=connectivity,
        parallel_derivative_model="gkw_upwind" if twist_shift else "matrix",
        collision_frequency=(args.collision_frequency if args.collision_frequency > 0.0 else None),
    )
    spectral = build_exb_pseudospectral_precompute(fourier)
    seeded_initial = _initial_state(
        precompute,
        args.initial_amplitude,
        args.seed,
        zonal_fraction=args.initial_zonal_fraction,
    )
    checkpoint_contract = _checkpoint_contract(
        args, fourier, seeded_initial.shape, seeded_initial.dtype
    )
    if args.restart_from is None:
        initial = seeded_initial
        start_time = 0.0
        lineage = {
            "schema_version": 1,
            "seed": args.seed,
            "initial_amplitude": args.initial_amplitude,
            "initial_zonal_fraction": args.initial_zonal_fraction,
            "segment_end_times": [],
        }
    else:
        initial, start_time, lineage = _load_checkpoint(
            args.restart_from,
            checkpoint_contract,
            {
                "seed": args.seed,
                "initial_amplitude": args.initial_amplitude,
                "initial_zonal_fraction": args.initial_zonal_fraction,
            },
        )
    if start_time >= args.final_time:
        raise ValueError("final-time must be greater than the restart checkpoint time")
    response_function = (
        gyrokinetic_energy_response
        if args.flux_moment == "gx_total_energy"
        else gyrokinetic_heat_response
    )

    def observe(state):
        phi = solve_adiabatic_electron_phi(state, precompute.field)
        heat = response_function(
            state,
            velocity,
            geometry.B,
            ion,
            precompute.rhs.flr_factors.bessel_j0,
        )
        flux = jnp.sum(
            radial_flux_spectrum(
                phi, heat, fourier.ky, w_z=geometry.w_z, parseval=fourier.parseval
            )
        )
        total_rms = jnp.sqrt(jnp.mean(jnp.abs(phi) ** 2))
        nonzonal_rms = jnp.sqrt(jnp.mean(jnp.abs(phi[..., 1:]) ** 2))
        by_ky = jnp.sqrt(jnp.mean(jnp.abs(phi) ** 2, axis=(0, 1)))
        return jnp.concatenate((jnp.asarray([flux, total_rms, nonzonal_rms]), by_ky))

    result = integrate_nonlinear_adaptive(
        initial,
        args.final_time - start_time,
        precompute,
        spectral,
        store_history=False,
        observation_fn=observe,
        observation_stride=args.diagnostic_stride,
    )
    if result.observations is None or result.observation_times is None:
        raise RuntimeError("nonlinear trajectory produced no diagnostics")
    observations = jnp.asarray(result.observations)
    if observations.shape[0] < 3:
        raise RuntimeError(
            "nonlinear trajectory produced fewer than three samples; increase final-time"
        )
    absolute_times = result.observation_times + start_time
    flux = observations[:, 0]
    total_rms = observations[:, 1]
    nonzonal_rms = observations[:, 2]
    by_ky = observations[:, 3:]
    statistics = correlated_flux_statistics(
        absolute_times,
        flux,
        start_fraction=args.start_fraction,
        block_duration=args.stationary_block_duration,
    )
    relative_standard_error = float(
        statistics.standard_error / jnp.maximum(jnp.abs(statistics.mean), 1.0e-14)
    )
    window_start = min(
        int(absolute_times.shape[0] * args.start_fraction),
        absolute_times.shape[0] - 1,
    )
    window_duration = float(absolute_times[-1] - absolute_times[window_start])
    floor = jnp.asarray(1.0e-14, dtype=total_rms.dtype)
    phi_diagnostics = {
        "phi_rms_initial": total_rms[0],
        "phi_rms_final": total_rms[-1],
        "phi_rms_ratio": total_rms[-1] / jnp.maximum(total_rms[0], floor),
        "nonzonal_phi_rms_initial": nonzonal_rms[0],
        "nonzonal_phi_rms_final": nonzonal_rms[-1],
        "nonzonal_phi_rms_ratio": nonzonal_rms[-1] / jnp.maximum(nonzonal_rms[0], floor),
        "phi_rms_by_ky_initial": by_ky[0],
        "phi_rms_by_ky_final": by_ky[-1],
        "phi_rms_ratio_by_ky": by_ky[-1] / jnp.maximum(by_ky[0], floor),
    }
    candidate_start = min(
        int(absolute_times.shape[0] * args.start_fraction), absolute_times.shape[0] - 2
    )
    candidate_phi = _candidate_window_amplitude_growth(
        nonzonal_rms[candidate_start:],
        absolute_times[candidate_start:],
    )
    stationary = bool(
        np.isfinite(np.asarray(result.state)).all()
        and statistics.n_samples >= args.min_stationary_samples
        and statistics.n_blocks >= args.min_stationary_blocks
        and window_duration >= args.min_stationary_window_duration
        and abs(float(statistics.relative_window_drift)) <= args.max_relative_drift
        and relative_standard_error <= args.max_relative_standard_error
        and float(phi_diagnostics["nonzonal_phi_rms_ratio"]) >= args.min_phi_rms_ratio
        and abs(float(candidate_phi["candidate_nonzonal_phi_growth_rate"]))
        <= args.max_absolute_phi_growth_rate
    )
    report_lineage = lineage | {
        "segment_end_times": [*lineage["segment_end_times"], args.final_time]
    }
    payload = {
        "schema_version": 1,
        "producer": "jax-fluxtube-gk/nonlinear-heat-flux",
        "normalization": (
            "gx_Q_over_Q_GB" if args.flux_moment == "gx_total_energy" else "jax_fluxtube_gk_native"
        ),
        "case": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        | {
            "output": str(args.output),
            "density_gradient_R_over_Ln": density_gradient,
            "temperature_gradient_R_over_LT": temperature_gradient,
            "kx": np.asarray(fourier.kx).tolist(),
            "ky": np.asarray(fourier.ky).tolist(),
            "jax_enable_x64": bool(jax.config.x64_enabled),
        },
        "n_steps": result.n_steps,
        "start_time": start_time,
        "end_time": args.final_time,
        "stationary": stationary,
        "trajectory_lineage": report_lineage,
        "state_rms_initial": float(jnp.sqrt(jnp.mean(jnp.abs(result.history[0]) ** 2))),
        "state_rms_final": float(jnp.sqrt(jnp.mean(jnp.abs(result.state) ** 2))),
        **{key: np.asarray(value).tolist() for key, value in phi_diagnostics.items()},
        **{key: np.asarray(value).tolist() for key, value in candidate_phi.items()},
        "max_abs_heat_flux": float(jnp.max(jnp.abs(flux))),
        "relative_standard_error": relative_standard_error,
        "stationary_window_duration": window_duration,
        "statistics": {key: float(value) for key, value in asdict(statistics).items()},
        "times": np.asarray(absolute_times).tolist(),
        "heat_flux": np.asarray(flux).tolist(),
        "nonzonal_phi_rms": np.asarray(nonzonal_rms).tolist(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.checkpoint_output is not None:
        _write_checkpoint(
            args.checkpoint_output,
            result.state,
            args.final_time,
            checkpoint_contract,
            report_lineage,
        )
    print(f"wrote {output}; stationary={stationary}; n_steps={result.n_steps}")
    if args.require_stationary and not stationary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
