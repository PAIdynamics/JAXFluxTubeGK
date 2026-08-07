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

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    GeometryScalarParams,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_exb_pseudospectral_precompute,
    build_fourier_grid,
    build_linear_residual_precompute,
    build_parallel_grid,
    build_s_alpha_geometry,
    build_velocity_grid,
    gyrokinetic_heat_response,
    integrate_nonlinear_adaptive,
    radial_flux_spectrum,
    saturated_radial_flux_statistics,
    solve_adiabatic_electron_phi,
)


def _hyperdiffusion(fourier, coefficient: float):
    kx_scale = jnp.max(jnp.abs(fourier.kx))
    ky_scale = jnp.max(jnp.abs(fourier.ky))
    return coefficient * (
        (jnp.abs(fourier.kx)[:, None] / kx_scale) ** 4
        + (jnp.abs(fourier.ky)[None, :] / ky_scale) ** 4
    )


def _initial_state(precompute, amplitude: float, seed: int):
    # maxwellian is (species,vpar,mu,z); Fourier topology comes from FLR J0.
    fourier_shape = precompute.rhs.flr_factors.bessel_j0.shape[-2:]
    shape = precompute.rhs.maxwellian.shape + fourier_shape
    noise = jax.random.normal(jax.random.key(seed), shape)
    state = amplitude * precompute.rhs.maxwellian[..., None, None] * noise
    if precompute.n_species == 1:
        state = state[0]
    # Enforce the real-field Hermitian constraint on the ky=0 line.
    center = fourier_shape[0] // 2
    ky_zero = state[..., :, 0]
    for offset in range(1, center + 1):
        ky_zero = ky_zero.at[..., center - offset].set(
            jnp.conj(ky_zero[..., center + offset])
        )
    ky_zero = ky_zero.at[..., center].set(jnp.real(ky_zero[..., center]))
    return state.at[..., :, 0].set(ky_zero).astype(jnp.complex128)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--n-z", type=int, default=12)
    parser.add_argument("--n-vpar", type=int, default=12)
    parser.add_argument("--n-mu", type=int, default=6)
    parser.add_argument("--n-kx", type=int, default=9)
    parser.add_argument("--n-ky", type=int, default=5)
    parser.add_argument("--kx-max", type=float, default=0.8)
    parser.add_argument("--ky-min", type=float, default=0.1)
    parser.add_argument("--hyperdiffusion", type=float, default=0.05)
    parser.add_argument("--initial-amplitude", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    parser.add_argument("--max-relative-drift", type=float, default=0.2)
    parser.add_argument("--max-relative-standard-error", type=float, default=0.1)
    parser.add_argument("--require-stationary", action="store_true")
    args = parser.parse_args(argv)
    if args.n_kx < 3 or args.n_kx % 2 == 0 or args.n_ky < 2:
        parser.error("n-kx must be odd and at least 3; n-ky must be at least 2")
    if min(args.n_z, args.n_vpar, args.n_mu) < 2 or args.final_time <= 0.0:
        parser.error("phase-space sizes must be at least 2 and final-time positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=args.n_vpar,
            n_mu=args.n_mu,
            vpar_max=3.5,
            mu_max=6.0,
            backend="finite_difference",
        )
    )
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=args.n_z, z_min=-0.5, z_max=0.5, topology="periodic")
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=args.n_kx,
            n_ky=args.n_ky,
            kx_max=args.kx_max,
            ky_values=tuple(args.ky_min * index for index in range(args.n_ky)),
        )
    )
    geometry = build_s_alpha_geometry(
        parallel, GeometryScalarParams(q=1.4, shat=0.8, eps=0.18)
    )
    ion = SpeciesParams(1.0, 1.0, 1.0, 1.0, 0.8, 2.49)
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        ion,
        electron_params=AdiabaticElectronParams(1.0, 1.0),
        perpendicular_damping=_hyperdiffusion(fourier, args.hyperdiffusion),
    )
    spectral = build_exb_pseudospectral_precompute(fourier)
    initial = _initial_state(precompute, args.initial_amplitude, args.seed)
    result = integrate_nonlinear_adaptive(
        initial, args.final_time, precompute, spectral, store_history=True
    )
    if result.history.shape[0] < 3:
        raise RuntimeError(
            "nonlinear trajectory produced fewer than three samples; increase final-time"
        )
    phi_history = jax.vmap(solve_adiabatic_electron_phi, in_axes=(0, None))(
        result.history, precompute.field
    )
    heat_history = jax.vmap(gyrokinetic_heat_response, in_axes=(0, None, None, None, None))(
        result.history,
        velocity,
        geometry.B,
        ion,
        precompute.rhs.flr_factors.bessel_j0,
    )
    statistics = saturated_radial_flux_statistics(
        phi_history,
        heat_history,
        result.times,
        fourier.ky,
        start_fraction=args.start_fraction,
        w_z=geometry.w_z,
        parseval=fourier.parseval,
    )
    flux = jax.vmap(
        lambda phi, heat: jnp.sum(
            radial_flux_spectrum(
                phi, heat, fourier.ky, w_z=geometry.w_z, parseval=fourier.parseval
            )
        )
    )(phi_history, heat_history)
    relative_standard_error = float(
        statistics.standard_error / jnp.maximum(jnp.abs(statistics.mean), 1.0e-14)
    )
    stationary = bool(
        np.isfinite(np.asarray(result.state)).all()
        and abs(float(statistics.relative_window_drift)) <= args.max_relative_drift
        and relative_standard_error <= args.max_relative_standard_error
    )
    payload = {
        "schema_version": 1,
        "producer": "optimal-fusion/nonlinear-heat-flux",
        "case": vars(args) | {"output": str(args.output)},
        "n_steps": result.n_steps,
        "stationary": stationary,
        "relative_standard_error": relative_standard_error,
        "statistics": {key: float(value) for key, value in asdict(statistics).items()},
        "times": np.asarray(result.times).tolist(),
        "heat_flux": np.asarray(flux).tolist(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}; stationary={stationary}; n_steps={result.n_steps}")
    if args.require_stationary and not stationary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
