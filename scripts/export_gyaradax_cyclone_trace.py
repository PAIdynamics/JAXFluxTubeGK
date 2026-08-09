"""Export and compare Gyaradax Cyclone selected-ky traces.

Run from the repository root after installing Gyaradax runtime dependencies:

    uv run --extra dev --extra reference python scripts/export_gyaradax_cyclone_trace.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")


@dataclass(frozen=True)
class TraceProfile:
    label: str
    n_z: int
    n_vpar: int
    n_mu: int
    steps_per_window: int
    n_windows: int
    tolerance: float
    output: Path
    comparison_output: Path


_TRACE_PROFILES = {
    "reduced": TraceProfile(
        label="reduced",
        n_z=8,
        n_vpar=6,
        n_mu=4,
        steps_per_window=2,
        n_windows=3,
        tolerance=2.0e-2,
        output=Path("figures/gyaradax_cyclone_trace_reduced.csv"),
        comparison_output=Path("figures/gyaradax_cyclone_trace_comparison.csv"),
    ),
    "production-control-smoke": TraceProfile(
        label="production-control smoke",
        n_z=48,
        n_vpar=32,
        n_mu=8,
        steps_per_window=20,
        n_windows=4,
        tolerance=2.0e-2,
        output=Path("figures/gyaradax_cyclone_trace_production_control_smoke.csv"),
        comparison_output=Path(
            "figures/gyaradax_cyclone_trace_production_control_smoke_comparison.csv"
        ),
    ),
    "production-control": TraceProfile(
        label="production-control",
        n_z=48,
        n_vpar=32,
        n_mu=8,
        steps_per_window=20,
        n_windows=80,
        tolerance=2.0e-2,
        output=Path("figures/gyaradax_cyclone_trace_production_control.csv"),
        comparison_output=Path(
            "figures/gyaradax_cyclone_trace_production_control_comparison.csv"
        ),
    ),
}


def main() -> None:
    args = _parse_args()
    from jax_fluxtube_gk.external import announce_external_path

    announce_external_path("Gyaradax source", args.gyaradax_root)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(args.gyaradax_root))

    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp

    from gyaradax.backends import create_ops
    from gyaradax.geometry import compute_geometry
    from gyaradax.integrals import calculate_phi
    from gyaradax.params import GKParams
    from gyaradax.simulate import gk_run
    from gyaradax.solver import default_state, init_f, linear_precompute, mode_amplitude

    from jax_fluxtube_gk import (
        compare_cyclone_base_case_traces,
        run_cyclone_base_case_trace,
        write_cyclone_trace_csv,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output = Path(args.comparison_output)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    geometry = _build_gyaradax_geometry(compute_geometry, args)
    params = _build_gyaradax_params(GKParams, geometry, args)
    trace = _run_gyaradax_trace(
        jnp,
        create_ops,
        calculate_phi,
        gk_run,
        default_state,
        init_f,
        linear_precompute,
        mode_amplitude,
        geometry,
        params,
        args,
    )
    write_cyclone_trace_csv(output, trace)

    current = run_cyclone_base_case_trace(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        dt=args.dt,
        nperiod=args.nperiod,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        initial_profile=args.finit,
    )
    fields = (
        "times",
        "physical_amplitude",
        "window_growth",
        "fitted_growth",
        "physical_phi_norm",
        "physical_state_norm",
        "physical_rhs_norm",
    )
    report = compare_cyclone_base_case_traces(
        current,
        trace,
        tolerance=args.tolerance,
        field_names=fields,
    )
    _write_comparison_csv(comparison_output, report, args.tolerance)

    status = "PASS" if bool(report.passed) else "OPEN"
    print(f"wrote {output}")
    print(f"wrote {comparison_output}")
    label = _TRACE_PROFILES[args.profile].label
    print(
        f"{label} Gyaradax/CycloneTrace physical comparison {status}: "
        f"max_abs_error={float(report.max_abs_error):.8e}, "
        f"tolerance={args.tolerance:.8e}"
    )


def _build_gyaradax_geometry(compute_geometry, args):
    # Gyaradax's single-mode krhomax is normalized by kthnorm internally.
    kthnorm = abs(args.q / (2.0 * 3.141592653589793 * args.eps))
    return compute_geometry(
        q=args.q,
        shat=args.shat,
        eps=args.eps,
        ns=args.n_z,
        nkx=1,
        nky=1,
        nvpar=args.n_vpar,
        nmu=args.n_mu,
        vpar_max=args.vpar_max,
        nperiod=args.nperiod,
        kxmax=0.0,
        krhomax=args.ky * kthnorm,
        ikxspace=5,
        geom_type="s-alpha",
    )


def _build_gyaradax_params(gk_params_cls, geometry, args):
    return gk_params_cls(
        dt=args.dt,
        naverage=args.steps_per_window,
        disp_par=args.disp_par,
        disp_vp=args.disp_vp,
        disp_x=0.0,
        disp_y=0.0,
        idisp=2,
        non_linear=False,
        finit=args.finit,
        amp_init=1.0e-4,
        adiabatic_electrons=True,
        rlt=args.rlt,
        rln=args.rln,
        q=args.q,
        shat=args.shat,
        eps=args.eps,
        kthnorm=float(geometry["kthnorm"]),
        Rref=100.0,
        d2X=1.0,
        signB=1.0,
        dvp=float(geometry["dvp"]),
        sgr_dist=float(geometry["sgr_dist"]),
        kxmax=float(geometry["kxmax"]),
        kymax=float(geometry["kymax"]),
        dgrid=1.0,
        tgrid=1.0,
        backend="jax",
        mixed_precision=False,
    )


def _run_gyaradax_trace(
    jnp,
    create_ops,
    calculate_phi,
    gk_run,
    default_state,
    init_f,
    linear_precompute,
    mode_amplitude,
    geometry,
    params,
    args,
):
    from jax_fluxtube_gk import CycloneTrace

    df = init_f(geometry, finit=args.finit, amp_init_real=1.0e-4)
    state = default_state(nky=1)
    phi0 = calculate_phi(geometry, df, params=params)
    amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
    state = replace(state, window_start_amp=amp0)
    pre = linear_precompute(geometry, params)
    ops = create_ops(pre, backend="jax", use_z2z=False, mixed_precision=False)

    times = []
    raw_amplitudes = []
    physical_amplitudes = []
    window_growths = []
    fitted_growths = []
    phi_norms = []
    state_norms = []
    rhs_norms = []
    log_normalizations = []
    previous_physical = None

    def append_snapshot():
        nonlocal previous_physical
        phi = calculate_phi(geometry, df, params=params, pre=pre)
        amplitude = mode_amplitude(phi, geometry, params.norm_eps)[0]
        log_normalization = -jnp.log(state.accumulated_norm_factor[0])
        physical = amplitude * jnp.exp(log_normalization)
        if previous_physical is None:
            window_growth = jnp.asarray(0.0, dtype=jnp.float64)
        else:
            window_growth = (
                jnp.log(jnp.maximum(physical, 1.0e-300))
                - jnp.log(jnp.maximum(previous_physical, 1.0e-300))
            ) / (args.steps_per_window * params.dt)
        rhs = ops.linear_rhs(df, phi, geometry, params, pre)
        times.append(float(state.time))
        raw_amplitudes.append(amplitude)
        physical_amplitudes.append(physical)
        window_growths.append(window_growth)
        log_normalizations.append(log_normalization)
        phi_norms.append(_l2_norm(jnp, phi[:, 0, 0]))
        state_norms.append(_l2_norm(jnp, df))
        rhs_norms.append(_l2_norm(jnp, rhs))
        fitted_growths.append(
            _fit_growth(
                jnp,
                jnp.asarray(times, dtype=jnp.float64),
                jnp.asarray(physical_amplitudes, dtype=jnp.float64),
            )
        )
        previous_physical = physical

    append_snapshot()
    for _ in range(args.n_windows):
        df, _phi, _fluxes, state = gk_run(
            df,
            geometry,
            params,
            state,
            n_steps=args.steps_per_window,
            pre=pre,
        )
        append_snapshot()

    return CycloneTrace(
        times=jnp.asarray(times, dtype=jnp.float64),
        raw_amplitude=jnp.asarray(raw_amplitudes, dtype=jnp.float64),
        physical_amplitude=jnp.asarray(physical_amplitudes, dtype=jnp.float64),
        window_growth=jnp.asarray(window_growths, dtype=jnp.float64),
        fitted_growth=jnp.asarray(fitted_growths, dtype=jnp.float64),
        phi_norm=jnp.asarray(phi_norms, dtype=jnp.float64),
        state_norm=jnp.asarray(state_norms, dtype=jnp.float64),
        rhs_norm=jnp.asarray(rhs_norms, dtype=jnp.float64),
        log_normalization=jnp.asarray(log_normalizations, dtype=jnp.float64),
        source="gyaradax",
        notes=f"{args.profile} Gyaradax s-alpha selected-ky trace; finit={args.finit}",
    )


def _l2_norm(jnp, values):
    return jnp.sqrt(jnp.mean(jnp.abs(values) ** 2))


def _fit_growth(jnp, times, amplitudes):
    if times.shape[0] < 2:
        return jnp.asarray(0.0, dtype=jnp.float64)
    log_amplitude = jnp.log(jnp.maximum(amplitudes, 1.0e-300))
    centered_time = times - jnp.mean(times)
    centered_log = log_amplitude - jnp.mean(log_amplitude)
    return jnp.sum(centered_time * centered_log) / jnp.sum(centered_time**2)


def _write_comparison_csv(path: Path, report, tolerance: float) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("field", "max_abs_error", "tolerance", "status", "notes"))
        status = "PASS" if bool(report.passed) else "OPEN"
        for field, error in zip(report.field_names, report.field_errors, strict=True):
            writer.writerow((field, float(error), tolerance, status, report.notes))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=tuple(_TRACE_PROFILES),
        default="reduced",
        help=(
            "named resolution/time profile; explicit size and output flags "
            "override the selected profile"
        ),
    )
    parser.add_argument("--gyaradax-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--n-z", type=int, default=None)
    parser.add_argument("--n-vpar", type=int, default=None)
    parser.add_argument("--n-mu", type=int, default=None)
    parser.add_argument("--steps-per-window", type=int, default=None)
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--nperiod", type=int, default=5)
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--shat", type=float, default=0.78)
    parser.add_argument("--eps", type=float, default=0.19)
    parser.add_argument("--rln", type=float, default=2.2)
    parser.add_argument("--rlt", type=float, default=6.9)
    parser.add_argument("--ky", type=float, default=0.5)
    parser.add_argument("--vpar-max", type=float, default=3.0)
    parser.add_argument("--disp-par", type=float, default=1.0)
    parser.add_argument("--disp-vp", type=float, default=0.2)
    parser.add_argument("--finit", choices=("cosine2", "cosine"), default="cosine2")
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()
    return _apply_profile_defaults(args)


def _apply_profile_defaults(args):
    profile = _TRACE_PROFILES[args.profile]
    args.n_z = profile.n_z if args.n_z is None else args.n_z
    args.n_vpar = profile.n_vpar if args.n_vpar is None else args.n_vpar
    args.n_mu = profile.n_mu if args.n_mu is None else args.n_mu
    if args.steps_per_window is None:
        args.steps_per_window = profile.steps_per_window
    args.n_windows = profile.n_windows if args.n_windows is None else args.n_windows
    args.tolerance = profile.tolerance if args.tolerance is None else args.tolerance
    args.output = profile.output if args.output is None else args.output
    if args.comparison_output is None:
        args.comparison_output = profile.comparison_output
    return args


if __name__ == "__main__":
    main()
