"""Run a revision-pinned Gyaradax kinetic-electron linear reference in scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> None:
    args = _parse_args()
    root = args.gyaradax_root.expanduser().resolve()
    revision = _git_revision(root)
    if args.expected_revision is not None and revision != args.expected_revision:
        raise RuntimeError(
            f"Gyaradax revision mismatch: found {revision}, expected {args.expected_revision}"
        )
    print(f"external Gyaradax source: {root} @ {revision}")
    sys.path.insert(0, str(root))

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from gyaradax.geometry import compute_geometry
    from gyaradax.params import GKParams
    from gyaradax.simulate import gk_run
    from gyaradax.solver import default_state, init_f, linear_precompute

    geometry = compute_geometry(
        q=args.q,
        shat=args.shat,
        eps=args.eps,
        ns=args.n_z,
        nvpar=args.n_vpar,
        nmu=args.n_mu,
        vpar_max=args.vpar_max,
        nkx=1,
        nky=1,
        nperiod=args.nperiod,
        kxmax=args.ky,
        krhomax=args.ky,
        signB=1.0,
        Rref=1.0,
        geom_type="s-alpha",
    )
    masses = jnp.asarray([1.0, args.electron_mass])
    params = GKParams(
        dt=args.dt,
        naverage=args.steps_per_window,
        non_linear=False,
        adaptive_dt=True,
        adiabatic_electrons=False,
        disp_par=1.0,
        disp_vp=0.1,
        disp_x=0.0,
        disp_y=0.0,
        finit="cosine2",
        amp_init=1.0e-3,
        mas=masses,
        tmp=jnp.ones(2),
        de=jnp.ones(2),
        signz=jnp.asarray([1.0, -1.0]),
        vthrat=jnp.sqrt(1.0 / masses),
        rlt=jnp.asarray([0.0, args.electron_temperature_gradient]),
        rln=jnp.asarray([args.density_gradient, args.density_gradient]),
        dgrid=1.0,
        tgrid=1.0,
        sgr_dist=float(geometry["sgr_dist"]),
        dvp=float(geometry["dvp"]),
        kxmax=float(np.max(np.abs(np.asarray(geometry["kxrh"])))) or 1.0,
        kymax=float(np.max(np.asarray(geometry["krho"]))) or 1.0,
        nlapar=args.nlapar,
        nlbpar=args.nlbpar,
        beta=args.beta,
        drive_scale=1.0,
        idisp=2,
        cfl_safety=0.9,
        mixed_precision=False,
        backend="jax",
    )
    distribution = init_f(
        geometry, finit="cosine2", amp_init_real=params.amp_init, n_species=2
    )
    precompute = linear_precompute(geometry, params)
    state = default_state(nky=1)
    times: list[float] = []
    phases: list[float] = []
    growth_rates: list[float] = []
    dt_min = np.inf
    dt_max = 0.0
    for _ in range(args.n_windows):
        distribution, phi, _fluxes, state, dt_info = gk_run(
            distribution,
            geometry,
            params,
            state,
            args.steps_per_window,
            pre=precompute,
            return_dt_info=True,
        )
        times.append(float(state.time))
        phases.append(float(jnp.angle(phi[args.n_z // 2, 0, 0])))
        growth_rates.append(float(jnp.asarray(state.last_growth_rate)[0]))
        used = np.asarray(dt_info["dt_used"], dtype=float)
        dt_min = min(dt_min, float(np.min(used)))
        dt_max = max(dt_max, float(np.max(used)))

    times_array = np.asarray(times)
    phases_array = np.unwrap(np.asarray(phases))
    late = times_array > 0.5 * times_array[-1]
    frequency = float(np.polyfit(times_array[late], phases_array[late], 1)[0])
    final_mode = np.asarray(phi[:, 0, 0], dtype=complex)
    final_mode = final_mode / np.linalg.norm(final_mode)
    final_mode = final_mode * np.exp(-1j * np.angle(final_mode[args.n_z // 2]))
    payload = {
        "schema_version": 1,
        "producer": "gyaradax",
        "revision": revision,
        "case": {
            "q": args.q,
            "shat": args.shat,
            "eps": args.eps,
            "ky": args.ky,
            "ky_internal": float(np.asarray(geometry["krho"])[0]),
            "wave_number_convention": "ky is GKW kthrho; ky_internal is solver krho",
            "density_gradient": args.density_gradient,
            "electron_temperature_gradient": args.electron_temperature_gradient,
            "electron_mass": args.electron_mass,
            "n_z": args.n_z,
            "n_vpar": args.n_vpar,
            "n_mu": args.n_mu,
            "nperiod": args.nperiod,
            "field_model": "electromagnetic" if args.nlapar else "kinetic",
            "nlapar": args.nlapar,
            "nlbpar": args.nlbpar,
            "beta": args.beta,
        },
        "steps_per_window": args.steps_per_window,
        "n_windows": args.n_windows,
        "final_time": float(times_array[-1]),
        "growth_rate": growth_rates[-1],
        "frequency": frequency,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "times": times,
        "window_growth_rates": growth_rates,
        "unwrapped_probe_phases": phases_array.tolist(),
        "mode_structure_real": final_mode.real.tolist(),
        "mode_structure_imag": final_mode.imag.tolist(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"gamma={payload['growth_rate']:+.8f}, omega={frequency:+.8f}")


def _git_revision(root: Path) -> str:
    if not (root / ".git").exists():
        raise ValueError(f"Gyaradax root is not a Git checkout: {root}")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gyaradax-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--n-z", type=int, default=32)
    parser.add_argument("--n-vpar", type=int, default=32)
    parser.add_argument("--n-mu", type=int, default=16)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--n-windows", type=int, default=200)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--ky", type=float, default=0.7)
    parser.add_argument("--density-gradient", type=float, default=2.2)
    parser.add_argument("--electron-temperature-gradient", type=float, default=6.9)
    parser.add_argument("--electron-mass", type=float, default=0.01)
    parser.add_argument("--vpar-max", type=float, default=3.0)
    parser.add_argument("--nlapar", action="store_true")
    parser.add_argument("--nlbpar", action="store_true")
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.nlbpar and not args.nlapar:
        parser.error("--nlbpar requires --nlapar")
    if args.nlapar and args.beta <= 0.0:
        parser.error("--nlapar requires a positive --beta")
    if not args.nlapar and args.beta != 0.0:
        parser.error("--beta requires --nlapar")
    return args


if __name__ == "__main__":
    main()
