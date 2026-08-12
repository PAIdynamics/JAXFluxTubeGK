"""Export the reduced Hermite-Laguerre moment-RHS fixture.

This is the no-external-binary fallback for the multi-ky physics discriminator.
It writes the same portable per-ky mode-structure CSV used by independent
fixture-comparison tools.

Run from the repository root:

    JAX_ENABLE_X64=1 uv run python examples/export_moment_rhs_fixture.py \
        --ky-values 0.3,0.5 \
        --output fixtures/s_alpha_moment_rhs_mode_structure_fixture.csv
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

from jax_fluxtube_gk.validation.cyclone_gkw import (
    run_s_alpha_moment_rhs_mode_structure_fixture,
)
from jax_fluxtube_gk.validation.fixture_io import write_per_ky_mode_structure_fixture_csv


def main() -> None:
    args = _parse_args()
    fixture = run_s_alpha_moment_rhs_mode_structure_fixture(
        ky_values=_parse_float_tuple(args.ky_values),
        n_z=args.n_z,
        n_hermite=args.n_hermite,
        n_laguerre=args.n_laguerre,
        nperiod=args.nperiod,
        dt=args.dt,
        steps_per_window=args.steps_per_window,
        n_windows=args.n_windows,
        growth_window_fraction=args.growth_window_fraction,
        density_gradient=args.density_gradient,
        temperature_gradient=args.temperature_gradient,
        tau=args.tau,
        magnetic_shear=args.magnetic_shear,
        drift_scale=args.drift_scale,
        drive_scale=args.drive_scale,
        streaming_scale=args.streaming_scale,
        nu_hyper_m=args.nu_hyper_m,
        p_hyper_m=args.p_hyper_m,
        normalize_each_window=not args.no_normalize_each_window,
        initial_profile=args.initial_profile,
        initial_width=args.initial_width,
    )
    write_per_ky_mode_structure_fixture_csv(args.output, fixture)
    print(
        f"wrote {fixture.ky.shape[0]} ky rows, "
        f"n_z={fixture.z.shape[0]}, source={fixture.source}"
    )
    print(args.output)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ky-values", default="0.3,0.5")
    parser.add_argument("--n-z", type=int, default=96)
    parser.add_argument("--n-hermite", type=int, default=48)
    parser.add_argument("--n-laguerre", type=int, default=16)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps-per-window", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=40)
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--density-gradient", type=float, default=0.8)
    parser.add_argument("--temperature-gradient", type=float, default=2.49)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--magnetic-shear", type=float, default=0.8)
    parser.add_argument("--drift-scale", type=float, default=0.18)
    parser.add_argument("--drive-scale", type=float, default=1.0)
    parser.add_argument("--streaming-scale", type=float, default=1.0)
    parser.add_argument("--nu-hyper-m", type=float, default=1.0)
    parser.add_argument("--p-hyper-m", type=int)
    parser.add_argument("--no-normalize-each-window", action="store_true")
    parser.add_argument(
        "--initial-profile",
        choices=("gaussian", "cosine", "cosine2"),
        default="gaussian",
    )
    parser.add_argument("--initial-width", type=float, default=0.35)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/s_alpha_moment_rhs_mode_structure_fixture.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
