"""Export a GX .big.nc complex-field diagnostic to a per-ky fixture CSV.

Run from the repository root after a GX run that retained the .big.nc stream:

    uv run --extra dev python examples/export_gx_mode_structure_fixture.py \
        --gx-big-output path/to/run.big.nc \
        --gx-growth-output path/to/run.out.nc \
        --ky-values 0.3,0.5 \
        --gx-z-coordinate theta_over_2pi
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from stellarator_gk import (
    load_gx_mode_structure_fixture,
    write_per_ky_mode_structure_fixture_csv,
)


def main() -> None:
    args = _parse_args()
    fixture = load_gx_mode_structure_fixture(
        args.gx_big_output,
        growth_reference_path=args.gx_growth_output,
        ikx=args.ikx,
        time_index=args.time_index,
        ky_values=_parse_float_tuple(args.ky_values) if args.ky_values else None,
        average_fraction=args.average_fraction,
        drop_zonal=not args.keep_zonal,
        z_scale=_gx_z_scale(args.gx_z_coordinate),
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


def _gx_z_scale(name: str) -> float:
    if name == "theta":
        return 1.0
    if name == "theta_over_2pi":
        return 1.0 / (2.0 * np.pi)
    raise ValueError(f"unsupported GX z-coordinate convention {name!r}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gx-big-output", type=Path, required=True)
    parser.add_argument("--gx-growth-output", type=Path)
    parser.add_argument("--ky-values", default="0.3,0.5")
    parser.add_argument("--ikx", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--average-fraction", type=float, default=0.5)
    parser.add_argument("--keep-zonal", action="store_true")
    parser.add_argument(
        "--gx-z-coordinate",
        choices=("theta", "theta_over_2pi"),
        default="theta_over_2pi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/gx_cyclone_mode_structure_fixture.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
