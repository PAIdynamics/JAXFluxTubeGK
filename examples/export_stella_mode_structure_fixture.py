"""Export a stella .out.nc linear run to a per-ky mode-structure fixture CSV.

Run from the repository root after a stella run with potential and omega
diagnostics enabled:

    uv run python examples/export_stella_mode_structure_fixture.py \
        --stella-output path/to/input.out.nc \
        --ky-values 0.1,0.2,0.3 \
        --output fixtures/w7x_itg_external_mode_structure_fixture.csv
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
    load_stella_mode_structure_fixture,
    write_per_ky_mode_structure_fixture_csv,
)


def main() -> None:
    args = _parse_args()
    fixture = load_stella_mode_structure_fixture(
        args.stella_output,
        ikx=args.ikx,
        tube_index=args.tube_index,
        time_index=args.time_index,
        ky_values=_parse_float_tuple(args.ky_values) if args.ky_values else None,
        average_fraction=args.average_fraction,
        drop_zonal=not args.keep_zonal,
        ky_tolerance=args.ky_tolerance,
        z_scale=_stella_z_scale(args.stella_z_coordinate),
        growth_scale=args.growth_scale,
        frequency_scale=args.frequency_scale,
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


def _stella_z_scale(name: str) -> float:
    if name == "zed_over_2pi":
        return 1.0 / (2.0 * np.pi)
    if name == "zed":
        return 1.0
    raise ValueError(f"unsupported stella z-coordinate convention {name!r}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stella-output", type=Path, required=True)
    parser.add_argument("--ky-values", default="0.1,0.2,0.3")
    parser.add_argument("--ikx", type=int, default=0)
    parser.add_argument("--tube-index", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--average-fraction", type=float, default=0.5)
    parser.add_argument("--ky-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--growth-scale", type=float, default=1.0)
    parser.add_argument("--frequency-scale", type=float, default=1.0)
    parser.add_argument("--keep-zonal", action="store_true")
    parser.add_argument(
        "--stella-z-coordinate",
        choices=("zed", "zed_over_2pi"),
        default="zed_over_2pi",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/w7x_itg_external_mode_structure_fixture.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
