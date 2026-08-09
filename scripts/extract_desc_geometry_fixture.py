#!/usr/bin/env python3
"""Extract a DESC example equilibrium onto the jax_fluxtube_gk flux-tube grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np

from jax_fluxtube_gk import (
    build_boozer_parallel_grid,
    desc_geometry_arrays_from_equilibrium,
    desc_geometry_arrays_from_path,
)


def main() -> int:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.desc_root is not None:
        sys.path.insert(0, str(args.desc_root.resolve()))

    parallel_grid = build_boozer_parallel_grid(
        n_z=args.n_z,
        n_turns=args.n_turns,
        center=args.zeta_center,
    )
    if args.desc_path is None:
        arrays = _arrays_from_example(args, parallel_grid)
        source = args.example
    else:
        arrays = desc_geometry_arrays_from_path(
            args.desc_path,
            parallel_grid,
            rho=args.rho,
            alpha=args.alpha,
            iota=args.iota,
            file_format=args.file_format,
            index=args.family_index,
        )
        source = str(args.desc_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "z": np.asarray(parallel_grid.z),
        "w_z": np.asarray(parallel_grid.w_z),
        "rho_input": np.asarray(args.rho),
        "alpha_input": np.asarray(args.alpha),
        "source": np.asarray(source),
    }
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    np.savez(args.output, **payload)

    print(f"wrote {args.output}")
    print(f"source={source} rho={args.rho:g} alpha={args.alpha:g} n_z={args.n_z}")
    print("arrays: " + ", ".join(sorted(arrays)))
    return 0


def _arrays_from_example(args, parallel_grid):
    try:
        from desc.examples import get
    except ImportError as exc:
        raise SystemExit(
            "Could not import DESC. Install desc-opt or pass --desc-root "
            "pointing at a DESC checkout with its Python dependencies available. "
            f"Original error: {exc}"
        ) from exc
    eq = get(args.example)
    return desc_geometry_arrays_from_equilibrium(
        eq,
        parallel_grid,
        rho=args.rho,
        alpha=args.alpha,
        iota=args.iota,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desc-root", type=Path, help="optional path to a DESC checkout")
    parser.add_argument("--example", default="DSHAPE", help="DESC example equilibrium name")
    parser.add_argument("--desc-path", type=Path, help="DESC HDF5/pickle equilibrium path")
    parser.add_argument("--file-format", choices=("hdf5", "pickle"), help="DESC file format")
    parser.add_argument("--family-index", type=int, default=-1, help="index if file contains a family")
    parser.add_argument("--rho", type=float, default=0.5, help="flux-surface label")
    parser.add_argument("--alpha", type=float, default=0.0, help="field-line label")
    parser.add_argument("--iota", type=float, default=None, help="optional iota override")
    parser.add_argument("--n-z", type=int, default=33, help="parallel grid nodes")
    parser.add_argument("--n-turns", type=int, default=1, help="toroidal turns in zeta")
    parser.add_argument("--zeta-center", type=float, default=0.0, help="center of zeta grid")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/desc_geometry_dshape_rho05_alpha0.npz"),
        help="output .npz fixture path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
