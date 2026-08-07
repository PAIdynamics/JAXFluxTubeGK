#!/usr/bin/env python3
"""Summarize a revision-pinned GX nonlinear heat-flux output in JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def summarize_heat_flux(times, heat_flux, *, start_fraction: float = 0.5) -> dict:
    """Return late-window statistics for one GX species heat-flux trace."""

    times = np.asarray(times, dtype=float)
    heat_flux = np.asarray(heat_flux, dtype=float)
    if times.ndim != 1 or heat_flux.shape != times.shape or times.size < 2:
        raise ValueError("time and heat flux must be matching one-dimensional traces")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1)")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(heat_flux)):
        raise ValueError("GX heat-flux traces must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("GX diagnostic times must be strictly increasing")
    start = min(int(times.size * start_fraction), times.size - 1)
    if times.size - start < 2:
        raise ValueError("stationarity window must contain at least two samples")
    window_time = times[start:]
    window_flux = heat_flux[start:]
    mean = float(np.mean(window_flux))
    standard_deviation = float(np.std(window_flux, ddof=1))
    centered_time = window_time - np.mean(window_time)
    slope = float(
        np.sum(centered_time * (window_flux - mean)) / np.sum(centered_time**2)
    )
    relative_drift = float(
        slope * (window_time[-1] - window_time[0]) / max(abs(mean), 1.0e-14)
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / np.sqrt(window_flux.size),
        "relative_window_drift": relative_drift,
        "n_samples": int(window_flux.size),
        "window_start_time": float(window_time[0]),
        "window_end_time": float(window_time[-1]),
    }


def read_gx_heat_flux(path: Path, species_index: int = 0):
    """Read GX ``time`` and ``HeatFlux_st`` arrays without importing GX."""

    from netCDF4 import Dataset

    with Dataset(path, mode="r") as dataset:
        times = np.asarray(dataset.groups["Grids"].variables["time"][:], dtype=float)
        all_fluxes = np.asarray(
            dataset.groups["Diagnostics"].variables["HeatFlux_st"][:], dtype=float
        )
    if all_fluxes.ndim != 2 or not 0 <= species_index < all_fluxes.shape[1]:
        raise ValueError("GX HeatFlux_st must have shape (time, species)")
    return times, all_fluxes[:, species_index]


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-root", type=Path, required=True)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--species-index", type=int, default=0)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    gx_root = args.gx_root.expanduser().resolve()
    revision = _git_revision(gx_root)
    if args.expected_revision is not None and revision != args.expected_revision:
        raise RuntimeError(
            f"GX revision mismatch: found {revision}, expected {args.expected_revision}"
        )
    netcdf = args.netcdf.expanduser().resolve()
    times, heat_flux = read_gx_heat_flux(netcdf, args.species_index)
    payload = {
        "schema_version": 1,
        "producer": "gx-nonlinear-heat-flux",
        "normalization": "gx_Q_over_Q_GB",
        "revision": revision,
        "source_netcdf": str(netcdf),
        "species_index": args.species_index,
        "start_fraction": args.start_fraction,
        "statistics": summarize_heat_flux(
            times, heat_flux, start_fraction=args.start_fraction
        ),
        "times": times.tolist(),
        "heat_flux": heat_flux.tolist(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
