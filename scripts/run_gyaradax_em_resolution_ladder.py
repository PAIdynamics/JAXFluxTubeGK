#!/usr/bin/env python3
"""Run an artifact-free local/Gyaradax electromagnetic resolution ladder."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

from jax_fluxtube_gk import (
    compare_tem_resolution_ladder,
    gyaradax_tem_case_spec,
    load_tem_external_reference,
    run_reduced_tem_linear_smoke,
)


RESOLUTION_PROFILES = {
    "reduced": ((8, 8, 4), (12, 12, 6), (16, 16, 8)),
    "production": ((16, 16, 8), (24, 24, 12), (32, 32, 16)),
}


def _parse_resolution(value: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must use n_z x n_vpar x n_mu") from exc
    if len(values) != 3 or min(values) < 2:
        raise argparse.ArgumentTypeError("resolution must contain three integers at least two")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gyaradax-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument(
        "--profile",
        choices=tuple(RESOLUTION_PROFILES),
        default="reduced",
        help="named rung set used when --resolution is not supplied",
    )
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        action="append",
        dest="resolutions",
        help="one n_z x n_vpar x n_mu rung; repeat for multiple rungs",
    )
    parser.add_argument("--n-windows", type=int, default=500)
    parser.add_argument("--steps-per-window", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--growth-convergence-tolerance", type=float, default=0.05)
    parser.add_argument("--frequency-convergence-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.resolutions is None:
        args.resolutions = list(RESOLUTION_PROFILES[args.profile])
    if len(args.resolutions) < 2:
        parser.error("at least two --resolution rungs are required")
    if args.n_windows < 3 or args.steps_per_window < 1:
        parser.error("n-windows must be at least three and steps-per-window positive")
    if args.dt <= 0.0 or args.beta <= 0.0:
        parser.error("dt and beta must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    producer = Path(__file__).with_name("run_gyaradax_tem_reference.py")
    results = []
    references = []
    rung_payloads = []

    for n_z, n_vpar, n_mu in args.resolutions:
        label = f"{n_z}x{n_vpar}x{n_mu}"
        reference_path = output_dir / f"gyaradax-em-{label}.json"
        command = [
            sys.executable,
            str(producer),
            "--gyaradax-root",
            str(args.gyaradax_root),
            "--output",
            str(reference_path),
            "--n-z",
            str(n_z),
            "--n-vpar",
            str(n_vpar),
            "--n-mu",
            str(n_mu),
            "--n-windows",
            str(args.n_windows),
            "--steps-per-window",
            str(args.steps_per_window),
            "--dt",
            str(args.dt),
            "--nlapar",
            "--nlbpar",
            "--beta",
            str(args.beta),
        ]
        if args.expected_revision is not None:
            command.extend(("--expected-revision", args.expected_revision))
        subprocess.run(command, check=True)

        spec = replace(
            gyaradax_tem_case_spec(),
            n_z=n_z,
            n_vpar=n_vpar,
            n_mu=n_mu,
        )
        result = run_reduced_tem_linear_smoke(
            spec,
            field_model="electromagnetic",
            beta=args.beta,
            dt=args.dt,
            steps_per_window=args.steps_per_window,
            n_windows=args.n_windows,
            allow_timestep_above_conservative_bound=True,
        )
        reference = load_tem_external_reference(reference_path)
        results.append(result)
        references.append(reference)
        rung_payloads.append(
            {
                "resolution": [n_z, n_vpar, n_mu],
                "local_growth_rate": result.growth_rate,
                "local_frequency": result.frequency,
                "local_late_growth_delta": result.late_window_growth_delta,
                "local_estimated_cfl_dt": result.estimated_cfl_dt,
                "reference_growth_rate": reference.growth_rate,
                "reference_frequency": reference.frequency,
                "reference_path": str(reference_path),
            }
        )

    report = compare_tem_resolution_ladder(
        tuple(results),
        tuple(references),
        growth_convergence_tolerance=args.growth_convergence_tolerance,
        frequency_convergence_tolerance=args.frequency_convergence_tolerance,
    )
    payload = {
        "schema_version": 1,
        "producer": "jax-fluxtube-gk/gyaradax-electromagnetic-resolution-ladder",
        "profile": args.profile if args.resolutions == list(RESOLUTION_PROFILES[args.profile]) else "custom",
        "beta": args.beta,
        "dt": args.dt,
        "steps_per_window": args.steps_per_window,
        "n_windows": args.n_windows,
        "rungs": rung_payloads,
        "report": asdict(report),
    }
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary}")
    print(
        "passed={} local finest changes growth={:.6g} frequency={:.6g}".format(
            report.passed,
            report.local_growth_relative_change,
            report.local_frequency_relative_change,
        )
    )
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
