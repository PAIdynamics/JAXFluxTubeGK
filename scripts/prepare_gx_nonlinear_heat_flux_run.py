#!/usr/bin/env python3
"""Prepare a pinned, case-matched external GX nonlinear heat-flux run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PINNED_GX_REVISION = "bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5"


def _replace_scalar(text: str, section: str, name: str, value: str) -> str:
    pattern = re.compile(rf"(?ms)(^\s*\[{re.escape(section)}\]\s*$.*?)(?=^\s*\[|\Z)")
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"GX input lacks [{section}] section")
    block = match.group(1)
    assignment = re.compile(rf"(?m)^(\s*{re.escape(name)}\s*=\s*)[^#\n]*(.*)$")
    if not assignment.search(block):
        block = block.rstrip() + f"\n {name} = {value}\n\n"
    else:
        block = assignment.sub(rf"\g<1>{value}\g<2>", block, count=1)
    return text[: match.start(1)] + block + text[match.end(1) :]


def patch_gx_nonlinear_input(
    text: str,
    *,
    ntheta: int = 24,
    nx: int = 32,
    ny: int = 16,
    nhermite: int = 8,
    nlaguerre: int = 4,
    y0: float = 10.0,
    final_time: float = 500.0,
    random_seed: int = 19,
    nwrite: int = 20,
) -> str:
    """Return a GX s-alpha input matched to the local nonlinear physics/box."""

    integer_controls = (ntheta, nx, ny, nhermite, nlaguerre, random_seed, nwrite)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_controls):
        raise ValueError("GX dimensions, seed, and diagnostic cadence must be integers")
    if min(integer_controls) < 1:
        raise ValueError("GX dimensions, seed, and diagnostic cadence must be positive")
    if not all(math.isfinite(value) and value > 0.0 for value in (y0, final_time)):
        raise ValueError("GX y0 and final time must be positive")
    updates = (
        ("Dimensions", "ntheta", str(ntheta)),
        ("Dimensions", "nperiod", "1"),
        ("Dimensions", "nx", str(nx)),
        ("Dimensions", "ny", str(ny)),
        ("Dimensions", "nhermite", str(nhermite)),
        ("Dimensions", "nlaguerre", str(nlaguerre)),
        ("Dimensions", "nspecies", "1"),
        ("Domain", "y0", repr(float(y0))),
        ("Domain", "boundary", '"linked"'),
        ("Domain", "jtwist", "1"),
        ("Physics", "beta", "0.0"),
        ("Physics", "nonlinear_mode", "true"),
        ("Time", "t_max", repr(float(final_time))),
        ("Time", "nstep", "100000000"),
        ("Initialization", "random_init", "true"),
        ("Initialization", "random_seed", str(random_seed)),
        ("Initialization", "init_amp", "1.0e-3"),
        ("Geometry", "geo_option", '"s-alpha"'),
        ("Geometry", "eps", "0.18"),
        ("Geometry", "Rmaj", "2.77778"),
        ("Geometry", "qinp", "1.4"),
        ("Geometry", "shat", "0.8"),
        ("species", "vnewk", "[0.0, 0.0]"),
        ("Dissipation", "hypercollisions", "true"),
        ("Dissipation", "hyper", "true"),
        ("Dissipation", "D_hyper", "0.05"),
        ("Dissipation", "p_hyper", "2"),
        ("Restart", "restart", "false"),
        ("Restart", "save_for_restart", "true"),
        ("Diagnostics", "nwrite", str(nwrite)),
        ("Diagnostics", "fluxes", "true"),
    )
    for section, name, value in updates:
        text = _replace_scalar(text, section, name, value)
    return text


def prepare_gx_nonlinear_heat_flux_run(
    gx_root: Path,
    output_dir: Path,
    *,
    expected_revision: str = PINNED_GX_REVISION,
    gx_executable: str = "gx",
    overwrite: bool = False,
    **controls,
) -> dict:
    """Write a patched input and self-contained run/summarization manifest."""

    gx_root = Path(gx_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("GX nonlinear output must be outside the repository")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=gx_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise RuntimeError(f"GX revision mismatch: found {revision}, expected {expected_revision}")
    source = gx_root / "unit_tests/inputs/cyc_nl.in"
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = output_dir / "jax_fluxtube_gk_cyclone_nonlinear.in"
    manifest_path = output_dir / "gx_nonlinear_run.json"
    if not overwrite and (prepared.exists() or manifest_path.exists()):
        raise FileExistsError("GX nonlinear run already exists; pass --overwrite")
    patched = patch_gx_nonlinear_input(source.read_text(encoding="utf-8"), **controls)
    prepared.write_text(patched, encoding="utf-8")
    digest = hashlib.sha256(patched.encode()).hexdigest()
    netcdf = prepared.with_suffix(".nc")
    summary = output_dir / "gx_nonlinear_heat_flux.json"
    run_command = (
        f"cd {shlex.quote(str(output_dir))} && {shlex.quote(gx_executable)} {prepared.name}"
    )
    summary_command = (
        f"{shlex.quote(str(ROOT / '.venv/bin/python'))} "
        f"{shlex.quote(str(ROOT / 'scripts/summarize_gx_nonlinear_heat_flux.py'))} "
        f"--gx-root {shlex.quote(str(gx_root))} --expected-revision {revision} "
        f"--netcdf {shlex.quote(str(netcdf))} --run-manifest {shlex.quote(str(manifest_path))} "
        f"--output {shlex.quote(str(summary))}"
    )
    manifest = {
        "schema_version": 1,
        "status": "pending_external_gx_run",
        "gx_revision": revision,
        "source_input": str(source),
        "prepared_input": str(prepared),
        "prepared_input_sha256": digest,
        "expected_netcdf": str(netcdf),
        "summary_output": str(summary),
        "run_command": run_command,
        "summary_command": summary_command,
        "case_contract": {
            "ntheta": int(controls.get("ntheta", 24)),
            "nx": int(controls.get("nx", 32)),
            "ny": int(controls.get("ny", 16)),
            "nhermite": int(controls.get("nhermite", 8)),
            "nlaguerre": int(controls.get("nlaguerre", 4)),
            "final_time": float(controls.get("final_time", 500.0)),
            "random_seed": int(controls.get("random_seed", 19)),
            "nwrite": int(controls.get("nwrite", 20)),
            "geometry": "s-alpha",
            "q": 1.4,
            "shat": 0.8,
            "eps": 0.18,
            "rmaj_over_lref": 2.77778,
            "fprim": 0.8,
            "tprim": 2.49,
            "ky_min": 1.0 / float(controls.get("y0", 10.0)),
            "boundary": "linked",
            "electrostatic": True,
            "hyperdiffusion": 0.05,
            "hyperdiffusion_order": 4,
            "collision_frequency": 0.0,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", default=PINNED_GX_REVISION)
    parser.add_argument("--gx-executable", default="gx")
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nhermite", type=int, default=8)
    parser.add_argument("--nlaguerre", type=int, default=4)
    parser.add_argument("--y0", type=float, default=10.0)
    parser.add_argument("--final-time", type=float, default=500.0)
    parser.add_argument("--random-seed", type=int, default=19)
    parser.add_argument("--nwrite", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    controls = vars(args).copy()
    for key in ("gx_root", "output_dir", "expected_revision", "gx_executable", "overwrite"):
        controls.pop(key)
    manifest = prepare_gx_nonlinear_heat_flux_run(
        args.gx_root,
        args.output_dir,
        expected_revision=args.expected_revision,
        gx_executable=args.gx_executable,
        overwrite=args.overwrite,
        **controls,
    )
    print(manifest["run_command"])
    print(manifest["summary_command"])


if __name__ == "__main__":
    main()
