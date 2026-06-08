"""Prepare the external GX W7-X run needed for benchmark parity.

The repository contains the GX W7-X input deck and VMEC file, but not a
matching retained ``.big.nc`` field diagnostic.  This helper writes a patched
GX input plus metadata that tells the user how to run GX, export the resulting
mode-structure fixture, and compare it against the committed reduced W7-X
``stellarator_gk`` fixture.
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_gx_mode_structure_run import patch_gx_diagnostics_text  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "relevant-codes/gx/benchmarks/linear/ITG_w7x/"
    "itg_w7x_adiabatic_electrons.in"
)
DEFAULT_VMEC = ROOT / "relevant-codes/gx/benchmarks/linear/ITG_w7x/wout_w7x.nc"
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/gx_w7x_mode_structure_run"
DEFAULT_EXTERNAL_FIXTURE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_OBSERVED_FIXTURE = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"
DEFAULT_COMPARISON_OUTPUT = ROOT / "figures/w7x_itg_external_mode_structure_comparison.csv"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = prepare_gx_w7x_mode_structure_run(
        args.input,
        args.output_dir,
        vmec_file=args.vmec_file,
        nwrite_big=args.nwrite_big,
        gx_executable=args.gx_executable,
        ky_values=args.ky_values,
        gx_z_coordinate=args.gx_z_coordinate,
        external_fixture=args.external_fixture,
        observed_fixture=args.observed_fixture,
        comparison_output=args.comparison_output,
        copy_vmec=args.copy_vmec,
        overwrite=args.overwrite,
    )
    print(metadata["prepared_input"])
    print(metadata["copy_vmec_command"])
    print(metadata["run_command"])
    return 0


def prepare_gx_w7x_mode_structure_run(
    input_path: Path,
    output_dir: Path,
    *,
    vmec_file: Path,
    nwrite_big: int,
    gx_executable: str,
    ky_values: str,
    gx_z_coordinate: str,
    external_fixture: Path,
    observed_fixture: Path,
    comparison_output: Path,
    copy_vmec: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Patch the GX W7-X input and write an external benchmark workflow."""

    source = Path(input_path)
    destination_dir = Path(output_dir)
    vmec_file = Path(vmec_file)
    if not source.exists():
        raise FileNotFoundError(source)
    if not vmec_file.exists():
        raise FileNotFoundError(vmec_file)
    if nwrite_big < 1:
        raise ValueError("nwrite_big must be positive")

    destination_dir.mkdir(parents=True, exist_ok=True)
    prepared_input = destination_dir / source.name
    vmec_destination = destination_dir / vmec_file.name
    if prepared_input.exists() and not overwrite:
        raise FileExistsError(f"{prepared_input} already exists; pass --overwrite")
    if copy_vmec and vmec_destination.exists() and not overwrite:
        raise FileExistsError(f"{vmec_destination} already exists; pass --overwrite")

    patched = patch_gx_diagnostics_text(source.read_text(), nwrite_big=nwrite_big)
    prepared_input.write_text(patched)
    if copy_vmec:
        shutil.copy2(vmec_file, vmec_destination)

    run_base = prepared_input.with_suffix("").name
    gx_big_output = destination_dir / f"{run_base}.big.nc"
    gx_growth_output = destination_dir / f"{run_base}.out.nc"
    copy_vmec_command = (
        f"cp {shlex.quote(_display_path(vmec_file))} "
        f"{shlex.quote(_display_path(vmec_destination))}"
    )
    run_command = (
        f"cd {shlex.quote(_display_path(destination_dir))} && "
        f"{shlex.quote(gx_executable)} {shlex.quote(prepared_input.name)}"
    )
    export_command = (
        "uv run python examples/export_gx_mode_structure_fixture.py "
        f"--gx-big-output {shlex.quote(_display_path(gx_big_output))} "
        f"--gx-growth-output {shlex.quote(_display_path(gx_growth_output))} "
        f"--ky-values {shlex.quote(ky_values)} "
        f"--gx-z-coordinate {shlex.quote(gx_z_coordinate)} "
        f"--output {shlex.quote(_display_path(external_fixture))}"
    )
    comparison_command = (
        "JAX_ENABLE_X64=1 uv run python examples/compare_mode_structure_fixtures.py "
        f"--observed {shlex.quote(_display_path(observed_fixture))} "
        f"--reference {shlex.quote(_display_path(external_fixture))} "
        f"--ky-values {shlex.quote(ky_values)} "
        "--require-profile "
        f"--output {shlex.quote(_display_path(comparison_output))}"
    )
    metadata: dict[str, object] = {
        "benchmark_name": "w7x_itg_external_gx_mode_structure_reference",
        "status": "pending_external_gx_run",
        "source_input": _display_path(source),
        "prepared_input": _display_path(prepared_input),
        "vmec_source": _display_path(vmec_file),
        "vmec_destination": _display_path(vmec_destination),
        "vmec_copied": bool(copy_vmec),
        "nwrite_big": int(nwrite_big),
        "diagnostics": {"omega": True, "fields": True, "moments": True},
        "gx_executable": gx_executable,
        "gx_big_output": _display_path(gx_big_output),
        "gx_growth_output": _display_path(gx_growth_output),
        "ky_values": ky_values,
        "gx_z_coordinate": gx_z_coordinate,
        "external_fixture": _display_path(external_fixture),
        "observed_fixture": _display_path(observed_fixture),
        "comparison_output": _display_path(comparison_output),
        "copy_vmec_command": copy_vmec_command,
        "run_command": run_command,
        "export_command": export_command,
        "comparison_command": comparison_command,
        "notes": (
            "The committed W7-X fixture is real-geometry/reduced-solver. "
            "Run these commands externally to replace or compare it against "
            "a retained GX W7-X complex mode-structure reference."
        ),
    }
    (destination_dir / "mode_structure_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (destination_dir / "README.md").write_text(_readme_text(metadata))
    return metadata


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _readme_text(metadata: dict[str, object]) -> str:
    return "\n".join(
        (
            "# GX W7-X Mode-Structure Reference Run",
            "",
            "This directory contains the patched GX W7-X linear ITG input for",
            "producing the external complex mode-structure reference required",
            "to upgrade the reduced W7-X fixture into a full code-to-code gate.",
            "",
            "1. Copy the VMEC file into the run directory:",
            "",
            f"   `{metadata['copy_vmec_command']}`",
            "",
            "2. Run GX externally:",
            "",
            f"   `{metadata['run_command']}`",
            "",
            "3. Export the retained GX field diagnostic to a portable fixture:",
            "",
            f"   `{metadata['export_command']}`",
            "",
            "4. Compare the current solver fixture against the external one:",
            "",
            f"   `{metadata['comparison_command']}`",
            "",
            "The compact `.out.nc` file alone is not enough for profile parity;",
            "the complex profiles are read from `Diagnostics/Phi` in `.big.nc`.",
            "",
        )
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vmec-file", type=Path, default=DEFAULT_VMEC)
    parser.add_argument("--nwrite-big", type=int, default=100)
    parser.add_argument("--gx-executable", default="path/to/gx")
    parser.add_argument("--ky-values", default="0.1,0.2,0.3")
    parser.add_argument(
        "--gx-z-coordinate",
        choices=("theta", "theta_over_2pi"),
        default="theta_over_2pi",
    )
    parser.add_argument("--external-fixture", type=Path, default=DEFAULT_EXTERNAL_FIXTURE)
    parser.add_argument("--observed-fixture", type=Path, default=DEFAULT_OBSERVED_FIXTURE)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--copy-vmec", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
