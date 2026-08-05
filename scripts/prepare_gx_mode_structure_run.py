"""Prepare a GX run directory that retains full-field mode structures.

The local GX Cyclone benchmark includes compact ``.out.nc`` spectra but not a
retained ``.big.nc`` field diagnostic.  This helper copies a GX input file,
forces the diagnostics needed by ``examples/export_gx_mode_structure_fixture.py``,
and writes a small metadata/README bundle with the external commands to run.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("fixtures/gx_cyclone_mode_structure_run")
DEFAULT_FIXTURE_OUTPUT = Path("fixtures/gx_cyclone_mode_structure_fixture.csv")


def patch_gx_diagnostics_text(
    text: str,
    *,
    nwrite_big: int,
    omega: bool = True,
    fields: bool = True,
    moments: bool = True,
) -> str:
    """Return GX input text with retained field diagnostics enabled."""

    if nwrite_big < 1:
        raise ValueError("nwrite_big must be positive")
    desired = {
        "nwrite_big": str(int(nwrite_big)),
        "omega": _toml_bool(omega),
        "fields": _toml_bool(fields),
        "moments": _toml_bool(moments),
    }
    lines = text.splitlines(keepends=True)
    start, end = _section_bounds(lines, "Diagnostics")
    if start is None:
        prefix = "" if text.endswith("\n") or not text else "\n"
        diagnostic_lines = ["[Diagnostics]\n"]
        diagnostic_lines.extend(f" {key} = {value}\n" for key, value in desired.items())
        return text + prefix + "".join(diagnostic_lines)

    patched = list(lines)
    seen: set[str] = set()
    for index in range(start, end):
        line = patched[index]
        for key, value in desired.items():
            if _is_toml_key_line(line, key):
                patched[index] = _replace_toml_key_line(line, key, value)
                seen.add(key)
                break

    insert_at = end
    missing = [key for key in desired if key not in seen]
    if missing:
        insertion = [f" {key} = {desired[key]}\n" for key in missing]
        patched[insert_at:insert_at] = insertion
    return "".join(patched)


def prepare_gx_mode_structure_run(
    input_path: Path,
    output_dir: Path,
    *,
    nwrite_big: int,
    gx_executable: str,
    ky_values: str,
    gx_z_coordinate: str,
    fixture_output: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Copy and patch a GX input file and write run/export metadata."""

    source = Path(input_path)
    destination_dir = Path(output_dir)
    fixture_output = Path(fixture_output)
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir.mkdir(parents=True, exist_ok=True)

    output_input = destination_dir / source.name
    if output_input.exists() and not overwrite:
        raise FileExistsError(f"{output_input} already exists; pass --overwrite")

    patched = patch_gx_diagnostics_text(
        source.read_text(),
        nwrite_big=nwrite_big,
        omega=True,
        fields=True,
        moments=True,
    )
    output_input.write_text(patched)

    run_base = output_input.with_suffix("").name
    gx_big_output = destination_dir / f"{run_base}.big.nc"
    gx_growth_output = destination_dir / f"{run_base}.out.nc"
    run_command = (
        f"cd {shlex.quote(str(destination_dir))} && "
        f"{shlex.quote(gx_executable)} {shlex.quote(output_input.name)}"
    )
    export_command = (
        "uv run python examples/export_gx_mode_structure_fixture.py "
        f"--gx-big-output {shlex.quote(str(gx_big_output))} "
        f"--gx-growth-output {shlex.quote(str(gx_growth_output))} "
        f"--ky-values {shlex.quote(ky_values)} "
        f"--gx-z-coordinate {shlex.quote(gx_z_coordinate)} "
        f"--output {shlex.quote(str(fixture_output))}"
    )
    gate_command = (
        "JAX_ENABLE_X64=1 uv run python "
        "examples/run_cyclone_mode_structure_gate.py "
        f"--reference-fixture {shlex.quote(str(fixture_output))} "
        "--profile gx-salpha-input --target-convention gx-salpha "
        "--ky-input-convention internal_krho --require-profile "
        "--resample-reference-to-solver-z --periodic-z"
    )
    metadata: dict[str, object] = {
        "source_input": str(source),
        "prepared_input": str(output_input),
        "nwrite_big": int(nwrite_big),
        "diagnostics": {"omega": True, "fields": True, "moments": True},
        "gx_executable": gx_executable,
        "gx_big_output": str(gx_big_output),
        "gx_growth_output": str(gx_growth_output),
        "ky_values": ky_values,
        "gx_z_coordinate": gx_z_coordinate,
        "fixture_output": str(fixture_output),
        "run_command": run_command,
        "export_command": export_command,
        "gate_command": gate_command,
    }
    metadata_path = destination_dir / "mode_structure_run_metadata.json"
    readme_path = destination_dir / "README.md"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    readme_path.write_text(_readme_text(metadata))
    return metadata


def _readme_text(metadata: dict[str, object]) -> str:
    return "\n".join(
        (
            "# GX Mode-Structure Fixture Run",
            "",
            "This directory contains a patched GX input that retains the full-field",
            "`.big.nc` diagnostics needed for the multi-ky mode-structure gate.",
            "",
            "1. Run GX externally:",
            "",
            f"   `{metadata['run_command']}`",
            "",
            "2. Export the retained GX field diagnostic to the portable fixture:",
            "",
            f"   `{metadata['export_command']}`",
            "",
            "3. Compare the current solver against that fixture:",
            "",
            f"   `{metadata['gate_command']}`",
            "",
            "The compact `.out.nc` file alone is not enough for this gate; the",
            "required complex profiles are stored in `Diagnostics/Phi` in `.big.nc`.",
            "",
        )
    )


def _section_bounds(lines: list[str], name: str) -> tuple[int | None, int]:
    section_re = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")
    start: int | None = None
    for index, line in enumerate(lines):
        match = section_re.match(line.strip())
        if match and match.group(1).strip() == name:
            start = index + 1
            break
    if start is None:
        return None, len(lines)
    end = len(lines)
    for index in range(start, len(lines)):
        if section_re.match(lines[index].strip()):
            end = index
            break
    return start, end


def _is_toml_key_line(line: str, key: str) -> bool:
    return re.match(rf"^\s*{re.escape(key)}\s*=", line) is not None


def _replace_toml_key_line(line: str, key: str, value: str) -> str:
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    comment_match = re.search(r"\s+#.*$", body)
    comment = comment_match.group(0) if comment_match else ""
    indent = re.match(r"^\s*", body).group(0)
    return f"{indent}{key} = {value}{comment}{newline}"


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nwrite-big", type=int, default=1000)
    parser.add_argument("--gx-executable", default="path/to/gx")
    parser.add_argument("--ky-values", default="0.3,0.5")
    parser.add_argument(
        "--gx-z-coordinate",
        choices=("theta", "theta_over_2pi"),
        default="theta_over_2pi",
    )
    parser.add_argument("--fixture-output", type=Path, default=DEFAULT_FIXTURE_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from stellarator_gk.external import announce_external_path

    announce_external_path("GX input", args.input)
    metadata = prepare_gx_mode_structure_run(
        args.input,
        args.output_dir,
        nwrite_big=args.nwrite_big,
        gx_executable=args.gx_executable,
        ky_values=args.ky_values,
        gx_z_coordinate=args.gx_z_coordinate,
        fixture_output=args.fixture_output,
        overwrite=args.overwrite,
    )
    print(metadata["prepared_input"])
    print(metadata["run_command"])


if __name__ == "__main__":
    main()
