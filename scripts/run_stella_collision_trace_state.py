"""Run one parameterized state through an already instrumented stella executable."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_stella_collision_field_particle_discriminator import (
    stella_collision_input,
)


def run_trace_state(
    executable: Path,
    output_dir: Path,
    *,
    initial_amplitude: float,
    initial_width: float,
    overwrite: bool = False,
) -> Path:
    """Write and run a distinct collision state in caller-owned scratch space."""

    executable = Path(executable).resolve()
    output_dir = Path(output_dir).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "collision_trace_state.in"
    metadata_path = output_dir / "collision_trace_state.json"
    if (input_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError("trace state exists; pass --overwrite")
    input_path.write_text(
        stella_collision_input(
            field_particle=True,
            initial_amplitude=initial_amplitude,
            initial_width=initial_width,
        )
    )
    subprocess.run([str(executable), input_path.name], cwd=output_dir, check=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "stella_collision_parameterized_trace_state",
                "status": "native_trace_state_completed",
                "executable": str(executable),
                "input": str(input_path),
                "initial_amplitude": initial_amplitude,
                "initial_width": initial_width,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-amplitude", type=float, required=True)
    parser.add_argument("--initial-width", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = run_trace_state(
        args.executable,
        args.output_dir,
        initial_amplitude=args.initial_amplitude,
        initial_width=args.initial_width,
        overwrite=args.overwrite,
    )
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
