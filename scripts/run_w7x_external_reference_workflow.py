"""Drive or audit the external GX W7-X mode-structure reference workflow.

This script is the executable wrapper around the external production-claim
blocker.  It can copy the VMEC file, run GX when a real executable is provided,
export the retained ``.big.nc`` mode-structure stream once it exists, and write
a machine-readable status when the local machine cannot perform the external
GPU run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "fixtures/gx_w7x_mode_structure_run/mode_structure_run_metadata.json"
DEFAULT_STATUS = ROOT / "fixtures/gx_w7x_mode_structure_run/external_reference_status.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_w7x_external_reference_workflow(args)
    _write_json(args.status_output, report)
    print(f"{'PASS' if report['passed'] else 'OPEN'}: {report['status']}")
    print(args.status_output)
    return 0 if report["passed"] or not args.require_pass else 2


def run_w7x_external_reference_workflow(args: argparse.Namespace) -> dict[str, object]:
    """Run the available pieces of the external W7-X reference workflow."""

    metadata = _load_json(args.metadata)
    paths = _workflow_paths(metadata, args)
    actions: list[str] = []
    commands: list[str] = []

    if args.copy_vmec:
        _copy_vmec_if_possible(paths, actions)

    if paths["external_fixture"].exists():
        status = "external_fixture_available"
        passed = True
    elif paths["gx_big_output"].exists() and paths["gx_growth_output"].exists():
        if args.export_outputs:
            export_status = _export_external_fixture(paths, metadata, args, commands)
            status = export_status
            passed = paths["external_fixture"].exists()
            if not passed:
                actions.append("inspect GX export failure and regenerate external fixture")
        else:
            status = "gx_outputs_ready_for_export"
            passed = False
            actions.append("rerun with export enabled to create the external fixture CSV")
    elif args.run_gx:
        run_status = _run_gx_if_possible(paths, args, commands, actions)
        status = run_status
        passed = False
    else:
        status = _blocked_or_pending_status(paths, actions)
        passed = False

    report = {
        "benchmark_name": "w7x_itg_external_reference_workflow",
        "status": status,
        "passed": passed,
        "metadata": _display_path(args.metadata),
        "run_directory": _display_path(paths["run_dir"]),
        "prepared_input": _display_path(paths["prepared_input"]),
        "prepared_input_exists": paths["prepared_input"].exists(),
        "vmec_source": _display_path(paths["vmec_source"]),
        "vmec_source_exists": paths["vmec_source"].exists(),
        "vmec_destination": _display_path(paths["vmec_destination"]),
        "vmec_destination_exists": paths["vmec_destination"].exists(),
        "gx_executable": _display_path(paths["gx_executable"]),
        "gx_executable_exists": _is_executable(paths["gx_executable"]),
        "gx_executable_placeholder": _is_placeholder_executable(paths["gx_executable"]),
        "gx_big_output": _display_path(paths["gx_big_output"]),
        "gx_big_output_exists": paths["gx_big_output"].exists(),
        "gx_growth_output": _display_path(paths["gx_growth_output"]),
        "gx_growth_output_exists": paths["gx_growth_output"].exists(),
        "external_fixture": _display_path(paths["external_fixture"]),
        "external_fixture_exists": paths["external_fixture"].exists(),
        "copy_vmec": bool(args.copy_vmec),
        "run_gx": bool(args.run_gx),
        "export_outputs": bool(args.export_outputs),
        "commands": commands,
        "required_actions": actions,
    }
    return report


def _workflow_paths(metadata: dict[str, object], args: argparse.Namespace) -> dict[str, Path]:
    run_dir = _resolve(metadata["prepared_input"]).parent
    gx_executable = _select_gx_executable(metadata, args)
    return {
        "run_dir": run_dir,
        "prepared_input": _resolve(metadata["prepared_input"]),
        "vmec_source": _resolve(metadata["vmec_source"]),
        "vmec_destination": _resolve(metadata["vmec_destination"]),
        "gx_executable": gx_executable,
        "gx_big_output": _resolve(metadata["gx_big_output"]),
        "gx_growth_output": _resolve(metadata["gx_growth_output"]),
        "external_fixture": _resolve(metadata["external_fixture"]),
    }


def _select_gx_executable(metadata: dict[str, object], args: argparse.Namespace) -> Path:
    if args.gx_executable is not None:
        return args.gx_executable
    env_value = os.environ.get("GX_EXECUTABLE")
    if env_value:
        return Path(env_value)
    local_candidate = ROOT / "relevant-codes/gx/gx"
    if local_candidate.exists():
        return local_candidate
    return _resolve(metadata.get("gx_executable", "path/to/gx"))


def _copy_vmec_if_possible(paths: dict[str, Path], actions: list[str]) -> None:
    if not paths["vmec_source"].exists():
        actions.append(f"make VMEC source available at {_display_path(paths['vmec_source'])}")
        return
    paths["vmec_destination"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["vmec_source"], paths["vmec_destination"])


def _run_gx_if_possible(
    paths: dict[str, Path],
    args: argparse.Namespace,
    commands: list[str],
    actions: list[str],
) -> str:
    if not paths["prepared_input"].exists():
        actions.append(f"prepare GX input at {_display_path(paths['prepared_input'])}")
        return "blocked_missing_prepared_input"
    if not paths["vmec_destination"].exists():
        actions.append(
            "copy VMEC into the GX run directory with --copy-vmec before running GX"
        )
        return "blocked_missing_vmec_in_run_directory"
    if not _is_executable(paths["gx_executable"]):
        actions.append("build GX on a CUDA/NVIDIA-capable system or pass --gx-executable")
        return "blocked_missing_gx_executable"

    command = [str(paths["gx_executable"]), paths["prepared_input"].name]
    commands.append(_format_command(command, cwd=paths["run_dir"]))
    result = subprocess.run(
        command,
        cwd=paths["run_dir"],
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
    )
    _write_text(args.status_output.with_suffix(".gx.stdout.log"), result.stdout)
    _write_text(args.status_output.with_suffix(".gx.stderr.log"), result.stderr)
    if result.returncode != 0:
        actions.append("inspect GX stdout/stderr logs and rerun external reference workflow")
        return "gx_run_failed"
    if not paths["gx_big_output"].exists() or not paths["gx_growth_output"].exists():
        actions.append("GX run completed but retained .big.nc/.out.nc outputs are missing")
        return "gx_run_missing_expected_outputs"
    return "gx_outputs_ready_for_export"


def _export_external_fixture(
    paths: dict[str, Path],
    metadata: dict[str, object],
    args: argparse.Namespace,
    commands: list[str],
) -> str:
    command = [
        sys.executable,
        "examples/export_gx_mode_structure_fixture.py",
        "--gx-big-output",
        str(paths["gx_big_output"]),
        "--gx-growth-output",
        str(paths["gx_growth_output"]),
        "--ky-values",
        str(metadata["ky_values"]),
        "--gx-z-coordinate",
        str(metadata["gx_z_coordinate"]),
        "--output",
        str(paths["external_fixture"]),
    ]
    commands.append(_format_command(command, cwd=ROOT))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
    )
    _write_text(args.status_output.with_suffix(".export.stdout.log"), result.stdout)
    _write_text(args.status_output.with_suffix(".export.stderr.log"), result.stderr)
    return "external_fixture_exported" if result.returncode == 0 else "external_fixture_export_failed"


def _blocked_or_pending_status(paths: dict[str, Path], actions: list[str]) -> str:
    if not paths["prepared_input"].exists():
        actions.append(f"prepare GX input at {_display_path(paths['prepared_input'])}")
        return "blocked_missing_prepared_input"
    if not paths["vmec_source"].exists():
        actions.append(f"make VMEC source available at {_display_path(paths['vmec_source'])}")
        return "blocked_missing_vmec_source"
    if not _is_executable(paths["gx_executable"]):
        actions.append("build GX on a CUDA/NVIDIA-capable system or pass --gx-executable")
        actions.append("rerun this script with --copy-vmec --run-gx on that system")
        return "blocked_missing_gx_executable"
    actions.append("rerun this script with --copy-vmec --run-gx")
    return "pending_external_gx_run"


def _is_executable(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def _is_placeholder_executable(path: Path) -> bool:
    return str(path).replace("\\", "/").endswith("path/to/gx")


def _format_command(command: list[str], *, cwd: Path) -> str:
    prefix = f"cd {_display_path(cwd)} && "
    return prefix + " ".join(command)


def _resolve(path_like) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--gx-executable", type=Path)
    parser.add_argument("--copy-vmec", action="store_true")
    parser.add_argument("--run-gx", action="store_true")
    parser.add_argument("--no-export", dest="export_outputs", action="store_false")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--require-pass", action="store_true")
    parser.set_defaults(export_outputs=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
