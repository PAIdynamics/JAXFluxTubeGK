"""Ingest returned GX W7-X external-reference outputs.

Run this from the repository root after a GX/CUDA machine has produced the
retained W7-X ``.big.nc`` and ``.out.nc`` files.  The command copies the
returned files into the prepared run directory when requested, exports the
portable per-ky fixture CSV, runs the W7-X mode-structure gate, and refreshes
the production-readiness ledger.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "fixtures/gx_w7x_mode_structure_run/mode_structure_run_metadata.json"
DEFAULT_STATUS = ROOT / "fixtures/gx_w7x_mode_structure_run/external_reference_ingest_status.json"
DEFAULT_EXTERNAL_GATE_DIR = ROOT / "fixtures/w7x_itg_convergence_study/external_mode_structure_gate"
DEFAULT_READINESS_OUTPUT = ROOT / "fixtures/w7x_itg_convergence_study/production_readiness_gate.json"
DEFAULT_PRODUCTION_TIMING = ROOT / "fixtures/w7x_itg_convergence_study/production_cpu_timing.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = ingest_w7x_external_reference_outputs(args)
    _write_json(args.status_output, report)
    print(f"{'PASS' if report['passed'] else 'OPEN'}: {report['status']}")
    print(args.status_output)
    return 0 if report["passed"] or not args.require_pass else 2


def ingest_w7x_external_reference_outputs(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Ingest, export, compare, and refresh readiness for returned GX outputs."""

    metadata = _load_json(args.metadata)
    paths = _paths(metadata, args)
    missing = [
        name
        for name in ("gx_big_output", "gx_growth_output")
        if not paths[name].exists()
    ]
    if missing:
        return _base_report(
            metadata=metadata,
            paths=paths,
            args=args,
            status="blocked_missing_external_outputs",
            passed=False,
            required_actions=[
                "copy the returned GX .big.nc/.out.nc files into the prepared "
                "run directory or pass --gx-big-output/--gx-growth-output"
            ],
            missing_outputs=missing,
        )

    copied = _copy_outputs_if_requested(paths, args.copy_outputs)
    export = _export_external_fixture(paths, metadata)
    if not paths["external_fixture"].exists():
        return _base_report(
            metadata=metadata,
            paths=paths,
            args=args,
            status="external_fixture_export_failed",
            passed=False,
            required_actions=[
                "inspect the returned GX NetCDF files and rerun the ingest command"
            ],
            copied_outputs=copied,
            export=export,
        )

    gate = None
    if not args.skip_gate:
        gate = _run_mode_structure_gate(paths, args)
        if not gate.get("passed"):
            return _base_report(
                metadata=metadata,
                paths=paths,
                args=args,
                status="external_mode_structure_gate_open",
                passed=False,
                required_actions=[
                    "inspect the W7-X mode-structure gate output and resolve "
                    "external parity before making a production claim"
                ],
                copied_outputs=copied,
                export=export,
                external_mode_structure_gate=gate,
            )

    readiness = None
    if not args.skip_readiness:
        readiness = _run_production_readiness_gate(paths, args)

    passed = bool(gate and gate.get("passed")) and bool(readiness and readiness.get("passed"))
    status = _status_after_ingest(gate, readiness, args)
    actions = []
    if gate is None:
        actions.append("run the W7-X mode-structure gate against the exported fixture")
    if readiness is None:
        actions.append("refresh the W7-X production-readiness gate")
    if readiness is not None and not readiness.get("passed"):
        actions.extend(str(item) for item in readiness.get("required_actions", ()))
    return _base_report(
        metadata=metadata,
        paths=paths,
        args=args,
        status=status,
        passed=passed,
        required_actions=actions,
        copied_outputs=copied,
        export=export,
        external_mode_structure_gate=gate,
        production_readiness_gate=readiness,
    )


def _paths(metadata: dict[str, object], args: argparse.Namespace) -> dict[str, Path]:
    return {
        "gx_big_output": _resolve(args.gx_big_output or metadata["gx_big_output"]),
        "gx_growth_output": _resolve(args.gx_growth_output or metadata["gx_growth_output"]),
        "target_gx_big_output": _resolve(metadata["gx_big_output"]),
        "target_gx_growth_output": _resolve(metadata["gx_growth_output"]),
        "external_fixture": _resolve(args.output_fixture or metadata["external_fixture"]),
        "observed_fixture": _resolve(args.observed_fixture),
        "external_gate_dir": _resolve(args.external_gate_dir),
        "production_readiness_output": _resolve(args.production_readiness_output),
        "production_timing": _resolve(args.production_timing),
    }


def _copy_outputs_if_requested(paths: dict[str, Path], copy_outputs: bool) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    if not copy_outputs:
        return copied
    for source_name, target_name in (
        ("gx_big_output", "target_gx_big_output"),
        ("gx_growth_output", "target_gx_growth_output"),
    ):
        source = paths[source_name]
        target = paths[target_name]
        if source.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        paths[source_name] = target
        copied.append({"source": _display_path(source), "target": _display_path(target)})
    return copied


def _export_external_fixture(
    paths: dict[str, Path],
    metadata: dict[str, object],
) -> dict[str, object]:
    from jax_fluxtube_gk import (
        load_gx_mode_structure_fixture,
        write_per_ky_mode_structure_fixture_csv,
    )

    fixture = load_gx_mode_structure_fixture(
        paths["gx_big_output"],
        growth_reference_path=paths["gx_growth_output"],
        ky_values=_parse_float_tuple(str(metadata["ky_values"])),
        z_scale=_gx_z_scale(str(metadata["gx_z_coordinate"])),
    )
    write_per_ky_mode_structure_fixture_csv(paths["external_fixture"], fixture)
    return {
        "status": "external_fixture_exported",
        "source_big": _display_path(paths["gx_big_output"]),
        "source_growth": _display_path(paths["gx_growth_output"]),
        "output_fixture": _display_path(paths["external_fixture"]),
        "ky_count": int(fixture.ky.shape[0]),
        "n_z": int(fixture.z.shape[0]),
    }


def _run_mode_structure_gate(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, object]:
    from examples.run_w7x_mode_structure_gate import main as run_gate

    command = [
        "--observed-fixture",
        str(paths["observed_fixture"]),
        "--reference-fixture",
        str(paths["external_fixture"]),
        "--output-dir",
        str(paths["external_gate_dir"]),
        "--ky-values",
        args.ky_values,
    ]
    if args.resample_reference_to_observed_z:
        command.append("--resample-reference-to-observed-z")
    run_gate(command)
    return _load_json(paths["external_gate_dir"] / "gate_status.json")


def _run_production_readiness_gate(
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    module = _load_script_module("run_w7x_production_readiness_gate")
    return module.run_w7x_production_readiness_gate(
        observed_fixture=paths["observed_fixture"],
        reference_fixture=paths["external_fixture"],
        external_gate_dir=paths["external_gate_dir"],
        output_path=paths["production_readiness_output"],
        production_timing_path=paths["production_timing"],
        ky_values=args.ky_values,
    )


def _base_report(
    *,
    metadata: dict[str, object],
    paths: dict[str, Path],
    args: argparse.Namespace,
    status: str,
    passed: bool,
    required_actions: list[str],
    **extra,
) -> dict[str, object]:
    report = {
        "benchmark_name": "w7x_itg_external_reference_ingest",
        "status": status,
        "passed": passed,
        "metadata": _display_path(args.metadata),
        "gx_big_output": _display_path(paths["gx_big_output"]),
        "gx_big_output_exists": paths["gx_big_output"].exists(),
        "gx_growth_output": _display_path(paths["gx_growth_output"]),
        "gx_growth_output_exists": paths["gx_growth_output"].exists(),
        "target_gx_big_output": _display_path(paths["target_gx_big_output"]),
        "target_gx_growth_output": _display_path(paths["target_gx_growth_output"]),
        "external_fixture": _display_path(paths["external_fixture"]),
        "external_fixture_exists": paths["external_fixture"].exists(),
        "observed_fixture": _display_path(paths["observed_fixture"]),
        "external_gate_dir": _display_path(paths["external_gate_dir"]),
        "ky_values": args.ky_values,
        "metadata_ky_values": metadata.get("ky_values"),
        "copy_outputs": bool(args.copy_outputs),
        "resample_reference_to_observed_z": bool(args.resample_reference_to_observed_z),
        "required_actions": required_actions,
    }
    report.update(extra)
    return report


def _status_after_ingest(
    gate: dict[str, object] | None,
    readiness: dict[str, object] | None,
    args: argparse.Namespace,
) -> str:
    if args.skip_gate:
        return "external_fixture_exported_gate_skipped"
    if gate is not None and not gate.get("passed"):
        return "external_mode_structure_gate_open"
    if args.skip_readiness:
        return "external_parity_passed_readiness_skipped"
    if readiness is not None and readiness.get("passed"):
        return "pass"
    if readiness is not None:
        return "external_parity_passed_readiness_open"
    return "external_fixture_exported"


def _load_script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def _gx_z_scale(name: str) -> float:
    if name == "theta":
        return 1.0
    if name == "theta_over_2pi":
        import numpy as np

        return 1.0 / (2.0 * np.pi)
    raise ValueError(f"unsupported GX z-coordinate convention {name!r}")


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    metadata = _load_json(DEFAULT_METADATA) if DEFAULT_METADATA.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--status-output", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--gx-big-output", type=Path)
    parser.add_argument("--gx-growth-output", type=Path)
    parser.add_argument("--output-fixture", type=Path)
    parser.add_argument("--observed-fixture", type=Path, default=ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv")
    parser.add_argument("--external-gate-dir", type=Path, default=DEFAULT_EXTERNAL_GATE_DIR)
    parser.add_argument("--production-readiness-output", type=Path, default=DEFAULT_READINESS_OUTPUT)
    parser.add_argument("--production-timing", type=Path, default=DEFAULT_PRODUCTION_TIMING)
    parser.add_argument("--ky-values", default=str(metadata.get("ky_values", "0.1,0.2,0.3")))
    parser.add_argument("--copy-outputs", action="store_true")
    parser.add_argument("--resample-reference-to-observed-z", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
