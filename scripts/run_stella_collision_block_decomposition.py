"""Split native stella collision blocks into vpar and mu operator paths."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.prepare_stella_collision_field_particle_trace_run import (
    TEST_PARTICLE_BLOCK_TRACE_FILENAME,
)
from scripts.run_stella_collision_field_particle_discriminator import (
    stella_collision_input,
)
from scripts.summarize_stella_collision_test_particle_blocks import (
    COLUMNS,
    SCHEMA,
    _block_arrays,
    _read,
)


def _component_metrics(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, object]:
    n_vpar = arrays[0].shape[2]
    boundary = np.asarray((0, n_vpar - 1)) if n_vpar > 1 else np.asarray((0,))
    interior = np.arange(1, n_vpar - 1)

    def norm(values: np.ndarray) -> float:
        return float(np.linalg.norm(values))

    all_values = np.concatenate([array.ravel() for array in arrays])
    total = norm(all_values)
    by_block = {
        name: norm(array)
        for name, array in zip(("lower", "diagonal", "upper"), arrays, strict=True)
    }
    boundary_values = np.concatenate([array[:, :, boundary].ravel() for array in arrays])
    interior_values = np.concatenate([array[:, :, interior].ravel() for array in arrays])
    by_pair = {}
    for target in range(arrays[0].shape[0]):
        for background in range(arrays[0].shape[1]):
            values = np.concatenate([array[target, background].ravel() for array in arrays])
            by_pair[f"target_{target + 1}_background_{background + 1}"] = norm(values)
    return {
        "frobenius": total,
        "imaginary_max_abs": float(np.max(np.abs(all_values.imag))),
        "by_block_frobenius": by_block,
        "by_pair_frobenius": by_pair,
        "vpar_boundary_frobenius": norm(boundary_values),
        "vpar_interior_frobenius": norm(interior_values),
        "vpar_boundary_fraction": norm(boundary_values) / max(total, np.finfo(float).tiny),
    }


def summarize_block_decomposition(
    full_trace: Path,
    vpar_trace: Path,
    *,
    expected_revision: str,
) -> dict[str, object]:
    """Validate and summarize an additive native ``mu_operator`` split."""

    full_values = _read(full_trace, SCHEMA, len(COLUMNS))
    vpar_values = _read(vpar_trace, SCHEMA, len(COLUMNS))
    full = _block_arrays(full_values)
    vpar = _block_arrays(vpar_values)
    if any(not np.array_equal(left, right) for left, right in zip(full[3:], vpar[3:], strict=True)):
        raise ValueError("block decomposition traces use different index axes")
    if not np.array_equal(full_values[:, :6], vpar_values[:, :6]):
        raise ValueError("block decomposition traces use different row ordering")
    if not np.array_equal(full_values[:, 12], vpar_values[:, 12]):
        raise ValueError("block decomposition traces use different timesteps")

    full_arrays = full[:3]
    vpar_arrays = vpar[:3]
    mu_arrays = tuple(total - base for total, base in zip(full_arrays, vpar_arrays, strict=True))
    reconstructed = tuple(base + mu for base, mu in zip(vpar_arrays, mu_arrays, strict=True))
    closure = np.concatenate(
        [
            (observed - expected).ravel()
            for observed, expected in zip(reconstructed, full_arrays, strict=True)
        ]
    )
    full_flat = np.concatenate([array.ravel() for array in full_arrays])
    relative_l2 = float(
        np.linalg.norm(closure) / max(np.linalg.norm(full_flat), np.finfo(float).tiny)
    )
    maximum = float(np.max(np.abs(closure)))
    if maximum > 1.0e-14:
        raise ValueError("mu-operator decomposition does not reconstruct the full blocks")
    component_metrics = {
        "full": _component_metrics(full_arrays),
        "vpar_path": _component_metrics(vpar_arrays),
        "mu_path": _component_metrics(mu_arrays),
    }
    full_norm = component_metrics["full"]["frobenius"]
    component_metrics["vpar_path"]["fraction_of_full_frobenius"] = (
        component_metrics["vpar_path"]["frobenius"] / full_norm
    )
    component_metrics["mu_path"]["fraction_of_full_frobenius"] = (
        component_metrics["mu_path"]["frobenius"] / full_norm
    )
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_test_particle_block_decomposition",
        "status": "native_block_operator_split_passed",
        "stella_source_revision": expected_revision,
        "full_trace": str(Path(full_trace).resolve()),
        "vpar_trace": str(Path(vpar_trace).resolve()),
        "rows_per_trace": int(full_values.shape[0]),
        "grid": {
            "nz": int(full[3].size),
            "nspecies": int(full[4].size),
            "nvpar": int(full_arrays[0].shape[2]),
            "nmu": int(full_arrays[0].shape[3]),
        },
        "components": component_metrics,
        "metrics": {
            "additive_closure_relative_l2": relative_l2,
            "additive_closure_max_abs": maximum,
        },
        "scope": (
            "native operator-path localization; independent local finite-difference "
            "coefficient construction pending"
        ),
    }


def run_block_decomposition(
    executable: Path,
    output_dir: Path,
    *,
    expected_revision: str,
    overwrite: bool = False,
) -> Path:
    """Run identical native cases with ``mu_operator`` enabled and disabled."""

    executable = Path(executable).resolve()
    output_dir = Path(output_dir).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = {}
    base_input = stella_collision_input(field_particle=True)
    marker = "  mu_operator = .true."
    if base_input.count(marker) != 1:
        raise ValueError("expected one mu_operator marker in stella input")
    for name, enabled in (("full", True), ("vpar", False)):
        input_path = output_dir / f"collision_blocks_{name}.in"
        trace_path = output_dir / f"collision_blocks_{name}.dat"
        log_path = output_dir / f"collision_blocks_{name}.log"
        if any(path.exists() for path in (input_path, trace_path, log_path)) and not overwrite:
            raise FileExistsError(f"block decomposition case {name!r} exists; pass --overwrite")
        payload = base_input if enabled else base_input.replace(marker, "  mu_operator = .false.")
        input_path.write_text(payload, encoding="utf-8")
        completed = subprocess.run(
            [str(executable), input_path.name],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        generated = output_dir / TEST_PARTICLE_BLOCK_TRACE_FILENAME
        if not generated.is_file():
            raise FileNotFoundError(generated)
        generated.replace(trace_path)
        traces[name] = trace_path
    report = summarize_block_decomposition(
        traces["full"],
        traces["vpar"],
        expected_revision=expected_revision,
    )
    report_path = output_dir / "collision_block_decomposition.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run_block_decomposition(
        args.executable,
        args.output_dir,
        expected_revision=args.expected_revision,
        overwrite=args.overwrite,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
