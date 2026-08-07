"""Run pair-resolved signed field-particle traces with a patched stella build."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from scripts.prepare_stella_collision_field_particle_trace_run import (
    COMPONENT_TRACE_FILENAME,
    FACTOR_TRACE_FILENAME,
    TEST_PARTICLE_MATRIX_TRACE_FILENAME,
    TRACE_FILENAME,
)
from scripts.run_stella_collision_field_particle_discriminator import stella_collision_input
from scripts.summarize_stella_collision_field_particle_components import (
    summarize_component_trace,
)
from scripts.summarize_stella_collision_field_particle_factors import summarize_factor_trace
from scripts.summarize_stella_collision_field_particle_trace import summarize_trace
from scripts.summarize_stella_collision_test_particle_matrix import (
    COLUMNS as MATRIX_COLUMNS,
    SCHEMA as MATRIX_SCHEMA,
    _dense_matrices,
)


CHANNELS = {
    "all": (1.0, 1.0, 1.0, 1.0),
    "ion_ion": (1.0, 0.0, 0.0, 0.0),
    "ion_electron": (0.0, 1.0, 0.0, 0.0),
    "electron_electron": (0.0, 0.0, 1.0, 0.0),
    "electron_ion": (0.0, 0.0, 0.0, 1.0),
}


def _trace_state_and_rhs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.atleast_2d(np.loadtxt(path))
    if values.shape[1] != 13 or not np.isfinite(values).all():
        raise ValueError(f"invalid signed collision trace {path}")
    before = values[:, 9] + 1j * values[:, 10]
    rhs = values[:, 11] + 1j * values[:, 12]
    return before, rhs


def summarize_channel_traces(
    trace_paths: dict[str, Path],
    *,
    expected_revision: str,
    component_paths: dict[str, Path] | None = None,
    factor_paths: dict[str, Path] | None = None,
    matrix_paths: dict[str, Path] | None = None,
) -> dict[str, object]:
    """Validate common inputs and report isolated-channel closure."""

    if set(trace_paths) != set(CHANNELS):
        raise ValueError("channel trace set is incomplete")
    summaries = {
        name: summarize_trace(path, expected_revision=expected_revision)
        for name, path in trace_paths.items()
    }
    arrays = {name: _trace_state_and_rhs(path) for name, path in trace_paths.items()}
    common_input = arrays["all"][0]
    if any(not np.array_equal(before, common_input) for before, _rhs in arrays.values()):
        raise ValueError("collision channel traces do not share an identical input state")
    isolated_sum = sum(arrays[name][1] for name in CHANNELS if name != "all")
    full_rhs = arrays["all"][1]
    full_norm = max(float(np.linalg.norm(full_rhs)), np.finfo(float).tiny)
    report = {
        "schema_version": 1,
        "benchmark": "stella_collision_pair_resolved_field_particle_trace",
        "status": "pair_resolved_native_traces_passed",
        "stella_source_revision": expected_revision,
        "channels": summaries,
        "metrics": {
            "isolated_sum_to_full_relative_l2": float(
                np.linalg.norm(isolated_sum - full_rhs) / full_norm
            ),
            "identical_input_state": True,
        },
        "scope": "native pair-resolved targets; local common-grid parity pending",
    }
    if component_paths is not None:
        if set(component_paths) != set(CHANNELS):
            raise ValueError("channel component trace set is incomplete")
        report["component_channels"] = {
            name: summarize_component_trace(
                component_paths[name],
                trace_paths[name],
                expected_revision=expected_revision,
            )
            for name in CHANNELS
        }
        report["metrics"]["component_reconstruction_passed"] = True
        report["scope"] = (
            "native pair-resolved Laguerre--Legendre action targets; "
            "local coefficient parity pending"
        )
    if factor_paths is not None:
        if set(factor_paths) != set(CHANNELS):
            raise ValueError("channel factor trace set is incomplete")
        report["factor_channels"] = {
            name: summarize_factor_trace(
                factor_paths[name],
                trace_paths[name],
                expected_revision=expected_revision,
            )
            for name in CHANNELS
        }
        report["metrics"]["local_jax_factor_replay_passed"] = True
        report["scope"] = (
            "native pair-resolved Laguerre--Legendre factors and local JAX "
            "action replay; local coefficient construction pending"
        )
    if matrix_paths is not None:
        if set(matrix_paths) != set(CHANNELS):
            raise ValueError("channel test-particle matrix trace set is incomplete")
        matrix_sets = {}
        for name, path in matrix_paths.items():
            header = Path(path).read_text(encoding="utf-8").splitlines()[:1]
            if not header or header[0].strip() != MATRIX_SCHEMA:
                raise ValueError(f"invalid collision matrix schema for channel {name}")
            values = np.atleast_2d(np.loadtxt(path))
            if values.shape[1] != len(MATRIX_COLUMNS) or not np.isfinite(values).all():
                raise ValueError(f"invalid collision matrix trace for channel {name}")
            matrix_sets[name] = _dense_matrices(values)
        keys = set(matrix_sets["all"])
        if any(set(matrices) != keys for matrices in matrix_sets.values()):
            raise ValueError("collision channel matrix traces have different mode sets")
        full_parts = []
        closure_parts = []
        channel_effects = {name: [] for name in CHANNELS if name != "all"}
        for key in sorted(keys):
            full = matrix_sets["all"][key]
            identity = np.eye(full.shape[0], dtype=full.dtype)
            reconstructed = identity.copy()
            for name in CHANNELS:
                if name == "all":
                    continue
                effect = matrix_sets[name][key] - identity
                reconstructed += effect
                channel_effects[name].append(effect.reshape(-1))
            full_parts.append(full.reshape(-1))
            closure_parts.append((reconstructed - full).reshape(-1))
        full_values = np.concatenate(full_parts)
        closure = np.concatenate(closure_parts)
        full_scale = max(float(np.linalg.norm(full_values)), np.finfo(float).tiny)
        report["test_particle_matrix_channels"] = {
            name: {
                "effect_frobenius": float(
                    np.linalg.norm(np.concatenate(channel_effects[name]))
                )
            }
            for name in channel_effects
        }
        matrix_closure_error = float(np.linalg.norm(closure) / full_scale)
        if matrix_closure_error > 1.0e-12:
            raise ValueError(
                "isolated test-particle matrices do not reconstruct the all-channel matrix"
            )
        report["metrics"]["test_particle_matrix_isolated_sum_relative_l2"] = (
            matrix_closure_error
        )
        report["metrics"]["test_particle_matrix_channel_decomposition_passed"] = True
        report["scope"] = (
            "native pair-resolved field-particle and test-particle targets; "
            "local coefficient construction pending"
        )
    return report


def run_channel_traces(
    output_dir: Path,
    patched_stella_executable: Path,
    *,
    expected_revision: str,
    overwrite: bool = False,
) -> dict[str, object]:
    """Execute the full and four isolated native collision-channel cases."""

    output_dir = Path(output_dir).resolve()
    executable = Path(patched_stella_executable).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_paths: dict[str, Path] = {}
    component_paths: dict[str, Path] = {}
    factor_paths: dict[str, Path] = {}
    matrix_paths: dict[str, Path] = {}
    for name, knobs in CHANNELS.items():
        input_path = output_dir / f"collision_{name}.in"
        trace_path = output_dir / f"collision_{name}_field_particle_trace.dat"
        if (input_path.exists() or trace_path.exists()) and not overwrite:
            raise FileExistsError(f"channel {name!r} exists; pass --overwrite")
        input_path.write_text(
            stella_collision_input(field_particle=True, collision_knobs=knobs),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [str(executable), input_path.name],
            cwd=output_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        (output_dir / f"collision_{name}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        generated_trace = output_dir / TRACE_FILENAME
        if not generated_trace.is_file():
            raise FileNotFoundError(generated_trace)
        generated_trace.replace(trace_path)
        trace_paths[name] = trace_path
        generated_components = output_dir / COMPONENT_TRACE_FILENAME
        if not generated_components.is_file():
            raise FileNotFoundError(generated_components)
        component_path = output_dir / f"collision_{name}_field_particle_components.dat"
        generated_components.replace(component_path)
        component_paths[name] = component_path
        generated_factors = output_dir / FACTOR_TRACE_FILENAME
        if not generated_factors.is_file():
            raise FileNotFoundError(generated_factors)
        factor_path = output_dir / f"collision_{name}_field_particle_factors.dat"
        generated_factors.replace(factor_path)
        factor_paths[name] = factor_path
        generated_matrix = output_dir / TEST_PARTICLE_MATRIX_TRACE_FILENAME
        if not generated_matrix.is_file():
            raise FileNotFoundError(generated_matrix)
        matrix_path = output_dir / f"collision_{name}_test_particle_matrix.dat"
        generated_matrix.replace(matrix_path)
        matrix_paths[name] = matrix_path

    report = summarize_channel_traces(
        trace_paths,
        expected_revision=expected_revision,
        component_paths=component_paths,
        factor_paths=factor_paths,
        matrix_paths=matrix_paths,
    )
    report["patched_stella_executable"] = str(executable)
    report_path = output_dir / "collision_channel_trace_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patched-stella-executable", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = run_channel_traces(
        args.output_dir,
        args.patched_stella_executable,
        expected_revision=args.expected_revision,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
