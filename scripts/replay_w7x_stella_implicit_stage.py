"""Replay a stage-resolved stella W7-X implicit trace in JAX."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scripts.compare_w7x_stella_rhs_trace_to_solver_balance import load_stella_array_trace
from scripts.replay_w7x_stella_state_in_solver import (
    apply_stella_coefficient_contract,
    build_native_stella_setup,
    replay_cases,
)
from stellarator_gk import (
    build_implicit_parallel_response_precompute,
    implicit_parallel_response_step,
    linear_residual,
    mirror_force,
    parallel_field_drive,
    parallel_streaming,
    semi_lagrangian_mirror_step,
    solve_field_from_state,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = (
    ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"
)
DEFAULT_EXPLICIT_SUMMARY = (
    ROOT / "fixtures/w7x_ky03_stella_rhs_trace_summary/rhs_trace_summary.json"
)
DEFAULT_OUTPUT = Path("/private/tmp/stellarator_gk_w7x_implicit_stage_replay.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--explicit-summary", type=Path, default=DEFAULT_EXPLICIT_SUMMARY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = replay_implicit_stage(
        trace=args.trace,
        explicit_summary=args.explicit_summary,
        geometry=args.geometry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def load_implicit_stage_trace(path: Path) -> dict[str, np.ndarray]:
    """Load trace rows into canonical PDF and field arrays."""

    rows = np.genfromtxt(path, names=True, dtype=None, encoding=None)
    result: dict[str, np.ndarray] = {}
    for stage in np.unique(rows["stage"]):
        selected = rows[rows["stage"] == stage]
        if selected["record"][0] == "pdf":
            values = np.empty((32, 8, 257), dtype=np.complex128)
            for row in selected:
                values[row["iv"] - 1, row["imu"] - 1, row["iz"] + 128] = (
                    row["real"] + 1j * row["imag"]
                )
        else:
            values = np.empty(257, dtype=np.complex128)
            for row in selected:
                values[row["iz"] + 128] = row["real"] + 1j * row["imag"]
        result[str(stage)] = values
    return result


def replay_implicit_stage(*, trace: Path, explicit_summary: Path, geometry: Path):
    """Return fixed and best-fit errors for the final PDF and potential."""

    stages = load_implicit_stage_trace(trace)
    summary = json.loads(Path(explicit_summary).read_text(encoding="utf-8"))
    native_trace = load_stella_array_trace(Path(summary["trace_path"]), summary)
    case = replace(
        next(case for case in replay_cases() if case.name == "replay_stella_native_32x8"),
        parallel_derivative_model="matrix",
    )
    setup = build_native_stella_setup(case, Path(geometry), native_trace)
    precompute = apply_stella_coefficient_contract(case, setup["precompute"], Path(geometry))
    state = jnp.asarray(stages["input_pdf"][:, :, :-1, None, None])
    response = build_implicit_parallel_response_precompute(
        precompute,
        0.1,
        spatial_scheme="stella_near_centered",
    )
    final_state = implicit_parallel_response_step(state, precompute, response)
    final_phi = solve_field_from_state(final_state, precompute)
    mirror_input = jnp.asarray(stages["mirror_input_pdf"][:, :, :-1, None, None])
    mirror_coefficient = precompute.rhs.mirror_force_coeff
    if mirror_coefficient.ndim == 3:
        mirror_coefficient = mirror_coefficient[0]
    mirror_final = semi_lagrangian_mirror_step(
        mirror_input,
        0.1,
        setup["velocity"].vpar,
        mirror_coefficient,
        interpolation="stella_cubic",
    )
    explicit_states = {
        name: jnp.asarray(stages[name][:, :, :-1, None, None])
        for name in (
            "explicit_input_pdf",
            "explicit_state1_pdf",
            "explicit_state2_pdf",
        )
    }

    def explicit_increment(value):
        phi = solve_field_from_state(value, precompute)
        residual = linear_residual(value, precomputed=precompute)
        residual -= parallel_streaming(
            value, precompute.rhs.D_z, precompute.rhs.parallel_streaming_coeff
        )
        residual -= parallel_field_drive(phi, precompute.rhs.D_z, precompute.rhs)
        residual -= mirror_force(
            value, precompute.rhs.D_vpar, precompute.rhs.mirror_force_coeff
        )
        return 0.1 * residual

    explicit_rhs = {
        f"rhs{index}": explicit_increment(explicit_states[state_name])
        for index, state_name in enumerate(
            ("explicit_input_pdf", "explicit_state1_pdf", "explicit_state2_pdf"),
            start=1,
        )
    }
    explicit_final = (
        explicit_states["explicit_input_pdf"] / 3.0
        + 0.5 * (explicit_states["explicit_input_pdf"] + explicit_rhs["rhs1"])
        + (explicit_states["explicit_state2_pdf"] + explicit_rhs["rhs3"]) / 6.0
    )
    return {
        "schema": "stellarator_gk_w7x_implicit_stage_replay_v1",
        "trace": str(Path(trace).resolve()),
        "distribution": _metrics(stages["final_pdf"][:, :, :-1], final_state[..., 0, 0]),
        "potential": _metrics(stages["final_phi"][:-1], final_phi[:, 0, 0]),
        "input_quasineutrality": _metrics(
            stages["input_phi"][:-1],
            solve_field_from_state(state, precompute)[:, 0, 0],
        ),
        "mirror_distribution": _metrics(
            stages["mirror_final_pdf"][:, :, :-1], mirror_final[..., 0, 0]
        ),
        "explicit_rhs": {
            name: _metrics(stages[f"explicit_{name}_pdf"][:, :, :-1], values[..., 0, 0])
            for name, values in explicit_rhs.items()
        },
        "explicit_distribution": _metrics(
            stages["explicit_final_pdf"][:, :, :-1], explicit_final[..., 0, 0]
        ),
    }


def _metrics(reference, candidate) -> dict[str, object]:
    reference = np.asarray(reference).reshape(-1)
    candidate = np.asarray(candidate).reshape(-1)
    scale = np.vdot(candidate, reference) / np.vdot(candidate, candidate)
    return {
        "relative_l2_error": float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference)),
        "best_fit_relative_l2_error": float(
            np.linalg.norm(scale * candidate - reference) / np.linalg.norm(reference)
        ),
        "best_fit_scale_real": float(scale.real),
        "best_fit_scale_imag": float(scale.imag),
    }


if __name__ == "__main__":
    raise SystemExit(main())
