"""Generate reduced optimization traces and publication-style PDF figures.

Run from the repository root:

    uv run --extra dev python examples/generate_optimization_figures.py
"""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jax_fluxtube_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from jax_fluxtube_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    OptimizationKnobs,
    ParallelGridSpec,
    SingleSurfaceOptimizationConfig,
    VelocityGridSpec,
    build_fourier_grid,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    single_surface_objective,
)


@dataclass(frozen=True)
class OptimizationCase:
    label: str
    ky_values: tuple[float, ...]
    selected_ky: int | None
    objective_kind: str
    scalar_mode: str
    knobs: OptimizationKnobs
    learning_rate: float
    target_growth: float = 0.0
    iterations: int = 1000
    dt: float = 0.01
    n_steps: int = 2


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    traces = [_run_case(case) for case in _cases()]
    _write_csv(output_dir / "optimization_traces.csv", traces)
    _write_line_pdf(
        output_dir / "optimization_objectives.pdf",
        title="Absolute objective error",
        y_label=r"absolute objective error",
        curves=[
            (
                f"Case {trace['label']}",
                trace["iteration"],
                trace["objective_error"],
            )
            for trace in traces
        ],
        y_scale="log",
    )
    _write_line_pdf(
        output_dir / "optimization_growth_rates.pdf",
        title="Growth-rate diagnostic changes",
        y_label=r"growth rate minus initial",
        curves=_growth_rate_curves(traces),
    )
    _write_line_pdf(
        output_dir / "optimization_geometry_knobs.pdf",
        title="Geometry knob changes during optimization",
        y_label="knob minus initial",
        curves=[
            (f"Case {trace['label']} q", trace["iteration"], trace["q"] - trace["q"][0])
            for trace in traces
        ]
        + [
            (
                f"Case {trace['label']} shat",
                trace["iteration"],
                trace["shat"] - trace["shat"][0],
            )
            for trace in traces
        ],
    )
    for path in (
        "optimization_objectives.pdf",
        "optimization_growth_rates.pdf",
        "optimization_geometry_knobs.pdf",
    ):
        print(output_dir / path)


def _cases() -> tuple[OptimizationCase, ...]:
    return (
        OptimizationCase(
            label="A",
            ky_values=(0.0, 0.35),
            selected_ky=1,
            objective_kind="selected_growth",
            scalar_mode="config",
            learning_rate=1.0e-3,
            knobs=OptimizationKnobs(
                density=0.9,
                temperature=1.2,
                density_gradient=0.8,
                temperature_gradient=2.1,
                q=1.25,
                shat=0.45,
                eps=0.17,
                rho=0.45,
                alpha=0.2,
                beta=0.03,
                pressure_gradient=0.04,
                equilibrium_coefficients=(0.04, -0.015),
            ),
        ),
        OptimizationCase(
            label="B",
            ky_values=(0.0, 0.25, 0.50),
            selected_ky=2,
            objective_kind="selected_growth",
            scalar_mode="config",
            learning_rate=1.0e-3,
            knobs=OptimizationKnobs(
                density=1.0,
                temperature=1.0,
                density_gradient=0.8,
                temperature_gradient=2.49,
                q=1.35,
                shat=0.65,
                eps=0.18,
                rho=0.55,
                alpha=0.55,
                beta=0.02,
                pressure_gradient=0.03,
                equilibrium_coefficients=(0.03, 0.02, -0.01),
            ),
        ),
        OptimizationCase(
            label="C",
            ky_values=(0.0, 0.25, 0.50),
            selected_ky=1,
            objective_kind="selected_growth",
            scalar_mode="max_nonzonal",
            learning_rate=1.0e-3,
            knobs=OptimizationKnobs(
                density=0.95,
                temperature=1.1,
                density_gradient=0.7,
                temperature_gradient=2.3,
                q=1.45,
                shat=0.75,
                eps=0.19,
                rho=0.65,
                alpha=0.85,
                beta=0.04,
                pressure_gradient=0.05,
                equilibrium_coefficients=(0.02, -0.01, 0.015),
            ),
        ),
    )


def _growth_rate_curves(traces):
    curves = []
    for trace in traces:
        selected_delta = trace["selected_growth"] - trace["selected_growth"][0]
        max_delta = trace["max_growth"] - trace["max_growth"][0]
        if np.allclose(selected_delta, max_delta):
            curves.append(
                (
                    f"Case {trace['label']} selected=max",
                    trace["iteration"],
                    selected_delta,
                )
            )
            continue
        curves.extend(
            (
                (
                    f"Case {trace['label']} selected",
                    trace["iteration"],
                    selected_delta,
                ),
                (
                    f"Case {trace['label']} max",
                    trace["iteration"],
                    max_delta,
                ),
            )
        )
    return curves


def _run_case(case: OptimizationCase) -> dict[str, np.ndarray | str]:
    velocity, parallel, fourier, connectivity, initial_state = _build_problem(case.ky_values)
    electrons = AdiabaticElectronParams(density=1.0, temperature=1.0, zonal_correction=False)
    config = SingleSurfaceOptimizationConfig(
        geometry_model="circular",
        dt=case.dt,
        n_steps=case.n_steps,
        selected_ky=case.selected_ky,
        objective_kind=case.objective_kind,
        store_history=False,
    )

    def objective(local_knobs):
        result = single_surface_objective(
            local_knobs,
            velocity,
            parallel,
            fourier,
            initial_state,
            electron_params=electrons,
            connectivity=connectivity,
            config=config,
        )
        residual = _objective_residual(result, case)
        cost = 0.5 * residual**2
        return cost, (result, residual)

    value_and_grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    knobs = case.knobs
    rows = []
    for iteration in range(case.iterations):
        (value, (result, residual)), gradient = value_and_grad(knobs)
        rows.append(
            {
                "iteration": iteration,
                "objective": float(value),
                "objective_error": float(jnp.abs(value)),
                "growth_residual": float(residual),
                "absolute_growth_residual": float(jnp.abs(residual)),
                "selected_growth": float(result.values.selected_growth_rate),
                "max_growth": float(jnp.max(result.values.growth_rate[1:])),
                "q": float(knobs.q),
                "shat": float(knobs.shat),
                "temperature_gradient": float(knobs.temperature_gradient),
                "density_gradient": float(knobs.density_gradient),
            }
        )
        knobs = _gradient_step(knobs, gradient, case.learning_rate)
    return {
        "label": case.label,
        **{
            key: np.asarray([row[key] for row in rows], dtype=float)
            for key in rows[0]
        },
    }


def _objective_residual(result, case: OptimizationCase):
    target = jnp.asarray(case.target_growth, dtype=result.values.growth_rate.dtype)
    if case.scalar_mode == "max_nonzonal":
        raw = jnp.max(result.values.growth_rate[1:])
    else:
        raw = result.scalar_objective
    return raw - target


def _build_problem(ky_values: tuple[float, ...]):
    velocity = build_velocity_grid(VelocityGridSpec(n_vpar=3, n_mu=3, vpar_max=1.5, mu_max=1.0))
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=5, z_min=-0.5 + 0.5 / 5, z_max=0.5 + 0.5 / 5)
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=3,
            n_ky=len(ky_values),
            kx_max=0.5,
            ky_values=ky_values,
            ikxspace=2,
        )
    )
    connectivity = build_mode_connectivity(fourier)
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    initial_state = 0.01 * (jnp.cos(index / 6.0) + 1j * jnp.sin(index / 8.0))
    return velocity, parallel, fourier, connectivity, initial_state


def _gradient_step(knobs: OptimizationKnobs, gradient: OptimizationKnobs, learning_rate: float):
    updated = jax.tree_util.tree_map(
        lambda parameter, grad: parameter - learning_rate * grad,
        knobs,
        gradient,
    )
    return replace(
        updated,
        density=jnp.maximum(updated.density, 1.0e-3),
        temperature=jnp.maximum(updated.temperature, 1.0e-3),
        eps=jnp.maximum(updated.eps, 1.0e-3),
        q=jnp.maximum(updated.q, 1.0e-3),
    )


def _write_csv(path: Path, traces: list[dict[str, np.ndarray | str]]) -> None:
    columns = (
        "case",
        "iteration",
        "objective",
        "objective_error",
        "growth_residual",
        "absolute_growth_residual",
        "selected_growth",
        "max_growth",
        "q",
        "shat",
        "temperature_gradient",
        "density_gradient",
    )
    lines = [",".join(columns)]
    for trace in traces:
        for row in range(len(trace["iteration"])):
            lines.append(
                ",".join(
                    [str(trace["label"])]
                    + [f"{float(trace[column][row]):.16e}" for column in columns[1:]]
                )
            )
    path.write_text("\n".join(lines) + "\n")


def _write_line_pdf(path: Path, *, title: str, y_label: str, curves, y_scale: str = "linear") -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")
    for index, (label, xs, ys) in enumerate(curves):
        x_values = np.asarray(xs, dtype=float)
        y_values = np.asarray(ys, dtype=float)
        if y_scale == "log":
            positive = y_values[np.isfinite(y_values) & (y_values > 0.0)]
            floor = 1.0e-16 if positive.size == 0 else max(float(np.min(positive)) * 0.1, 1.0e-300)
            y_values = np.where(y_values > 0.0, y_values, floor)
        stride = max(1, x_values.size // 1600)
        plot_indices = np.unique(np.r_[np.arange(0, x_values.size, stride), x_values.size - 1])
        plot_x = x_values[plot_indices]
        plot_y = y_values[plot_indices]
        ax.plot(
            plot_x,
            plot_y,
            marker=markers[index % len(markers)],
            markevery=max(1, plot_x.size // 12),
            linewidth=1.8,
            markersize=4.2,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(y_label)
    ax.set_yscale(y_scale)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True, which="major", color="0.86", linewidth=0.8)
    ax.grid(True, which="minor", color="0.93", linewidth=0.5)
    ax.minorticks_on()
    if y_scale != "log":
        ax.axhline(0.0, color="0.25", linewidth=0.8, alpha=0.7)
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.75",
    )
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
