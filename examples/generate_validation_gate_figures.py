"""Generate reduced validation-gate demonstration data and figures.

Run from the repository root:

    uv run --extra dev python examples/generate_validation_gate_figures.py
"""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from stellarator_gk import (
    FourierGridSpec,
    ParallelGridSpec,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_parallel_grid,
    load_gx_eik_geometry_reference,
    resample_gx_eik_geometry_reference,
    run_geometry_to_gx_eik_export_gate,
    run_gx_gist_external_eik_suite_gate,
    run_gx_eik_geometry_gate,
    run_reduced_cyclone_base_case_gate,
    run_reduced_rosenbluth_hinton_gate,
    run_rosenbluth_hinton_plateau_gate,
)


@dataclass(frozen=True)
class RhPlateauPoint:
    t_end: float
    t_start: float
    observed: float
    reference: float
    residual: float
    passed: bool
    notes: str


@dataclass(frozen=True)
class GateSummary:
    label: str
    gate: str
    status: str
    observed: float
    reference: float
    residual: float
    tolerance: float
    notes: str


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    rh_points = _run_rh_plateau_points()
    summary = _run_gate_summary(rh_points[-1])

    _write_rh_csv(output_dir / "rh_plateau_demo.csv", rh_points)
    _write_summary_csv(output_dir / "validation_gate_summary.csv", summary)
    _write_validation_pdf(output_dir / "validation_gate_status.pdf", rh_points, summary)

    for path in (
        "rh_plateau_demo.csv",
        "validation_gate_summary.csv",
        "validation_gate_status.pdf",
    ):
        print(output_dir / path)


def _run_rh_plateau_points() -> list[RhPlateauPoint]:
    points = []
    for t_end in (0.02, 0.04, 0.06, 0.08, 0.10):
        t_start = max(0.01, 0.5 * t_end)
        result = run_rosenbluth_hinton_plateau_gate(
            n_z=8,
            n_vpar=6,
            n_mu=4,
            t_end=t_end,
            t_start=t_start,
            diagnostic_interval=0.01,
            parallel_recurrence_rate=0.01,
            z_modal_damping=0.0,
        )
        points.append(
            RhPlateauPoint(
                t_end=t_end,
                t_start=t_start,
                observed=float(result.observed_value),
                reference=float(result.target.reference_value),
                residual=float(result.residual),
                passed=bool(result.passed),
                notes=result.notes,
            )
        )
    return points


def _run_gate_summary(rh_plateau_point: RhPlateauPoint) -> list[GateSummary]:
    reduced_rh = run_reduced_rosenbluth_hinton_gate(n_z=8, n_vpar=6, n_mu=4, n_steps=5)
    cyclone = run_reduced_cyclone_base_case_gate(n_z=8, n_vpar=6, n_mu=4, n_steps=5)
    eik = _run_eik_gate()
    desc_eik = _run_desc_eik_export_gate()
    gx_gist = _run_gx_gist_suite_gate()
    return [
        _summary_from_result("RH endpoint", reduced_rh),
        GateSummary(
            label="RH plateau",
            gate="rosenbluth_hinton_q13_eps005",
            status="PASS" if rh_plateau_point.passed else "OPEN",
            observed=rh_plateau_point.observed,
            reference=rh_plateau_point.reference,
            residual=rh_plateau_point.residual,
            tolerance=1.0e-3,
            notes=rh_plateau_point.notes,
        ),
        _summary_from_result("Cyclone", cyclone),
        _summary_from_result("GX/eik", eik),
        _summary_from_result("DESC/eik", desc_eik),
        _summary_from_result("GX/GIST", gx_gist),
    ]


def _summary_from_result(label, result) -> GateSummary:
    return GateSummary(
        label=label,
        gate=result.target.name,
        status="PASS" if bool(result.passed) else "OPEN",
        observed=float(result.observed_value),
        reference=float(result.target.reference_value),
        residual=float(result.residual),
        tolerance=float(result.target.tolerance),
        notes=result.notes,
    )


def _run_eik_gate():
    reference_path = Path(
        "relevant-codes/gx/geometry_modules/vmec/tests/"
        "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
    )
    reference = load_gx_eik_geometry_reference(reference_path)
    theta = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    sampled = resample_gx_eik_geometry_reference(reference, theta)
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    parallel = build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    return run_gx_eik_geometry_gate(sampled, parallel, fourier)


def _run_desc_eik_export_gate():
    fixture = np.load("fixtures/desc_geometry_dshape_rho05_alpha0.npz")
    parallel = _parallel_grid_from_fixture_z(fixture["z"])
    geometry = build_desc_geometry_from_arrays(
        parallel,
        theta=fixture["theta"],
        phi=fixture["phi"],
        alpha=fixture["alpha"],
        rho=fixture["rho"],
        B=fixture["B"],
        b_dot_grad_z=fixture["b_dot_grad_z"],
        grad_psi_sq=fixture["grad_psi_sq"],
        grad_alpha_sq=fixture["grad_alpha_sq"],
        grad_psi_dot_grad_alpha=fixture["grad_psi_dot_grad_alpha"],
        B_cross_gradB_dot_grad_psi=fixture["B_cross_gradB_dot_grad_psi"],
        B_cross_gradB_dot_grad_alpha=fixture["B_cross_gradB_dot_grad_alpha"],
        b_cross_kappa_dot_grad_psi=fixture["b_cross_kappa_dot_grad_psi"],
        b_cross_kappa_dot_grad_alpha=fixture["b_cross_kappa_dot_grad_alpha"],
    )
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.2, ky_values=(0.0, 0.35))
    )
    return run_geometry_to_gx_eik_export_gate(geometry, fourier)


def _run_gx_gist_suite_gate():
    paths = (
        Path(
            "relevant-codes/gx/geometry_modules/vmec/tests/"
            "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
        ),
        Path(
            "relevant-codes/gx/geometry_modules/vmec/tests/"
            "gist_gs2_wout_li383_1.4m.txt_highres_surf12_pol_10_nz0_10000"
        ),
        Path(
            "relevant-codes/gx/geometry_modules/vmec/tests/"
            "gist_gs2_wout_st_a34_i32v22_beta_35_scaledAUG.txt_highres_surf12_pol_10_nz0_10000"
        ),
    )
    return run_gx_gist_external_eik_suite_gate(paths, n_theta=17)


def _parallel_grid_from_fixture_z(z):
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _write_rh_csv(path: Path, points: list[RhPlateauPoint]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "t_end",
                "t_start",
                "observed",
                "reference",
                "normalized_residual",
                "status",
                "notes",
            )
        )
        for point in points:
            writer.writerow(
                (
                    point.t_end,
                    point.t_start,
                    point.observed,
                    point.reference,
                    point.residual,
                    "PASS" if point.passed else "OPEN",
                    point.notes,
                )
            )


def _write_summary_csv(path: Path, rows: list[GateSummary]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "label",
                "gate",
                "status",
                "observed",
                "reference",
                "normalized_residual",
                "tolerance",
                "notes",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row.label,
                    row.gate,
                    row.status,
                    row.observed,
                    row.reference,
                    row.residual,
                    row.tolerance,
                    row.notes,
                )
            )


def _write_validation_pdf(
    path: Path,
    rh_points: list[RhPlateauPoint],
    summary: list[GateSummary],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), constrained_layout=True)
    _draw_rh_panel(axes[0], rh_points)
    _draw_gate_panel(axes[1], summary)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_rh_panel(ax, points: list[RhPlateauPoint]) -> None:
    t_end = np.array([point.t_end for point in points])
    observed = np.array([point.observed for point in points])
    reference = points[0].reference
    ax.plot(t_end, observed, marker="o", linewidth=1.8, label="reduced gate")
    ax.axhline(reference, color="black", linestyle="--", linewidth=1.2, label="RH reference")
    ax.set_title("RH late-window metric")
    ax.set_xlabel(r"end time $t_{\mathrm{end}}$")
    ax.set_ylabel(r"$R_{\mathrm{RH}}$")
    ax.set_ylim(0.0, max(1.08, float(np.max(observed)) * 1.05))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", frameon=False)


def _draw_gate_panel(ax, summary: list[GateSummary]) -> None:
    labels = [row.label for row in summary]
    residuals = np.array([abs(row.residual) for row in summary])
    plot_values = np.maximum(residuals, 1.0e-6)
    colors = ["#c85a54" if row.status == "OPEN" else "#3b8f72" for row in summary]
    positions = np.arange(len(summary))
    ax.bar(positions, plot_values, color=colors, width=0.68)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="pass threshold")
    ax.set_yscale("log")
    ax.set_title("Normalized gate residuals")
    ax.set_ylabel(r"$|\mathcal{O}_{\mathrm{obs}}-\mathcal{O}_{\mathrm{ref}}|/\sigma$")
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylim(1.0e-6, max(1.0e4, float(np.max(plot_values)) * 2.0))
    ax.grid(True, axis="y", which="both", alpha=0.28)
    ax.legend(loc="upper right", frameon=False)
    for index, row in enumerate(summary):
        ax.text(
            index,
            plot_values[index] * 1.25,
            row.status,
            ha="center",
            va="bottom",
            fontsize=7,
        )


if __name__ == "__main__":
    main()
