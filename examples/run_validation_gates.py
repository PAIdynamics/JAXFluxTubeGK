"""Print the current reduced benchmark validation gates.

Run from the repository root:

    uv run --extra dev python examples/run_validation_gates.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from stellarator_gk import (
    FourierGridSpec,
    ParallelGridSpec,
    build_desc_geometry_from_arrays,
    build_fourier_grid,
    build_parallel_grid,
    load_gx_eik_geometry_reference,
    resample_gx_eik_geometry_reference,
    run_desc_gx_eik_external_geometry_gate,
    run_geometry_to_gx_eik_export_gate,
    run_gx_gist_external_eik_suite_gate,
    run_gx_eik_geometry_gate,
    run_cyclone_base_case_term_parity_audit,
    run_cyclone_base_case_trace,
    run_rosenbluth_hinton_plateau_gate,
    run_reduced_cyclone_base_case_gate,
    run_reduced_rosenbluth_hinton_gate,
)


def main() -> None:
    args = _parse_args()
    from stellarator_gk.external import announce_external_path

    announce_external_path("GX/GIST eik", args.eik_reference)
    if args.desc_root is not None:
        announce_external_path("DESC source", args.desc_root)
    for index, path in enumerate(args.gx_gist_reference):
        announce_external_path(f"GX/GIST suite input {index}", path)
    rh = run_reduced_rosenbluth_hinton_gate(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        n_steps=args.rh_steps,
        parallel_recurrence_rate=args.rh_disp_par,
    )
    cyclone = run_reduced_cyclone_base_case_gate(
        n_z=args.n_z,
        n_vpar=args.n_vpar,
        n_mu=args.n_mu,
        n_steps=args.cyclone_steps,
        parallel_recurrence_rate=args.cyclone_disp_par,
    )
    eik = _run_eik_gate(args)
    results = [rh, cyclone, eik]
    if args.desc_eik:
        results.append(_run_desc_eik_export_gate(args))
    if args.desc_gx_eik:
        results.append(_run_desc_gx_eik_external_gate(args))
    if args.gx_gist_suite:
        results.append(_run_gx_gist_suite_gate(args))
    if args.rh_plateau:
        results.append(_run_rh_plateau_gate(args))
    term_reports = []
    if args.cyclone_term_audit:
        term_reports.append(run_cyclone_base_case_term_parity_audit())
    traces = []
    if args.cyclone_trace:
        traces.append(run_cyclone_base_case_trace(n_windows=args.cyclone_trace_windows))

    print("# stellarator_gk reduced validation gates")
    print("gate status observed reference residual tolerance notes")
    for result in results:
        status = "PASS" if bool(result.passed) else "OPEN"
        print(
            f"{result.target.name} {status} "
            f"{float(result.observed_value): .8e} "
            f"{float(result.target.reference_value): .8e} "
            f"{float(result.residual): .8e} "
            f"{result.target.tolerance: .8e} "
            f"{result.notes}"
        )
    for report in term_reports:
        tolerance = 5.0e-13
        status = "PASS" if bool(report.passed) else "OPEN"
        print(
            f"cyclone_base_case_term_parity {status} "
            f"{float(report.max_abs_error): .8e} "
            f"{0.0: .8e} "
            f"{float(report.max_abs_error) / tolerance: .8e} "
            f"{tolerance: .8e} "
            f"{report.notes}"
        )
    for trace in traces:
        print("# cyclone trace")
        print("time raw_amp physical_amp window_growth fitted_growth phi_norm state_norm rhs_norm")
        for row in zip(
            trace.times,
            trace.raw_amplitude,
            trace.physical_amplitude,
            trace.window_growth,
            trace.fitted_growth,
            trace.phi_norm,
            trace.state_norm,
            trace.rhs_norm,
            strict=True,
        ):
            print(" ".join(f"{float(value): .8e}" for value in row))


def _run_eik_gate(args):
    reference = load_gx_eik_geometry_reference(args.eik_reference)
    theta = np.linspace(-np.pi, np.pi, args.eik_nodes, endpoint=False)
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


def _run_rh_plateau_gate(args):
    return run_rosenbluth_hinton_plateau_gate(
        n_z=args.rh_plateau_n_z,
        n_vpar=args.rh_plateau_n_vpar,
        n_mu=args.rh_plateau_n_mu,
        t_end=args.rh_t_end,
        t_start=args.rh_t_start,
        diagnostic_interval=args.rh_diagnostic_interval,
        parallel_recurrence_rate=args.rh_disp_par,
        velocity_recurrence_rate=args.rh_disp_vp,
        parallel_backend=args.rh_parallel_backend,
        velocity_backend=args.rh_velocity_backend,
        z_modal_damping=args.rh_z_modal_damping,
        vpar_modal_damping=args.rh_vpar_modal_damping,
        mu_modal_damping=args.rh_mu_modal_damping,
    )


def _run_desc_eik_export_gate(args):
    fixture = np.load(args.desc_fixture)
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


def _run_desc_gx_eik_external_gate(args):
    if args.desc_root is None or args.desc_path is None:
        raise ValueError("--desc-gx-eik requires --desc-root and --desc-path")
    desc_root = args.desc_root.resolve()
    print(f"DESC root: {desc_root}")
    if desc_root.exists() and str(desc_root) not in sys.path:
        sys.path.insert(0, str(desc_root))
    return run_desc_gx_eik_external_geometry_gate(
        args.desc_path,
        args.desc_gx_eik_reference,
        rho=args.desc_gx_eik_rho,
        alpha=args.desc_gx_eik_alpha,
    )


def _run_gx_gist_suite_gate(args):
    return run_gx_gist_external_eik_suite_gate(
        args.gx_gist_reference,
        n_theta=args.gx_gist_nodes,
    )


def _parallel_grid_from_fixture_z(z):
    z = np.asarray(z, dtype=float)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-z", type=int, default=8)
    parser.add_argument("--n-vpar", type=int, default=6)
    parser.add_argument("--n-mu", type=int, default=4)
    parser.add_argument("--rh-steps", type=int, default=5)
    parser.add_argument("--cyclone-steps", type=int, default=5)
    parser.add_argument("--rh-plateau", action="store_true")
    parser.add_argument("--cyclone-term-audit", action="store_true")
    parser.add_argument("--cyclone-trace", action="store_true")
    parser.add_argument("--cyclone-trace-windows", type=int, default=4)
    parser.add_argument("--desc-eik", action="store_true")
    parser.add_argument("--desc-gx-eik", action="store_true")
    parser.add_argument("--gx-gist-suite", action="store_true")
    parser.add_argument("--desc-root", type=Path)
    parser.add_argument("--rh-plateau-n-z", type=int, default=64)
    parser.add_argument("--rh-plateau-n-vpar", type=int, default=64)
    parser.add_argument("--rh-plateau-n-mu", type=int, default=16)
    parser.add_argument("--rh-t-end", type=float, default=100.0)
    parser.add_argument("--rh-t-start", type=float, default=80.0)
    parser.add_argument("--rh-diagnostic-interval", type=float, default=1.0)
    parser.add_argument("--rh-disp-par", type=float, default=0.01)
    parser.add_argument("--rh-disp-vp", type=float, default=0.08)
    parser.add_argument("--rh-parallel-backend", default="finite_difference")
    parser.add_argument("--rh-velocity-backend", default="finite_difference")
    parser.add_argument("--cyclone-disp-par", type=float, default=1.0)
    parser.add_argument("--rh-z-modal-damping", type=float, default=0.0)
    parser.add_argument("--rh-vpar-modal-damping", type=float, default=0.0)
    parser.add_argument("--rh-mu-modal-damping", type=float, default=0.0)
    parser.add_argument("--eik-nodes", type=int, default=17)
    parser.add_argument("--gx-gist-nodes", type=int, default=17)
    parser.add_argument(
        "--eik-reference",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--desc-fixture",
        type=Path,
        default=Path("fixtures/desc_geometry_dshape_rho05_alpha0.npz"),
    )
    parser.add_argument(
        "--desc-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--desc-gx-eik-reference",
        type=Path,
        default=Path("fixtures/gx_desc_dshape_rho05_alpha0.eik.out"),
    )
    parser.add_argument("--desc-gx-eik-rho", type=float, default=0.5)
    parser.add_argument("--desc-gx-eik-alpha", type=float, default=0.0)
    parser.add_argument(
        "--gx-gist-reference",
        type=Path,
        action="append",
        default=[],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
