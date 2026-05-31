"""Print the current reduced benchmark validation gates.

Run from the repository root:

    uv run --extra dev python examples/run_validation_gates.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

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
    run_geometry_to_gx_eik_export_gate,
    run_gx_gist_external_eik_suite_gate,
    run_gx_eik_geometry_gate,
    run_rosenbluth_hinton_plateau_gate,
    run_reduced_cyclone_base_case_gate,
    run_reduced_rosenbluth_hinton_gate,
)


def main() -> None:
    args = _parse_args()
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
    if args.gx_gist_suite:
        results.append(_run_gx_gist_suite_gate(args))
    if args.rh_plateau:
        results.append(_run_rh_plateau_gate(args))

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
    parser.add_argument("--desc-eik", action="store_true")
    parser.add_argument("--gx-gist-suite", action="store_true")
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
        default=Path(
            "relevant-codes/gx/geometry_modules/vmec/tests/"
            "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
        ),
    )
    parser.add_argument(
        "--desc-fixture",
        type=Path,
        default=Path("fixtures/desc_geometry_dshape_rho05_alpha0.npz"),
    )
    parser.add_argument(
        "--gx-gist-reference",
        type=Path,
        action="append",
        default=[
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
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
