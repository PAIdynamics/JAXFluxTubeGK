"""Run a reduced linear stellarator ``ky`` scan from DESC-style/eik geometry.

Example from the repository root:

    uv run --extra dev python examples/run_stellarator_linear_scan.py \
        --output-dir runs/dshape_linear_scan

The default path uses the bundled DESC DSHAPE fixture.  A real DESC
equilibrium can be used with ``--geometry-source desc-path --desc-path ...``.
GX/GIST/GS2 eik geometry tables can be used with
``--geometry-source eik --eik-reference ...``.  A stella ``.geometry`` file
can be used with ``--geometry-source stella-geometry --stella-geometry ...``.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stellarator_gk_matplotlib")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stellarator_gk import (
    AdiabaticElectronParams,
    FourierGridSpec,
    FluxTubeGeometry,
    PerKyModeStructureFixture,
    ParallelGridSpec,
    SpeciesParams,
    VelocityGridSpec,
    build_boozer_parallel_grid,
    build_desc_geometry_from_arrays,
    build_desc_geometry_from_path,
    build_fourier_grid,
    build_flux_tube_geometry_from_gx_eik_reference,
    build_linear_residual_precompute,
    build_mode_connectivity,
    build_parallel_grid,
    build_velocity_grid,
    estimate_linear_cfl_dt,
    integrate_fixed_step,
    k_perp_squared,
    kperp2_weighted_average,
    linear_residual,
    load_gx_eik_geometry_reference,
    mode_chain_amplitude,
    normalize_by_ky_amplitude,
    real_frequency,
    resample_gx_eik_geometry_reference,
    run_desc_gx_eik_external_geometry_gate,
    run_stellarator_geometry_preflight,
    solve_field_from_state,
    weighted_quasilinear_proxy,
    write_per_ky_mode_structure_fixture_csv,
)

DESC_FIXTURE_KEYS = (
    "theta",
    "phi",
    "rho",
    "alpha",
    "B",
    "b_dot_grad_z",
    "grad_psi_sq",
    "grad_alpha_sq",
    "grad_psi_dot_grad_alpha",
    "B_cross_gradB_dot_grad_psi",
    "B_cross_gradB_dot_grad_alpha",
    "b_cross_kappa_dot_grad_psi",
    "b_cross_kappa_dot_grad_alpha",
)

GEOMETRY_FIELDS = (
    "B",
    "F",
    "G",
    "E_y",
    "D_x",
    "D_y",
    "g_xx",
    "g_xy",
    "g_yy",
)

STELLA_GEOMETRY_COLUMNS = (
    "alpha",
    "zed",
    "zeta",
    "bmag",
    "b_dot_grad_zed",
    "g_yy",
    "g_xy",
    "g_xx",
    "B_cross_gradB_dot_grad_alpha",
    "b_cross_kappa_dot_grad_alpha",
    "B_cross_gradB_dot_grad_psi",
    "bmag_psi0",
)
STELLA_GLOBAL_COLUMNS = (
    "rhoc",
    "qinp",
    "shat",
    "rhotor",
    "aref",
    "bref",
    "dxdpsi",
    "dydalpha",
    "exb_nonlin",
    "flux_fac",
    "one_over_Grho",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.desc_root is not None:
        sys.path.insert(0, str(args.desc_root.resolve()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ky_values = _parse_float_tuple(args.ky_values)
    geometry, parallel, geometry_metadata = _load_geometry(args)
    velocity = build_velocity_grid(
        VelocityGridSpec(
            n_vpar=args.n_vpar,
            n_mu=args.n_mu,
            vpar_max=args.vpar_max,
            mu_max=args.mu_max,
            backend=args.velocity_backend,
        )
    )
    fourier = build_fourier_grid(
        FourierGridSpec(
            n_kx=args.n_kx,
            n_ky=len(ky_values),
            kx_max=args.kx_max,
            ky_values=ky_values,
            ikxspace=args.ikxspace,
        )
    )
    connectivity = build_mode_connectivity(fourier)

    geometry_audit = _geometry_audit(
        geometry,
        fourier,
        args,
        geometry_metadata,
    )
    if not geometry_audit["passed"]:
        _write_json(args.output_dir / "geometry_audit.json", geometry_audit)
        raise ValueError("geometry preflight failed; see geometry_audit.json")
    _write_geometry_outputs(args.output_dir, geometry, fourier, geometry_audit)

    scan = _run_scan(args, geometry, parallel, velocity, fourier, connectivity)
    _write_scan_outputs(args.output_dir, scan, geometry, fourier, args, geometry_metadata)

    print(
        "PASS: stellarator linear scan "
        f"n_ky={len(ky_values)} max_growth={float(np.max(scan['growth_rate'])):.8e} "
        f"ql_proxy={float(scan['quasilinear_proxy_total']):.8e}"
    )
    print(args.output_dir)
    return 0


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dshape_linear_scan"))
    parser.add_argument(
        "--geometry-source",
        choices=("fixture", "desc-path", "eik", "stella-geometry"),
        default="fixture",
        help=(
            "use the bundled .npz fixture, evaluate DESC, sample a GX/GIST/GS2 eik "
            "table, or import a stella .geometry table"
        ),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/desc_geometry_dshape_rho05_alpha0.npz"),
        help="DESC-sampled .npz fixture for --geometry-source fixture",
    )
    parser.add_argument("--desc-path", type=Path, help="DESC HDF5/pickle equilibrium path")
    parser.add_argument(
        "--desc-root",
        type=Path,
        help="optional DESC checkout to prepend to sys.path",
    )
    parser.add_argument("--file-format", choices=("hdf5", "pickle"), help="DESC file format")
    parser.add_argument("--family-index", type=int, default=-1)
    parser.add_argument("--eik-reference", type=Path, help="GX/GIST/GS2 eik geometry table")
    parser.add_argument(
        "--stella-geometry",
        type=Path,
        default=Path("fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.geometry"),
        help="stella .geometry table for --geometry-source stella-geometry",
    )
    parser.add_argument(
        "--keep-stella-endpoint",
        action="store_true",
        help="keep the duplicate +pi endpoint in stella .geometry instead of dropping it",
    )
    parser.add_argument("--external-eik-reference", type=Path)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--iota", type=float)
    parser.add_argument("--n-z", type=int)
    parser.add_argument(
        "--field-line-periods",
        "--n-turns",
        dest="field_line_periods",
        type=int,
        default=1,
    )
    parser.add_argument("--zeta-center", type=float, default=0.0)
    parser.add_argument("--n-vpar", type=int, default=3)
    parser.add_argument("--n-mu", type=int, default=3)
    parser.add_argument("--vpar-max", type=float, default=1.5)
    parser.add_argument("--mu-max", type=float, default=1.0)
    parser.add_argument(
        "--velocity-backend",
        choices=("chebyshev", "finite_difference"),
        default="chebyshev",
    )
    parser.add_argument("--n-kx", type=int, default=3)
    parser.add_argument("--kx-max", type=float, default=0.45)
    parser.add_argument("--ky-values", default="0.0,0.35")
    parser.add_argument("--ikxspace", type=int, default=2)
    parser.add_argument("--charge", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--density-gradient", type=float, default=0.8)
    parser.add_argument("--temperature-gradient", type=float, default=2.1)
    parser.add_argument("--electron-density", type=float, default=1.0)
    parser.add_argument("--electron-temperature", type=float, default=1.0)
    parser.add_argument("--zonal-correction", action="store_true")
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("matrix", "gkw_upwind", "gkw_igh"),
        default="matrix",
    )
    parser.add_argument("--parallel-recurrence-rate", type=float, default=0.0)
    parser.add_argument("--velocity-recurrence-rate", type=float, default=0.0)
    parser.add_argument("--perpendicular-damping", type=float)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps-per-window", type=int, default=1)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument(
        "--growth-diagnostic",
        choices=("late_fit", "late_mean_window", "endpoint"),
        default="late_fit",
    )
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    parser.add_argument("--no-normalize-each-window", action="store_true")
    parser.add_argument("--initial-amplitude", type=float, default=1.0e-2)
    parser.add_argument("--softplus-temperature", type=float)
    parser.add_argument("--kperp-epsilon", type=float, default=1.0e-12)
    return parser.parse_args(argv)


def _load_geometry(args):
    if args.geometry_source == "fixture":
        data = np.load(args.fixture)
        if args.field_line_periods != 1:
            raise ValueError("fixture geometry is stored for one field-line period")
        if abs(args.zeta_center) > 0.0:
            raise ValueError("fixture geometry is stored with zeta_center=0")
        fixture_n_z = int(data["z"].shape[0])
        if args.n_z is not None and args.n_z != fixture_n_z:
            raise ValueError(
                f"fixture n_z is fixed at {fixture_n_z}; use --geometry-source desc-path or eik "
                "to evaluate another parallel resolution"
            )
        parallel = _parallel_grid_from_z(data["z"])
        geometry = build_desc_geometry_from_arrays(
            parallel,
            **{key: data[key] for key in DESC_FIXTURE_KEYS},
        )
        return geometry, parallel, {
            "geometry_source": "fixture",
            "fixture": str(args.fixture),
            "source": _json_scalar(data["source"]) if "source" in data else str(args.fixture),
            "rho": float(np.ravel(data["rho"])[0]),
            "alpha": float(np.ravel(data["alpha"])[0]),
            "n_z": fixture_n_z,
            "field_line_periods": 1,
        }

    if args.geometry_source == "eik":
        if args.eik_reference is None:
            raise ValueError("--eik-reference is required for --geometry-source eik")
        n_z = 33 if args.n_z is None else args.n_z
        theta = np.linspace(
            -np.pi * args.field_line_periods,
            np.pi * args.field_line_periods,
            n_z,
            endpoint=False,
        )
        reference = load_gx_eik_geometry_reference(args.eik_reference)
        sampled = resample_gx_eik_geometry_reference(reference, theta)
        parallel = _parallel_grid_from_theta(theta)
        geometry = build_flux_tube_geometry_from_gx_eik_reference(sampled, parallel)
        return geometry, parallel, {
            "geometry_source": "eik",
            "eik_reference": str(args.eik_reference),
            "source": sampled.source,
            "rho": None,
            "radial_coordinate": "external_eik_table",
            "alpha": args.alpha,
            "n_z": n_z,
            "field_line_periods": args.field_line_periods,
            "theta_min": float(theta[0]),
            "theta_max": float(theta[-1]),
            "theta_endpoint": "excluded",
        }

    if args.geometry_source == "stella-geometry":
        return _load_stella_geometry(args)

    if args.desc_path is None:
        raise ValueError("--desc-path is required for --geometry-source desc-path")
    n_z = 33 if args.n_z is None else args.n_z
    parallel = build_boozer_parallel_grid(
        n_z=n_z,
        n_turns=args.field_line_periods,
        center=args.zeta_center,
    )
    geometry = build_desc_geometry_from_path(
        args.desc_path,
        parallel,
        rho=args.rho,
        alpha=args.alpha,
        iota=args.iota,
        file_format=args.file_format,
        index=args.family_index,
    )
    return geometry, parallel, {
        "geometry_source": "desc-path",
        "desc_path": str(args.desc_path),
        "rho": args.rho,
        "alpha": args.alpha,
        "iota": args.iota,
        "n_z": n_z,
        "field_line_periods": args.field_line_periods,
        "zeta_center": args.zeta_center,
    }


def _geometry_audit(geometry, fourier, args, geometry_metadata):
    include_mirror_fd_check = args.geometry_source != "eik"
    report = run_stellarator_geometry_preflight(
        geometry,
        fourier,
        include_mirror_fd_check=include_mirror_fd_check,
    )
    fields = {name: np.asarray(getattr(geometry, name)) for name in GEOMETRY_FIELDS}
    field_stats_from_report = {
        name: {
            "min": float(report.field_min[index]),
            "max": float(report.field_max[index]),
            "mean": float(report.field_mean[index]),
            "finite": bool(np.all(np.isfinite(values))),
        }
        for index, (name, values) in enumerate(fields.items())
    }
    checks = {
        name: bool(value)
        for name, value in zip(
            report.check_names,
            np.asarray(report.check_passed, dtype=bool),
            strict=True,
        )
    }

    external_eik_gate = None
    if args.external_eik_reference is not None:
        if args.geometry_source != "desc-path":
            raise ValueError(
                "--external-eik-reference is supported for --geometry-source desc-path; "
                "the fixture and imported-eik paths use the internal export contract"
            )
        external_gate = run_desc_gx_eik_external_geometry_gate(
            args.desc_path,
            args.external_eik_reference,
            rho=args.rho,
            alpha=args.alpha,
            zeta_center=args.zeta_center,
            file_format=args.file_format,
            index=args.family_index,
            fourier_grid=fourier,
        )
        external_eik_gate = _gate_summary(external_gate)
        checks["external_eik_gate"] = bool(external_gate.passed)

    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "geometry": geometry_metadata,
        "z": {
            "n_z": int(np.asarray(geometry.z).shape[0]),
            "min": float(np.min(np.asarray(geometry.z))),
            "max": float(np.max(np.asarray(geometry.z))),
            "weight_sum": float(np.sum(np.asarray(geometry.w_z))),
        },
        "field_stats": field_stats_from_report,
        "kperp2": {
            "min": float(report.kperp2_min),
            "max": float(report.kperp2_max),
            "mean": float(report.kperp2_mean),
        },
        "eik_export_gate": {
            "name": "stellarator_geometry_preflight_eik_export",
            "quantity": "max_abs_eik_export_error",
            "observed_value": float(report.eik_export_error),
            "reference_value": 0.0,
            "tolerance": 1.0e-12,
            "passed": bool(checks["gx_eik_export_contract"]),
            "notes": report.notes,
        },
        "mirror_fd_error": float(report.mirror_fd_error),
        "mirror_fd_check_enabled": include_mirror_fd_check,
        "external_eik_gate": external_eik_gate,
    }


def _parallel_grid_from_theta(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.shape[0] < 2:
        raise ValueError("theta must be a one-dimensional grid with at least two points")
    z = theta / (2.0 * np.pi)
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _load_stella_geometry(args):
    rows, global_header = _read_stella_geometry_file(args.stella_geometry)
    original_n_z = int(rows.shape[0])
    zed = rows[:, STELLA_GEOMETRY_COLUMNS.index("zed")]
    zeta = rows[:, STELLA_GEOMETRY_COLUMNS.index("zeta")]
    full_zeta = np.asarray(zeta, dtype=float)
    dropped_endpoint = False
    if not args.keep_stella_endpoint and _has_duplicate_stella_endpoint(rows):
        rows = rows[:-1]
        zed = zed[:-1]
        zeta = zeta[:-1]
        dropped_endpoint = True
    n_z = int(rows.shape[0])
    if args.n_z is not None and args.n_z != n_z:
        raise ValueError(
            f"stella geometry provides n_z={n_z} after endpoint handling; "
            "rerun stella/export an eik table for a different resolution"
        )

    z = (
        np.linspace(-0.5, 0.5, n_z, endpoint=False, dtype=float)
        if dropped_endpoint
        else zed / (2.0 * np.pi)
    )
    parallel = _parallel_grid_from_z(z)
    solver_z = np.asarray(parallel.z, dtype=float)
    theta = 2.0 * np.pi * solver_z
    phi = np.asarray(rows[:, STELLA_GEOMETRY_COLUMNS.index("zeta")], dtype=float)
    rho_value = _stella_rho_from_header(global_header)
    B = np.asarray(rows[:, STELLA_GEOMETRY_COLUMNS.index("bmag")], dtype=float)
    # stella's b.Gz multiplies d/dzed.  The solver grid below differentiates
    # with respect to zed/(2*pi), so F must be scaled by 1/(2*pi).
    F = np.asarray(rows[:, STELLA_GEOMETRY_COLUMNS.index("b_dot_grad_zed")], dtype=float) / (
        2.0 * np.pi
    )
    D_z = np.asarray(parallel.D_z, dtype=float)
    G = -F * (D_z @ B) / B
    bxgb_gy = np.asarray(
        rows[:, STELLA_GEOMETRY_COLUMNS.index("B_cross_gradB_dot_grad_alpha")],
        dtype=float,
    )
    bxkappa_gy = np.asarray(
        rows[:, STELLA_GEOMETRY_COLUMNS.index("b_cross_kappa_dot_grad_alpha")],
        dtype=float,
    )
    geometry = FluxTubeGeometry(
        z=parallel.z,
        w_z=parallel.w_z,
        theta=jnp.asarray(theta, dtype=jnp.float64),
        phi=jnp.asarray(phi, dtype=jnp.float64),
        rho=jnp.full_like(parallel.z, rho_value),
        B=jnp.asarray(B, dtype=jnp.float64),
        F=jnp.asarray(F, dtype=jnp.float64),
        G=jnp.asarray(G, dtype=jnp.float64),
        E_y=jnp.asarray(bxgb_gy, dtype=jnp.float64),
        D_x=jnp.asarray(
            rows[:, STELLA_GEOMETRY_COLUMNS.index("B_cross_gradB_dot_grad_psi")],
            dtype=jnp.float64,
        ),
        D_y=jnp.asarray(bxgb_gy + bxkappa_gy, dtype=jnp.float64),
        g_xx=jnp.asarray(rows[:, STELLA_GEOMETRY_COLUMNS.index("g_xx")], dtype=jnp.float64),
        g_xy=jnp.asarray(rows[:, STELLA_GEOMETRY_COLUMNS.index("g_xy")], dtype=jnp.float64),
        g_yy=jnp.asarray(rows[:, STELLA_GEOMETRY_COLUMNS.index("g_yy")], dtype=jnp.float64),
        radial_coordinate="rho",
        source="stella-geometry",
    )
    field_line_turns = (float(np.max(full_zeta)) - float(np.min(full_zeta))) / (2.0 * np.pi)
    sampled_zeta_turns = (float(np.max(zeta)) - float(np.min(zeta))) / (2.0 * np.pi)
    metadata = {
        "geometry_source": "stella-geometry",
        "stella_geometry": str(args.stella_geometry),
        "source": str(args.stella_geometry),
        "rho": rho_value,
        "alpha": float(rows[0, STELLA_GEOMETRY_COLUMNS.index("alpha")]),
        "n_z": n_z,
        "original_n_z": original_n_z,
        "dropped_periodic_endpoint": dropped_endpoint,
        "z_coordinate": "zed_over_2pi",
        "z_min": float(solver_z[0]),
        "z_max": float(solver_z[-1]),
        "theta_min": float(np.min(theta)),
        "theta_max": float(np.max(theta)),
        "zeta_min": float(np.min(phi)),
        "zeta_max": float(np.max(phi)),
        "zeta_turns": field_line_turns,
        "sampled_zeta_turns": sampled_zeta_turns,
        "field_line_periods": field_line_turns,
        "b_dot_grad_z_scaling": "F = stella b.Gz / (2*pi)",
        "global_header": {
            name: float(value)
            for name, value in zip(STELLA_GLOBAL_COLUMNS, global_header, strict=True)
        },
    }
    return geometry, parallel, metadata


def _read_stella_geometry_file(path: Path):
    global_header = None
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                stripped = stripped[1:].strip()
                if not stripped:
                    continue
            try:
                values = [float(item) for item in stripped.split()]
            except ValueError:
                continue
            if len(values) == len(STELLA_GLOBAL_COLUMNS) and global_header is None:
                global_header = tuple(values)
            elif len(values) >= len(STELLA_GEOMETRY_COLUMNS):
                rows.append(values[: len(STELLA_GEOMETRY_COLUMNS)])
    if global_header is None:
        raise ValueError(f"{path} is missing the stella global geometry header")
    if not rows:
        raise ValueError(f"{path} contains no stella geometry rows")
    return np.asarray(rows, dtype=float), global_header


def _has_duplicate_stella_endpoint(rows) -> bool:
    if rows.shape[0] < 3:
        return False
    first = rows[0]
    last = rows[-1]
    zed_span = abs((last[STELLA_GEOMETRY_COLUMNS.index("zed")] - first[STELLA_GEOMETRY_COLUMNS.index("zed")]) - 2.0 * np.pi)
    periodic_columns = (
        "bmag",
        "b_dot_grad_zed",
        "g_yy",
        "g_xx",
        "B_cross_gradB_dot_grad_alpha",
        "b_cross_kappa_dot_grad_alpha",
        "bmag_psi0",
    )
    periodic_match = all(
        np.isclose(
            first[STELLA_GEOMETRY_COLUMNS.index(name)],
            last[STELLA_GEOMETRY_COLUMNS.index(name)],
            rtol=1.0e-8,
            atol=1.0e-10,
        )
        for name in periodic_columns
    )
    return bool(zed_span <= 1.0e-3 and periodic_match)


def _stella_rho_from_header(global_header: tuple[float, ...]) -> float:
    values = dict(zip(STELLA_GLOBAL_COLUMNS, global_header, strict=True))
    return float(values["rhoc"])


def _run_scan(args, geometry, parallel, velocity, fourier, connectivity):
    species = SpeciesParams(
        charge=args.charge,
        mass=args.mass,
        density=args.density,
        temperature=args.temperature,
        density_gradient=args.density_gradient,
        temperature_gradient=args.temperature_gradient,
    )
    electrons = AdiabaticElectronParams(
        density=args.electron_density,
        temperature=args.electron_temperature,
        zonal_correction=args.zonal_correction,
    )
    precompute = build_linear_residual_precompute(
        velocity,
        parallel,
        fourier,
        geometry,
        species,
        electron_params=electrons,
        perpendicular_damping=args.perpendicular_damping,
        parallel_recurrence_rate=args.parallel_recurrence_rate,
        velocity_recurrence_rate=args.velocity_recurrence_rate,
        mode_connectivity=connectivity,
        parallel_derivative_model=args.parallel_derivative_model,
    )
    state = _initial_state(velocity, parallel, fourier, args.initial_amplitude)
    solve_phi = jax.jit(lambda state_value: solve_field_from_state(state_value, precompute))
    advance_window = jax.jit(
        lambda state_value: integrate_fixed_step(
            state_value,
            args.dt,
            args.steps_per_window,
            linear_residual,
            precompute,
            store_history=False,
        ).state
    )

    times = []
    log_amplitudes = []
    raw_amplitudes = []
    phi_samples = []
    log_normalization = jnp.zeros((fourier.ky.shape[0],), dtype=jnp.float64)

    def snapshot(time_value, state_value, accumulated_log):
        phi_value = solve_phi(state_value)
        amplitude = mode_chain_amplitude(phi_value, w_z=geometry.w_z, connectivity=connectivity)
        floor = jnp.asarray(1.0e-300, dtype=amplitude.dtype)
        times.append(float(time_value))
        raw_amplitudes.append(np.asarray(amplitude, dtype=float))
        log_amplitudes.append(np.asarray(jnp.log(jnp.maximum(amplitude, floor)) + accumulated_log))
        phi_samples.append(phi_value)
        return phi_value, amplitude

    _, amplitude = snapshot(0.0, state, log_normalization)
    normalize_each_window = not args.no_normalize_each_window
    for window in range(args.n_windows):
        state = advance_window(state)
        time_value = (window + 1) * args.steps_per_window * args.dt
        _, amplitude = snapshot(time_value, state, log_normalization)
        if normalize_each_window:
            normalized = normalize_by_ky_amplitude(
                state,
                amplitude,
                log_normalization=log_normalization,
            )
            state = normalized.state
            log_normalization = normalized.log_normalization

    times_array = np.asarray(times, dtype=float)
    log_amplitude_array = np.asarray(log_amplitudes, dtype=float)
    raw_amplitude_array = np.asarray(raw_amplitudes, dtype=float)
    late_start = _late_start_index(len(times_array), args.growth_window_fraction)
    growth = _growth_from_log_amplitudes(
        times_array,
        log_amplitude_array,
        args.growth_diagnostic,
        late_start,
    )
    frequency = np.asarray(
        real_frequency(
            phi_samples[late_start],
            phi_samples[-1],
            times_array[late_start],
            times_array[-1],
            w_z=geometry.w_z,
            connectivity=connectivity,
        ),
        dtype=float,
    )
    final_phi = phi_samples[-1]
    final_amplitude = mode_chain_amplitude(final_phi, w_z=geometry.w_z, connectivity=connectivity)
    mode_structure = final_phi / jnp.maximum(
        final_amplitude,
        jnp.asarray(1.0e-300, dtype=final_amplitude.dtype),
    )[None, None, :]
    kperp2 = k_perp_squared(geometry, fourier)
    kperp_average = np.asarray(
        kperp2_weighted_average(kperp2, final_phi, w_z=geometry.w_z, connectivity=connectivity),
        dtype=float,
    )
    active_ky = np.asarray(fourier.ky, dtype=float) > 0.0
    ql_contribution = np.where(
        active_ky,
        np.maximum(growth, 0.0) / (kperp_average + args.kperp_epsilon),
        0.0,
    )
    ql_total = float(
        weighted_quasilinear_proxy(
            jnp.asarray(growth),
            jnp.asarray(kperp_average),
            active_mask=jnp.asarray(active_ky),
            epsilon=args.kperp_epsilon,
            softplus_temperature=args.softplus_temperature,
        )
    )
    return {
        "species": species,
        "electron_params": electrons,
        "times": times_array,
        "log_amplitude": log_amplitude_array,
        "raw_amplitude": raw_amplitude_array,
        "window_growth": np.diff(log_amplitude_array, axis=0) / np.diff(times_array)[:, None],
        "growth_rate": growth,
        "frequency": frequency,
        "final_phi": np.asarray(final_phi),
        "mode_structure": np.asarray(mode_structure),
        "kperp2_average": kperp_average,
        "quasilinear_contribution": ql_contribution,
        "quasilinear_proxy_total": ql_total,
        "log_normalization": np.asarray(log_normalization, dtype=float),
        "estimated_cfl_dt": float(estimate_linear_cfl_dt(precompute)),
    }


def _write_geometry_outputs(output_dir: Path, geometry, fourier, audit):
    _write_json(output_dir / "geometry_audit.json", audit)
    kperp2 = np.asarray(k_perp_squared(geometry, fourier), dtype=float)
    with (output_dir / "geometry_audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "z_index",
                "z",
                "theta",
                "phi",
                "rho",
                "B",
                "F",
                "G",
                "E_y",
                "D_x",
                "D_y",
                "g_xx",
                "g_xy",
                "g_yy",
                "kperp2_min",
                "kperp2_max",
            )
        )
        arrays = {name: np.asarray(getattr(geometry, name)) for name in GEOMETRY_FIELDS}
        z = np.asarray(geometry.z)
        theta = np.asarray(geometry.theta)
        phi = np.asarray(geometry.phi)
        rho = np.asarray(geometry.rho)
        for index, z_value in enumerate(z):
            writer.writerow(
                (
                    index,
                    float(z_value),
                    float(theta[index]),
                    float(phi[index]),
                    float(rho[index]),
                    *(float(arrays[name][index]) for name in GEOMETRY_FIELDS),
                    float(np.min(kperp2[index])),
                    float(np.max(kperp2[index])),
                )
            )


def _write_scan_outputs(output_dir: Path, scan, geometry, fourier, args, geometry_metadata):
    ky = np.asarray(fourier.ky, dtype=float)
    with (output_dir / "ky_growth.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "ky_index",
                "ky",
                "growth_rate",
                "frequency",
                "kperp2_average",
                "quasilinear_contribution",
                "raw_final_amplitude",
                "log_normalization",
            )
        )
        for index, ky_value in enumerate(ky):
            writer.writerow(
                (
                    index,
                    ky_value,
                    scan["growth_rate"][index],
                    scan["frequency"][index],
                    scan["kperp2_average"][index],
                    scan["quasilinear_contribution"][index],
                    scan["raw_amplitude"][-1, index],
                    scan["log_normalization"][index],
                )
            )

    with (output_dir / "convergence_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "sample_index",
                "time",
                "ky_index",
                "ky",
                "log_amplitude",
                "raw_amplitude",
                "window_growth",
            )
        )
        for sample_index, time_value in enumerate(scan["times"]):
            for ky_index, ky_value in enumerate(ky):
                window_growth = (
                    ""
                    if sample_index == 0
                    else scan["window_growth"][sample_index - 1, ky_index]
                )
                writer.writerow(
                    (
                        sample_index,
                        time_value,
                        ky_index,
                        ky_value,
                        scan["log_amplitude"][sample_index, ky_index],
                        scan["raw_amplitude"][sample_index, ky_index],
                        window_growth,
                    )
                )

    fixture = PerKyModeStructureFixture(
        ky=jnp.asarray(ky, dtype=jnp.float64),
        z=geometry.z,
        phi=jnp.asarray(scan["mode_structure"][:, fourier.ixzero, :].T),
        growth_rate=jnp.asarray(scan["growth_rate"]),
        frequency=jnp.asarray(scan["frequency"]),
        source=f"stellarator-linear-scan:{geometry_metadata['geometry_source']}",
        normalization="unit_mode_chain_amplitude",
        metadata=(
            ("growth_diagnostic", args.growth_diagnostic),
            ("steps_per_window", args.steps_per_window),
            ("n_windows", args.n_windows),
            ("dt", args.dt),
        ),
    )
    write_per_ky_mode_structure_fixture_csv(output_dir / "mode_structures.csv", fixture)

    _write_json(
        output_dir / "convergence_metadata.json",
        {
            "growth_diagnostic": args.growth_diagnostic,
            "growth_window_fraction": args.growth_window_fraction,
            "steps_per_window": args.steps_per_window,
            "n_windows": args.n_windows,
            "dt": args.dt,
            "normalize_each_window": not args.no_normalize_each_window,
            "estimated_cfl_dt": scan["estimated_cfl_dt"],
            "finite_growth": bool(np.all(np.isfinite(scan["growth_rate"]))),
            "finite_frequency": bool(np.all(np.isfinite(scan["frequency"]))),
            "max_abs_late_window_growth_delta": _late_window_delta(scan["window_growth"]),
        },
    )
    _write_json(
        output_dir / "quasilinear_proxy.json",
        {
            "total": scan["quasilinear_proxy_total"],
            "per_ky": [
                {
                    "ky": float(ky_value),
                    "growth_rate": float(scan["growth_rate"][index]),
                    "kperp2_average": float(scan["kperp2_average"][index]),
                    "contribution": float(scan["quasilinear_contribution"][index]),
                }
                for index, ky_value in enumerate(ky)
            ],
            "softplus_temperature": args.softplus_temperature,
            "kperp_epsilon": args.kperp_epsilon,
        },
    )
    _write_json(
        output_dir / "run_config.json",
        {
            "geometry": geometry_metadata,
            "n_vpar": args.n_vpar,
            "n_mu": args.n_mu,
            "velocity_backend": args.velocity_backend,
            "n_kx": args.n_kx,
            "kx_max": args.kx_max,
            "ky_values": list(ky),
            "ikxspace": args.ikxspace,
            "parallel_derivative_model": args.parallel_derivative_model,
            "species": {
                "charge": args.charge,
                "mass": args.mass,
                "density": args.density,
                "temperature": args.temperature,
                "density_gradient": args.density_gradient,
                "temperature_gradient": args.temperature_gradient,
            },
            "adiabatic_electrons": {
                "density": args.electron_density,
                "temperature": args.electron_temperature,
                "zonal_correction": args.zonal_correction,
            },
        },
    )


def _parallel_grid_from_z(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.shape[0] < 2:
        raise ValueError("fixture z must be a one-dimensional grid with at least two points")
    dz = z[1] - z[0]
    return build_parallel_grid(
        ParallelGridSpec(n_z=len(z), z_min=float(z[0]), z_max=float(z[-1] + dz))
    )


def _initial_state(velocity, parallel, fourier, amplitude: float):
    shape = (
        velocity.vpar.shape[0],
        velocity.mu.shape[0],
        parallel.z.shape[0],
        fourier.kx.shape[0],
        fourier.ky.shape[0],
    )
    index = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    z_profile = (1.0 + 0.25 * jnp.cos(2.0 * jnp.pi * parallel.z))[None, None, :, None, None]
    return amplitude * z_profile * (jnp.cos(index / 7.0) + 1j * jnp.sin(index / 11.0))


def _growth_from_log_amplitudes(times, log_amplitudes, diagnostic: str, late_start: int):
    if diagnostic == "endpoint":
        return (log_amplitudes[-1] - log_amplitudes[0]) / (times[-1] - times[0])
    if diagnostic == "late_mean_window":
        window_growth = np.diff(log_amplitudes, axis=0) / np.diff(times)[:, None]
        start = min(max(late_start, 0), window_growth.shape[0] - 1)
        return np.mean(window_growth[start:], axis=0)
    return _fit_log_amplitudes(times[late_start:], log_amplitudes[late_start:])


def _fit_log_amplitudes(times, log_amplitudes):
    centered_time = times - np.mean(times)
    centered_log = log_amplitudes - np.mean(log_amplitudes, axis=0)
    denominator = np.sum(centered_time**2)
    return np.sum(centered_time[:, None] * centered_log, axis=0) / denominator


def _late_start_index(n_samples: int, fraction: float) -> int:
    if n_samples < 2:
        raise ValueError("at least two samples are required")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("growth_window_fraction must lie in [0, 1)")
    return max(0, min(int(n_samples * fraction), n_samples - 2))


def _late_window_delta(window_growth):
    if window_growth.shape[0] < 2:
        return 0.0
    half = max(1, window_growth.shape[0] // 2)
    left = np.mean(window_growth[-2 * half : -half], axis=0)
    right = np.mean(window_growth[-half:], axis=0)
    return float(np.max(np.abs(right - left)))


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("ky_values must contain at least one value")
    return values


def _gate_summary(gate):
    return {
        "name": gate.target.name,
        "quantity": gate.target.quantity,
        "observed_value": float(gate.observed_value),
        "reference_value": float(gate.target.reference_value),
        "tolerance": float(gate.target.tolerance),
        "passed": bool(gate.passed),
        "notes": gate.notes,
    }


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape"):
        return _to_jsonable(np.asarray(value))
    return value


def _json_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.tolist()


if __name__ == "__main__":
    raise SystemExit(main())
