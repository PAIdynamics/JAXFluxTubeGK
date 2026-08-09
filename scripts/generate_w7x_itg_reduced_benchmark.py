"""Generate the reduced W7-X ITG stellarator benchmark fixture.

This fixture uses the real GX/GIST W7-X eik geometry table and the local GX
W7-X ITG input deck as provenance.  The emitted growth rates and mode
structures are reduced solver-regression diagnostics until a matching external
GX, GKW, GS2, or stella time-history fixture is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/w7x_itg_reduced_benchmark"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from jax_fluxtube_gk.external import announce_external_path

    announce_external_path("GX/GIST eik", args.eik_reference)
    announce_external_path("GX input", args.gx_input)
    sys.path.insert(0, str(ROOT))
    from examples.run_stellarator_linear_scan import main as run_scan

    scan_args = [
        "--geometry-source",
        "eik",
        "--eik-reference",
        _relative_or_absolute(args.eik_reference),
        "--output-dir",
        str(args.output_dir),
        "--n-z",
        str(args.n_z),
        "--field-line-periods",
        str(args.field_line_periods),
        "--ky-values",
        args.ky_values,
        "--n-kx",
        str(args.n_kx),
        "--kx-max",
        str(args.kx_max),
        "--ikxspace",
        str(args.ikxspace),
        "--n-vpar",
        str(args.n_vpar),
        "--n-mu",
        str(args.n_mu),
        "--vpar-max",
        str(args.vpar_max),
        "--mu-max",
        str(args.mu_max),
        "--density",
        str(args.density),
        "--temperature",
        str(args.temperature),
        "--density-gradient",
        str(args.density_gradient),
        "--temperature-gradient",
        str(args.temperature_gradient),
        "--electron-density",
        str(args.electron_density),
        "--electron-temperature",
        str(args.electron_temperature),
        "--dt",
        str(args.dt),
        "--steps-per-window",
        str(args.steps_per_window),
        "--n-windows",
        str(args.n_windows),
        "--growth-diagnostic",
        args.growth_diagnostic,
        "--growth-window-fraction",
        str(args.growth_window_fraction),
    ]
    if args.velocity_backend != "chebyshev":
        scan_args.extend(["--velocity-backend", args.velocity_backend])
    if args.parallel_derivative_model != "matrix":
        scan_args.extend(["--parallel-derivative-model", args.parallel_derivative_model])
    if args.perpendicular_damping is not None:
        scan_args.extend(["--perpendicular-damping", str(args.perpendicular_damping)])

    result = run_scan(scan_args)
    _write_metadata(args)
    print(args.output_dir / "benchmark_metadata.json")
    return result


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eik-reference", type=Path, required=True)
    parser.add_argument("--gx-input", type=Path, required=True)
    parser.add_argument("--n-z", type=int, default=33)
    parser.add_argument("--field-line-periods", type=int, default=1)
    parser.add_argument("--ky-values", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--n-kx", type=int, default=3)
    parser.add_argument("--kx-max", type=float, default=0.3)
    parser.add_argument("--ikxspace", type=int, default=2)
    parser.add_argument("--n-vpar", type=int, default=4)
    parser.add_argument("--n-mu", type=int, default=4)
    parser.add_argument("--vpar-max", type=float, default=2.0)
    parser.add_argument("--mu-max", type=float, default=1.5)
    parser.add_argument(
        "--velocity-backend",
        choices=("chebyshev", "finite_difference"),
        default="chebyshev",
    )
    parser.add_argument(
        "--parallel-derivative-model",
        choices=("matrix", "gkw_upwind", "gkw_igh"),
        default="matrix",
    )
    parser.add_argument("--perpendicular-damping", type=float)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--density-gradient", type=float, default=1.0)
    parser.add_argument("--temperature-gradient", type=float, default=3.0)
    parser.add_argument("--electron-density", type=float, default=1.0)
    parser.add_argument("--electron-temperature", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps-per-window", type=int, default=1)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument(
        "--growth-diagnostic",
        choices=("late_fit", "late_mean_window", "endpoint"),
        default="late_fit",
    )
    parser.add_argument("--growth-window-fraction", type=float, default=0.5)
    return parser.parse_args(argv)


def _write_metadata(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_name": "w7x_itg_adiabatic_electrons_reduced",
        "validation_status": "real_external_geometry_internal_reduced_solver_regression",
        "external_growth_frequency_mode_structure_reference": None,
        "external_reference_workflow": {
            "status": "prepared_gx_run_pending_external_execution",
            "run_prep_dir": "fixtures/gx_w7x_mode_structure_run",
            "run_prep_metadata": (
                "fixtures/gx_w7x_mode_structure_run/mode_structure_run_metadata.json"
            ),
            "expected_external_fixture": "fixtures/w7x_itg_external_mode_structure_fixture.csv",
            "comparison_output": "figures/w7x_itg_external_mode_structure_comparison.csv",
        },
        "notes": (
            "The geometry and GX input provenance are real W7-X references. "
            "The growth, frequency, and mode-structure files are generated by "
            "the current reduced jax_fluxtube_gk linear RHS and should be used "
            "as regression artifacts until a matching external code output is added."
        ),
        "external_geometry_reference": {
            "format": "GX/GIST/GS2 eik table",
            "path": _relative_or_absolute(args.eik_reference),
            "sha256": _sha256_if_exists(args.eik_reference),
        },
        "gx_input_reference": {
            "path": _relative_or_absolute(args.gx_input),
            "sha256": _sha256_if_exists(args.gx_input),
            "vmec_file": "wout_w7x.nc",
            "torflux": 0.64,
            "alpha": 0.0,
            "nperiod": 1,
            "vmec_npol": 6.0,
            "ntheta": 256,
            "nky": 28,
            "nkx": 1,
            "nhermite": 16,
            "nlaguerre": 8,
            "ion_density_gradient": 1.0,
            "ion_temperature_gradient": 3.0,
            "adiabatic_electron_tau_fac": 1.0,
        },
        "reduced_solver_controls": {
            "n_z": args.n_z,
            "field_line_periods": args.field_line_periods,
            "ky_values": [
                float(item.strip()) for item in args.ky_values.split(",") if item.strip()
            ],
            "n_kx": args.n_kx,
            "kx_max": args.kx_max,
            "ikxspace": args.ikxspace,
            "n_vpar": args.n_vpar,
            "n_mu": args.n_mu,
            "vpar_max": args.vpar_max,
            "mu_max": args.mu_max,
            "velocity_backend": args.velocity_backend,
            "parallel_derivative_model": args.parallel_derivative_model,
            "perpendicular_damping": args.perpendicular_damping,
            "dt": args.dt,
            "steps_per_window": args.steps_per_window,
            "n_windows": args.n_windows,
            "growth_diagnostic": args.growth_diagnostic,
            "growth_window_fraction": args.growth_window_fraction,
        },
        "species": {
            "ion_density": args.density,
            "ion_temperature": args.temperature,
            "ion_density_gradient": args.density_gradient,
            "ion_temperature_gradient": args.temperature_gradient,
            "electron_density": args.electron_density,
            "electron_temperature": args.electron_temperature,
            "electron_model": "adiabatic",
        },
        "artifact_contract": {
            "required_files": [
                "geometry_audit.json",
                "geometry_audit.csv",
                "ky_growth.csv",
                "mode_structures.csv",
                "convergence_history.csv",
                "convergence_metadata.json",
                "quasilinear_proxy.json",
                "run_config.json",
                "benchmark_metadata.json",
            ],
            "geometry_preflight_required": True,
            "mirror_fd_check_enabled": False,
        },
    }
    _write_json(args.output_dir / "benchmark_metadata.json", payload)


def _sha256_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
