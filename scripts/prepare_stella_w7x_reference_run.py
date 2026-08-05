"""Prepare the matched stella W7-X reference run for mode-structure parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "fixtures/stella_w7x_mode_structure_run"
DEFAULT_VMEC = ROOT / "relevant-codes/gx/benchmarks/linear/ITG_w7x/wout_w7x.nc"
DEFAULT_EIK = (
    ROOT
    / "relevant-codes/gx/geometry_modules/vmec/tests/"
    "gist_gs2_wout_w7x_standardConfig_highres_surf12_pol_10_nz0_10000"
)
DEFAULT_STELLA_EXECUTABLE = ROOT / "relevant-codes/stella/stella"
DEFAULT_EXTERNAL_FIXTURE = ROOT / "fixtures/w7x_itg_external_mode_structure_fixture.csv"
DEFAULT_OBSERVED_FIXTURE = ROOT / "fixtures/w7x_itg_reduced_benchmark/mode_structures.csv"
DEFAULT_COMPARISON_OUTPUT = ROOT / "figures/w7x_itg_external_mode_structure_comparison.csv"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = prepare_stella_w7x_reference_run(
        output_dir=args.output_dir,
        vmec_file=args.vmec_file,
        eik_reference=args.eik_reference,
        stella_executable=args.stella_executable,
        external_fixture=args.external_fixture,
        observed_fixture=args.observed_fixture,
        comparison_output=args.comparison_output,
        torflux=args.torflux,
        alpha=args.alpha,
        gx_npol=args.gx_npol,
        nfp=args.nfp,
        q_value=args.q_value,
        ky_values=args.ky_values,
        export_ky_values=args.export_ky_values,
        nzed=args.nzed,
        nmu=args.nmu,
        nvgrid=args.nvgrid,
        tend=args.tend,
        delt=args.delt,
        nwrite=args.nwrite,
        average_fraction=args.average_fraction,
        copy_vmec=args.copy_vmec,
        overwrite=args.overwrite,
    )
    print(metadata["prepared_input"])
    print(metadata["run_command"])
    print(metadata["export_command"])
    return 0


def prepare_stella_w7x_reference_run(
    *,
    output_dir: Path,
    vmec_file: Path,
    eik_reference: Path,
    stella_executable: Path,
    external_fixture: Path,
    observed_fixture: Path,
    comparison_output: Path,
    torflux: float,
    alpha: float,
    gx_npol: float,
    nfp: float,
    q_value: float | None,
    ky_values: str,
    export_ky_values: str,
    nzed: int,
    nmu: int,
    nvgrid: int,
    tend: float,
    delt: float,
    nwrite: int,
    average_fraction: float,
    copy_vmec: bool = True,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write a stella input matched to the GX W7-X production gate."""

    output_dir = Path(output_dir)
    vmec_file = Path(vmec_file)
    eik_reference = Path(eik_reference)
    stella_executable = Path(stella_executable)
    if not vmec_file.exists():
        raise FileNotFoundError(vmec_file)
    if not eik_reference.exists():
        raise FileNotFoundError(eik_reference)
    if nzed < 2 or nmu < 1 or nvgrid < 1:
        raise ValueError("nzed, nmu, and nvgrid must be positive production-grid sizes")
    if nwrite < 1:
        raise ValueError("nwrite must be positive")
    if tend <= 0.0 or delt <= 0.0:
        raise ValueError("tend and delt must be positive")
    if not 0.0 <= average_fraction < 1.0:
        raise ValueError("average_fraction must lie in [0, 1)")

    ky_grid = _parse_float_list(ky_values)
    export_ky_grid = _parse_float_list(export_ky_values)
    if len(ky_grid) < 2:
        raise ValueError("ky_values must contain at least two modes")
    if any(value < 0.0 for value in ky_grid + export_ky_grid):
        raise ValueError("ky values must be nonnegative")
    if sorted(ky_grid) != ky_grid:
        raise ValueError("ky_values must be sorted")
    if not set(export_ky_grid).issubset(set(ky_grid)):
        raise ValueError("export_ky_values must be a subset of ky_values")

    q_from_eik = _read_gx_eik_q(eik_reference)
    q_used = float(q_value) if q_value is not None else q_from_eik
    nfield_periods = float(gx_npol) * float(q_used) * float(nfp)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = "stella_w7x_adiabatic_electrons.in"
    prepared_input = output_dir / input_name
    vmec_destination = output_dir / "wout_w7x.nc"
    if prepared_input.exists() and not overwrite:
        raise FileExistsError(f"{prepared_input} already exists; pass --overwrite")
    if copy_vmec and vmec_destination.exists() and not overwrite:
        raise FileExistsError(f"{vmec_destination} already exists; pass --overwrite")

    prepared_input.write_text(
        _stella_input_text(
            vmec_filename=vmec_destination.name,
            torflux=torflux,
            alpha=alpha,
            nfield_periods=nfield_periods,
            ky_grid=ky_grid,
            nzed=nzed,
            nmu=nmu,
            nvgrid=nvgrid,
            tend=tend,
            delt=delt,
            nwrite=nwrite,
        )
    )
    if copy_vmec:
        shutil.copy2(vmec_file, vmec_destination)

    run_base = prepared_input.with_suffix("").name
    stella_output = output_dir / f"{run_base}.out.nc"
    run_script = output_dir / "run_stella_reference.sh"
    export_script = output_dir / "export_stella_fixture.sh"
    run_command = f"bash {shlex.quote(_display_path(run_script))}"
    export_command = (
        "uv run python examples/export_stella_mode_structure_fixture.py "
        f"--stella-output {shlex.quote(_display_path(stella_output))} "
        f"--ky-values {shlex.quote(export_ky_values)} "
        f"--average-fraction {average_fraction:g} "
        "--stella-z-coordinate zed_over_2pi "
        f"--output {shlex.quote(_display_path(external_fixture))}"
    )
    comparison_command = (
        "JAX_ENABLE_X64=1 uv run python examples/compare_mode_structure_fixtures.py "
        f"--observed {shlex.quote(_display_path(observed_fixture))} "
        f"--reference {shlex.quote(_display_path(external_fixture))} "
        f"--ky-values {shlex.quote(export_ky_values)} "
        "--require-profile "
        "--resample-reference-to-observed-z "
        f"--output {shlex.quote(_display_path(comparison_output))}"
    )
    metadata: dict[str, object] = {
        "benchmark_name": "w7x_itg_external_stella_mode_structure_reference",
        "status": "prepared_stella_run_pending_execution",
        "prepared_input": _display_path(prepared_input),
        "stella_output": _display_path(stella_output),
        "stella_executable": _display_path(stella_executable),
        "vmec_source": _display_path(vmec_file),
        "vmec_source_sha256": _sha256(vmec_file),
        "vmec_destination": _display_path(vmec_destination),
        "vmec_copied": bool(copy_vmec),
        "eik_reference": _display_path(eik_reference),
        "eik_q_used": q_used,
        "gx_npol": float(gx_npol),
        "nfp": float(nfp),
        "nfield_periods": nfield_periods,
        "geometry": {
            "geometry_option": "vmec",
            "vmec_filename": vmec_destination.name,
            "torflux": float(torflux),
            "alpha0": float(alpha),
            "zeta_center": 0.0,
            "zed_equal_arc": True,
            "field_line_match": "nfield_periods = gx_npol * q_eik * nfp",
        },
        "species": {
            "kinetic_species": 1,
            "ion_density": 1.0,
            "ion_temperature": 1.0,
            "ion_density_gradient": 1.0,
            "ion_temperature_gradient": 3.0,
            "electron_model": "adiabatic",
            "adiabatic_option": "field-line-average-term",
            "tite": 1.0,
            "nine": 1.0,
        },
        "grid": {
            "ky_values": ky_grid,
            "export_ky_values": export_ky_grid,
            "kx_values": [0.0],
            "naky": len(ky_grid),
            "nakx": 1,
            "nzed": int(nzed),
            "nmu": int(nmu),
            "nvgrid": int(nvgrid),
            "stella_vpa_points": 2 * int(nvgrid),
        },
        "time": {
            "tend": float(tend),
            "delt": float(delt),
            "nwrite": int(nwrite),
            "growth_average_fraction": float(average_fraction),
        },
        "external_fixture": _display_path(external_fixture),
        "observed_fixture": _display_path(observed_fixture),
        "comparison_output": _display_path(comparison_output),
        "run_command": run_command,
        "export_command": export_command,
        "comparison_command": comparison_command,
        "notes": (
            "This is the matched CPU continuum stella W7-X reference path. "
            "It uses the GX benchmark VMEC rather than stella's bundled W7-X "
            "VMEC because the existing solver gate is tied to the GX VMEC/eik "
            "provenance."
        ),
    }
    (output_dir / "mode_structure_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "README.md").write_text(_readme_text(metadata))
    run_script.write_text(_run_script_text(input_name, output_dir))
    export_script.write_text(_export_script_text(metadata, output_dir))
    return metadata


def _stella_input_text(
    *,
    vmec_filename: str,
    torflux: float,
    alpha: float,
    nfield_periods: float,
    ky_grid: list[float],
    nzed: int,
    nmu: int,
    nvgrid: int,
    tend: float,
    delt: float,
    nwrite: int,
) -> str:
    aky_min = min(ky_grid)
    aky_max = max(ky_grid)
    return f"""! Matched W7-X ITG reference for stellarator_gk.
! Generated by scripts/prepare_stella_w7x_reference_run.py.
! Physics provenance: GX ITG_w7x adiabatic-electron input.

&geometry_options
  geometry_option = 'vmec'
/

&geometry_vmec
  alpha0 = {alpha:.16g}
  zeta_center = 0.0
  nfield_periods = {nfield_periods:.16g}
  torflux = {torflux:.16g}
  vmec_filename = '{vmec_filename}'
/

&gyrokinetic_terms
  include_nonlinear = .false.
/

&diagnostics
  nsave = 100
  nwrite = {nwrite:d}
  save_for_restart = .false.
  write_all_time_traces = .true.
  write_all_spectra_kxkyz = .true.
  write_all_spectra_kxky = .true.
  write_all_velocity_space = .false.
  write_all_potential = .true.
  write_all_omega = .true.
  write_all_distribution = .false.
  write_all_fluxes = .true.
/

&initialise_distribution
  initialise_distribution_option = 'default'
  phiinit = 1.0e-10
/

&initialise_distribution_maxwellian
  width0 = 1.0
/

&species_options
  nspec = 1
/

&species_parameters_1
  z = 1.0
  mass = 1.0
  dens = 1.0
  temp = 1.0
  fprim = 1.0
  tprim = 3.0
  type = 'ion'
/

&adiabatic_electron_response
  adiabatic_option = 'field-line-average-term'
  tite = 1.0
  nine = 1.0
/

&kxky_grid_option
  grid_option = 'range'
/

&kxky_grid_range
  aky_min = {aky_min:.16g}
  aky_max = {aky_max:.16g}
  nakx = 1
  naky = {len(ky_grid):d}
/

&z_grid
  nperiod = 1
  nzed = {nzed:d}
  zed_equal_arc = .true.
/

&velocity_grids
  nmu = {nmu:d}
  nvgrid = {nvgrid:d}
/

&time_trace_options
  tend = {tend:.16g}
/

&time_step
  delt = {delt:.16g}
/

&dissipation_and_collisions_options
  hyper_dissipation = .false.
  include_collisions = .false.
/

&parallelisation
  lu_option = 'local'
/
"""


def _read_gx_eik_q(path: Path) -> float:
    first_line = Path(path).read_text().splitlines()[0]
    values = [float(item) for item in first_line.split()]
    if len(values) < 5:
        raise ValueError(f"could not parse GX eik q from {path}")
    return values[4]


def _readme_text(metadata: dict[str, object]) -> str:
    return "\n".join(
        (
            "# stella W7-X Mode-Structure Reference Run",
            "",
            "This directory contains the matched stella W7-X linear ITG input",
            "for the external continuum-code mode-structure reference.",
            "",
            "Matched controls:",
            "",
            f"- VMEC source: `{metadata['vmec_source']}`",
            f"- torflux: `{metadata['geometry']['torflux']}`",
            f"- alpha0: `{metadata['geometry']['alpha0']}`",
            f"- nfield_periods: `{metadata['nfield_periods']}`",
            f"- electron model: `{metadata['species']['electron_model']}`",
            f"- ky grid: `{metadata['grid']['ky_values']}`",
            f"- nzed/nmu/nvgrid: `{metadata['grid']['nzed']}` / "
            f"`{metadata['grid']['nmu']}` / `{metadata['grid']['nvgrid']}`",
            f"- tend/delt/growth window: `{metadata['time']['tend']}` / "
            f"`{metadata['time']['delt']}` / "
            f"`{metadata['time']['growth_average_fraction']}`",
            "",
            "Run stella:",
            "",
            f"```bash\n{metadata['run_command']}\n```",
            "",
            "Export the portable reference fixture:",
            "",
            f"```bash\n{metadata['export_command']}\n```",
            "",
            "Compare against the current solver fixture:",
            "",
            f"```bash\n{metadata['comparison_command']}\n```",
            "",
            "The production parity claim remains open until the stella run",
            "completes and the exported fixture passes the W7-X mode-structure",
            "gate.",
            "",
        )
    )


def _run_script_text(input_name: str, output_dir: Path) -> str:
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            _repo_root_shell_assignment(output_dir),
            'STELLA_EXECUTABLE="${STELLA_EXECUTABLE:?Set STELLA_EXECUTABLE to the pinned stella binary}"',
            'cd "${SCRIPT_DIR}"',
            f'"${{STELLA_EXECUTABLE}}" {shlex.quote(input_name)}',
            "",
        )
    )


def _export_script_text(metadata: dict[str, object], output_dir: Path) -> str:
    return "\n".join(
        (
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            _repo_root_shell_assignment(output_dir),
            'cd "${REPO_ROOT}"',
            metadata["export_command"],
            metadata["comparison_command"],
            "uv run python scripts/run_w7x_production_readiness_gate.py",
            "",
        )
    )


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one comma-separated float")
    return parsed


def _repo_root_shell_assignment(output_dir: Path) -> str:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    try:
        relative = output_path.relative_to(ROOT)
    except ValueError:
        return f"REPO_ROOT={shlex.quote(str(ROOT))}"
    up_levels = "/".join(".." for _ in relative.parts) or "."
    return f'REPO_ROOT="$(cd "${{SCRIPT_DIR}}/{up_levels}" && pwd)"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vmec-file", type=Path, default=DEFAULT_VMEC)
    parser.add_argument("--eik-reference", type=Path, default=DEFAULT_EIK)
    parser.add_argument("--stella-executable", type=Path, default=DEFAULT_STELLA_EXECUTABLE)
    parser.add_argument("--external-fixture", type=Path, default=DEFAULT_EXTERNAL_FIXTURE)
    parser.add_argument("--observed-fixture", type=Path, default=DEFAULT_OBSERVED_FIXTURE)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--torflux", type=float, default=0.64)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--gx-npol", type=float, default=6.0)
    parser.add_argument("--nfp", type=float, default=5.0)
    parser.add_argument("--q-value", type=float)
    parser.add_argument("--ky-values", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--export-ky-values", default="0.1,0.2,0.3")
    parser.add_argument("--nzed", type=int, default=256)
    parser.add_argument("--nmu", type=int, default=8)
    parser.add_argument("--nvgrid", type=int, default=16)
    parser.add_argument("--tend", type=float, default=200.0)
    parser.add_argument("--delt", type=float, default=0.1)
    parser.add_argument("--nwrite", type=int, default=10)
    parser.add_argument("--average-fraction", type=float, default=0.5)
    parser.add_argument("--copy-vmec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
