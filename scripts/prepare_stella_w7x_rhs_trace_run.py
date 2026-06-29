"""Prepare a non-destructive stella W7-X ``ky=0.3`` RHS trace run.

The generated run copies the local stella checkout, patches only the copy, and
adds a narrow diagnostic around ``add_explicit_gyrokinetic_terms``.  The patch
writes the selected ``iky=4``/``ikx=1`` complex ``pdf/g`` state, ``phi``, total
RHS, and RHS deltas for mirror, magnetic-drift, drive, and parallel-streaming
terms.

The dedicated RHS trace input forces mirror and parallel streaming through the
explicit RHS path by default.  The production stella W7-X comparison input keeps
those two terms implicit, so it cannot expose their deltas inside
``add_explicit_gyrokinetic_terms``.  stella's explicit RHS is native ``rhs*dt``;
the trace preserves that unit so the Python comparator can decide when to
divide by ``code_dt``.  Trace format v2 also records stella's velocity
quadrature weights for direct array-weighted parity checks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STELLA_SOURCE = ROOT / "relevant-codes/stella"
DEFAULT_REFERENCE_RUN = ROOT / "fixtures/stella_w7x_mode_structure_run"
DEFAULT_INPUT = DEFAULT_REFERENCE_RUN / "stella_w7x_adiabatic_electrons.in"
DEFAULT_VMEC = DEFAULT_REFERENCE_RUN / "wout_w7x.nc"
DEFAULT_OUTPUT_ROOT = Path("/tmp/stellarator_gk_stella_w7x_rhs_trace")
TRACE_FILENAME = "stellarator_gk_w7x_ky03_rhs_trace.dat"
TARGET_RELATIVE_SOURCE = Path(
    "STELLA_CODE/gyrokinetic_equation/gyrokinetic_equation_explicit.f90"
)


@dataclass(frozen=True)
class PreparedStellaW7xRhsTraceRun:
    """Paths produced by :func:`prepare_stella_w7x_rhs_trace_run`."""

    output_root: Path
    patched_source_root: Path
    patched_explicit_source: Path
    run_dir: Path
    input_file: Path
    vmec_file: Path
    build_script: Path
    run_script: Path
    metadata_file: Path
    trace_step: int
    trace_iky: int
    trace_ikx: int
    trace_filename: str = TRACE_FILENAME


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    prepared = prepare_stella_w7x_rhs_trace_run(
        stella_source=args.stella_source,
        input_file=args.input_file,
        vmec_file=args.vmec_file,
        output_root=args.output_root,
        trace_step=args.trace_step,
        trace_iky=args.trace_iky,
        trace_ikx=args.trace_ikx,
        tend=args.tend,
        delt=args.delt,
        force_explicit_stream_mirror=not args.preserve_implicit_stream_mirror,
        overwrite=args.overwrite,
    )
    print(prepared.metadata_file)
    print(prepared.build_script)
    print(prepared.run_script)
    return 0


def prepare_stella_w7x_rhs_trace_run(
    *,
    stella_source: Path = DEFAULT_STELLA_SOURCE,
    input_file: Path = DEFAULT_INPUT,
    vmec_file: Path = DEFAULT_VMEC,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    trace_step: int = 2000,
    trace_iky: int = 4,
    trace_ikx: int = 1,
    tend: float = 200.0,
    delt: float = 0.1,
    force_explicit_stream_mirror: bool = True,
    overwrite: bool = False,
) -> PreparedStellaW7xRhsTraceRun:
    """Copy stella, patch the copy, and write a matched RHS-trace run."""

    stella_source = Path(stella_source).resolve()
    input_file = Path(input_file).resolve()
    vmec_file = Path(vmec_file).resolve()
    output_root = Path(output_root).resolve()
    if not (stella_source / TARGET_RELATIVE_SOURCE).is_file():
        raise FileNotFoundError(stella_source / TARGET_RELATIVE_SOURCE)
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if not vmec_file.is_file():
        raise FileNotFoundError(vmec_file)
    if output_root == stella_source:
        raise ValueError("output_root must be different from stella_source")
    if trace_step < 1 or trace_iky < 1 or trace_ikx < 1:
        raise ValueError("trace_step, trace_iky, and trace_ikx must be positive Fortran indices")
    if tend <= 0.0 or delt <= 0.0:
        raise ValueError("tend and delt must be positive")
    expected_step = int(round(tend / delt))
    if abs(expected_step * delt - tend) > 1.0e-10:
        raise ValueError("tend/delt must be an integer for the matched trace run")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)
    patched_source_root = output_root / "stella"
    run_dir = output_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(stella_source, patched_source_root, ignore=_ignore_stella_copy_entries)

    patched_explicit = patched_source_root / TARGET_RELATIVE_SOURCE
    patch_stella_explicit_rhs_trace(
        patched_explicit,
        trace_step=trace_step,
        trace_iky=trace_iky,
        trace_ikx=trace_ikx,
        trace_filename=TRACE_FILENAME,
    )
    prepared_input = run_dir / "stella_w7x_ky03_rhs_trace.in"
    write_rhs_trace_input(
        input_file,
        prepared_input,
        nstep=trace_step,
        delt=delt,
        vmec_filename="wout_w7x.nc",
        force_explicit_stream_mirror=force_explicit_stream_mirror,
    )
    vmec_destination = run_dir / "wout_w7x.nc"
    shutil.copy2(vmec_file, vmec_destination)
    build_script = output_root / "build_stella_rhs_trace.sh"
    run_script = output_root / "run_stella_rhs_trace.sh"
    metadata_file = output_root / "rhs_trace_run_metadata.json"
    metadata = _metadata_payload(
        output_root=output_root,
        patched_source_root=patched_source_root,
        patched_explicit=patched_explicit,
        run_dir=run_dir,
        input_file=prepared_input,
        vmec_file=vmec_destination,
        build_script=build_script,
        run_script=run_script,
        trace_step=trace_step,
        trace_iky=trace_iky,
        trace_ikx=trace_ikx,
        tend=tend,
        delt=delt,
        force_explicit_stream_mirror=force_explicit_stream_mirror,
    )
    build_script.write_text(_build_script_text(), encoding="utf-8")
    run_script.write_text(_run_script_text(prepared_input.name), encoding="utf-8")
    build_script.chmod(0o755)
    run_script.chmod(0o755)
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_root / "README.md").write_text(_readme_text(metadata), encoding="utf-8")
    return PreparedStellaW7xRhsTraceRun(
        output_root=output_root,
        patched_source_root=patched_source_root,
        patched_explicit_source=patched_explicit,
        run_dir=run_dir,
        input_file=prepared_input,
        vmec_file=vmec_destination,
        build_script=build_script,
        run_script=run_script,
        metadata_file=metadata_file,
        trace_step=trace_step,
        trace_iky=trace_iky,
        trace_ikx=trace_ikx,
    )


def patch_stella_explicit_rhs_trace(
    source_path: Path,
    *,
    trace_step: int,
    trace_iky: int,
    trace_ikx: int,
    trace_filename: str = TRACE_FILENAME,
) -> bool:
    """Insert the focused RHS trace hook into copied stella explicit RHS source."""

    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    if "stellarator_gk W7-X ky=0.3 RHS trace patch" in text:
        return False
    text = _insert_module_trace_state(
        text,
        trace_step=trace_step,
        trace_iky=trace_iky,
        trace_ikx=trace_ikx,
        trace_filename=trace_filename,
    )
    text = _patch_add_explicit_imports_and_locals(text)
    text = _patch_term_call(text, "advance_mirror_explicit(pdf, rhs)", "mirror_force")
    text = _patch_term_call(text, "advance_wdrifty_explicit(pdf, phi, bpar, rhs)", "magnetic_drift_y")
    text = _patch_term_call(text, "advance_wdriftx_explicit(pdf, phi, bpar, rhs)", "magnetic_drift_x")
    text = _patch_term_call(text, "advance_wstar_explicit(phi, rhs)", "equilibrium_drive_wstar")
    text = _patch_term_call(
        text,
        "advance_parallel_streaming_explicit(pdf, phi, bpar, rhs)",
        "parallel_streaming",
    )
    final_marker = "      ! if advancing apar, need to convert input pdf back from g to gbar\n"
    final_patch = (
        "      if (stellarator_gk_trace_active(istep)) then\n"
        "         call stellarator_gk_write_complex_state('rhs_total', 'total', istep, rhs)\n"
        "      end if\n"
        "      if (allocated(stellarator_gk_rhs_before)) deallocate (stellarator_gk_rhs_before)\n\n"
    )
    if final_marker not in text:
        raise ValueError("could not find add_explicit final conversion marker")
    text = text.replace(final_marker, final_patch + final_marker, 1)
    helper_marker = "   !****************************************************************************\n   !                                      Title\n"
    if helper_marker not in text:
        raise ValueError("could not find helper insertion marker after add_explicit")
    text = text.replace(helper_marker, STELLA_RHS_TRACE_HELPERS + helper_marker, 1)
    source_path.write_text(text, encoding="utf-8")
    return True


def write_rhs_trace_input(
    input_path: Path,
    output_path: Path,
    *,
    nstep: int,
    delt: float,
    vmec_filename: str,
    force_explicit_stream_mirror: bool = True,
) -> None:
    """Write a stella input that stops exactly at the traced step."""

    text = Path(input_path).read_text(encoding="utf-8")
    text = _set_namelist_value(text, "time_trace_options", "nstep", str(int(nstep)))
    text = _set_namelist_value(text, "time_trace_options", "tend", "-1.0")
    text = _set_namelist_value(text, "time_step", "delt", f"{delt:g}")
    text = _set_namelist_value(text, "diagnostics", "nsave", "0")
    text = _set_namelist_value(text, "diagnostics", "save_for_restart", ".false.")
    if force_explicit_stream_mirror:
        text = _set_namelist_value(text, "numerical_algorithms", "stream_implicit", "F")
        text = _set_namelist_value(text, "numerical_algorithms", "mirror_implicit", "F")
        text = _set_namelist_value(text, "numerical_algorithms", "mirror_semi_lagrange", "F")
        text = _set_namelist_value(text, "numerical_algorithms", "fully_explicit", "T")
    text = _set_namelist_value(
        text,
        "geometry_vmec",
        "vmec_filename",
        f"'{vmec_filename}'",
    )
    output_path.write_text(
        "! Matched W7-X ky=0.3 RHS/source trace input for stellarator_gk.\n"
        "! Generated by scripts/prepare_stella_w7x_rhs_trace_run.py.\n"
        f"! force_explicit_stream_mirror = {str(force_explicit_stream_mirror).lower()}.\n"
        + text,
        encoding="utf-8",
    )


def _insert_module_trace_state(
    text: str,
    *,
    trace_step: int,
    trace_iky: int,
    trace_ikx: int,
    trace_filename: str,
) -> str:
    marker = "   private\n"
    patch = f"""
   ! stellarator_gk W7-X ky=0.3 RHS trace patch.
   ! This diagnostic is intentionally narrow and intended for serial matched
   ! W7-X reference runs.  It writes stella-native explicit RHS increments
   ! (rhs*dt), not continuous-time RHS values.
   integer, parameter :: stellarator_gk_trace_step = {trace_step}
   integer, parameter :: stellarator_gk_trace_iky = {trace_iky}
   integer, parameter :: stellarator_gk_trace_ikx = {trace_ikx}
   integer, parameter :: stellarator_gk_trace_unit = 9304
   character(len=*), parameter :: stellarator_gk_trace_filename = '{trace_filename}'
   logical :: stellarator_gk_trace_file_initialised = .false.

"""
    if marker not in text:
        raise ValueError("could not find module private marker")
    return text.replace(marker, marker + patch, 1)


def _patch_add_explicit_imports_and_locals(text: str) -> str:
    old = "      use parallelisation_layouts, only: vmu_lo\n"
    new = (
        "      use parallelisation_layouts, only: vmu_lo\n"
        "      use parallelisation_layouts, only: iv_idx, imu_idx, is_idx\n"
    )
    if old not in text:
        raise ValueError("could not find vmu_lo import in add_explicit")
    text = text.replace(old, new, 1)

    old = "      complex, dimension(:, :), allocatable :: rhs_ky_swap\n"
    new = (
        "      complex, dimension(:, :), allocatable :: rhs_ky_swap\n"
        "      complex, dimension(:, :, :, :, :), allocatable :: stellarator_gk_rhs_before\n"
    )
    if old not in text:
        raise ValueError("could not find rhs_ky_swap local declaration")
    text = text.replace(old, new, 1)

    marker = "      restart_time_step = .false.\n"
    patch = (
        "      if (stellarator_gk_trace_active(istep)) then\n"
        "         call stellarator_gk_write_complex_state('pdf_g', 'input_pdf', istep, pdf)\n"
        "         call stellarator_gk_write_phi_trace(istep, phi)\n"
        "      end if\n"
        "\n"
    )
    if marker not in text:
        raise ValueError("could not find restart_time_step marker")
    return text.replace(marker, marker + patch, 1)


def _patch_term_call(text: str, call_text: str, term_name: str) -> str:
    pattern = re.compile(rf"(?m)^(?P<indent>\s*)call {re.escape(call_text)}\s*$")
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"could not find term call: {call_text}")
    indent = match.group("indent")
    new = (
        f"{indent}if (stellarator_gk_trace_active(istep)) then\n"
        f"{indent}   call stellarator_gk_capture_rhs_before(rhs, stellarator_gk_rhs_before)\n"
        f"{indent}end if\n"
        f"{indent}call {call_text}\n"
        f"{indent}if (stellarator_gk_trace_active(istep)) then\n"
        f"{indent}   call stellarator_gk_write_rhs_delta('{term_name}', istep, rhs, stellarator_gk_rhs_before)\n"
        f"{indent}end if\n"
    )
    return text[: match.start()] + new + text[match.end() :]


STELLA_RHS_TRACE_HELPERS = r"""
   !****************************************************************************
   !              stellarator_gk selected-mode RHS trace helpers
   !****************************************************************************
   logical function stellarator_gk_trace_active(istep)

      use parameters_physics, only: full_flux_surface

      implicit none

      integer, intent(in) :: istep

      stellarator_gk_trace_active = (.not. full_flux_surface) .and. &
           (istep == stellarator_gk_trace_step)

   end function stellarator_gk_trace_active

   subroutine stellarator_gk_capture_rhs_before(rhs, rhs_before)

      use parallelisation_layouts, only: vmu_lo
      use grids_z, only: nzgrid

      implicit none

      complex, dimension(:, :, -nzgrid:, :, vmu_lo%llim_proc:), intent(in) :: rhs
      complex, dimension(:, :, :, :, :), allocatable, intent(in out) :: rhs_before

      if (.not. allocated(rhs_before)) then
         allocate (rhs_before(size(rhs, 1), size(rhs, 2), -nzgrid:nzgrid, &
              size(rhs, 4), vmu_lo%llim_proc:vmu_lo%ulim_alloc))
      end if
      rhs_before = rhs

   end subroutine stellarator_gk_capture_rhs_before

   subroutine stellarator_gk_write_rhs_delta(term_name, istep, rhs, rhs_before)

      use parallelisation_layouts, only: vmu_lo
      use grids_z, only: nzgrid, ntubes
      use grids_kxky, only: naky, nakx

      implicit none

      character(len=*), intent(in) :: term_name
      integer, intent(in) :: istep
      complex, dimension(:, :, -nzgrid:, :, vmu_lo%llim_proc:), intent(in) :: rhs
      complex, dimension(:, :, :, :, :), allocatable, intent(in) :: rhs_before
      complex, dimension(:, :, :, :, :), allocatable :: delta

      if (.not. allocated(rhs_before)) return
      if (stellarator_gk_trace_iky > naky .or. stellarator_gk_trace_ikx > nakx) return
      allocate (delta(size(rhs, 1), size(rhs, 2), -nzgrid:nzgrid, &
           ntubes, vmu_lo%llim_proc:vmu_lo%ulim_alloc))
      delta = rhs - rhs_before
      call stellarator_gk_write_complex_state('rhs_delta', term_name, istep, delta)
      deallocate (delta)

   end subroutine stellarator_gk_write_rhs_delta

   subroutine stellarator_gk_write_complex_state(record_type, term_name, istep, values)

      use mp, only: proc0
      use parallelisation_layouts, only: vmu_lo
      use parallelisation_layouts, only: iv_idx, imu_idx, is_idx
      use grids_z, only: nzgrid, ntubes
      use grids_kxky, only: naky, nakx
      use grids_velocity, only: vpa, mu, wgts_vpa, wgts_mu

      implicit none

      character(len=*), intent(in) :: record_type, term_name
      integer, intent(in) :: istep
      complex, dimension(:, :, -nzgrid:, :, vmu_lo%llim_proc:), intent(in) :: values

      integer :: iz, it, ivmu, iv, imu, is

      if (.not. proc0) return
      if (stellarator_gk_trace_iky > naky .or. stellarator_gk_trace_ikx > nakx) return
      call stellarator_gk_open_trace_file()
      do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
         iv = iv_idx(vmu_lo, ivmu)
         imu = imu_idx(vmu_lo, ivmu)
         is = is_idx(vmu_lo, ivmu)
         do it = 1, ntubes
            do iz = -nzgrid, nzgrid
               call stellarator_gk_write_trace_row(record_type, term_name, istep, iz, it, &
                    ivmu, iv, imu, is, vpa(iv), mu(imu), wgts_vpa(iv), wgts_mu(1, iz, imu), &
                    values(stellarator_gk_trace_iky, stellarator_gk_trace_ikx, iz, it, ivmu))
            end do
         end do
      end do
      close (stellarator_gk_trace_unit)

   end subroutine stellarator_gk_write_complex_state

   subroutine stellarator_gk_write_phi_trace(istep, phi)

      use mp, only: proc0
      use grids_z, only: nzgrid, ntubes
      use grids_kxky, only: naky, nakx

      implicit none

      integer, intent(in) :: istep
      complex, dimension(:, :, -nzgrid:, :), intent(in) :: phi

      integer :: iz, it

      if (.not. proc0) return
      if (stellarator_gk_trace_iky > naky .or. stellarator_gk_trace_ikx > nakx) return
      call stellarator_gk_open_trace_file()
      do it = 1, ntubes
         do iz = -nzgrid, nzgrid
            call stellarator_gk_write_trace_row('phi', 'field_phi', istep, iz, it, &
                 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, &
                 phi(stellarator_gk_trace_iky, stellarator_gk_trace_ikx, iz, it))
         end do
      end do
      close (stellarator_gk_trace_unit)

   end subroutine stellarator_gk_write_phi_trace

   subroutine stellarator_gk_open_trace_file()

      implicit none

      if (.not. stellarator_gk_trace_file_initialised) then
         open (unit=stellarator_gk_trace_unit, file=stellarator_gk_trace_filename, &
              status='replace', action='write')
         write (stellarator_gk_trace_unit, '(A)') &
              'record step term iky ikx iz it ivmu iv imu is vpa mu ' // &
              'wgts_vpa wgts_mu code_time code_dt real imag'
         stellarator_gk_trace_file_initialised = .true.
      else
         open (unit=stellarator_gk_trace_unit, file=stellarator_gk_trace_filename, &
              status='old', position='append', action='write')
      end if

   end subroutine stellarator_gk_open_trace_file

   subroutine stellarator_gk_write_trace_row(record_type, term_name, istep, iz, it, &
        ivmu, iv, imu, is, vpa_value, mu_value, wgts_vpa_value, wgts_mu_value, value)

      use grids_time, only: code_dt, code_time

      implicit none

      character(len=*), intent(in) :: record_type, term_name
      integer, intent(in) :: istep, iz, it, ivmu, iv, imu, is
      real, intent(in) :: vpa_value, mu_value, wgts_vpa_value, wgts_mu_value
      complex, intent(in) :: value

      write (stellarator_gk_trace_unit, &
           '(A,1X,I0,1X,A,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0,' // &
           '1X,ES24.16E3,1X,ES24.16E3,1X,ES24.16E3,1X,ES24.16E3,' // &
           '1X,ES24.16E3,1X,ES24.16E3,' // &
           '1X,ES24.16E3,1X,ES24.16E3)') &
           trim(record_type), istep, trim(term_name), stellarator_gk_trace_iky, &
           stellarator_gk_trace_ikx, iz, it, ivmu, iv, imu, is, vpa_value, &
           mu_value, wgts_vpa_value, wgts_mu_value, code_time, code_dt, real(value), aimag(value)

   end subroutine stellarator_gk_write_trace_row

"""


def _set_namelist_value(text: str, namelist: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?ims)^&{re.escape(namelist)}\b(?P<body>.*?)(?=^/)", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        suffix = "" if text.endswith("\n") else "\n"
        return f"{text}{suffix}\n&{namelist}\n  {key} = {value}\n/\n"
    body = match.group("body")
    key_pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*)(.*)$")
    if key_pattern.search(body):
        body = key_pattern.sub(rf"\g<1>{value}\3", body, count=1)
    else:
        body = f"\n  {key} = {value}" + body
    return text[: match.start("body")] + body + text[match.end("body") :]


def _ignore_stella_copy_entries(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        "build",
        "build_cmake",
        "stella",
        "*.o",
        "*.mod",
    }
    return {name for name in names if name in ignored or name.endswith((".o", ".mod"))}


def _metadata_payload(
    *,
    output_root: Path,
    patched_source_root: Path,
    patched_explicit: Path,
    run_dir: Path,
    input_file: Path,
    vmec_file: Path,
    build_script: Path,
    run_script: Path,
    trace_step: int,
    trace_iky: int,
    trace_ikx: int,
    tend: float,
    delt: float,
    force_explicit_stream_mirror: bool,
) -> dict[str, object]:
    return {
        "benchmark_name": "w7x_ky03_stella_rhs_trace_run",
        "status": "prepared_patched_stella_run_pending_execution",
        "output_root": _display_path(output_root),
        "patched_source_root": _display_path(patched_source_root),
        "patched_explicit_source": _display_path(patched_explicit),
        "run_dir": _display_path(run_dir),
        "input_file": _display_path(input_file),
        "vmec_file": _display_path(vmec_file),
        "build_script": _display_path(build_script),
        "run_script": _display_path(run_script),
        "trace_output": _display_path(run_dir / TRACE_FILENAME),
        "trace_step": int(trace_step),
        "trace_time": float(trace_step) * float(delt),
        "trace_iky_fortran": int(trace_iky),
        "trace_ikx_fortran": int(trace_ikx),
        "trace_ky_value": 0.3,
        "trace_kx_value": 0.0,
        "rhs_units": "stella_native_rhs_times_code_dt",
        "force_explicit_stream_mirror": bool(force_explicit_stream_mirror),
        "trace_input_note": (
            "mirror and parallel streaming are forced explicit for this RHS term "
            "audit"
            if force_explicit_stream_mirror
            else "mirror and parallel streaming retain the input implicit settings"
        ),
        "matched_tend": float(tend),
        "matched_delt": float(delt),
        "terms": (
            "pdf_g",
            "phi",
            "mirror_force",
            "magnetic_drift_y",
            "magnetic_drift_x",
            "equilibrium_drive_wstar",
            "parallel_streaming",
            "rhs_total",
        ),
        "next_action": (
            "build the patched stella copy, run the generated input, then ingest "
            "the trace against fixtures/w7x_ky03_rhs_model_balance/"
        ),
    }


def _build_script_text() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/stella"
BUILD_DIR="${SOURCE_DIR}/COMPILATION/build_cmake"
JOBS="${JOBS:-4}"
cmake "${SOURCE_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target stella -j "${JOBS}"
"""


def _run_script_text(input_name: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SOURCE_DIR="${{SCRIPT_DIR}}/stella"
RUN_DIR="${{SCRIPT_DIR}}/run"
STELLA_EXECUTABLE="${{STELLA_EXECUTABLE:-${{SOURCE_DIR}}/COMPILATION/build_cmake/COMPILATION/stella}}"
cd "${{RUN_DIR}}"
"${{STELLA_EXECUTABLE}}" {input_name}
"""


def _readme_text(metadata: dict[str, object]) -> str:
    return "\n".join(
        (
            "# W7-X ky=0.3 stella RHS Trace Run",
            "",
            "This directory is generated by",
            "`uv run python scripts/prepare_stella_w7x_rhs_trace_run.py --overwrite`.",
            "",
            "It contains a copied stella source tree patched only around",
            "`add_explicit_gyrokinetic_terms`, plus a matched W7-X input that",
            "stops exactly at the selected trace step.",
            "",
            "By default this diagnostic input forces stella's mirror and parallel",
            "streaming terms onto the explicit RHS path.  The production W7-X",
            "growth-rate run keeps those terms implicit, which is why this trace",
            "is a term-balance diagnostic rather than a production growth run.",
            "",
            "Build and run:",
            "",
            "```bash",
            f"bash {metadata['build_script']}",
            f"bash {metadata['run_script']}",
            "```",
            "",
            f"The trace file will be `{metadata['trace_output']}`.",
            "Rows are space-separated and store stella-native `rhs*dt` values.",
            "",
        )
    )


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella-source", type=Path, default=DEFAULT_STELLA_SOURCE)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--vmec-file", type=Path, default=DEFAULT_VMEC)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trace-step", type=int, default=2000)
    parser.add_argument("--trace-iky", type=int, default=4)
    parser.add_argument("--trace-ikx", type=int, default=1)
    parser.add_argument("--tend", type=float, default=200.0)
    parser.add_argument("--delt", type=float, default=0.1)
    parser.add_argument(
        "--preserve-implicit-stream-mirror",
        action="store_true",
        help=(
            "Keep the source input's implicit mirror/streaming switches.  This "
            "prepares a production-like trace, but mirror/streaming deltas will "
            "not be present in add_explicit_gyrokinetic_terms."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
