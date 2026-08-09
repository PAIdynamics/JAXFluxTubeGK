"""Prepare a non-destructive, stage-resolved stella implicit-step trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.prepare_stella_w7x_rhs_trace_run import (
    _build_script_text,
    _git_revision,
    _ignore_stella_copy_entries,
    _run_script_text,
    write_rhs_trace_input,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "fixtures/stella_w7x_mode_structure_run/stella_w7x_adiabatic_electrons.in"
DEFAULT_OUTPUT_ROOT = Path("/private/tmp/jax_fluxtube_gk_stella_w7x_implicit_trace")
TARGET = Path("STELLA_CODE/gyrokinetic_equation/gk_implicit_terms.f90")
MIRROR_TARGET = Path(
    "STELLA_CODE/gyrokinetic_equation/gyrokinetic_equation_implicit.f90"
)
EXPLICIT_TARGET = Path(
    "STELLA_CODE/gyrokinetic_equation/gyrokinetic_equation_explicit.f90"
)
TRACE_FILENAME = "jax_fluxtube_gk_w7x_implicit_stage_trace.dat"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella-source", type=Path, required=True)
    parser.add_argument("--vmec-file", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    metadata = prepare_stella_w7x_implicit_trace_run(
        stella_source=args.stella_source,
        vmec_file=args.vmec_file,
        input_file=args.input_file,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(metadata)
    return 0


def prepare_stella_w7x_implicit_trace_run(
    *,
    stella_source: Path,
    vmec_file: Path,
    input_file: Path = DEFAULT_INPUT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
) -> Path:
    """Copy and instrument stella for a one-step production-implicit trace."""

    stella_source = Path(stella_source).resolve()
    vmec_file = Path(vmec_file).resolve()
    input_file = Path(input_file).resolve()
    output_root = Path(output_root).resolve()
    for required in (
        stella_source / TARGET,
        stella_source / MIRROR_TARGET,
        stella_source / EXPLICIT_TARGET,
        vmec_file,
        input_file,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    source_copy = output_root / "stella"
    run_dir = output_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copytree(stella_source, source_copy, ignore=_ignore_stella_copy_entries)
    patched_source = source_copy / TARGET
    patch_stella_implicit_stage_trace(patched_source)
    patch_stella_mirror_stage_trace(source_copy / MIRROR_TARGET)
    patch_stella_explicit_stage_trace(source_copy / EXPLICIT_TARGET)

    prepared_input = run_dir / "stella_w7x_implicit_trace.in"
    write_rhs_trace_input(
        input_file,
        prepared_input,
        nstep=1,
        delt=0.1,
        vmec_filename="wout_w7x.nc",
        force_explicit_stream_mirror=False,
    )
    shutil.copy2(vmec_file, run_dir / "wout_w7x.nc")
    build_script = output_root / "build_stella_implicit_trace.sh"
    run_script = output_root / "run_stella_implicit_trace.sh"
    build_script.write_text(_build_script_text(), encoding="utf-8")
    run_script.write_text(_run_script_text(prepared_input.name), encoding="utf-8")
    build_script.chmod(0o755)
    run_script.chmod(0o755)

    metadata = output_root / "implicit_trace_run_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "benchmark_name": "w7x_stella_implicit_stage_trace",
                "stella_source": str(stella_source),
                "stella_source_revision": _git_revision(stella_source),
                "patched_source": str(patched_source),
                "input_file": str(prepared_input),
                "vmec_source": str(vmec_file),
                "trace_output": str(run_dir / TRACE_FILENAME),
                "trace_iky_fortran": 4,
                "trace_ikx_fortran": 1,
                "trace_implicit_call": 1,
                "stages": [
                    "input_pdf",
                    "mirror_input_pdf",
                    "mirror_final_pdf",
                    "explicit_input_pdf",
                    "explicit_rhs1_pdf",
                    "explicit_state1_pdf",
                    "explicit_rhs2_pdf",
                    "explicit_state2_pdf",
                    "explicit_rhs3_pdf",
                    "explicit_final_pdf",
                    "input_phi",
                    "inhomogeneous_pdf",
                    "inhomogeneous_phi",
                    "response_phi",
                    "final_pdf",
                    "final_phi",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def patch_stella_implicit_stage_trace(source_path: Path) -> bool:
    """Patch copied ``gk_implicit_terms.f90`` at stable stage boundaries."""

    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    if "jax_fluxtube_gk implicit stage trace patch" in text:
        return False
    module_marker = "   private\n"
    module_state = f"""
   ! jax_fluxtube_gk implicit stage trace patch
   integer, parameter :: jax_fluxtube_gk_trace_iky = 4
   integer, parameter :: jax_fluxtube_gk_trace_ikx = 1
   integer, parameter :: jax_fluxtube_gk_trace_unit = 9314
   character(len=*), parameter :: jax_fluxtube_gk_trace_filename = '{TRACE_FILENAME}'
   integer :: jax_fluxtube_gk_implicit_call = 0
   logical :: jax_fluxtube_gk_trace_initialised = .false.

"""
    text = _replace_once(text, module_marker, module_marker + module_state)

    entry_marker = "      if (debug) write (*, *) 'implicit_solve::advance_implicit_terms'\n"
    text = _replace_once(
        text,
        entry_marker,
        entry_marker + "      jax_fluxtube_gk_implicit_call = jax_fluxtube_gk_implicit_call + 1\n",
    )
    input_marker = "      g1 = g\n      g2 = g\n"
    text = _replace_once(
        text,
        input_marker,
        input_marker
        + "      call jax_fluxtube_gk_trace_pdf('input_pdf', g1)\n"
        + "      call jax_fluxtube_gk_trace_phi('input_phi', phi)\n",
    )
    inhom_pdf_marker = "         ! We now have g_{inh}^{n+1, i+1} stored in g\n"
    text = _replace_once(
        text,
        inhom_pdf_marker,
        "         call jax_fluxtube_gk_trace_pdf('inhomogeneous_pdf', g)\n"
        + inhom_pdf_marker,
    )
    inhom_phi_marker = (
        "         ! Solve response_matrix*(phi^{n+1}-phi^{n*}) = "
        "phi_{inh}^{n+1}-phi^{n*}\n"
    )
    text = _replace_once(
        text,
        inhom_phi_marker,
        "         call jax_fluxtube_gk_trace_phi('inhomogeneous_phi', phi)\n"
        + inhom_phi_marker,
    )
    response_marker = "         call invert_parstream_response(phi, apar, bpar)\n"
    text = _replace_once(
        text,
        response_marker,
        response_marker + "         call jax_fluxtube_gk_trace_phi('response_phi', phi)\n",
    )
    final_marker = "         itt = itt + 1\n"
    text = _replace_once(
        text,
        final_marker,
        "         call jax_fluxtube_gk_trace_pdf('final_pdf', g)\n"
        "         call jax_fluxtube_gk_trace_phi('final_phi', phi)\n"
        + final_marker,
    )
    text = _replace_once(
        text,
        "end module gk_implicit_terms\n",
        IMPLICIT_TRACE_HELPERS + "\nend module gk_implicit_terms\n",
    )
    source_path.write_text(text, encoding="utf-8")
    return True


def patch_stella_mirror_stage_trace(source_path: Path) -> bool:
    """Trace the PDF immediately around stella's first implicit mirror advance."""

    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    if "jax_fluxtube_gk mirror stage trace patch" in text:
        return False
    module_marker = "   private\n"
    module_state = f"""
   ! jax_fluxtube_gk mirror stage trace patch
   integer, parameter :: jax_fluxtube_gk_mirror_trace_iky = 4
   integer, parameter :: jax_fluxtube_gk_mirror_trace_ikx = 1
   integer, parameter :: jax_fluxtube_gk_mirror_trace_unit = 9315
   character(len=*), parameter :: jax_fluxtube_gk_mirror_trace_filename = '{TRACE_FILENAME}'
   logical :: jax_fluxtube_gk_mirror_trace_initialised = .false.

"""
    text = _replace_once(text, module_marker, module_marker + module_state)
    mirror_block = """         if (mirror_implicit .and. include_mirror) then
            call advance_mirror_implicit(collisions_implicit, g, apar)
            fields_updated = .false.
         end if
"""
    traced_block = """         if (mirror_implicit .and. include_mirror) then
            call jax_fluxtube_gk_trace_mirror_pdf('mirror_input_pdf', istep, g)
            call advance_mirror_implicit(collisions_implicit, g, apar)
            call jax_fluxtube_gk_trace_mirror_pdf('mirror_final_pdf', istep, g)
            fields_updated = .false.
         end if
"""
    if mirror_block not in text:
        raise ValueError("could not find stella's implicit mirror block")
    # The same block appears in both ADI orderings.  Patch only the first,
    # explicit-then-implicit branch used by the reference (flip_flop=false).
    text = text.replace(mirror_block, traced_block, 1)
    text = _replace_once(
        text,
        "end module gyrokinetic_equation_implicit\n",
        MIRROR_TRACE_HELPER + "\nend module gyrokinetic_equation_implicit\n",
    )
    source_path.write_text(text, encoding="utf-8")
    return True


def patch_stella_explicit_stage_trace(source_path: Path) -> bool:
    """Trace stella's first three-stage SSP RK3 explicit advance."""

    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    if "jax_fluxtube_gk explicit stage trace patch" in text:
        return False
    module_marker = "   private\n"
    module_state = f"""
   ! jax_fluxtube_gk explicit stage trace patch
   integer, parameter :: jax_fluxtube_gk_explicit_trace_iky = 4
   integer, parameter :: jax_fluxtube_gk_explicit_trace_ikx = 1
   integer, parameter :: jax_fluxtube_gk_explicit_trace_unit = 9316
   character(len=*), parameter :: jax_fluxtube_gk_explicit_trace_filename = '{TRACE_FILENAME}'
   logical :: jax_fluxtube_gk_explicit_trace_initialised = .false.

"""
    text = _replace_once(text, module_marker, module_marker + module_state)
    rk3_start = text.index("   subroutine advance_explicit_rk3")
    rk3_end_marker = "   end subroutine advance_explicit_rk3\n"
    rk3_end = text.index(rk3_end_marker, rk3_start) + len(rk3_end_marker)
    rk3 = text[rk3_start:rk3_end]
    rk3 = _replace_once(
        rk3,
        "      g0 = g\n",
        "      g0 = g\n"
        "      call jax_fluxtube_gk_trace_explicit_pdf('explicit_input_pdf', istep, g0)\n",
    )
    rk3 = _replace_once(
        rk3,
        "            call add_explicit_gyrokinetic_terms(g0, g1, restart_time_step, istep)\n",
        "            call add_explicit_gyrokinetic_terms(g0, g1, restart_time_step, istep)\n"
        "            call jax_fluxtube_gk_trace_explicit_pdf('explicit_rhs1_pdf', istep, g1)\n",
    )
    rk3 = _replace_once(
        rk3,
        "            g1 = g0 + g1\n",
        "            g1 = g0 + g1\n"
        "            call jax_fluxtube_gk_trace_explicit_pdf('explicit_state1_pdf', istep, g1)\n",
    )
    rk3 = _replace_once(
        rk3,
        "            call add_explicit_gyrokinetic_terms(g1, g2, restart_time_step, istep)\n",
        "            call add_explicit_gyrokinetic_terms(g1, g2, restart_time_step, istep)\n"
        "            call jax_fluxtube_gk_trace_explicit_pdf('explicit_rhs2_pdf', istep, g2)\n",
    )
    rk3 = _replace_once(
        rk3,
        "            g2 = g1 + g2\n",
        "            g2 = g1 + g2\n"
        "            call jax_fluxtube_gk_trace_explicit_pdf('explicit_state2_pdf', istep, g2)\n",
    )
    rk3 = _replace_once(
        rk3,
        "            call add_explicit_gyrokinetic_terms(g2, g, restart_time_step, istep)\n",
        "            call add_explicit_gyrokinetic_terms(g2, g, restart_time_step, istep)\n"
        "            call jax_fluxtube_gk_trace_explicit_pdf('explicit_rhs3_pdf', istep, g)\n",
    )
    rk3 = _replace_once(
        rk3,
        "      g = g0 / 3.+0.5 * g1 + (g2 + g) / 6.\n",
        "      g = g0 / 3.+0.5 * g1 + (g2 + g) / 6.\n"
        "      call jax_fluxtube_gk_trace_explicit_pdf('explicit_final_pdf', istep, g)\n",
    )
    text = text[:rk3_start] + rk3 + text[rk3_end:]
    text = _replace_once(
        text,
        "end module gyrokinetic_equation_explicit\n",
        EXPLICIT_TRACE_HELPER + "\nend module gyrokinetic_equation_explicit\n",
    )
    source_path.write_text(text, encoding="utf-8")
    return True


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one patch marker, found {text.count(old)}: {old!r}")
    return text.replace(old, new, 1)


IMPLICIT_TRACE_HELPERS = r"""
   subroutine jax_fluxtube_gk_trace_pdf(stage, values)
      use mp, only: proc0
      use parallelisation_layouts, only: vmu_lo, iv_idx, imu_idx, is_idx
      use grids_z, only: nzgrid
      use grids_kxky, only: naky, nakx
      use grids_velocity, only: vpa, mu
      implicit none
      character(len=*), intent(in) :: stage
      complex, dimension(:, :, -nzgrid:, :, vmu_lo%llim_proc:), intent(in) :: values
      integer :: iz, ivmu, iv, imu, is
      if (.not. proc0 .or. jax_fluxtube_gk_implicit_call /= 1) return
      if (jax_fluxtube_gk_trace_iky > naky .or. jax_fluxtube_gk_trace_ikx > nakx) return
      call jax_fluxtube_gk_open_implicit_trace()
      do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
         iv = iv_idx(vmu_lo, ivmu); imu = imu_idx(vmu_lo, ivmu); is = is_idx(vmu_lo, ivmu)
         do iz = -nzgrid, nzgrid
            write (jax_fluxtube_gk_trace_unit, '(A,1X,A,8(1X,I0),4(1X,ES24.16E3))') &
                 'pdf', trim(stage), jax_fluxtube_gk_implicit_call, jax_fluxtube_gk_trace_iky, &
                 jax_fluxtube_gk_trace_ikx, iz, ivmu, iv, imu, is, vpa(iv), mu(imu), &
                 real(values(jax_fluxtube_gk_trace_iky, jax_fluxtube_gk_trace_ikx, iz, 1, ivmu)), &
                 aimag(values(jax_fluxtube_gk_trace_iky, jax_fluxtube_gk_trace_ikx, iz, 1, ivmu))
         end do
      end do
      close (jax_fluxtube_gk_trace_unit)
   end subroutine jax_fluxtube_gk_trace_pdf

   subroutine jax_fluxtube_gk_trace_phi(stage, values)
      use mp, only: proc0
      use grids_z, only: nzgrid
      use grids_kxky, only: naky, nakx
      implicit none
      character(len=*), intent(in) :: stage
      complex, dimension(:, :, -nzgrid:, :), intent(in) :: values
      integer :: iz
      if (.not. proc0 .or. jax_fluxtube_gk_implicit_call /= 1) return
      if (jax_fluxtube_gk_trace_iky > naky .or. jax_fluxtube_gk_trace_ikx > nakx) return
      call jax_fluxtube_gk_open_implicit_trace()
      do iz = -nzgrid, nzgrid
         write (jax_fluxtube_gk_trace_unit, '(A,1X,A,8(1X,I0),4(1X,ES24.16E3))') &
              'phi', trim(stage), jax_fluxtube_gk_implicit_call, jax_fluxtube_gk_trace_iky, &
              jax_fluxtube_gk_trace_ikx, iz, 0, 0, 0, 0, 0.0, 0.0, &
              real(values(jax_fluxtube_gk_trace_iky, jax_fluxtube_gk_trace_ikx, iz, 1)), &
              aimag(values(jax_fluxtube_gk_trace_iky, jax_fluxtube_gk_trace_ikx, iz, 1))
      end do
      close (jax_fluxtube_gk_trace_unit)
   end subroutine jax_fluxtube_gk_trace_phi

   subroutine jax_fluxtube_gk_open_implicit_trace()
      implicit none
      logical :: trace_exists
      if (.not. jax_fluxtube_gk_trace_initialised) then
         inquire (file=jax_fluxtube_gk_trace_filename, exist=trace_exists)
         if (trace_exists) then
            open (unit=jax_fluxtube_gk_trace_unit, file=jax_fluxtube_gk_trace_filename, &
                 status='old', position='append', action='write')
         else
            open (unit=jax_fluxtube_gk_trace_unit, file=jax_fluxtube_gk_trace_filename, &
                 status='replace', action='write')
            write (jax_fluxtube_gk_trace_unit, '(A)') &
                 'record stage implicit_call iky ikx iz ivmu iv imu is vpa mu real imag'
         end if
         jax_fluxtube_gk_trace_initialised = .true.
      else
         open (unit=jax_fluxtube_gk_trace_unit, file=jax_fluxtube_gk_trace_filename, &
              status='old', position='append', action='write')
      end if
   end subroutine jax_fluxtube_gk_open_implicit_trace
"""


MIRROR_TRACE_HELPER = r"""
   subroutine jax_fluxtube_gk_trace_mirror_pdf(stage, istep, values)
      use mp, only: proc0
      use parallelisation_layouts, only: vmu_lo, iv_idx, imu_idx, is_idx
      use grids_z, only: nzgrid
      use grids_kxky, only: naky, nakx
      use grids_velocity, only: vpa, mu
      implicit none
      character(len=*), intent(in) :: stage
      integer, intent(in) :: istep
      complex, dimension(:, :, -nzgrid:, :, vmu_lo%llim_proc:), intent(in) :: values
      integer :: iz, ivmu, iv, imu, is
      logical :: trace_exists
      if (.not. proc0 .or. istep /= 1) return
      if (jax_fluxtube_gk_mirror_trace_iky > naky .or. jax_fluxtube_gk_mirror_trace_ikx > nakx) return
      if (.not. jax_fluxtube_gk_mirror_trace_initialised) then
         inquire (file=jax_fluxtube_gk_mirror_trace_filename, exist=trace_exists)
         if (trace_exists) then
            open (unit=jax_fluxtube_gk_mirror_trace_unit, file=jax_fluxtube_gk_mirror_trace_filename, &
                 status='old', position='append', action='write')
         else
            open (unit=jax_fluxtube_gk_mirror_trace_unit, file=jax_fluxtube_gk_mirror_trace_filename, &
                 status='replace', action='write')
            write (jax_fluxtube_gk_mirror_trace_unit, '(A)') &
                 'record stage implicit_call iky ikx iz ivmu iv imu is vpa mu real imag'
         end if
         jax_fluxtube_gk_mirror_trace_initialised = .true.
      else
         open (unit=jax_fluxtube_gk_mirror_trace_unit, file=jax_fluxtube_gk_mirror_trace_filename, &
              status='old', position='append', action='write')
      end if
      do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
         iv = iv_idx(vmu_lo, ivmu); imu = imu_idx(vmu_lo, ivmu); is = is_idx(vmu_lo, ivmu)
         do iz = -nzgrid, nzgrid
            write (jax_fluxtube_gk_mirror_trace_unit, '(A,1X,A,8(1X,I0),4(1X,ES24.16E3))') &
                 'pdf', trim(stage), 1, jax_fluxtube_gk_mirror_trace_iky, &
                 jax_fluxtube_gk_mirror_trace_ikx, iz, ivmu, iv, imu, is, vpa(iv), mu(imu), &
                 real(values(jax_fluxtube_gk_mirror_trace_iky, jax_fluxtube_gk_mirror_trace_ikx, iz, 1, ivmu)), &
                 aimag(values(jax_fluxtube_gk_mirror_trace_iky, jax_fluxtube_gk_mirror_trace_ikx, iz, 1, ivmu))
         end do
      end do
      close (jax_fluxtube_gk_mirror_trace_unit)
   end subroutine jax_fluxtube_gk_trace_mirror_pdf
"""


EXPLICIT_TRACE_HELPER = r"""
   subroutine jax_fluxtube_gk_trace_explicit_pdf(stage, istep, values)
      use mp, only: proc0
      use parallelisation_layouts, only: vmu_lo, iv_idx, imu_idx, is_idx
      use grids_z, only: nzgrid
      use grids_kxky, only: naky, nakx
      use grids_velocity, only: vpa, mu
      implicit none
      character(len=*), intent(in) :: stage
      integer, intent(in) :: istep
      complex, dimension(:, :, -nzgrid:, :, vmu_lo%llim_proc:), intent(in) :: values
      integer :: iz, ivmu, iv, imu, is
      if (.not. proc0 .or. istep /= 1) return
      if (jax_fluxtube_gk_explicit_trace_iky > naky .or. jax_fluxtube_gk_explicit_trace_ikx > nakx) return
      if (.not. jax_fluxtube_gk_explicit_trace_initialised) then
         open (unit=jax_fluxtube_gk_explicit_trace_unit, file=jax_fluxtube_gk_explicit_trace_filename, &
              status='replace', action='write')
         write (jax_fluxtube_gk_explicit_trace_unit, '(A)') &
              'record stage implicit_call iky ikx iz ivmu iv imu is vpa mu real imag'
         jax_fluxtube_gk_explicit_trace_initialised = .true.
      else
         open (unit=jax_fluxtube_gk_explicit_trace_unit, file=jax_fluxtube_gk_explicit_trace_filename, &
              status='old', position='append', action='write')
      end if
      do ivmu = vmu_lo%llim_proc, vmu_lo%ulim_proc
         iv = iv_idx(vmu_lo, ivmu); imu = imu_idx(vmu_lo, ivmu); is = is_idx(vmu_lo, ivmu)
         do iz = -nzgrid, nzgrid
            write (jax_fluxtube_gk_explicit_trace_unit, '(A,1X,A,8(1X,I0),4(1X,ES24.16E3))') &
                 'pdf', trim(stage), 1, jax_fluxtube_gk_explicit_trace_iky, &
                 jax_fluxtube_gk_explicit_trace_ikx, iz, ivmu, iv, imu, is, vpa(iv), mu(imu), &
                 real(values(jax_fluxtube_gk_explicit_trace_iky, jax_fluxtube_gk_explicit_trace_ikx, iz, 1, ivmu)), &
                 aimag(values(jax_fluxtube_gk_explicit_trace_iky, jax_fluxtube_gk_explicit_trace_ikx, iz, 1, ivmu))
         end do
      end do
      close (jax_fluxtube_gk_explicit_trace_unit)
   end subroutine jax_fluxtube_gk_trace_explicit_pdf
"""


if __name__ == "__main__":
    raise SystemExit(main())
