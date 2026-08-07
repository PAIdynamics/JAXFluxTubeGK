"""Prepare a scratch stella build tracing the signed field-particle increment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.prepare_stella_w7x_rhs_trace_run import (
    _build_script_text,
    _git_revision,
    _ignore_stella_copy_entries,
    _run_script_text,
)
from scripts.run_stella_collision_field_particle_discriminator import (
    stella_collision_input,
)


TARGET = Path("STELLA_CODE/dissipation/collisions_fokkerplanck.f90")
TRACE_FILENAME = "stellarator_gk_collision_field_particle_trace.dat"


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one collision trace marker, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_stella_collision_field_particle_trace(source_path: Path) -> bool:
    """Instrument the aggregate signed field-particle RHS at a stable boundary."""

    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    if "stellarator_gk collision field-particle trace patch" in text:
        return False

    declaration = "      complex, dimension(:, :, :), allocatable :: g_in\n"
    text = _replace_once(
        text,
        declaration,
        declaration
        + "      ! stellarator_gk collision field-particle trace patch\n"
        + "      complex, dimension(:, :, :), allocatable :: stellarator_gk_fieldpart_input\n",
    )
    snapshot = """      g = g_in

      ! RHS is g^{***} + Ze/T*<phi^{n+1}>*F0 + sum_jlm psi_jlm^{n+1}*delta_jl
"""
    text = _replace_once(
        text,
        snapshot,
        """      g = g_in
      if (fieldpart) then
         allocate (stellarator_gk_fieldpart_input(nvpa, nmu, &
              kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
         stellarator_gk_fieldpart_input = g
      end if

      ! RHS is g^{***} + Ze/T*<phi^{n+1}>*F0 + sum_jlm psi_jlm^{n+1}*delta_jl
""",
    )
    end_fieldpart = """      end if

      deallocate (flds)
"""
    text = _replace_once(
        text,
        end_fieldpart,
        """      end if
      if (fieldpart) then
         call stellarator_gk_trace_field_particle_increment( &
              stellarator_gk_fieldpart_input, g)
         deallocate (stellarator_gk_fieldpart_input)
      end if

      deallocate (flds)
""",
    )
    text = _replace_once(
        text,
        "end module collisions_fokkerplanck\n",
        FIELD_PARTICLE_TRACE_HELPER + "\nend module collisions_fokkerplanck\n",
    )
    source_path.write_text(text, encoding="utf-8")
    return True


FIELD_PARTICLE_TRACE_HELPER = f"""
   subroutine stellarator_gk_trace_field_particle_increment(before, after)
      use mp, only: proc0
      use grids_time, only: code_dt
      use grids_velocity, only: nvpa, nmu, vpa, mu
      use parallelisation_layouts, only: kxkyz_lo
      use parallelisation_layouts, only: iky_idx, ikx_idx, iz_idx, is_idx, it_idx
      implicit none
      complex, dimension(:, :, kxkyz_lo%llim_proc:), intent(in) :: before, after
      integer :: unit, ikxkyz, iv, imu, iky, ikx, iz, is, it
      complex :: increment
      if (.not. proc0) return
      open(newunit=unit, file='{TRACE_FILENAME}', status='replace', action='write')
      write(unit, '(a)') '# schema=stellarator_gk_stella_collision_fieldpart_trace_v1'
      write(unit, '(a)') '# iv imu iky ikx iz tube species vpa mu before_re before_im rhs_re rhs_im'
      do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
         iky = iky_idx(kxkyz_lo, ikxkyz)
         ikx = ikx_idx(kxkyz_lo, ikxkyz)
         iz = iz_idx(kxkyz_lo, ikxkyz)
         it = it_idx(kxkyz_lo, ikxkyz)
         is = is_idx(kxkyz_lo, ikxkyz)
         do iv = 1, nvpa
            do imu = 1, nmu
               increment = (after(iv, imu, ikxkyz) - before(iv, imu, ikxkyz)) / code_dt
               write(unit, *) iv, imu, iky, ikx, iz, it, is, vpa(iv), mu(imu), &
                    real(before(iv, imu, ikxkyz)), aimag(before(iv, imu, ikxkyz)), &
                    real(increment), aimag(increment)
            end do
         end do
      end do
      close(unit)
   end subroutine stellarator_gk_trace_field_particle_increment
"""


def prepare_trace_run(
    stella_source: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy, patch, and prepare a serial one-step stella trace build."""

    stella_source = Path(stella_source).resolve()
    output_root = Path(output_root).resolve()
    target = stella_source / TARGET
    if not target.is_file():
        raise FileNotFoundError(target)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    source_copy = output_root / "stella"
    run_dir = output_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copytree(stella_source, source_copy, ignore=_ignore_stella_copy_entries)
    patched_source = source_copy / TARGET
    patch_stella_collision_field_particle_trace(patched_source)

    input_path = run_dir / "collision_field_particle_trace.in"
    input_path.write_text(stella_collision_input(field_particle=True), encoding="utf-8")
    build_script = output_root / "build_stella_collision_trace.sh"
    run_script = output_root / "run_stella_collision_trace.sh"
    build_script.write_text(_build_script_text(), encoding="utf-8")
    run_script.write_text(_run_script_text(input_path.name), encoding="utf-8")
    build_script.chmod(0o755)
    run_script.chmod(0o755)

    metadata = output_root / "collision_field_particle_trace_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "stella_collision_signed_field_particle_trace",
                "status": "prepared_patched_stella_run_pending_execution",
                "stella_source": str(stella_source),
                "stella_source_revision": _git_revision(stella_source),
                "patched_source": str(patched_source),
                "build_script": str(build_script),
                "run_script": str(run_script),
                "trace_output": str(run_dir / TRACE_FILENAME),
                "trace_quantity": "aggregate signed field-particle RHS before final implicit inversion",
                "serial_execution_required": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stella-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(
        prepare_trace_run(
            args.stella_source,
            args.output_root,
            overwrite=args.overwrite,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
