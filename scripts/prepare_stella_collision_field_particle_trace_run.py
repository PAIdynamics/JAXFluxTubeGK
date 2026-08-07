"""Prepare a scratch stella build tracing the signed field-particle increment."""

from __future__ import annotations

import argparse
import json
import shlex
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
COMPONENT_TRACE_FILENAME = "stellarator_gk_collision_field_particle_components.dat"
FACTOR_TRACE_FILENAME = "stellarator_gk_collision_field_particle_factors.dat"
PRIMITIVE_TRACE_FILENAME = "stellarator_gk_collision_field_particle_primitives.dat"


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
    text = _replace_once(
        text,
        "      complex, dimension(:, :), allocatable :: g0spitzer\n",
        "      complex, dimension(:, :), allocatable :: g0spitzer\n"
        "      complex, dimension(:, :), allocatable :: stellarator_gk_component_input\n"
        "      complex :: stellarator_gk_component_increment\n"
        "      complex :: stellarator_gk_factor_increment, stellarator_gk_psi\n"
        "      real :: stellarator_gk_response_basis, stellarator_gk_response_sign\n"
        "      integer :: stellarator_gk_component_unit, stellarator_gk_factor_unit\n"
        "      integer :: stellarator_gk_primitive_unit\n",
    )
    text = _replace_once(
        text,
        "      integer :: ikxkyz, iky, ikx, iz, is, iv, it, ia\n",
        "      integer :: ikxkyz, iky, ikx, iz, is, iv, imu, it, ia\n",
    )
    text = _replace_once(
        text,
        "   subroutine advance_implicit_fp(phi, apar, bpar, g)\n\n"
        "      use mp, only: sum_allreduce\n",
        "   subroutine advance_implicit_fp(phi, apar, bpar, g)\n\n"
        "      use mp, only: sum_allreduce\n"
        "      use mp, only: proc0\n",
    )
    text = _replace_once(
        text,
        "      use mp, only: proc0\n      use calculations_finite_differences, only: tridag\n",
        "      use mp, only: proc0\n"
        "      use calculations_finite_differences, only: tridag\n"
        "      use geometry, only: bmag\n",
    )
    text = _replace_once(
        text,
        "      use grids_velocity, only: nmu, nvpa\n"
        "      use grids_velocity, only: vpa\n"
        "      use grids_velocity, only: set_vpa_weights\n",
        "      use grids_velocity, only: nmu, nvpa\n"
        "      use grids_velocity, only: vpa, mu\n"
        "      use grids_velocity, only: set_vpa_weights\n",
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
    component_start = """      ! add field particle contribution to RHS:
      if (fieldpart) then
         do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
"""
    text = _replace_once(
        text,
        component_start,
        f"""      ! add field particle contribution to RHS:
      if (fieldpart) then
         allocate (stellarator_gk_component_input(nvpa, nmu))
         if (proc0) then
            open(newunit=stellarator_gk_component_unit, file='{COMPONENT_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(stellarator_gk_component_unit, '(a)') &
                 '# schema=stellarator_gk_stella_collision_fieldpart_components_v1'
            write(stellarator_gk_component_unit, '(a)') &
                 '# iv imu iky ikx iz tube species l m j vpa mu before_re before_im rhs_re rhs_im'
            open(newunit=stellarator_gk_factor_unit, file='{FACTOR_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(stellarator_gk_factor_unit, '(a)') &
                 '# schema=stellarator_gk_stella_collision_fieldpart_factors_v1'
            write(stellarator_gk_factor_unit, '(a)') &
                 '# iv imu iky ikx iz tube target background l m j vpa mu psi_re psi_im basis rhs_re rhs_im'
            open(newunit=stellarator_gk_primitive_unit, file='{PRIMITIVE_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(stellarator_gk_primitive_unit, '(a)') &
                 '# schema=stellarator_gk_stella_collision_fieldpart_primitives_v1'
            write(stellarator_gk_primitive_unit, '(a)') &
                 '# iv imu iky ikx iz tube target background l m j vpa mu bmag frequency clm legendre gyroaverage mass_factor delta_j sign basis'
         end if
         do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
""",
    )
    component_snapshot = """               jj1 = ij - 1

               if (density_conservation_tp .and. (jj1 == 0) .and. (ll1 == 0)) then
"""
    text = _replace_once(
        text,
        component_snapshot,
        """               jj1 = ij - 1
               stellarator_gk_component_input = g(:, :, ikxkyz)

               if (density_conservation_tp .and. (jj1 == 0) .and. (ll1 == 0)) then
""",
    )
    component_end = """               end if

            end do
         end do

      end if
"""
    text = _replace_once(
        text,
        component_end,
        """               end if
               if (proc0) then
                  do iv = 1, nvpa
                     do imu = 1, nmu
                        stellarator_gk_component_increment = &
                             (g(iv, imu, ikxkyz) - &
                              stellarator_gk_component_input(iv, imu)) / code_dt
                        write(stellarator_gk_component_unit, *) iv, imu, iky, ikx, &
                             iz, it, is, ll1, mm1, jj1, vpa(iv), mu(imu), &
                             real(stellarator_gk_component_input(iv, imu)), &
                             aimag(stellarator_gk_component_input(iv, imu)), &
                             real(stellarator_gk_component_increment), &
                             aimag(stellarator_gk_component_increment)
                     end do
                  end do
                  if (.not. density_conservation_tp) then
                     do isb = 1, nspec
                        stellarator_gk_psi = flds(iky, ikx, iz, it, &
                             2 + (is - 1) * ((jmax + 1) * (lmax + 1)**2 * nspec) &
                             + (idx1 - 1) * nspec + (isb - 1))
                        do iv = 1, nvpa
                           do imu = 1, nmu
                              stellarator_gk_response_basis = spec(is)%vnew(isb) &
                                   * clm * legendre_vpamu(ll1, mm1, iv, imu, iz) &
                                   * jm(imu, abs(mm1), iky, ikx, iz, is) &
                                   * (spec(is)%mass / spec(isb)%mass)**(-1.5) &
                                   * deltaj(ll1, jj1, is, isb, iv, imu, ia, iz)
                              stellarator_gk_response_sign = 1.0
                              if (mm1 < 0) stellarator_gk_response_sign = (-1)**mm1
                              stellarator_gk_response_basis = &
                                   stellarator_gk_response_sign * stellarator_gk_response_basis
                              stellarator_gk_factor_increment = stellarator_gk_psi &
                                   * stellarator_gk_response_basis
                              write(stellarator_gk_factor_unit, *) iv, imu, iky, &
                                   ikx, iz, it, is, isb, ll1, mm1, jj1, vpa(iv), &
                                   mu(imu), real(stellarator_gk_psi), &
                                   aimag(stellarator_gk_psi), &
                                   stellarator_gk_response_basis, &
                                   real(stellarator_gk_factor_increment), &
                                   aimag(stellarator_gk_factor_increment)
                              write(stellarator_gk_primitive_unit, *) iv, imu, &
                                   iky, ikx, iz, it, is, isb, ll1, mm1, jj1, &
                                   vpa(iv), mu(imu), bmag(ia, iz), &
                                   spec(is)%vnew(isb), clm, &
                                   legendre_vpamu(ll1, mm1, iv, imu, iz), &
                                   jm(imu, abs(mm1), iky, ikx, iz, is), &
                                   (spec(is)%mass / spec(isb)%mass)**(-1.5), &
                                   deltaj(ll1, jj1, is, isb, iv, imu, ia, iz), &
                                   stellarator_gk_response_sign, &
                                   stellarator_gk_response_basis
                           end do
                        end do
                     end do
                  end if
               end if

            end do
         end do
         if (proc0) close(stellarator_gk_component_unit)
         if (proc0) close(stellarator_gk_factor_unit)
         if (proc0) close(stellarator_gk_primitive_unit)
         deallocate (stellarator_gk_component_input)

      end if
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
    build_script_text = _build_script_text().replace(
        'cmake "${SOURCE_DIR}" -B "${BUILD_DIR}"',
        'cmake "${SOURCE_DIR}" -B "${BUILD_DIR}" '
        + shlex.quote(f"-DFORTRAN_GIT_WORKING_TREE={stella_source}"),
    )
    build_script.write_text(build_script_text, encoding="utf-8")
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
                "component_trace_output": str(run_dir / COMPONENT_TRACE_FILENAME),
                "factor_trace_output": str(run_dir / FACTOR_TRACE_FILENAME),
                "primitive_trace_output": str(run_dir / PRIMITIVE_TRACE_FILENAME),
                "trace_quantity": "aggregate signed field-particle RHS before final implicit inversion",
                "component_trace_quantity": "signed (j,l,m) field-particle RHS components",
                "factor_trace_quantity": "pair-resolved scalar responses and velocity bases",
                "primitive_trace_quantity": "independent factors of each response basis",
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
