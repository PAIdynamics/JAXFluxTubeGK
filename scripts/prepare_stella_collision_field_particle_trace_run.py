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
TRACE_FILENAME = "jax_fluxtube_gk_collision_field_particle_trace.dat"
COMPONENT_TRACE_FILENAME = "jax_fluxtube_gk_collision_field_particle_components.dat"
FACTOR_TRACE_FILENAME = "jax_fluxtube_gk_collision_field_particle_factors.dat"
PRIMITIVE_TRACE_FILENAME = "jax_fluxtube_gk_collision_field_particle_primitives.dat"
QUADRATURE_TRACE_FILENAME = "jax_fluxtube_gk_collision_velocity_quadrature.dat"
DRIVER_TRACE_FILENAME = "jax_fluxtube_gk_collision_field_particle_drivers.dat"
TEST_PARTICLE_MATRIX_TRACE_FILENAME = (
    "jax_fluxtube_gk_collision_test_particle_matrix.dat"
)
FINAL_STATE_TRACE_FILENAME = "jax_fluxtube_gk_collision_final_state.dat"
TEST_PARTICLE_PRIMITIVE_TRACE_FILENAME = (
    "jax_fluxtube_gk_collision_test_particle_primitives.dat"
)
TEST_PARTICLE_BLOCK_TRACE_FILENAME = "jax_fluxtube_gk_collision_test_particle_blocks.dat"


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one collision trace marker, found {count}: {old!r}")
    return text.replace(old, new, 1)


def patch_stella_collision_field_particle_trace(source_path: Path) -> bool:
    """Instrument the aggregate signed field-particle RHS at a stable boundary."""

    source_path = Path(source_path)
    text = source_path.read_text(encoding="utf-8")
    if "jax_fluxtube_gk collision field-particle trace patch" in text:
        return False

    declaration = "      complex, dimension(:, :, :), allocatable :: g_in\n"
    text = _replace_once(
        text,
        declaration,
        declaration
        + "      ! jax_fluxtube_gk collision field-particle trace patch\n"
        + "      complex, dimension(:, :, :), allocatable :: jax_fluxtube_gk_fieldpart_input\n",
    )
    text = _replace_once(
        text,
        "      complex, dimension(:, :), allocatable :: g0spitzer\n",
        "      complex, dimension(:, :), allocatable :: g0spitzer\n"
        "      complex, dimension(:, :), allocatable :: jax_fluxtube_gk_component_input\n"
        "      complex :: jax_fluxtube_gk_component_increment\n"
        "      complex :: jax_fluxtube_gk_factor_increment, jax_fluxtube_gk_psi\n"
        "      real :: jax_fluxtube_gk_response_basis, jax_fluxtube_gk_response_sign\n"
        "      real :: jax_fluxtube_gk_driver_basis, jax_fluxtube_gk_driver_clm\n"
        "      integer :: jax_fluxtube_gk_component_unit, jax_fluxtube_gk_factor_unit\n"
        "      integer :: jax_fluxtube_gk_primitive_unit, jax_fluxtube_gk_quadrature_unit\n"
        "      integer :: jax_fluxtube_gk_driver_unit\n",
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
        "      use grids_velocity, only: vpa, mu, wgts_vpa, wgts_mu\n"
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
         allocate (jax_fluxtube_gk_fieldpart_input(nvpa, nmu, &
              kxkyz_lo%llim_proc:kxkyz_lo%ulim_alloc))
         jax_fluxtube_gk_fieldpart_input = g
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
         allocate (jax_fluxtube_gk_component_input(nvpa, nmu))
         if (proc0) then
            open(newunit=jax_fluxtube_gk_component_unit, file='{COMPONENT_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(jax_fluxtube_gk_component_unit, '(a)') &
                 '# schema=jax_fluxtube_gk_stella_collision_fieldpart_components_v1'
            write(jax_fluxtube_gk_component_unit, '(a)') &
                 '# iv imu iky ikx iz tube species l m j vpa mu before_re before_im rhs_re rhs_im'
            open(newunit=jax_fluxtube_gk_factor_unit, file='{FACTOR_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(jax_fluxtube_gk_factor_unit, '(a)') &
                 '# schema=jax_fluxtube_gk_stella_collision_fieldpart_factors_v1'
            write(jax_fluxtube_gk_factor_unit, '(a)') &
                 '# iv imu iky ikx iz tube target background l m j vpa mu psi_re psi_im basis rhs_re rhs_im'
            open(newunit=jax_fluxtube_gk_primitive_unit, file='{PRIMITIVE_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(jax_fluxtube_gk_primitive_unit, '(a)') &
                 '# schema=jax_fluxtube_gk_stella_collision_fieldpart_primitives_v1'
            write(jax_fluxtube_gk_primitive_unit, '(a)') &
                 '# iv imu iky ikx iz tube target background l m j vpa mu bmag frequency clm legendre gyroaverage mass_factor delta_j sign basis'
            open(newunit=jax_fluxtube_gk_driver_unit, file='{DRIVER_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(jax_fluxtube_gk_driver_unit, '(a)') &
                 '# schema=jax_fluxtube_gk_stella_collision_fieldpart_drivers_v1'
            write(jax_fluxtube_gk_driver_unit, '(a)') &
                 '# iv imu iky ikx iz tube target background l m j vpa mu measure clm legendre gyroaverage delta_j maxwellian psijnorm sign driver'
            open(newunit=jax_fluxtube_gk_quadrature_unit, file='{QUADRATURE_TRACE_FILENAME}', &
                 status='replace', action='write')
            write(jax_fluxtube_gk_quadrature_unit, '(a)') &
                 '# schema=jax_fluxtube_gk_stella_collision_velocity_quadrature_v1'
            write(jax_fluxtube_gk_quadrature_unit, '(a)') &
                 '# iv imu iz vpa mu bmag w_vpa w_mu'
            do iz = -nzgrid, nzgrid
               do iv = 1, nvpa
                  do imu = 1, nmu
                     write(jax_fluxtube_gk_quadrature_unit, *) iv, imu, iz, &
                          vpa(iv), mu(imu), bmag(ia, iz), wgts_vpa(iv), &
                          wgts_mu(ia, iz, imu)
                  end do
               end do
            end do
            close(jax_fluxtube_gk_quadrature_unit)
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
               jax_fluxtube_gk_component_input = g(:, :, ikxkyz)

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
                        jax_fluxtube_gk_component_increment = &
                             (g(iv, imu, ikxkyz) - &
                              jax_fluxtube_gk_component_input(iv, imu)) / code_dt
                        write(jax_fluxtube_gk_component_unit, *) iv, imu, iky, ikx, &
                             iz, it, is, ll1, mm1, jj1, vpa(iv), mu(imu), &
                             real(jax_fluxtube_gk_component_input(iv, imu)), &
                             aimag(jax_fluxtube_gk_component_input(iv, imu)), &
                             real(jax_fluxtube_gk_component_increment), &
                             aimag(jax_fluxtube_gk_component_increment)
                     end do
                  end do
                  if (.not. density_conservation_tp) then
                     do isb = 1, nspec
                        jax_fluxtube_gk_psi = flds(iky, ikx, iz, it, &
                             2 + (is - 1) * ((jmax + 1) * (lmax + 1)**2 * nspec) &
                             + (idx1 - 1) * nspec + (isb - 1))
                        do iv = 1, nvpa
                           do imu = 1, nmu
                              jax_fluxtube_gk_response_basis = spec(is)%vnew(isb) &
                                   * clm * legendre_vpamu(ll1, mm1, iv, imu, iz) &
                                   * jm(imu, abs(mm1), iky, ikx, iz, is) &
                                   * (spec(is)%mass / spec(isb)%mass)**(-1.5) &
                                   * deltaj(ll1, jj1, is, isb, iv, imu, ia, iz)
                              jax_fluxtube_gk_response_sign = 1.0
                              if (mm1 < 0) jax_fluxtube_gk_response_sign = (-1)**mm1
                              jax_fluxtube_gk_response_basis = &
                                   jax_fluxtube_gk_response_sign * jax_fluxtube_gk_response_basis
                              jax_fluxtube_gk_factor_increment = jax_fluxtube_gk_psi &
                                   * jax_fluxtube_gk_response_basis
                              write(jax_fluxtube_gk_factor_unit, *) iv, imu, iky, &
                                   ikx, iz, it, is, isb, ll1, mm1, jj1, vpa(iv), &
                                   mu(imu), real(jax_fluxtube_gk_psi), &
                                   aimag(jax_fluxtube_gk_psi), &
                                   jax_fluxtube_gk_response_basis, &
                                   real(jax_fluxtube_gk_factor_increment), &
                                   aimag(jax_fluxtube_gk_factor_increment)
                              write(jax_fluxtube_gk_primitive_unit, *) iv, imu, &
                                   iky, ikx, iz, it, is, isb, ll1, mm1, jj1, &
                                   vpa(iv), mu(imu), bmag(ia, iz), &
                                   spec(is)%vnew(isb), clm, &
                                   legendre_vpamu(ll1, mm1, iv, imu, iz), &
                                   jm(imu, abs(mm1), iky, ikx, iz, is), &
                                   (spec(is)%mass / spec(isb)%mass)**(-1.5), &
                                   deltaj(ll1, jj1, is, isb, iv, imu, ia, iz), &
                                   jax_fluxtube_gk_response_sign, &
                                   jax_fluxtube_gk_response_basis
                              jax_fluxtube_gk_driver_clm = (-1)**mm1 * &
                                   sqrt(((2 * ll1 + 1) * &
                                   gamma(ll1 + mm1 + 1.)) / &
                                   (4 * pi * gamma(ll1 - mm1 + 1.)))
                              jax_fluxtube_gk_driver_basis = &
                                   jax_fluxtube_gk_response_sign * &
                                   wgts_vpa(iv) * wgts_mu(ia, iz, imu) * &
                                   jax_fluxtube_gk_driver_clm * &
                                   legendre_vpamu(ll1, -mm1, iv, imu, iz) * &
                                   jm(imu, abs(mm1), iky, ikx, iz, isb) * &
                                   deltaj(ll1, jj1, isb, is, iv, imu, ia, iz) / &
                                   mw(iv, imu, iz, isb) / &
                                   psijnorm(ll1, jj1, is, isb, iz)
                              write(jax_fluxtube_gk_driver_unit, *) iv, imu, &
                                   iky, ikx, iz, it, is, isb, ll1, mm1, jj1, &
                                   vpa(iv), mu(imu), &
                                   wgts_vpa(iv) * wgts_mu(ia, iz, imu), &
                                   jax_fluxtube_gk_driver_clm, &
                                   legendre_vpamu(ll1, -mm1, iv, imu, iz), &
                                   jm(imu, abs(mm1), iky, ikx, iz, isb), &
                                   deltaj(ll1, jj1, isb, is, iv, imu, ia, iz), &
                                   mw(iv, imu, iz, isb), &
                                   psijnorm(ll1, jj1, is, isb, iz), &
                                   jax_fluxtube_gk_response_sign, &
                                   jax_fluxtube_gk_driver_basis
                           end do
                        end do
                     end do
                  end if
               end if

            end do
         end do
         if (proc0) close(jax_fluxtube_gk_component_unit)
         if (proc0) close(jax_fluxtube_gk_factor_unit)
         if (proc0) close(jax_fluxtube_gk_primitive_unit)
         if (proc0) close(jax_fluxtube_gk_driver_unit)
         deallocate (jax_fluxtube_gk_component_input)

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
         call jax_fluxtube_gk_trace_field_particle_increment( &
              jax_fluxtube_gk_fieldpart_input, g)
         deallocate (jax_fluxtube_gk_fieldpart_input)
      end if

      deallocate (flds)
""",
    )
    text = _replace_once(
        text,
        "      !fields_updated = .false.\n\n"
        "      deallocate (g_in)\n",
        "      !fields_updated = .false.\n\n"
        "      call jax_fluxtube_gk_trace_collision_final_state(g_in, g)\n\n"
        "      deallocate (g_in)\n",
    )
    text = _replace_once(
        text,
        "end module collisions_fokkerplanck\n",
        FIELD_PARTICLE_TRACE_HELPER
        + "\n"
        + FINAL_STATE_TRACE_HELPER
        + "\nend module collisions_fokkerplanck\n",
    )
    text = _replace_once(
        text,
        "   subroutine init_fp_diffmatrix\n\n"
        "      use grids_time, only: code_dt\n",
        "   subroutine init_fp_diffmatrix\n\n"
        "      use mp, only: proc0\n"
        "      use grids_time, only: code_dt\n",
    )
    text = _replace_once(
        text,
        "      integer :: nc, nb, lldab, bm_colind, bm_rowind\n",
        "      integer :: nc, nb, lldab, bm_colind, bm_rowind\n"
        "      integer :: jax_fluxtube_gk_matrix_unit\n"
        "      integer :: jax_fluxtube_gk_tp_primitive_unit\n"
        "      integer :: jax_fluxtube_gk_tp_block_unit\n"
        "      integer :: jax_fluxtube_gk_matrix_row, jax_fluxtube_gk_matrix_col\n"
        "      integer :: jax_fluxtube_gk_matrix_band_row\n",
    )
    matrix_factorization = (
        "      ! AVB: LU factorise cdiffmat, using LAPACK's zgbtrf routine for banded matrices\n"
        "      nc = nvpa * nmu\n"
        "      nb = nmu + 1\n"
        "      lldab = 3 * (nmu + 1) + 1\n"
    )
    block_assembly = "      ! construct full block matrix\n"
    text = _replace_once(
        text,
        block_assembly,
        f"""      if (proc0) then
         open(newunit=jax_fluxtube_gk_tp_block_unit, &
              file='{TEST_PARTICLE_BLOCK_TRACE_FILENAME}', &
              status='replace', action='write')
         write(jax_fluxtube_gk_tp_block_unit, '(a)') &
              '# schema=jax_fluxtube_gk_stella_collision_test_particle_blocks_v1'
         write(jax_fluxtube_gk_tp_block_unit, '(a)') &
              '# iz target background iv row_mu col_mu lower_re lower_im ' // &
              'diagonal_re diagonal_im upper_re upper_im code_dt'
         do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
            iky = iky_idx(kxkyz_lo, ikxkyz)
            ikx = ikx_idx(kxkyz_lo, ikxkyz)
            iz = iz_idx(kxkyz_lo, ikxkyz)
            is = is_idx(kxkyz_lo, ikxkyz)
            if (iky /= 1 .or. ikx /= 1) cycle
            do isb = 1, nspec
               do iv = 1, nvpa
                  do imu = 1, nmu
                     do imu2 = 1, nmu
                        write(jax_fluxtube_gk_tp_block_unit, *) &
                             iz, is, isb, iv, imu, imu2, &
                             real(aa_blcs(iv, imu, imu2, ikxkyz, isb)), &
                             aimag(aa_blcs(iv, imu, imu2, ikxkyz, isb)), &
                             real(bb_blcs(iv, imu, imu2, ikxkyz, isb)), &
                             aimag(bb_blcs(iv, imu, imu2, ikxkyz, isb)), &
                             real(cc_blcs(iv, imu, imu2, ikxkyz, isb)), &
                             aimag(cc_blcs(iv, imu, imu2, ikxkyz, isb)), code_dt
                     end do
                  end do
               end do
            end do
         end do
         close(jax_fluxtube_gk_tp_block_unit)
      end if

{block_assembly}""",
    )
    text = _replace_once(
        text,
        matrix_factorization,
        f"""      ! jax_fluxtube_gk collision test-particle matrix trace patch
      nc = nvpa * nmu
      nb = nmu + 1
      if (proc0) then
         open(newunit=jax_fluxtube_gk_matrix_unit, &
              file='{TEST_PARTICLE_MATRIX_TRACE_FILENAME}', &
              status='replace', action='write')
         write(jax_fluxtube_gk_matrix_unit, '(a)') &
              '# schema=jax_fluxtube_gk_stella_collision_test_particle_matrix_v1'
         write(jax_fluxtube_gk_matrix_unit, '(a)') &
              '# iky ikx iz species row col matrix_re matrix_im kperp2 code_dt'
         do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
            iky = iky_idx(kxkyz_lo, ikxkyz)
            ikx = ikx_idx(kxkyz_lo, ikxkyz)
            iz = iz_idx(kxkyz_lo, ikxkyz)
            is = is_idx(kxkyz_lo, ikxkyz)
            do jax_fluxtube_gk_matrix_col = 1, nc
               do jax_fluxtube_gk_matrix_row = &
                    max(1, jax_fluxtube_gk_matrix_col - nb), &
                    min(nc, jax_fluxtube_gk_matrix_col + nb)
                  jax_fluxtube_gk_matrix_band_row = 2 * nb + 1 &
                       + jax_fluxtube_gk_matrix_row - jax_fluxtube_gk_matrix_col
                  write(jax_fluxtube_gk_matrix_unit, *) iky, ikx, iz, is, &
                       jax_fluxtube_gk_matrix_row, jax_fluxtube_gk_matrix_col, &
                       real(cdiffmat_band(jax_fluxtube_gk_matrix_band_row, &
                            jax_fluxtube_gk_matrix_col, iky, ikx, iz, is)), &
                       aimag(cdiffmat_band(jax_fluxtube_gk_matrix_band_row, &
                            jax_fluxtube_gk_matrix_col, iky, ikx, iz, is)), &
                       kperp2(iky, ikx, ia, iz), code_dt
               end do
            end do
         end do
         close(jax_fluxtube_gk_matrix_unit)
         open(newunit=jax_fluxtube_gk_tp_primitive_unit, &
              file='{TEST_PARTICLE_PRIMITIVE_TRACE_FILENAME}', &
              status='replace', action='write')
         write(jax_fluxtube_gk_tp_primitive_unit, '(a)') &
              '# schema=jax_fluxtube_gk_stella_collision_test_particle_primitives_v1'
         write(jax_fluxtube_gk_tp_primitive_unit, '(a)') &
              '# iv imu iz target background vpa mu bmag target_mass ' // &
              'background_mass frequency speed maxwell nupa nuD nux ' // &
              'target_smz dvpa dmu code_dt deflknob eiediffknob ' // &
              'eideflknob nuxfac cfac'
         do is = 1, nspec
            do isb = 1, nspec
               do iz = -nzgrid, nzgrid
                  do iv = 1, nvpa
                     do imu = 1, nmu
                        write(jax_fluxtube_gk_tp_primitive_unit, *) &
                             iv, imu, iz, is, isb, vpa(iv), mu(imu), &
                             bmag(ia, iz), spec(is)%mass, spec(isb)%mass, &
                             spec(is)%vnew(isb), velvpamu(iv, imu, iz), &
                             mw(iv, imu, iz, is), nupa(iv, imu, iz, is, isb), &
                             nuD(iv, imu, iz, is, isb), &
                             nux(iv, imu, iz, is, isb), spec(is)%smz, dvpa, &
                             dmu(min(imu, nmu - 1)), code_dt, deflknob, &
                             eiediffknob, eideflknob, nuxfac, cfac
                     end do
                  end do
               end do
            end do
         end do
         close(jax_fluxtube_gk_tp_primitive_unit)
      end if

{matrix_factorization}""",
    )
    source_path.write_text(text, encoding="utf-8")
    return True


FIELD_PARTICLE_TRACE_HELPER = f"""
   subroutine jax_fluxtube_gk_trace_field_particle_increment(before, after)
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
      write(unit, '(a)') '# schema=jax_fluxtube_gk_stella_collision_fieldpart_trace_v1'
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
   end subroutine jax_fluxtube_gk_trace_field_particle_increment
"""


FINAL_STATE_TRACE_HELPER = f"""
   subroutine jax_fluxtube_gk_trace_collision_final_state(before, after)
      use mp, only: proc0
      use grids_velocity, only: nvpa, nmu
      use parallelisation_layouts, only: kxkyz_lo
      use parallelisation_layouts, only: iky_idx, ikx_idx, iz_idx, is_idx, it_idx
      implicit none
      complex, dimension(:, :, kxkyz_lo%llim_proc:), intent(in) :: before, after
      integer :: unit, ikxkyz, iv, imu, iky, ikx, iz, is, it
      if (.not. proc0) return
      open(newunit=unit, file='{FINAL_STATE_TRACE_FILENAME}', &
           status='replace', action='write')
      write(unit, '(a)') '# schema=jax_fluxtube_gk_stella_collision_final_state_v1'
      write(unit, '(a)') &
           '# iv imu iky ikx iz tube species input_re input_im output_re output_im'
      do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
         iky = iky_idx(kxkyz_lo, ikxkyz)
         ikx = ikx_idx(kxkyz_lo, ikxkyz)
         iz = iz_idx(kxkyz_lo, ikxkyz)
         it = it_idx(kxkyz_lo, ikxkyz)
         is = is_idx(kxkyz_lo, ikxkyz)
         do iv = 1, nvpa
            do imu = 1, nmu
               write(unit, *) iv, imu, iky, ikx, iz, it, is, &
                    real(before(iv, imu, ikxkyz)), aimag(before(iv, imu, ikxkyz)), &
                    real(after(iv, imu, ikxkyz)), aimag(after(iv, imu, ikxkyz))
            end do
         end do
      end do
      close(unit)
   end subroutine jax_fluxtube_gk_trace_collision_final_state
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
                "quadrature_trace_output": str(run_dir / QUADRATURE_TRACE_FILENAME),
                "driver_trace_output": str(run_dir / DRIVER_TRACE_FILENAME),
                "test_particle_matrix_trace_output": str(
                    run_dir / TEST_PARTICLE_MATRIX_TRACE_FILENAME
                ),
                "final_state_trace_output": str(run_dir / FINAL_STATE_TRACE_FILENAME),
                "test_particle_primitive_trace_output": str(
                    run_dir / TEST_PARTICLE_PRIMITIVE_TRACE_FILENAME
                ),
                "test_particle_block_trace_output": str(
                    run_dir / TEST_PARTICLE_BLOCK_TRACE_FILENAME
                ),
                "trace_quantity": "aggregate signed field-particle RHS before final implicit inversion",
                "component_trace_quantity": "signed (j,l,m) field-particle RHS components",
                "factor_trace_quantity": "pair-resolved scalar responses and velocity bases",
                "primitive_trace_quantity": "independent factors of each response basis",
                "quadrature_trace_quantity": "velocity nodes and integrate_vmu weights",
                "driver_trace_quantity": "normalized background-space driver coefficients",
                "test_particle_matrix_trace_quantity": (
                    "unfactorized banded I-dt*C_test matrix in dense row/column coordinates"
                ),
                "final_state_trace_quantity": (
                    "input and final phase-space state across the complete implicit collision solve"
                ),
                "test_particle_primitive_trace_quantity": (
                    "analytic collision frequencies, Maxwellian, and velocity-grid assembly inputs"
                ),
                "test_particle_block_trace_quantity": (
                    "pair-resolved lower, diagonal, and upper velocity blocks before band packing"
                ),
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
