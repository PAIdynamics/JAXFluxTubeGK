from scripts.prepare_stella_collision_field_particle_trace_run import (
    COMPONENT_TRACE_FILENAME,
    FACTOR_TRACE_FILENAME,
    PRIMITIVE_TRACE_FILENAME,
    TRACE_FILENAME,
    patch_stella_collision_field_particle_trace,
    prepare_trace_run,
)


def _minimal_collision_source() -> str:
    return """module collisions_fokkerplanck
contains
   subroutine advance_implicit_fp(phi, apar, bpar, g)

      use mp, only: sum_allreduce
      use calculations_finite_differences, only: tridag
      use grids_velocity, only: nmu, nvpa
      use grids_velocity, only: vpa
      use grids_velocity, only: set_vpa_weights
      complex, dimension(:, :, :), allocatable :: g_in
      complex, dimension(:, :), allocatable :: g0spitzer
      integer :: ikxkyz, iky, ikx, iz, is, iv, it, ia
      g = g_in

      ! RHS is g^{***} + Ze/T*<phi^{n+1}>*F0 + sum_jlm psi_jlm^{n+1}*delta_jl
      ! add field particle contribution to RHS:
      if (fieldpart) then
         do ikxkyz = kxkyz_lo%llim_proc, kxkyz_lo%ulim_proc
            do idx1 = 1, (jmax + 1) * (lmax + 1)**2
               jj1 = ij - 1

               if (density_conservation_tp .and. (jj1 == 0) .and. (ll1 == 0)) then
                  g = g + 1
               end if

            end do
         end do

      end if

      deallocate (flds)
   end subroutine advance_implicit_fp
end module collisions_fokkerplanck
"""


def test_trace_patch_captures_signed_increment_and_is_idempotent(tmp_path):
    source = tmp_path / "collisions_fokkerplanck.f90"
    source.write_text(_minimal_collision_source())

    assert patch_stella_collision_field_particle_trace(source)
    assert not patch_stella_collision_field_particle_trace(source)
    patched = source.read_text()

    assert "stellarator_gk_fieldpart_input = g" in patched
    assert "(after(iv, imu, ikxkyz) - before(iv, imu, ikxkyz)) / code_dt" in patched
    assert TRACE_FILENAME in patched
    assert COMPONENT_TRACE_FILENAME in patched
    assert FACTOR_TRACE_FILENAME in patched
    assert PRIMITIVE_TRACE_FILENAME in patched
    assert "rhs_re rhs_im" in patched
    assert "ll1, mm1, jj1" in patched
    assert "stellarator_gk_factor_increment = stellarator_gk_psi" in patched
    assert "legendre_vpamu(ll1, mm1, iv, imu, iz)" in patched
    assert "stellarator_gk_response_sign" in patched


def test_prepare_trace_run_writes_only_to_scratch_copy(tmp_path):
    source_root = tmp_path / "source"
    target = source_root / "STELLA_CODE/dissipation/collisions_fokkerplanck.f90"
    target.parent.mkdir(parents=True)
    original = _minimal_collision_source()
    target.write_text(original)
    output = tmp_path / "prepared"

    metadata = prepare_trace_run(source_root, output)

    assert target.read_text() == original
    assert metadata.is_file()
    assert (output / "build_stella_collision_trace.sh").is_file()
    assert (
        f"-DFORTRAN_GIT_WORKING_TREE={source_root}"
        in (output / "build_stella_collision_trace.sh").read_text()
    )
    assert (output / "run/collision_field_particle_trace.in").is_file()
    assert (
        "stellarator_gk collision field-particle trace patch"
        in (output / "stella" / target.relative_to(source_root)).read_text()
    )
