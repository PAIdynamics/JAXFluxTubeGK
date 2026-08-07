from scripts.prepare_stella_collision_field_particle_trace_run import (
    TRACE_FILENAME,
    patch_stella_collision_field_particle_trace,
    prepare_trace_run,
)


def _minimal_collision_source() -> str:
    return """module collisions_fokkerplanck
contains
   subroutine advance_implicit_fp(phi, apar, bpar, g)
      complex, dimension(:, :, :), allocatable :: g_in
      g = g_in

      ! RHS is g^{***} + Ze/T*<phi^{n+1}>*F0 + sum_jlm psi_jlm^{n+1}*delta_jl
      if (fieldpart) then
         g = g + 1
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
    assert "rhs_re rhs_im" in patched


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
    assert (output / "run/collision_field_particle_trace.in").is_file()
    assert "stellarator_gk collision field-particle trace patch" in (
        output / "stella" / target.relative_to(source_root)
    ).read_text()
