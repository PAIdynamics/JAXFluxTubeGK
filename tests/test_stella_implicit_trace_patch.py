from pathlib import Path

from scripts.prepare_stella_w7x_implicit_trace_run import (
    TRACE_FILENAME,
    patch_stella_implicit_stage_trace,
    patch_stella_explicit_stage_trace,
    patch_stella_mirror_stage_trace,
)
from scripts.replay_w7x_stella_implicit_stage import _metrics


def test_implicit_trace_patch_instruments_all_response_stages(tmp_path: Path):
    source = tmp_path / "gk_implicit_terms.f90"
    source.write_text(
        "module gk_implicit_terms\n"
        "   private\n"
        "contains\n"
        "   subroutine advance_implicit_terms(g, phi, apar, bpar)\n"
        "      if (debug) write (*, *) 'implicit_solve::advance_implicit_terms'\n"
        "      g1 = g\n"
        "      g2 = g\n"
        "         ! We now have g_{inh}^{n+1, i+1} stored in g\n"
        "         ! Solve response_matrix*(phi^{n+1}-phi^{n*}) = "
        "phi_{inh}^{n+1}-phi^{n*}\n"
        "         call invert_parstream_response(phi, apar, bpar)\n"
        "         itt = itt + 1\n"
        "   end subroutine advance_implicit_terms\n"
        "end module gk_implicit_terms\n",
        encoding="utf-8",
    )

    assert patch_stella_implicit_stage_trace(source) is True
    assert patch_stella_implicit_stage_trace(source) is False
    patched = source.read_text(encoding="utf-8")
    for stage in (
        "input_pdf",
        "input_phi",
        "inhomogeneous_pdf",
        "inhomogeneous_phi",
        "response_phi",
        "final_pdf",
        "final_phi",
    ):
        assert stage in patched
    assert TRACE_FILENAME in patched
    assert "jax_fluxtube_gk_implicit_call /= 1" in patched


def test_implicit_stage_metrics_report_fixed_and_phase_aligned_errors():
    metrics = _metrics([1.0 + 1.0j, 2.0 - 1.0j], [2.0 + 2.0j, 4.0 - 2.0j])
    assert metrics["relative_l2_error"] == 1.0
    assert metrics["best_fit_relative_l2_error"] < 1.0e-14
    assert metrics["best_fit_scale_real"] == 0.5
    assert abs(metrics["best_fit_scale_imag"]) < 1.0e-14


def test_mirror_trace_patch_instruments_first_mirror_stage(tmp_path: Path):
    source = tmp_path / "gyrokinetic_equation_implicit.f90"
    source.write_text(
        "module gyrokinetic_equation_implicit\n"
        "   private\n"
        "contains\n"
        "         if (mirror_implicit .and. include_mirror) then\n"
        "            call advance_mirror_implicit(collisions_implicit, g, apar)\n"
        "            fields_updated = .false.\n"
        "         end if\n"
        "end module gyrokinetic_equation_implicit\n",
        encoding="utf-8",
    )

    assert patch_stella_mirror_stage_trace(source) is True
    assert patch_stella_mirror_stage_trace(source) is False
    patched = source.read_text(encoding="utf-8")
    assert "mirror_input_pdf" in patched
    assert "mirror_final_pdf" in patched
    assert TRACE_FILENAME in patched


def test_explicit_trace_patch_instruments_rk3_stages(tmp_path: Path):
    source = tmp_path / "gyrokinetic_equation_explicit.f90"
    source.write_text(
            "module gyrokinetic_equation_explicit\n"
            "   private\n"
            "contains\n"
            "   subroutine advance_explicit_rk3(g, restart_time_step, istep)\n"
        "      g0 = g\n"
        "            call add_explicit_gyrokinetic_terms(g0, g1, restart_time_step, istep)\n"
        "            g1 = g0 + g1\n"
        "            call add_explicit_gyrokinetic_terms(g1, g2, restart_time_step, istep)\n"
        "            g2 = g1 + g2\n"
        "            call add_explicit_gyrokinetic_terms(g2, g, restart_time_step, istep)\n"
            "      g = g0 / 3.+0.5 * g1 + (g2 + g) / 6.\n"
            "   end subroutine advance_explicit_rk3\n"
            "end module gyrokinetic_equation_explicit\n",
        encoding="utf-8",
    )

    assert patch_stella_explicit_stage_trace(source) is True
    assert patch_stella_explicit_stage_trace(source) is False
    patched = source.read_text(encoding="utf-8")
    for stage in ("input", "rhs1", "state1", "rhs2", "state2", "rhs3", "final"):
        assert f"explicit_{stage}_pdf" in patched
