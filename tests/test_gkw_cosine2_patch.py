from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_gkw_cosine2_run import (
    PATCHED_SELECTOR,
    add_cosin2_branch,
    add_multitime_velocity_slice_patch,
    add_rhs_trace_exp_integration_patch,
    add_rhs_trace_linear_terms_patch,
    add_rhs_trace_matdat_patch,
    add_selected_state_dump_patch,
    add_state_trace_patch,
    prepare_gkw_cosine2_run,
    write_cosin2_input,
)
from stellarator_gk import (
    CycloneSourceTermTrace,
    GkwStateTrace,
    GkwVelocitySpaceSliceSeries,
    SelectedModeStateTrace,
    compare_gkw_state_trace_to_source_term_trace,
    compare_selected_mode_state_traces,
    load_gkw_parallel_phi_trace,
    load_gkw_selected_mode_rhs_trace,
    load_gkw_selected_mode_state_trace,
    load_gkw_state_trace,
    load_gkw_time_dat_trace,
    load_gkw_velocity_space_slice_series,
)


ROOT = Path(__file__).resolve().parents[1]


_MINIMAL_INIT = """subroutine init_fdisi
    select case(finit)

    case('cosine')
       do ix = 1, nx
          fdisi(indx(imod,ix,i,j,k,is)) = amp_init *      &
               &    de(is) * cos(2*pi*sgr(ix,i))
       end do

    case('sine')
       do ix = 1, nx
          fdisi(indx(imod,ix,i,j,k,is)) = amp_init *      &
               &    de(is) * sin(2*pi*sgr(ix,i))
       end do

    end select
end subroutine init_fdisi
"""

_MINIMAL_DIAGNOSTIC = """module diagnostic
contains

subroutine write_output

  use control,      only : output3d, lphi_diagnostics

  ! 2D outputs: CHECK THESE WORK WITH PARALLEL_S
  call write_file_output

end subroutine write_output

subroutine velocity_space_output

  use io,           only : output_slice

""" + "  character (len=18) :: file_fmt \n" + """
  file_fmt='(257(1x,e13.5))'
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr1.dat',FMT=file_fmt)
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr2.dat',FMT=file_fmt)
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr3.dat',FMT=file_fmt)
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr4.dat',FMT=file_fmt)

end subroutine velocity_space_output

end module diagnostic
"""

_MINIMAL_MATDAT = """module matdat
implicit none
private
public :: compress_matrix, finish_matrix_section,print_matrix, iac, ii, iir
public :: iiy, iiz, jj, jjr, jjy, jjz, mat, matdat_allocate, matr, maty
public :: matz, n1, n2, n2r, n3, n3r, n4, n4r, nmat, nmaty, nmatz
public :: nmata, mata, jja
public :: put_element, put_elem_zonal, put_source, source
public :: put_element_correct_apar
integer :: n1=0, n2=0, n3=0, n4=0, nmat=0, nmatr=0
integer, allocatable :: ii(:), jj(:), iir(:), jjr(:)
complex, allocatable :: mat(:)
real, allocatable :: matr(:)
complex, allocatable :: source(:)
contains
subroutine matdat_allocate
integer :: ierr, i, nsolc, ntot
nsolc = 4
ntot = 16
allocate(source(nsolc),stat=ierr)
do i = 1, nsolc
  source(i) = (0.,0.)
end do
  allocate(mat(ntot),stat=ierr)
end subroutine matdat_allocate
subroutine put_element(iih,jjh,mat_elem,itime_est)
integer, intent(in) :: iih, jjh
integer, optional, intent(in) :: itime_est
complex, intent(in) :: mat_elem
nmat = nmat + 1
ii(nmat) = iih
jj(nmat) = jjh
  mat(nmat) = mat_elem
end subroutine put_element
subroutine put_source(iih,mat_elem)
integer, intent(in) :: iih
complex, intent(in) :: mat_elem
source(iih) = source(iih) + mat_elem
end subroutine put_source
subroutine finish_matrix_section(isel)
integer, intent(in) :: isel
select case(isel)
case(1)
  n1 = nmat
case(2)
  n2 = nmat
case(3)
  n3 = nmat
case(4)
  n4 = nmat
end select
end subroutine finish_matrix_section
subroutine compress_section(isel)
integer :: i, ireduced
    if ((ii(i).eq.ii(ireduced)).and.(jj(i).eq.jj(ireduced))) then
  mat(ireduced) = mat(ireduced) + mat(i)
else
      mat(ireduced) = mat(i)
endif
end subroutine compress_section
subroutine slow_heap_sort(n_elem,i_start)
integer, intent(in) :: n_elem, i_start
integer :: itmp, ind
complex :: ctmp
    ctmp = mat(ind) ; mat(ind) = mat(i_start) ; mat(i_start) = ctmp
end subroutine slow_heap_sort
subroutine siftd(b,e,iii)
integer, intent(in) :: b,e,iii
integer :: itmp, ind, ind2
complex :: ctmp
      ctmp = mat(ind2) ; mat(ind2) = mat(ind) ; mat(ind) = ctmp
end subroutine siftd
end module matdat
"""

_MINIMAL_LINEAR_TERMS = """module linear_terms
contains
subroutine calc_linear_terms
  use matdat,      only : finish_matrix_section
  if (ltrapping_arakawa) call igh(disp_par,disp_vp)
  if (wills .eq. 1) then
          call vpar_grad_df_4d_testnewbc(disp_par)
  endif
  if (wills .eq. 1) then
          call vpar_grad_df_4d_testnewbc(disp_par)
  endif
        call trapdf_2d(trapping,disp_vp)
        call trapdf_4d(trapping,disp_vp)
  if (lvdgradf) call vdgradf
  call finish_matrix_section(1)
  if (lve_grad_fm) call ve_grad_fm
  if (lvd_grad_phi_fm) call vd_grad_phi_fm
  if (willv .eq. 1) then
          call vpgrphi_3_newbc(landau,disp_fe)
  endif
  if (willv .eq. 1) then
          call vpgrphi_3_newbc(landau,disp_fe)
  endif
  call finish_matrix_section(2)
  if (lpoisson_int) call poisson_int
  call finish_matrix_section(3)
  call finish_matrix_section(4)
  if (neoclassics .and. lneoclassical) call neoclassical
end subroutine calc_linear_terms
end module linear_terms
"""

_MINIMAL_EXP_INTEGRATION = """module exp_integration
contains
subroutine explicit_integration
  ! done after calculate_fields to have the new potential
  call normalize(2,fdisi(1),nsolc)

  if (meth == 3) then
     call normalize(3,fdisk(1,1),nsolc)
  endif
end subroutine explicit_integration
end module exp_integration
"""


def test_prepare_gkw_cosine2_run_patches_only_the_copy(tmp_path: Path) -> None:
    source_root = tmp_path / "gkw"
    (source_root / "src").mkdir(parents=True)
    source_init = source_root / "src" / "init.f90"
    source_init.write_text(_MINIMAL_INIT, encoding="utf-8")
    source_diagnostic = source_root / "src" / "diagnostic.F90"
    source_diagnostic.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")
    input_path = tmp_path / "selected_ky_input.dat"
    input_path.write_text("&SPCGENERAL\n finit = 'cosine'\n /\n", encoding="utf-8")

    prepared = prepare_gkw_cosine2_run(
        source_root=source_root,
        output_root=tmp_path / "patched-gkw",
        input_path=input_path,
    )

    patched_init = prepared.patched_init.read_text(encoding="utf-8")
    assert prepared.selector == PATCHED_SELECTOR
    assert "case('cosin2')" in patched_init
    assert "1.E0 + cos(2*pi*sgr(ix,i))" in patched_init
    assert "case('cosin2')" not in source_init.read_text(encoding="utf-8")
    assert "distr1_" not in source_diagnostic.read_text(encoding="utf-8")
    assert "finit = 'cosin2'" in prepared.patched_input.read_text(encoding="utf-8")
    assert prepared.patched_diagnostic is None
    assert not prepared.multi_time_distr
    assert (prepared.output_root / "README_stellarator_gk_cosin2.md").is_file()


def test_prepare_gkw_cosine2_run_can_patch_multitime_distr(tmp_path: Path) -> None:
    source_root = tmp_path / "gkw"
    (source_root / "src").mkdir(parents=True)
    source_init = source_root / "src" / "init.f90"
    source_init.write_text(_MINIMAL_INIT, encoding="utf-8")
    source_diagnostic = source_root / "src" / "diagnostic.F90"
    source_diagnostic.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")
    input_path = tmp_path / "selected_ky_input.dat"
    input_path.write_text("&SPCGENERAL\n finit = 'cosine'\n /\n", encoding="utf-8")

    prepared = prepare_gkw_cosine2_run(
        source_root=source_root,
        output_root=tmp_path / "patched-gkw",
        input_path=input_path,
        multi_time_distr=True,
    )

    assert prepared.multi_time_distr
    assert prepared.patched_diagnostic is not None
    diagnostic_text = prepared.patched_diagnostic.read_text(encoding="utf-8")
    assert "stellarator_gk multi-time distr patch" in diagnostic_text
    assert "use control,      only : output3d, lphi_diagnostics, ntotstep" in diagnostic_text
    assert "call velocity_space_output(ntotstep)" in diagnostic_text
    assert "subroutine velocity_space_output(snapshot_index)" in diagnostic_text
    assert 'write(distr_file,\'("distr1_",I8.8,".dat")\') snapshot_index' in diagnostic_text
    assert "distr1.dat" in diagnostic_text
    assert "distr1_" not in source_diagnostic.read_text(encoding="utf-8")
    assert "velocity-space\nsnapshots" in (
        prepared.output_root / "README_stellarator_gk_cosin2.md"
    ).read_text(encoding="utf-8")


def test_prepare_gkw_cosine2_run_can_patch_state_trace(tmp_path: Path) -> None:
    source_root = tmp_path / "gkw"
    (source_root / "src").mkdir(parents=True)
    source_init = source_root / "src" / "init.f90"
    source_init.write_text(_MINIMAL_INIT, encoding="utf-8")
    source_diagnostic = source_root / "src" / "diagnostic.F90"
    source_diagnostic.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")
    input_path = tmp_path / "selected_ky_input.dat"
    input_path.write_text("&SPCGENERAL\n finit = 'cosine'\n /\n", encoding="utf-8")

    prepared = prepare_gkw_cosine2_run(
        source_root=source_root,
        output_root=tmp_path / "patched-gkw",
        input_path=input_path,
        state_trace=True,
    )

    assert prepared.state_trace
    assert prepared.patched_diagnostic is not None
    diagnostic_text = prepared.patched_diagnostic.read_text(encoding="utf-8")
    assert "stellarator_gk compact state trace patch" in diagnostic_text
    assert "call stellarator_gk_state_trace_output" in diagnostic_text
    assert "stellarator_gk_state_trace.dat" in diagnostic_text
    assert "stellarator_gk_state_trace.dat" not in source_diagnostic.read_text(encoding="utf-8")
    assert "compact state trace" in (
        prepared.output_root / "README_stellarator_gk_cosin2.md"
    ).read_text(encoding="utf-8")


def test_prepare_gkw_cosine2_run_can_patch_selected_state_dump(tmp_path: Path) -> None:
    source_root = tmp_path / "gkw"
    (source_root / "src").mkdir(parents=True)
    source_init = source_root / "src" / "init.f90"
    source_init.write_text(_MINIMAL_INIT, encoding="utf-8")
    source_diagnostic = source_root / "src" / "diagnostic.F90"
    source_diagnostic.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")
    input_path = tmp_path / "selected_ky_input.dat"
    input_path.write_text("&SPCGENERAL\n finit = 'cosine'\n /\n", encoding="utf-8")

    prepared = prepare_gkw_cosine2_run(
        source_root=source_root,
        output_root=tmp_path / "patched-gkw",
        input_path=input_path,
        selected_state_dump=True,
    )

    assert prepared.selected_state_dump
    assert prepared.patched_diagnostic is not None
    diagnostic_text = prepared.patched_diagnostic.read_text(encoding="utf-8")
    assert "stellarator_gk selected-state dump patch" in diagnostic_text
    assert "call stellarator_gk_selected_state_dump_output" in diagnostic_text
    assert "stellarator_gk_selected_state_" in diagnostic_text
    assert "stellarator_gk_selected_state_" not in source_diagnostic.read_text(
        encoding="utf-8"
    )
    assert "full selected\nsingle-mode state" in (
        prepared.output_root / "README_stellarator_gk_cosin2.md"
    ).read_text(encoding="utf-8")


def test_prepare_gkw_cosine2_run_can_patch_rhs_trace(tmp_path: Path) -> None:
    source_root = tmp_path / "gkw"
    (source_root / "src").mkdir(parents=True)
    source_init = source_root / "src" / "init.f90"
    source_init.write_text(_MINIMAL_INIT, encoding="utf-8")
    source_diagnostic = source_root / "src" / "diagnostic.F90"
    source_diagnostic.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")
    source_matdat = source_root / "src" / "matdat.F90"
    source_matdat.write_text(_MINIMAL_MATDAT, encoding="utf-8")
    source_linear_terms = source_root / "src" / "linear_terms.F90"
    source_linear_terms.write_text(_MINIMAL_LINEAR_TERMS, encoding="utf-8")
    source_exp_integration = source_root / "src" / "exp_integration.F90"
    source_exp_integration.write_text(_MINIMAL_EXP_INTEGRATION, encoding="utf-8")
    input_path = tmp_path / "selected_ky_input.dat"
    input_path.write_text("&SPCGENERAL\n finit = 'cosine'\n /\n", encoding="utf-8")

    prepared = prepare_gkw_cosine2_run(
        source_root=source_root,
        output_root=tmp_path / "patched-gkw",
        input_path=input_path,
        rhs_trace=True,
        rhs_trace_steps=(20, 40),
    )

    assert prepared.rhs_trace
    assert prepared.rhs_trace_steps == (20, 40)
    assert prepared.patched_diagnostic is None
    exp_integration_text = (
        prepared.output_root / "src" / "exp_integration.F90"
    ).read_text(encoding="utf-8")
    matdat_text = (prepared.output_root / "src" / "matdat.F90").read_text(
        encoding="utf-8"
    )
    linear_terms_text = (
        prepared.output_root / "src" / "linear_terms.F90"
    ).read_text(encoding="utf-8")
    assert "stellarator_gk rhs trace exp_integration patch" in exp_integration_text
    assert "call stellarator_gk_rhs_trace_output" in exp_integration_text
    assert "ntotstep .eq. 40" in exp_integration_text
    assert "stellarator_gk rhs trace matdat patch" in matdat_text
    assert "stellarator_gk_mat_term" in matdat_text
    assert "stellarator_gk_set_trace_term(7)" in linear_terms_text
    assert "stellarator_gk_rhs_trace_" not in source_exp_integration.read_text(
        encoding="utf-8"
    )
    assert "RHS/source actions" in (
        prepared.output_root / "README_stellarator_gk_cosin2.md"
    ).read_text(encoding="utf-8")


def test_cosin2_patch_is_idempotent(tmp_path: Path) -> None:
    init_path = tmp_path / "init.f90"
    init_path.write_text(_MINIMAL_INIT, encoding="utf-8")

    assert add_cosin2_branch(init_path)
    first_patch = init_path.read_text(encoding="utf-8")
    assert not add_cosin2_branch(init_path)
    assert init_path.read_text(encoding="utf-8") == first_patch


def test_multitime_velocity_slice_patch_is_idempotent(tmp_path: Path) -> None:
    diagnostic_path = tmp_path / "diagnostic.F90"
    diagnostic_path.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")

    assert add_multitime_velocity_slice_patch(diagnostic_path)
    first_patch = diagnostic_path.read_text(encoding="utf-8")
    assert not add_multitime_velocity_slice_patch(diagnostic_path)
    assert diagnostic_path.read_text(encoding="utf-8") == first_patch


def test_state_trace_patch_is_idempotent(tmp_path: Path) -> None:
    diagnostic_path = tmp_path / "diagnostic.F90"
    diagnostic_path.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")

    assert add_state_trace_patch(diagnostic_path)
    first_patch = diagnostic_path.read_text(encoding="utf-8")
    assert not add_state_trace_patch(diagnostic_path)
    assert diagnostic_path.read_text(encoding="utf-8") == first_patch


def test_selected_state_dump_patch_is_idempotent(tmp_path: Path) -> None:
    diagnostic_path = tmp_path / "diagnostic.F90"
    diagnostic_path.write_text(_MINIMAL_DIAGNOSTIC, encoding="utf-8")

    assert add_selected_state_dump_patch(diagnostic_path)
    first_patch = diagnostic_path.read_text(encoding="utf-8")
    assert not add_selected_state_dump_patch(diagnostic_path)
    assert diagnostic_path.read_text(encoding="utf-8") == first_patch


def test_rhs_trace_patches_are_idempotent(tmp_path: Path) -> None:
    exp_integration_path = tmp_path / "exp_integration.F90"
    matdat_path = tmp_path / "matdat.F90"
    linear_terms_path = tmp_path / "linear_terms.F90"
    exp_integration_path.write_text(_MINIMAL_EXP_INTEGRATION, encoding="utf-8")
    matdat_path.write_text(_MINIMAL_MATDAT, encoding="utf-8")
    linear_terms_path.write_text(_MINIMAL_LINEAR_TERMS, encoding="utf-8")

    assert add_rhs_trace_exp_integration_patch(exp_integration_path, (20, 40))
    assert add_rhs_trace_matdat_patch(matdat_path)
    assert add_rhs_trace_linear_terms_patch(linear_terms_path)
    first_exp_integration = exp_integration_path.read_text(encoding="utf-8")
    first_matdat = matdat_path.read_text(encoding="utf-8")
    first_linear_terms = linear_terms_path.read_text(encoding="utf-8")

    assert not add_rhs_trace_exp_integration_patch(exp_integration_path, (20, 40))
    assert not add_rhs_trace_matdat_patch(matdat_path)
    assert not add_rhs_trace_linear_terms_patch(linear_terms_path)
    assert exp_integration_path.read_text(encoding="utf-8") == first_exp_integration
    assert matdat_path.read_text(encoding="utf-8") == first_matdat
    assert linear_terms_path.read_text(encoding="utf-8") == first_linear_terms


def test_write_cosin2_input_rejects_missing_cosine_selector(tmp_path: Path) -> None:
    input_path = tmp_path / "input.dat"
    output_path = tmp_path / "patched.dat"
    input_path.write_text("&SPCGENERAL\n finit = 'noise'\n /\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finit='cosine'"):
        write_cosin2_input(input_path, output_path)


def test_patched_cosin2_gkw_fixtures_load() -> None:
    input_text = (ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_linear_input.dat").read_text(
        encoding="utf-8"
    )
    time_trace = load_gkw_time_dat_trace(ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat")
    phi_trace = load_gkw_parallel_phi_trace(
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_parallel_phi.dat",
        time_path=ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_time.dat",
    )
    state_trace = load_gkw_state_trace(
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_state_trace.dat"
    )

    assert "finit = 'cosin2'" in input_text
    assert time_trace.times.shape == (80,)
    assert phi_trace.phi_power.shape == (80, 48)
    assert state_trace.times.shape == (80,)
    assert np.isclose(float(time_trace.window_growth[-1]), 0.188741)
    assert np.isclose(float(state_trace.state_norm[-1]), 1.61128981617690e-02)
    assert np.isclose(float(state_trace.phi_norm[-1]), 1.44337567297406e-01)


def test_gkw_state_trace_loader_and_solver_comparison(tmp_path: Path) -> None:
    path = tmp_path / "stellarator_gk_state_trace.dat"
    path.write_text(
        "# step time state_norm phi_norm\n"
        "20 6.00000000000000e-02 1.2e-01 2.3e-01\n"
        "40 1.20000000000000e-01 1.4e-01 2.6e-01\n",
        encoding="utf-8",
    )
    gkw_trace = load_gkw_state_trace(path)
    solver_trace = CycloneSourceTermTrace(
        times=np.asarray([0.0, 0.06, 0.12]),
        phi_norm=np.asarray([0.0, 0.23, 0.26]),
        state_norm=np.asarray([0.0, 0.12, 0.14]),
        rhs_norm=np.asarray([0.0, 1.0, 1.0]),
        term_norms=np.zeros((3, 1)),
        reconstructed_rhs_norm=np.asarray([0.0, 1.0, 1.0]),
        reconstruction_error=np.zeros(3),
        log_normalization=np.zeros(3),
        term_names=("total",),
        source="synthetic-solver",
    )
    report = compare_gkw_state_trace_to_source_term_trace(gkw_trace, solver_trace)

    assert isinstance(gkw_trace, GkwStateTrace)
    np.testing.assert_array_equal(gkw_trace.steps, np.asarray([20, 40], dtype=np.int32))
    assert bool(report.passed)
    np.testing.assert_allclose(report.max_abs_error, 0.0, atol=1.0e-15)


def test_gkw_selected_mode_state_loader_and_comparison(tmp_path: Path) -> None:
    path = tmp_path / "stellarator_gk_selected_state_00000020.dat"
    rows = ["# step time iz imu ivpar real_f imag_f real_phi imag_phi"]
    for iz in (1, 2):
        phi = 0.1 * iz + 1j * 0.01 * iz
        for imu in (1, 2):
            for ivpar in (1, 2):
                value = (ivpar + 10 * imu + 100 * iz) + 1j * (ivpar - imu + iz)
                rows.append(
                    f"20 6.0e-2 {iz} {imu} {ivpar} "
                    f"{value.real:.16e} {value.imag:.16e} "
                    f"{phi.real:.16e} {phi.imag:.16e}"
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    gkw_trace = load_gkw_selected_mode_state_trace(path)
    solver_trace = SelectedModeStateTrace(
        steps=np.asarray([20], dtype=np.int32),
        times=np.asarray([0.06]),
        state=np.asarray(gkw_trace.state),
        phi=np.asarray(gkw_trace.phi),
        source="synthetic-solver",
    )
    report = compare_selected_mode_state_traces(gkw_trace, solver_trace)

    assert isinstance(gkw_trace, SelectedModeStateTrace)
    assert gkw_trace.state.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(gkw_trace.state[0, 1, 0, 0], 112 + 1j * 2)
    np.testing.assert_allclose(gkw_trace.phi[0, 1], 0.2 + 0.02j)
    assert bool(report.passed)
    np.testing.assert_allclose(report.max_abs_error, 0.0, atol=1.0e-15)
    assert "state_phase_aligned" in report.field_names


def test_selected_mode_state_comparison_allows_global_phase(tmp_path: Path) -> None:
    path = tmp_path / "stellarator_gk_selected_state_00000020.dat"
    path.write_text(
        "# step time iz imu ivpar real_f imag_f real_phi imag_phi\n"
        "20 6.0e-2 1 1 1 1.0 2.0 3.0 4.0\n",
        encoding="utf-8",
    )
    gkw_trace = load_gkw_selected_mode_state_trace(path)
    phase = np.exp(0.75j)
    solver_trace = SelectedModeStateTrace(
        steps=np.asarray([20], dtype=np.int32),
        times=np.asarray([0.06]),
        state=np.asarray(gkw_trace.state) * phase,
        phi=np.asarray(gkw_trace.phi) * phase,
        source="synthetic-solver-phase",
    )

    report = compare_selected_mode_state_traces(gkw_trace, solver_trace)

    assert bool(report.passed)
    assert float(report.field_errors[1]) > 0.0
    np.testing.assert_allclose(report.max_abs_error, 0.0, atol=1.0e-14)


def test_patched_cosin2_selected_mode_state_fixtures_load() -> None:
    trace = load_gkw_selected_mode_state_trace(
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_selected_state"
    )

    assert isinstance(trace, SelectedModeStateTrace)
    np.testing.assert_array_equal(trace.steps, np.asarray([20, 800, 1600], dtype=np.int32))
    np.testing.assert_allclose(trace.times, np.asarray([0.06, 2.4, 4.8]))
    assert trace.state.shape == (3, 32, 8, 48)
    assert trace.phi.shape == (3, 48)


def test_gkw_selected_mode_rhs_trace_loader(tmp_path: Path) -> None:
    path = tmp_path / "stellarator_gk_rhs_trace_00000020.dat"
    rows = [
        "# step time iz imu ivpar real_total imag_total "
        "real_untagged imag_untagged real_igh imag_igh real_trapdf imag_trapdf "
        "real_vdgradf imag_vdgradf real_hyper_collision imag_hyper_collision "
        "real_ve_grad_fm imag_ve_grad_fm real_vd_grad_phi_fm imag_vd_grad_phi_fm "
        "real_vpgrphi imag_vpgrphi real_field_eq imag_field_eq"
    ]
    for iz in (1, 2):
        for imu in (1, 2):
            for ivpar in (1, 2):
                terms = []
                total = 0.0 + 0.0j
                for term in range(9):
                    value = (
                        term
                        + 0.1 * ivpar
                        + 0.01 * imu
                        + 0.001 * iz
                        + 1j * (term + iz)
                    )
                    total += value
                    terms.extend((value.real, value.imag))
                rows.append(
                    " ".join(
                        [
                            "20",
                            "6.0e-2",
                            str(iz),
                            str(imu),
                            str(ivpar),
                            f"{total.real:.16e}",
                            f"{total.imag:.16e}",
                            *(f"{value:.16e}" for value in terms),
                        ]
                    )
                )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    trace = load_gkw_selected_mode_rhs_trace(path)

    assert trace.steps.tolist() == [20]
    assert trace.term_names[1] == "igh_or_term_i"
    assert trace.total_action.shape == (1, 2, 2, 2)
    assert trace.term_actions.shape == (1, 9, 2, 2, 2)
    np.testing.assert_allclose(trace.term_actions[0, 7, 1, 0, 0], 7.211 + 8j)


def test_patched_cosin2_rhs_trace_fixtures_load() -> None:
    trace = load_gkw_selected_mode_rhs_trace(
        ROOT / "fixtures/gkw_cyclone_selected_ky_cosin2_rhs_trace"
    )

    np.testing.assert_array_equal(trace.steps, np.asarray([20, 800, 1600], dtype=np.int32))
    np.testing.assert_allclose(trace.times, np.asarray([0.06, 2.4, 4.8]))
    assert trace.total_action.shape == (3, 32, 8, 48)
    assert trace.term_actions.shape == (3, 9, 32, 8, 48)
    assert trace.term_names[7] == "vpgrphi"
    assert float(np.max(np.abs(np.asarray(trace.total_action)))) > 0.0


def test_gkw_velocity_space_slice_series_loader_reads_suffixed_snapshots(tmp_path: Path) -> None:
    for snapshot_index, offset in ((20, 0.0), (40, 10.0)):
        suffix = f"{snapshot_index:08d}"
        values = (
            np.array([[1.0, 2.0], [3.0, 4.0]]) + offset,
            np.array([[5.0, 6.0], [7.0, 8.0]]) + offset,
            np.array([[0.1, 0.2], [0.3, 0.4]]) + offset,
            np.array([[0.5, 0.6], [0.7, 0.8]]) + offset,
        )
        for component, array in enumerate(values, start=1):
            np.savetxt(tmp_path / f"distr{component}_{suffix}.dat", array)
    time_path = tmp_path / "time.dat"
    time_path.write_text("6.0 0.1\n12.0 0.2\n", encoding="utf-8")

    series = load_gkw_velocity_space_slice_series(tmp_path, time_path=time_path)

    assert isinstance(series, GkwVelocitySpaceSliceSeries)
    np.testing.assert_array_equal(series.snapshot_indices, np.array([20, 40]))
    np.testing.assert_allclose(series.times, np.array([6.0, 12.0]))
    assert series.vpar.shape == (2, 2, 2)
    np.testing.assert_allclose(series.vpar[1, 0, 0], 11.0)
    np.testing.assert_allclose(series.imag_part[0, 1, 1], 0.4)
