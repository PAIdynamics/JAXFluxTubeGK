"""Prepare a non-destructive GKW tree with a solver-style cosine2 initializer.

The original GKW ``finit`` selector is ``character(len = 6)``.  The patched
branch therefore uses the six-character name ``cosin2`` while implementing the
same ``1 + cos(2*pi*s)`` profile used by the solver/Gyaradax ``cosine2``
initialization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "relevant-codes" / "gkw"
DEFAULT_INPUT = REPO_ROOT / "fixtures" / "gkw_cyclone_selected_ky_linear_input.dat"
DEFAULT_OUTPUT_ROOT = Path("/tmp/stellarator_gk_gkw_cosin2")
PATCHED_SELECTOR = "cosin2"


@dataclass(frozen=True)
class PreparedGkwCosine2Run:
    """Paths produced by :func:`prepare_gkw_cosine2_run`."""

    source_root: Path
    output_root: Path
    patched_init: Path
    patched_input: Path
    patched_diagnostic: Path | None = None
    multi_time_distr: bool = False
    state_trace: bool = False
    selected_state_dump: bool = False
    rhs_trace: bool = False
    rhs_trace_state_dump: bool = False
    rhs_trace_internal_apply: bool = False
    rhs_trace_igh_matrix_dump: bool = False
    rhs_trace_steps: tuple[int, ...] = (20, 800, 1600)
    selector: str = PATCHED_SELECTOR


COSIN2_CASE = """    case('cosin2')

       do imod = 1, nmod
          do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp
                      do ix = 1, nx

                         ! No initialisation of the zonal flow
                         if ((imod.eq.1).and.mode_box.and.(nmod.ne.1)) then
                            fdisi(indx(imod,ix,i,j,k,is)) = 0.
                         else
                            fdisi(indx(imod,ix,i,j,k,is)) = amp_init *      &
                                 &    de(is) * (1.E0 + cos(2*pi*sgr(ix,i)))
                         endif

                      end do
                   end do
                end do
             end do
          end do
       end do

"""


def prepare_gkw_cosine2_run(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    input_path: Path = DEFAULT_INPUT,
    *,
    overwrite: bool = False,
    multi_time_distr: bool = False,
    state_trace: bool = False,
    selected_state_dump: bool = False,
    rhs_trace: bool = False,
    rhs_trace_state_dump: bool = False,
    rhs_trace_internal_apply: bool = False,
    rhs_trace_igh_matrix_dump: bool = False,
    rhs_trace_steps: tuple[int, ...] = (20, 800, 1600),
) -> PreparedGkwCosine2Run:
    """Copy GKW, patch only the copy, and install a ``finit='cosin2'`` input.

    The source tree is never edited.  If ``output_root`` already exists,
    ``overwrite=True`` is required.
    """

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    input_path = Path(input_path).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"GKW source root not found: {source_root}")
    if not (source_root / "src" / "init.f90").is_file():
        raise FileNotFoundError(f"GKW init.f90 not found under: {source_root}")
    if not input_path.is_file():
        raise FileNotFoundError(f"GKW input file not found: {input_path}")
    if source_root == output_root:
        raise ValueError("output_root must be different from source_root")
    if rhs_trace_state_dump and not rhs_trace:
        raise ValueError("rhs_trace_state_dump requires rhs_trace=True")
    if rhs_trace_internal_apply and not rhs_trace:
        raise ValueError("rhs_trace_internal_apply requires rhs_trace=True")
    if rhs_trace_igh_matrix_dump and not rhs_trace:
        raise ValueError("rhs_trace_igh_matrix_dump requires rhs_trace=True")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    shutil.copytree(source_root, output_root, ignore=_ignore_generated_gkw_files)
    patched_init = output_root / "src" / "init.f90"
    add_cosin2_branch(patched_init)
    patched_diagnostic = None
    if rhs_trace:
        add_rhs_trace_matdat_patch(output_root / "src" / "matdat.F90")
        add_rhs_trace_linear_terms_patch(output_root / "src" / "linear_terms.F90")
        add_rhs_trace_exp_integration_patch(
            output_root / "src" / "exp_integration.F90",
            rhs_trace_steps,
            state_dump=rhs_trace_state_dump,
            internal_apply=rhs_trace_internal_apply,
            igh_matrix_dump=rhs_trace_igh_matrix_dump,
        )
    if multi_time_distr or state_trace or selected_state_dump:
        patched_diagnostic = output_root / "src" / "diagnostic.F90"
    if state_trace:
        add_state_trace_patch(patched_diagnostic)
    if selected_state_dump:
        add_selected_state_dump_patch(patched_diagnostic)
    if multi_time_distr:
        add_multitime_velocity_slice_patch(patched_diagnostic)
    patched_input = output_root / "input.dat"
    write_cosin2_input(input_path, patched_input)
    _write_run_readme(
        output_root,
        multi_time_distr=multi_time_distr,
        state_trace=state_trace,
        selected_state_dump=selected_state_dump,
        rhs_trace=rhs_trace,
        rhs_trace_state_dump=rhs_trace_state_dump,
        rhs_trace_internal_apply=rhs_trace_internal_apply,
        rhs_trace_igh_matrix_dump=rhs_trace_igh_matrix_dump,
        rhs_trace_steps=tuple(int(step) for step in rhs_trace_steps),
    )
    return PreparedGkwCosine2Run(
        source_root=source_root,
        output_root=output_root,
        patched_init=patched_init,
        patched_diagnostic=patched_diagnostic,
        patched_input=patched_input,
        multi_time_distr=multi_time_distr,
        state_trace=state_trace,
        selected_state_dump=selected_state_dump,
        rhs_trace=rhs_trace,
        rhs_trace_state_dump=rhs_trace_state_dump,
        rhs_trace_internal_apply=rhs_trace_internal_apply,
        rhs_trace_igh_matrix_dump=rhs_trace_igh_matrix_dump,
        rhs_trace_steps=tuple(int(step) for step in rhs_trace_steps),
    )


def add_cosin2_branch(init_path: Path) -> bool:
    """Insert the six-character ``cosin2`` branch into a copied GKW init file."""

    text = init_path.read_text(encoding="utf-8")
    if "case('cosin2')" in text:
        return False
    marker = "    case('sine')"
    if marker not in text:
        raise ValueError(f"could not find insertion marker {marker!r} in {init_path}")
    init_path.write_text(text.replace(marker, COSIN2_CASE + marker, 1), encoding="utf-8")
    return True


def write_cosin2_input(input_path: Path, output_path: Path) -> None:
    """Write an input.dat copy whose ``finit`` selector is ``'cosin2'``."""

    text = input_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?im)^(\s*finit\s*=\s*)(?:'cosine'|\"cosine\"|cosine)(\s*,?)")
    patched, count = pattern.subn(rf"\1'{PATCHED_SELECTOR}'\2", text, count=1)
    if count != 1:
        raise ValueError(f"could not replace finit='cosine' in {input_path}")
    output_path.write_text(patched, encoding="utf-8")


def add_multitime_velocity_slice_patch(diagnostic_path: Path) -> bool:
    """Patch copied GKW diagnostics to write ``distr*_<ntotstep>.dat`` snapshots."""

    text = diagnostic_path.read_text(encoding="utf-8")
    if "stellarator_gk multi-time distr patch" in text:
        return False

    control_import = "  use control,      only : output3d, lphi_diagnostics\n"
    control_import_patch = (
        "  use control,      only : output3d, lphi_diagnostics, ntotstep\n"
    )
    if control_import not in text:
        raise ValueError(f"could not find write_output control import in {diagnostic_path}")
    text = text.replace(control_import, control_import_patch, 1)

    call_marker = "  ! 2D outputs: CHECK THESE WORK WITH PARALLEL_S\n"
    call_patch = (
        "  ! stellarator_gk multi-time distr patch: write selected peak-phi\n"
        "  ! velocity-space slices at every normal diagnostic output window.\n"
        "  call velocity_space_output(ntotstep)\n\n"
    )
    if call_marker not in text:
        raise ValueError(f"could not find write_output insertion marker in {diagnostic_path}")
    text = text.replace(call_marker, call_patch + call_marker, 1)

    signature = "subroutine velocity_space_output\n"
    replacement = "subroutine velocity_space_output(snapshot_index)\n"
    if signature not in text:
        raise ValueError(f"could not find velocity_space_output signature in {diagnostic_path}")
    text = text.replace(signature, replacement, 1)

    declaration = "  character (len=18) :: file_fmt \n"
    if declaration not in text:
        raise ValueError(f"could not find file_fmt declaration in {diagnostic_path}")
    text = text.replace(
        declaration,
        "  integer, optional, intent(in) :: snapshot_index\n"
        + declaration
        + "  character (len=32) :: distr_file\n",
        1,
    )

    for index in range(1, 5):
        old = (
            f"  if (lwrite) call output_slice(global_vpar_mu,FILE='distr{index}.dat',"
            "FMT=file_fmt)"
        )
        new = (
            "  if (present(snapshot_index)) then\n"
            f"    write(distr_file,'(\"distr{index}_\",I8.8,\".dat\")') snapshot_index\n"
            "  else\n"
            f"    distr_file = 'distr{index}.dat'\n"
            "  endif\n"
            "  if (lwrite) call output_slice(global_vpar_mu,FILE=trim(distr_file),"
            "FMT=file_fmt)"
        )
        if old not in text:
            raise ValueError(f"could not find distr{index}.dat output call in {diagnostic_path}")
        text = text.replace(old, new, 1)

    diagnostic_path.write_text(text, encoding="utf-8")
    return True


def add_state_trace_patch(diagnostic_path: Path) -> bool:
    """Patch copied GKW diagnostics to write compact state/field norm traces."""

    text = diagnostic_path.read_text(encoding="utf-8")
    if "stellarator_gk compact state trace patch" in text:
        return False

    call_marker = "  ! 2D outputs: CHECK THESE WORK WITH PARALLEL_S\n"
    call_patch = (
        "  ! stellarator_gk compact state trace patch: write state and field\n"
        "  ! norms at every normal diagnostic output window.\n"
        "  call stellarator_gk_state_trace_output\n\n"
    )
    if call_marker not in text:
        raise ValueError(f"could not find write_output insertion marker in {diagnostic_path}")
    text = text.replace(call_marker, call_patch + call_marker, 1)

    module_marker = "end module diagnostic"
    if module_marker not in text:
        raise ValueError(f"could not find diagnostic module end marker in {diagnostic_path}")
    text = text.replace(module_marker, STATE_TRACE_SUBROUTINE + "\n" + module_marker, 1)
    diagnostic_path.write_text(text, encoding="utf-8")
    return True


def add_selected_state_dump_patch(diagnostic_path: Path) -> bool:
    """Patch copied GKW diagnostics to dump the full selected-mode state."""

    text = diagnostic_path.read_text(encoding="utf-8")
    if "stellarator_gk selected-state dump patch" in text:
        return False

    call_marker = "  ! 2D outputs: CHECK THESE WORK WITH PARALLEL_S\n"
    call_patch = (
        "  ! stellarator_gk selected-state dump patch: write imod=1, ix=1,\n"
        "  ! species=1 distribution and phi at every diagnostic output window.\n"
        "  call stellarator_gk_selected_state_dump_output\n\n"
    )
    if call_marker not in text:
        raise ValueError(f"could not find write_output insertion marker in {diagnostic_path}")
    text = text.replace(call_marker, call_patch + call_marker, 1)

    module_marker = "end module diagnostic"
    if module_marker not in text:
        raise ValueError(f"could not find diagnostic module end marker in {diagnostic_path}")
    text = text.replace(
        module_marker,
        SELECTED_STATE_DUMP_SUBROUTINE + "\n" + module_marker,
        1,
    )
    diagnostic_path.write_text(text, encoding="utf-8")
    return True


def add_rhs_trace_matdat_patch(matdat_path: Path) -> bool:
    """Patch copied GKW matrix storage to retain linear-term IDs."""

    text = matdat_path.read_text(encoding="utf-8")
    if "stellarator_gk rhs trace matdat patch" in text:
        return False

    public_marker = "public :: put_element_correct_apar\n"
    public_patch = (
        "public :: put_element_correct_apar\n"
        "public :: stellarator_gk_set_trace_term, stellarator_gk_mat_term\n"
        "public :: stellarator_gk_source_by_term, stellarator_gk_n_trace_terms\n"
    )
    if public_marker not in text:
        raise ValueError(f"could not find matdat public marker in {matdat_path}")
    text = text.replace(public_marker, public_patch, 1)

    source_decl = "complex, allocatable :: source(:)\n"
    source_patch = (
        "complex, allocatable :: source(:)\n"
        "! stellarator_gk rhs trace matdat patch\n"
        "integer, parameter :: stellarator_gk_n_trace_terms = 8\n"
        "integer :: stellarator_gk_current_trace_term = 0\n"
        "integer, allocatable :: stellarator_gk_mat_term(:)\n"
        "complex, allocatable :: stellarator_gk_source_by_term(:,:)\n"
    )
    if source_decl not in text:
        raise ValueError(f"could not find source declaration in {matdat_path}")
    text = text.replace(source_decl, source_patch, 1)

    source_init_pattern = r"do i = 1, nsolc[ \t]*\n  source\(i\) = \(0\.,0\.\)[ \t]*\nend do[ \t]*\n"
    source_init_match = re.search(source_init_pattern, text)
    if source_init_match is None:
        raise ValueError(f"could not find source initialization in {matdat_path}")
    source_init_patch = (
        source_init_match.group(0)
        + "allocate(stellarator_gk_source_by_term(nsolc,0:stellarator_gk_n_trace_terms),"
        "stat=ierr)\n"
        "if (ierr.ne.0) then\n"
        "  stop 'Could not allocate stellarator_gk_source_by_term in matdat'\n"
        "endif\n"
        "stellarator_gk_source_by_term = (0.,0.)\n"
    )
    text = text[: source_init_match.start()] + source_init_patch + text[source_init_match.end() :]

    mat_alloc = "  allocate(mat(ntot),stat=ierr)\n"
    mat_alloc_patch = (
        mat_alloc
        + "  allocate(stellarator_gk_mat_term(ntot),stat=ierr)\n"
        + "  if (ierr.ne.0) then\n"
        + "    stop 'Could not allocate stellarator_gk_mat_term in matdat'\n"
        + "  endif\n"
        + "  stellarator_gk_mat_term = 0\n"
    )
    if mat_alloc not in text:
        raise ValueError(f"could not find complex matrix allocation in {matdat_path}")
    text = text.replace(mat_alloc, mat_alloc_patch, 1)

    mat_store_pattern = r"  mat\(nmat\) = mat_elem[ \t]*\n"
    mat_store_match = re.search(mat_store_pattern, text)
    if mat_store_match is None:
        raise ValueError(f"could not find complex matrix store in {matdat_path}")
    mat_store_patch = (
        mat_store_match.group(0)
        + "  stellarator_gk_mat_term(nmat) = stellarator_gk_current_trace_term\n"
    )
    text = text[: mat_store_match.start()] + mat_store_patch + text[mat_store_match.end() :]

    source_store_pattern = r"source\(iih\) = source\(iih\) \+ mat_elem[ \t]*\n"
    source_store_match = re.search(source_store_pattern, text)
    if source_store_match is None:
        raise ValueError(f"could not find source store in {matdat_path}")
    source_store_patch = (
        source_store_match.group(0)
        + "stellarator_gk_source_by_term(iih,stellarator_gk_current_trace_term) = &\n"
        + "     & stellarator_gk_source_by_term(iih,stellarator_gk_current_trace_term) + mat_elem\n"
    )
    text = text[: source_store_match.start()] + source_store_patch + text[source_store_match.end() :]

    duplicate_check = "    if ((ii(i).eq.ii(ireduced)).and.(jj(i).eq.jj(ireduced))) then\n"
    duplicate_patch = (
        "    if ((ii(i).eq.ii(ireduced)).and.(jj(i).eq.jj(ireduced)) &\n"
        "        & .and.(stellarator_gk_mat_term(i).eq.stellarator_gk_mat_term(ireduced))) then\n"
    )
    if duplicate_check not in text:
        raise ValueError(f"could not find compression duplicate check in {matdat_path}")
    text = text.replace(duplicate_check, duplicate_patch, 1)

    reduced_copy = "      mat(ireduced) = mat(i)\n"
    reduced_copy_patch = reduced_copy + "      stellarator_gk_mat_term(ireduced) = stellarator_gk_mat_term(i)\n"
    if reduced_copy not in text:
        raise ValueError(f"could not find compression matrix copy in {matdat_path}")
    text = text.replace(reduced_copy, reduced_copy_patch, 1)

    heap_swap = "    ctmp = mat(ind) ; mat(ind) = mat(i_start) ; mat(i_start) = ctmp\n"
    heap_swap_patch = (
        heap_swap
        + "    itmp = stellarator_gk_mat_term(ind)\n"
        + "    stellarator_gk_mat_term(ind) = stellarator_gk_mat_term(i_start)\n"
        + "    stellarator_gk_mat_term(i_start) = itmp\n"
    )
    if heap_swap not in text:
        raise ValueError(f"could not find heap-sort matrix swap in {matdat_path}")
    text = text.replace(heap_swap, heap_swap_patch, 1)

    sift_swap = "      ctmp = mat(ind2) ; mat(ind2) = mat(ind) ; mat(ind) = ctmp\n"
    sift_swap_patch = (
        sift_swap
        + "      itmp = stellarator_gk_mat_term(ind2)\n"
        + "      stellarator_gk_mat_term(ind2) = stellarator_gk_mat_term(ind)\n"
        + "      stellarator_gk_mat_term(ind) = itmp\n"
    )
    if sift_swap not in text:
        raise ValueError(f"could not find sift matrix swap in {matdat_path}")
    text = text.replace(sift_swap, sift_swap_patch, 1)

    subroutine_marker = "subroutine put_element(iih,jjh,mat_elem,itime_est)\n"
    setter = (
        "subroutine stellarator_gk_set_trace_term(term_id)\n"
        "  integer, intent(in) :: term_id\n"
        "  if (term_id < 0 .or. term_id > stellarator_gk_n_trace_terms) then\n"
        "    stop 'stellarator_gk_set_trace_term: term out of range'\n"
        "  endif\n"
        "  stellarator_gk_current_trace_term = term_id\n"
        "end subroutine stellarator_gk_set_trace_term\n\n"
    )
    if subroutine_marker not in text:
        raise ValueError(f"could not find put_element marker in {matdat_path}")
    text = text.replace(subroutine_marker, setter + subroutine_marker, 1)

    matdat_path.write_text(text, encoding="utf-8")
    return True


def add_rhs_trace_linear_terms_patch(linear_terms_path: Path) -> bool:
    """Patch copied GKW linear-term assembly to tag matrix/source entries."""

    text = linear_terms_path.read_text(encoding="utf-8")
    if "stellarator_gk rhs trace linear_terms patch" in text:
        return False

    use_marker = "  use matdat,      only : finish_matrix_section\n"
    use_patch = (
        "  use matdat,      only : finish_matrix_section, &\n"
        "      & stellarator_gk_set_trace_term\n"
        "  ! stellarator_gk rhs trace linear_terms patch\n"
    )
    if use_marker not in text:
        raise ValueError(f"could not find calc_linear_terms matdat use in {linear_terms_path}")
    text = text.replace(use_marker, use_patch, 1)

    replacements = (
        ("  if (ltrapping_arakawa) call igh(disp_par,disp_vp)\n", 1, 1),
        ("          call vpar_grad_df_4d_testnewbc(disp_par)\n", 1, 2),
        ("        call trapdf_2d(trapping,disp_vp)\n", 2, 1),
        ("        call trapdf_4d(trapping,disp_vp)\n", 2, 1),
        ("  if (lvdgradf) call vdgradf\n", 3, 1),
        ("  if (lve_grad_fm) call ve_grad_fm\n", 5, 1),
        ("  if (lvd_grad_phi_fm) call vd_grad_phi_fm\n", 6, 1),
        ("          call vpgrphi_3_newbc(landau,disp_fe)\n", 7, 2),
        ("  if (lpoisson_int) call poisson_int\n", 8, 1),
    )
    for marker, term_id, count in replacements:
        if marker not in text:
            raise ValueError(f"could not find linear term marker {marker!r}")
        text = text.replace(
            marker,
            f"  call stellarator_gk_set_trace_term({term_id})\n" + marker,
            count,
        )
    reset_marker = "  if (neoclassics .and. lneoclassical) call neoclassical\n"
    if reset_marker not in text:
        raise ValueError(f"could not find neoclassical marker in {linear_terms_path}")
    text = text.replace(
        reset_marker,
        "  call stellarator_gk_set_trace_term(0)\n" + reset_marker,
        1,
    )

    linear_terms_path.write_text(text, encoding="utf-8")
    return True


def add_rhs_trace_exp_integration_patch(
    exp_integration_path: Path,
    steps: tuple[int, ...] = (20, 800, 1600),
    *,
    state_dump: bool = False,
    internal_apply: bool = False,
    igh_matrix_dump: bool = False,
) -> bool:
    """Patch copied GKW time advancement to dump selected-mode RHS term actions."""

    text = exp_integration_path.read_text(encoding="utf-8")
    if "stellarator_gk rhs trace exp_integration patch" in text:
        if state_dump and "stellarator_gk rhs trace state dump" not in text:
            raise ValueError(
                f"{exp_integration_path} is already patched without rhs trace state dump"
            )
        if internal_apply and "stellarator_gk rhs trace internal apply" not in text:
            raise ValueError(
                f"{exp_integration_path} is already patched without internal apply dump"
            )
        if igh_matrix_dump and "stellarator_gk igh matrix dump" not in text:
            raise ValueError(
                f"{exp_integration_path} is already patched without igh matrix dump"
            )
        return False

    if not steps:
        raise ValueError("rhs trace steps must not be empty")
    step_values = tuple(int(step) for step in steps)
    if any(step <= 0 for step in step_values):
        raise ValueError("rhs trace steps must be positive")

    call_marker = (
        "  ! done after calculate_fields to have the new potential\n"
        "  call normalize(2,fdisi(1),nsolc)\n\n"
    )
    call_patch = (
        call_marker
        + "  ! stellarator_gk rhs trace exp_integration patch: write selected-mode\n"
        + "  ! dtim-scaled term actions after end-of-window normalization.\n"
        + ("  call stellarator_gk_rhs_trace_state_output\n" if state_dump else "")
        + ("  call stellarator_gk_rhs_apply_output\n" if internal_apply else "")
        + ("  call stellarator_gk_igh_matrix_output\n" if igh_matrix_dump else "")
        + "  call stellarator_gk_rhs_trace_output\n\n"
    )
    if call_marker not in text:
        raise ValueError(
            f"could not find explicit normalization marker in {exp_integration_path}"
        )
    text = text.replace(call_marker, call_patch, 1)

    module_marker = "end module exp_integration"
    if module_marker not in text:
        raise ValueError(
            f"could not find exp_integration module end marker in {exp_integration_path}"
        )
    subroutine = ""
    if state_dump:
        subroutine += _rhs_trace_state_subroutine(step_values) + "\n"
    if internal_apply:
        subroutine += _rhs_trace_apply_subroutine(step_values) + "\n"
    if igh_matrix_dump:
        subroutine += _igh_matrix_dump_subroutine(step_values) + "\n"
    subroutine += _rhs_trace_subroutine(step_values)
    text = text.replace(module_marker, subroutine + "\n" + module_marker, 1)
    exp_integration_path.write_text(text, encoding="utf-8")
    return True


STATE_TRACE_SUBROUTINE = r"""
subroutine stellarator_gk_state_trace_output

  use control,      only : time, ntotstep
  use dist,         only : fdisi, nf, nsolc, phi, get_phi
  use grid,         only : nmod, nx, ns
  use io,           only : get_free_lun
  use mpiinterface, only : root_processor

  integer :: i, imod, ix, is, lun
  logical, save :: header_written = .false.
  real :: state_power, state_norm, phi_power, phi_norm

  state_power = 0.E0
  do i = 1, nf
     state_power = state_power + real(fdisi(i)*conjg(fdisi(i)))
  end do
  state_norm = sqrt(state_power / max(1, nf))

  call get_phi(fdisi(1:nsolc), phi)
  phi_power = 0.E0
  do imod = 1, nmod
     do ix = 1, nx
        do is = 1, ns
           phi_power = phi_power + real(phi(imod,ix,is)*conjg(phi(imod,ix,is)))
        end do
     end do
  end do
  phi_norm = sqrt(phi_power / max(1, nmod*nx*ns))

  if (root_processor) then
     call get_free_lun(lun)
     open(lun, FILE='stellarator_gk_state_trace.dat', STATUS='unknown', &
          & POSITION='append')
     if (.not. header_written) then
        write(lun,'(A)') '# step time state_norm phi_norm'
        header_written = .true.
     end if
     write(lun,'(I12,1X,3(1PE22.14,1X))') ntotstep, time, state_norm, phi_norm
     close(lun)
  end if

end subroutine stellarator_gk_state_trace_output
"""


SELECTED_STATE_DUMP_SUBROUTINE = r"""
subroutine stellarator_gk_selected_state_dump_output

  use control,      only : time, ntotstep
  use dist,         only : fdisi, nsolc, phi, get_phi, indx
  use grid,         only : ns, nmu, nvpar
  use io,           only : get_free_lun
  use mpiinterface, only : root_processor

  integer :: iz, imu, ivpar, lun
  character (len=64) :: dump_file
  complex :: fval, phival

  call get_phi(fdisi(1:nsolc), phi)

  if (root_processor) then
     write(dump_file,'("stellarator_gk_selected_state_",I8.8,".dat")') ntotstep
     call get_free_lun(lun)
     open(lun, FILE=trim(dump_file), STATUS='unknown')
     write(lun,'(A)') '# step time iz imu ivpar real_f imag_f real_phi imag_phi'
     do iz = 1, ns
        phival = phi(1,1,iz)
        do imu = 1, nmu
           do ivpar = 1, nvpar
              fval = fdisi(indx(1,1,iz,imu,ivpar,1))
              write(lun,'(I12,1X,1PE22.14,3(1X,I8),4(1X,1PE22.14))') &
                   & ntotstep, time, iz, imu, ivpar, real(fval), aimag(fval), &
                   & real(phival), aimag(phival)
           end do
        end do
     end do
     close(lun)
  end if

end subroutine stellarator_gk_selected_state_dump_output
"""


def _rhs_trace_subroutine(steps: tuple[int, ...]) -> str:
    keep_checks = "\n".join(f"  if (ntotstep .eq. {step}) keep_snapshot = .true." for step in steps)
    return rf"""
subroutine stellarator_gk_rhs_trace_output

  use control,      only : time, ntotstep, dtim
  use dist,         only : fdisi, indx, nsolc
  use grid,         only : ns, nmu, nvpar
  use io,           only : get_free_lun
  use matdat,       only : mat, ii, jj, n2, source, stellarator_gk_mat_term, &
       & stellarator_gk_source_by_term, stellarator_gk_n_trace_terms
  use mpiinterface, only : root_processor

  integer :: iz, imu, ivpar, irow, elem, term_id, term, lun
  character (len=64) :: trace_file
  complex :: action(0:stellarator_gk_n_trace_terms)
  complex :: total_action
  logical :: keep_snapshot

  keep_snapshot = .false.
{keep_checks}
  if (.not. keep_snapshot) return

  if (root_processor) then
     write(trace_file,'("stellarator_gk_rhs_trace_",I8.8,".dat")') ntotstep
     call get_free_lun(lun)
     open(lun, FILE=trim(trace_file), STATUS='unknown')
     write(lun,'(A)') '# step time iz imu ivpar real_total imag_total ' // &
          & 'real_untagged imag_untagged real_igh imag_igh ' // &
          & 'real_trapdf imag_trapdf real_vdgradf imag_vdgradf ' // &
          & 'real_hyper_collision imag_hyper_collision ' // &
          & 'real_ve_grad_fm imag_ve_grad_fm ' // &
          & 'real_vd_grad_phi_fm imag_vd_grad_phi_fm ' // &
          & 'real_vpgrphi imag_vpgrphi real_field_eq imag_field_eq'
     do iz = 1, ns
        do imu = 1, nmu
           do ivpar = 1, nvpar
              irow = indx(1,1,iz,imu,ivpar,1)
              action = (0.,0.)
              do term = 0, stellarator_gk_n_trace_terms
                 action(term) = action(term) + dtim*stellarator_gk_source_by_term(irow,term)
              end do
              do elem = 1, n2
                 if (ii(elem) .eq. irow) then
                    term_id = stellarator_gk_mat_term(elem)
                    if (term_id .lt. 0 .or. term_id .gt. stellarator_gk_n_trace_terms) then
                       term_id = 0
                    endif
                    if (jj(elem) .ge. 1 .and. jj(elem) .le. nsolc) then
                       action(term_id) = action(term_id) + dtim*mat(elem)*fdisi(jj(elem))
                    endif
                 endif
              end do
              total_action = (0.,0.)
              do term = 0, stellarator_gk_n_trace_terms
                 total_action = total_action + action(term)
              end do
              write(lun,'(I12,1X,1PE22.14,3(1X,I8),20(1X,1PE22.14))') &
                   & ntotstep, time, iz, imu, ivpar, real(total_action), &
                   & aimag(total_action), &
                   & (real(action(term)), aimag(action(term)), &
                   &  term = 0, stellarator_gk_n_trace_terms)
           end do
        end do
     end do
     close(lun)
  end if

end subroutine stellarator_gk_rhs_trace_output
"""


def _rhs_trace_state_subroutine(steps: tuple[int, ...]) -> str:
    keep_checks = "\n".join(f"  if (ntotstep .eq. {step}) keep_snapshot = .true." for step in steps)
    return rf"""
subroutine stellarator_gk_rhs_trace_state_output

  use control,      only : time, ntotstep
  use dist,         only : fdisi, indx, nsolc, phi, get_phi
  use grid,         only : ns, nmu, nvpar
  use io,           only : get_free_lun
  use mpiinterface, only : root_processor

  integer :: iz, imu, ivpar, lun
  character (len=64) :: state_file
  complex :: fval, phival
  logical :: keep_snapshot

  ! stellarator_gk rhs trace state dump: write the selected state at the same
  ! post-normalization timing used by stellarator_gk_rhs_trace_output.
  keep_snapshot = .false.
{keep_checks}
  if (.not. keep_snapshot) return

  call get_phi(fdisi(1:nsolc), phi)

  if (root_processor) then
     write(state_file,'("stellarator_gk_rhs_state_",I8.8,".dat")') ntotstep
     call get_free_lun(lun)
     open(lun, FILE=trim(state_file), STATUS='unknown')
     write(lun,'(A)') '# step time iz imu ivpar real_f imag_f real_phi imag_phi'
     do iz = 1, ns
        phival = phi(1,1,iz)
        do imu = 1, nmu
           do ivpar = 1, nvpar
              fval = fdisi(indx(1,1,iz,imu,ivpar,1))
              write(lun,'(I12,1X,1PE22.14,3(1X,I8),4(1X,1PE22.14))') &
                   & ntotstep, time, iz, imu, ivpar, real(fval), aimag(fval), &
                   & real(phival), aimag(phival)
           end do
        end do
     end do
     close(lun)
  end if

end subroutine stellarator_gk_rhs_trace_state_output
"""


def _rhs_trace_apply_subroutine(steps: tuple[int, ...]) -> str:
    keep_checks = "\n".join(f"  if (ntotstep .eq. {step}) keep_snapshot = .true." for step in steps)
    return rf"""
subroutine stellarator_gk_rhs_apply_output

  use control,      only : time, ntotstep
  use dist,         only : fdisi, indx, nsolc
  use grid,         only : ns, nmu, nvpar
  use io,           only : get_free_lun
  use mpiinterface, only : root_processor

  integer :: iz, imu, ivpar, irow, lun, ierr
  character (len=64) :: apply_file
  complex, allocatable :: rhs_internal(:)
  logical :: keep_snapshot

  ! stellarator_gk rhs trace internal apply: run GKW calculate_rhs on the
  ! post-normalization state and dump the actual selected-row RHS totals.
  keep_snapshot = .false.
{keep_checks}
  if (.not. keep_snapshot) return

  allocate(rhs_internal(nsolc), stat=ierr)
  if (ierr .ne. 0) stop 'Could not allocate rhs_internal in stellarator_gk_rhs_apply_output'
  call calculate_rhs(fdisi, rhs_internal)

  if (root_processor) then
     write(apply_file,'("stellarator_gk_rhs_apply_",I8.8,".dat")') ntotstep
     call get_free_lun(lun)
     open(lun, FILE=trim(apply_file), STATUS='unknown')
     write(lun,'(A)') '# step time iz imu ivpar real_calculate_rhs imag_calculate_rhs'
     do iz = 1, ns
        do imu = 1, nmu
           do ivpar = 1, nvpar
              irow = indx(1,1,iz,imu,ivpar,1)
              write(lun,'(I12,1X,1PE22.14,3(1X,I8),2(1X,1PE22.14))') &
                   & ntotstep, time, iz, imu, ivpar, real(rhs_internal(irow)), &
                   & aimag(rhs_internal(irow))
           end do
        end do
     end do
     close(lun)
  end if

  deallocate(rhs_internal)

end subroutine stellarator_gk_rhs_apply_output
"""


def _igh_matrix_dump_subroutine(steps: tuple[int, ...]) -> str:
    keep_checks = "\n".join(f"  if (ntotstep .eq. {step}) keep_snapshot = .true." for step in steps)
    return rf"""
subroutine stellarator_gk_igh_matrix_output

  use control,      only : time, ntotstep
  use dist,         only : indx, nsolc
  use grid,         only : ns, nmu, nvpar
  use io,           only : get_free_lun
  use matdat,       only : mat, ii, jj, n2, stellarator_gk_mat_term
  use mpiinterface, only : root_processor

  integer :: iz, imu, ivpar, elem, row_index, col_index, lun, ierr
  integer :: term_id
  integer, allocatable :: map_iz(:), map_imu(:), map_ivpar(:)
  character (len=64) :: matrix_file
  logical :: keep_snapshot

  ! stellarator_gk igh matrix dump: dump compressed selected-row matrix
  ! entries tagged as igh_or_term_i after matdat compression/sorting.
  keep_snapshot = .false.
{keep_checks}
  if (.not. keep_snapshot) return
  if (.not. root_processor) return

  allocate(map_iz(nsolc), map_imu(nsolc), map_ivpar(nsolc), stat=ierr)
  if (ierr .ne. 0) stop 'Could not allocate index maps in stellarator_gk_igh_matrix_output'
  map_iz = -1
  map_imu = -1
  map_ivpar = -1

  do iz = 1, ns
     do imu = 1, nmu
        do ivpar = 1, nvpar
           row_index = indx(1,1,iz,imu,ivpar,1)
           if (row_index .ge. 1 .and. row_index .le. nsolc) then
              map_iz(row_index) = iz
              map_imu(row_index) = imu
              map_ivpar(row_index) = ivpar
           endif
        end do
     end do
  end do

  write(matrix_file,'("stellarator_gk_igh_matrix_",I8.8,".dat")') ntotstep
  call get_free_lun(lun)
  open(lun, FILE=trim(matrix_file), STATUS='unknown')
  write(lun,'(A)') '# step time elem row_index col_index row_iz row_imu row_ivpar ' // &
       & 'col_iz col_imu col_ivpar term real_mat imag_mat'

  do elem = 1, n2
     row_index = ii(elem)
     col_index = jj(elem)
     term_id = stellarator_gk_mat_term(elem)
     if (term_id .eq. 1 .and. row_index .ge. 1 .and. row_index .le. nsolc) then
        if (map_iz(row_index) .gt. 0) then
           if (col_index .ge. 1 .and. col_index .le. nsolc) then
              write(lun,'(I12,1X,1PE22.14,10(1X,I12),2(1X,1PE22.14))') &
                   & ntotstep, time, elem, row_index, col_index, &
                   & map_iz(row_index), map_imu(row_index), map_ivpar(row_index), &
                   & map_iz(col_index), map_imu(col_index), map_ivpar(col_index), &
                   & term_id, real(mat(elem)), aimag(mat(elem))
           else
              write(lun,'(I12,1X,1PE22.14,10(1X,I12),2(1X,1PE22.14))') &
                   & ntotstep, time, elem, row_index, col_index, &
                   & map_iz(row_index), map_imu(row_index), map_ivpar(row_index), &
                   & -1, -1, -1, term_id, real(mat(elem)), aimag(mat(elem))
           endif
        endif
     endif
  end do
  close(lun)

  deallocate(map_iz, map_imu, map_ivpar)

end subroutine stellarator_gk_igh_matrix_output
"""


def _ignore_generated_gkw_files(_directory: str, names: list[str]) -> set[str]:
    generated_names = {
        ".DS_Store",
        "gkw.x",
        "input.out",
        "time.dat",
        "parallel_phi.dat",
        "phi.dat",
        "fluxes.dat",
        "screen.out",
        "stellarator_gk_state_trace.dat",
    }
    suffixes = (".o", ".mod", ".smod", ".log")
    return {
        name
        for name in names
        if name in generated_names
        or name.startswith("stellarator_gk_selected_state_")
        or name.startswith("stellarator_gk_rhs_trace_")
        or name.startswith("stellarator_gk_rhs_state_")
        or name.startswith("stellarator_gk_rhs_apply_")
        or name.startswith("stellarator_gk_igh_matrix_")
        or name.endswith(suffixes)
        or name == "__pycache__"
    }


def _write_run_readme(
    output_root: Path,
    *,
    multi_time_distr: bool = False,
    state_trace: bool = False,
    selected_state_dump: bool = False,
    rhs_trace: bool = False,
    rhs_trace_state_dump: bool = False,
    rhs_trace_internal_apply: bool = False,
    rhs_trace_igh_matrix_dump: bool = False,
    rhs_trace_steps: tuple[int, ...] = (20, 800, 1600),
) -> None:
    extra = ""
    if multi_time_distr:
        extra = """
The copied `src/diagnostic.F90` was also patched to write velocity-space
snapshots at every normal diagnostic output window:

```text
distr1_00000020.dat, ..., distr4_00000020.dat
distr1_00000040.dat, ..., distr4_00000040.dat
...
```

The unsuffixed `distr*.dat` files are still produced by GKW final output.
"""
    if state_trace:
        extra += """
The copied `src/diagnostic.F90` was also patched to write a compact state trace
at every normal diagnostic output window:

```text
stellarator_gk_state_trace.dat
```

The columns are `step`, `time`, `state_norm`, and `phi_norm`.
"""
    if selected_state_dump:
        extra += """
The copied `src/diagnostic.F90` was also patched to write the full selected
single-mode state at every normal diagnostic output window:

```text
stellarator_gk_selected_state_00000020.dat
stellarator_gk_selected_state_00000040.dat
...
```

Each row stores `step`, `time`, one-based `(z, mu, vpar)` indices, the complex
distribution value, and the selected-mode complex field at that `z`.
"""
    if rhs_trace:
        step_list = ", ".join(str(step) for step in rhs_trace_steps)
        extra += f"""
The copied `src/matdat.F90`, `src/linear_terms.F90`, and
`src/exp_integration.F90` were also patched to tag linear matrix entries by
source term and write selected-mode dtim-scaled RHS/source actions at steps
{step_list}:

```text
stellarator_gk_rhs_trace_00000020.dat
stellarator_gk_rhs_trace_00000800.dat
stellarator_gk_rhs_trace_00001600.dat
```

Each row stores `step`, `time`, one-based `(z, mu, vpar)` indices, the total
selected-row action, and the untagged, `igh`, `trapdf`, `vdgradf`,
`hyper_collision`, `ve_grad_fm`, `vd_grad_phi_fm`, `vpgrphi`, and field-equation
actions.  The matrix contribution mirrors the explicit complex
`calculate_rhs` convention and therefore sums compressed matrix entries through
section `n2`.
"""
        if rhs_trace_state_dump:
            extra += """
The RHS trace patch also writes same-timing selected states from
`exp_integration.F90`, immediately after the same normalization point used for
the RHS action trace:

```text
stellarator_gk_rhs_state_00000020.dat
stellarator_gk_rhs_state_00000800.dat
stellarator_gk_rhs_state_00001600.dat
```

These files use the selected-state row format and are intended as a
trace-timing discriminator against later diagnostic selected-state dumps.
"""
        if rhs_trace_internal_apply:
            extra += """
The RHS trace patch also writes an internal `calculate_rhs` total-action trace
on the same post-normalization state:

```text
stellarator_gk_rhs_apply_00000020.dat
stellarator_gk_rhs_apply_00000800.dat
stellarator_gk_rhs_apply_00001600.dat
```

These files record the selected-row totals returned by GKW's own
`calculate_rhs(fdisi, rhs_internal)` call, and are intended to discriminate
RHS trace reconstruction from the matrix-vector product actually applied by
GKW.
"""
        if rhs_trace_igh_matrix_dump:
            extra += """
The RHS trace patch also writes compressed selected-row `igh_or_term_i` matrix
entries after GKW matrix construction/compression:

```text
stellarator_gk_igh_matrix_00000020.dat
stellarator_gk_igh_matrix_00000800.dat
stellarator_gk_igh_matrix_00001600.dat
```

Each row stores the compressed element number, raw row/column indices,
resolved selected-mode one-based row/column `(z, mu, vpar)` coordinates when
available, the trace term id, and the complex matrix coefficient before the
`dtim` factor applied by `calculate_rhs`.
"""
    readme = output_root / "README_stellarator_gk_cosin2.md"
    readme.write_text(
        f"""# Patched GKW `cosin2` Run

This directory was copied from `{DEFAULT_SOURCE_ROOT}` by
`scripts/prepare_gkw_cosine2_run.py`.

Only the copied `src/init.f90` was patched.  The original GKW tree is unchanged.
The selector is named `{PATCHED_SELECTOR}` because GKW declares `finit` as
`character(len = 6)`.

The generated `input.dat` uses:

```fortran
finit = '{PATCHED_SELECTOR}'
```
{extra}

Build and run from this directory with the same local serial/no-FFT settings
used for the matched selected-ky reference, for example:

```bash
make FC=gfortran FFLAGS="-fdefault-real-8 -O2" FFTLIB=nofft PARALLEL=nompi LDFLAGS=""
./gkw.x
```
""",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy GKW to an output directory and add a cosin2 initializer."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--multi-time-distr",
        action="store_true",
        help="Patch copied GKW diagnostics to write distr*_<ntotstep>.dat snapshots.",
    )
    parser.add_argument(
        "--state-trace",
        action="store_true",
        help="Patch copied GKW diagnostics to write stellarator_gk_state_trace.dat.",
    )
    parser.add_argument(
        "--selected-state-dump",
        action="store_true",
        help="Patch copied GKW diagnostics to write stellarator_gk_selected_state_<step>.dat.",
    )
    parser.add_argument(
        "--rhs-trace",
        action="store_true",
        help="Patch copied GKW to write selected-mode stellarator_gk_rhs_trace_<step>.dat.",
    )
    parser.add_argument(
        "--rhs-trace-state-dump",
        action="store_true",
        help=(
            "With --rhs-trace, also write same-timing "
            "stellarator_gk_rhs_state_<step>.dat selected states."
        ),
    )
    parser.add_argument(
        "--rhs-trace-internal-apply",
        action="store_true",
        help=(
            "With --rhs-trace, also write "
            "stellarator_gk_rhs_apply_<step>.dat calculate_rhs totals."
        ),
    )
    parser.add_argument(
        "--rhs-trace-igh-matrix-dump",
        action="store_true",
        help=(
            "With --rhs-trace, also write compressed selected-row "
            "stellarator_gk_igh_matrix_<step>.dat coefficient dumps."
        ),
    )
    parser.add_argument(
        "--rhs-trace-steps",
        default="20,800,1600",
        help="Comma-separated positive ntotstep values for --rhs-trace output.",
    )
    return parser.parse_args()


def _parse_step_list(value: str) -> tuple[int, ...]:
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not steps:
        raise argparse.ArgumentTypeError("at least one RHS trace step is required")
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("RHS trace steps must be positive")
    return steps


def main() -> None:
    args = _parse_args()
    prepared = prepare_gkw_cosine2_run(
        source_root=args.source_root,
        output_root=args.output_root,
        input_path=args.input,
        overwrite=args.overwrite,
        multi_time_distr=args.multi_time_distr,
        state_trace=args.state_trace,
        selected_state_dump=args.selected_state_dump,
        rhs_trace=args.rhs_trace,
        rhs_trace_state_dump=args.rhs_trace_state_dump,
        rhs_trace_internal_apply=args.rhs_trace_internal_apply,
        rhs_trace_igh_matrix_dump=args.rhs_trace_igh_matrix_dump,
        rhs_trace_steps=_parse_step_list(args.rhs_trace_steps),
    )
    print(f"prepared patched GKW tree: {prepared.output_root}")
    print(f"patched initializer: {prepared.patched_init}")
    if prepared.patched_diagnostic is not None:
        print(f"patched diagnostics: {prepared.patched_diagnostic}")
    print(f"patched input: {prepared.patched_input}")
    print("run from the patched tree with:")
    print(
        '  make FC=gfortran FFLAGS="-fdefault-real-8 -O2" '
        'FFTLIB=nofft PARALLEL=nompi LDFLAGS=""'
    )
    print("  ./gkw.x")


if __name__ == "__main__":
    main()
