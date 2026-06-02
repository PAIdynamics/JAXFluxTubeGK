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
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite")
        shutil.rmtree(output_root)

    shutil.copytree(source_root, output_root, ignore=_ignore_generated_gkw_files)
    patched_init = output_root / "src" / "init.f90"
    add_cosin2_branch(patched_init)
    patched_diagnostic = None
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
        or name.endswith(suffixes)
        or name == "__pycache__"
    }


def _write_run_readme(
    output_root: Path,
    *,
    multi_time_distr: bool = False,
    state_trace: bool = False,
    selected_state_dump: bool = False,
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
    return parser.parse_args()


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
