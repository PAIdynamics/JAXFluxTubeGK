from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_gkw_cosine2_run import (
    PATCHED_SELECTOR,
    add_cosin2_branch,
    add_multitime_velocity_slice_patch,
    prepare_gkw_cosine2_run,
    write_cosin2_input,
)
from stellarator_gk import (
    GkwVelocitySpaceSliceSeries,
    load_gkw_parallel_phi_trace,
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

    assert "finit = 'cosin2'" in input_text
    assert time_trace.times.shape == (80,)
    assert phi_trace.phi_power.shape == (80, 48)
    assert np.isclose(float(time_trace.window_growth[-1]), 0.188741)


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
