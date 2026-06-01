from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_gkw_cosine2_run import (
    PATCHED_SELECTOR,
    add_cosin2_branch,
    prepare_gkw_cosine2_run,
    write_cosin2_input,
)
from stellarator_gk import load_gkw_parallel_phi_trace, load_gkw_time_dat_trace


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


def test_prepare_gkw_cosine2_run_patches_only_the_copy(tmp_path: Path) -> None:
    source_root = tmp_path / "gkw"
    (source_root / "src").mkdir(parents=True)
    source_init = source_root / "src" / "init.f90"
    source_init.write_text(_MINIMAL_INIT, encoding="utf-8")
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
    assert "finit = 'cosin2'" in prepared.patched_input.read_text(encoding="utf-8")
    assert (prepared.output_root / "README_stellarator_gk_cosin2.md").is_file()


def test_cosin2_patch_is_idempotent(tmp_path: Path) -> None:
    init_path = tmp_path / "init.f90"
    init_path.write_text(_MINIMAL_INIT, encoding="utf-8")

    assert add_cosin2_branch(init_path)
    first_patch = init_path.read_text(encoding="utf-8")
    assert not add_cosin2_branch(init_path)
    assert init_path.read_text(encoding="utf-8") == first_patch


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
