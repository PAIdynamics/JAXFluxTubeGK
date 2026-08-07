import numpy as np
import pytest

from examples.run_nonlinear_heat_flux import (
    _hyperdiffusion,
    _parse_args,
    _phi_rms_diagnostics,
)
from stellarator_gk import FourierGridSpec, build_fourier_grid


def test_nonlinear_heat_flux_runner_defaults_and_hyperdiffusion(tmp_path):
    args = _parse_args(["--output", str(tmp_path / "result.json")])
    grid = build_fourier_grid(
        FourierGridSpec(n_kx=5, n_ky=3, kx_max=0.8, ky_values=(0.0, 0.1, 0.2))
    )
    damping = _hyperdiffusion(grid, 0.05)

    assert args.n_kx == 9
    assert args.n_ky == 5
    assert args.min_phi_rms_ratio == pytest.approx(0.8)
    assert damping.shape == (5, 3)
    assert damping[grid.ixzero, 0] == pytest.approx(0.0)
    assert np.all(np.asarray(damping) >= 0.0)


def test_nonlinear_heat_flux_runner_rejects_even_kx(tmp_path):
    with pytest.raises(SystemExit):
        _parse_args(["--output", str(tmp_path / "result.json"), "--n-kx", "4"])


def test_phi_rms_diagnostics_separates_zonal_and_nonzonal_amplitudes():
    phi = np.ones((2, 3, 2, 3), dtype=np.complex128)
    phi[1, ..., 0] = 10.0
    phi[1, ..., 1:] = 0.5

    diagnostics = _phi_rms_diagnostics(phi)

    assert float(diagnostics["nonzonal_phi_rms_ratio"]) == pytest.approx(0.5)
    assert np.asarray(diagnostics["phi_rms_ratio_by_ky"]).tolist() == pytest.approx(
        [10.0, 0.5, 0.5]
    )
    assert float(diagnostics["phi_rms_ratio"]) > 1.0
