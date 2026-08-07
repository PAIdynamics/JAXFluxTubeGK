import numpy as np
import pytest
from types import SimpleNamespace

from examples.run_nonlinear_heat_flux import (
    _hyperdiffusion,
    _initial_state,
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
    assert args.initial_zonal_fraction == pytest.approx(0.0)
    assert args.gx_fprim * args.rmaj_over_lref == pytest.approx(2.222224)
    assert args.gx_tprim * args.rmaj_over_lref == pytest.approx(6.9166722)
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


def test_initial_state_does_not_seed_zonal_potential_by_default():
    precompute = SimpleNamespace(
        n_species=1,
        rhs=SimpleNamespace(
            maxwellian=np.ones((1, 2, 2, 3)),
            flr_factors=SimpleNamespace(bessel_j0=np.ones((1, 2, 2, 3, 3, 2))),
        ),
    )

    state = _initial_state(precompute, 1.0e-3, 17)

    np.testing.assert_allclose(np.asarray(state[..., 0]), 0.0, atol=0.0)
    assert np.max(np.abs(np.asarray(state[..., 1]))) > 0.0
