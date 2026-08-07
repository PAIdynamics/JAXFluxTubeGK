import numpy as np
import pytest
from types import SimpleNamespace

from examples.run_nonlinear_heat_flux import (
    _candidate_window_phi_growth,
    _checkpoint_contract,
    _hyperdiffusion,
    _initial_state,
    _load_checkpoint,
    _nonzonal_phi_rms_history,
    _parse_args,
    _phi_rms_diagnostics,
    _require_x64,
    _write_checkpoint,
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
    assert args.collision_frequency == pytest.approx(0.0)
    assert args.parallel_boundary_model == "twist_shift"
    assert args.ikxspace == 1
    assert args.min_stationary_samples == 100
    assert args.min_stationary_window_duration == pytest.approx(10.0)
    assert args.stationary_block_duration == pytest.approx(5.0)
    assert args.min_stationary_blocks == 6
    assert args.max_absolute_phi_growth_rate == pytest.approx(0.02)
    assert args.flux_moment == "nonadvective_heat"
    assert args.gx_fprim * args.rmaj_over_lref == pytest.approx(2.222224)
    assert args.gx_tprim * args.rmaj_over_lref == pytest.approx(6.9166722)
    assert damping.shape == (5, 3)
    assert damping[grid.ixzero, 0] == pytest.approx(0.0)
    assert np.all(np.asarray(damping) >= 0.0)


def test_nonlinear_heat_flux_runner_rejects_even_kx(tmp_path):
    with pytest.raises(SystemExit):
        _parse_args(["--output", str(tmp_path / "result.json"), "--n-kx", "4"])


def test_nonlinear_heat_flux_runner_requires_x64():
    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64=1"):
        _require_x64(False)
    _require_x64(True)


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


def test_candidate_window_phi_growth_recovers_exponential_rate():
    times = np.linspace(0.0, 10.0, 21)
    phi = np.ones((21, 2, 3, 2), dtype=np.complex128)
    phi[..., 0] = 20.0
    phi[..., 1] *= np.exp(0.07 * times)[:, None, None]

    diagnostics = _candidate_window_phi_growth(phi, times, 0.5)

    assert float(diagnostics["candidate_nonzonal_phi_growth_rate"]) == pytest.approx(0.07)
    assert float(diagnostics["candidate_nonzonal_phi_rms_ratio"]) == pytest.approx(
        np.exp(0.07 * 5.0)
    )
    np.testing.assert_allclose(
        _nonzonal_phi_rms_history(phi),
        np.exp(0.07 * times),
        rtol=2.0e-13,
    )


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


def test_nonlinear_checkpoint_roundtrip_and_contract_guard(tmp_path):
    args = _parse_args(["--output", str(tmp_path / "report.json")])
    grid = build_fourier_grid(FourierGridSpec(n_kx=3, n_ky=2, kx_max=0.6, ky_values=(0.0, 0.1)))
    state = np.arange(12).reshape(2, 3, 2).astype(np.complex128) * (1.0 + 0.5j)
    contract = _checkpoint_contract(args, grid, state.shape)
    lineage = {
        "schema_version": 1,
        "seed": 17,
        "initial_amplitude": 1.0e-3,
        "initial_zonal_fraction": 0.0,
        "segment_end_times": [3.5],
    }
    path = tmp_path / "restart.npz"

    _write_checkpoint(path, state, 3.5, contract, lineage)
    restored, time, restored_lineage = _load_checkpoint(path, contract)

    np.testing.assert_array_equal(np.asarray(restored), state)
    assert time == pytest.approx(3.5)
    assert restored_lineage == lineage
    assert contract["state_dtype"] == "complex128"
    with pytest.raises(ValueError, match="contract does not match"):
        _load_checkpoint(path, contract | {"hyperdiffusion": 0.2})
