import json

import numpy as np
import pytest

from scripts.merge_nonlinear_heat_flux_segments import merge_nonlinear_heat_flux_segments


def _segment(path, start, stop, *, hyperdiffusion=0.05):
    times = np.linspace(start, stop, 101)
    payload = {
        "schema_version": 1,
        "producer": "optimal-fusion/nonlinear-heat-flux",
        "normalization": "optimal_fusion_native",
        "case": {
            "n_z": 12,
            "n_vpar": 12,
            "n_mu": 6,
            "n_kx": 9,
            "n_ky": 5,
            "kx": list(np.linspace(-2.0, 2.0, 9)),
            "ky": list(np.linspace(0.0, 0.4, 5)),
            "ikxspace": 1,
            "parallel_boundary_model": "twist_shift",
            "parallel_recurrence_rate": 1.0,
            "rmaj_over_lref": 2.77778,
            "gx_fprim": 0.8,
            "gx_tprim": 2.49,
            "density_gradient_R_over_Ln": 2.222224,
            "temperature_gradient_R_over_LT": 6.9166722,
            "hyperdiffusion": hyperdiffusion,
            "collision_frequency": 0.0,
            "flux_moment": "nonadvective_heat",
        },
        "times": times.tolist(),
        "heat_flux": (5.0 + 0.02 * np.sin(times)).tolist(),
        "nonzonal_phi_rms": np.exp(1.0e-3 * times).tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merge_nonlinear_segments_recomputes_one_contiguous_window(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    second = _segment(tmp_path / "second.json", 40.0, 60.0)

    report = merge_nonlinear_heat_flux_segments((first, second))

    assert report["stationary"] is True
    assert report["start_time"] == pytest.approx(20.0)
    assert report["end_time"] == pytest.approx(60.0)
    assert len(report["times"]) == 201
    assert report["statistics"]["n_samples"] == 101
    assert report["candidate_nonzonal_phi_growth_rate"] == pytest.approx(1.0e-3)


def test_merge_nonlinear_segments_rejects_contract_changes_and_gaps(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    changed = _segment(tmp_path / "changed.json", 40.0, 60.0, hyperdiffusion=0.1)
    gap = _segment(tmp_path / "gap.json", 41.0, 60.0)

    with pytest.raises(ValueError, match="contracts do not match"):
        merge_nonlinear_heat_flux_segments((first, changed))
    with pytest.raises(ValueError, match="contiguous"):
        merge_nonlinear_heat_flux_segments((first, gap))
