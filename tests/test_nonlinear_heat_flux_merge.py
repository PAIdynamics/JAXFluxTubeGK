import json

import numpy as np
import pytest

from scripts.merge_nonlinear_heat_flux_segments import merge_nonlinear_heat_flux_segments


def _segment(path, start, stop, *, hyperdiffusion=0.05, schedule=None):
    times = np.linspace(start, stop, 101)
    payload = {
        "schema_version": 1,
        "producer": "jax-fluxtube-gk/nonlinear-heat-flux",
        "normalization": "jax_fluxtube_gk_native",
        "start_time": start,
        "end_time": stop,
        "trajectory_lineage": {
            "schema_version": 1,
            "seed": 17,
            "initial_amplitude": 1.0e-3,
            "initial_zonal_fraction": 0.0,
            "segment_end_times": [stop] if schedule is None else schedule,
        },
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
            "stationary_block_duration": 5.0,
            "min_stationary_blocks": 6,
        },
        "times": times.tolist(),
        "heat_flux": (5.0 + 0.02 * np.sin(times)).tolist(),
        "nonzonal_phi_rms": np.exp(1.0e-3 * times).tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merge_nonlinear_segments_recomputes_one_contiguous_window(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    second = _segment(tmp_path / "second.json", 40.0, 60.0, schedule=[40.0, 60.0])

    report = merge_nonlinear_heat_flux_segments((first, second), min_blocks=4)

    assert report["stationary"] is True
    assert report["start_time"] == pytest.approx(20.0)
    assert report["end_time"] == pytest.approx(60.0)
    assert len(report["times"]) == 201
    assert report["statistics"]["n_samples"] == 101
    assert report["candidate_nonzonal_phi_growth_rate"] == pytest.approx(1.0e-3)
    assert report["trajectory_lineage"]["segment_end_times"] == [40.0, 60.0]


def test_merge_nonlinear_segments_rejects_contract_changes_and_gaps(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    changed = _segment(
        tmp_path / "changed.json",
        40.0,
        60.0,
        hyperdiffusion=0.1,
        schedule=[40.0, 60.0],
    )
    gap = _segment(tmp_path / "gap.json", 41.0, 60.0, schedule=[40.0, 60.0])

    with pytest.raises(ValueError, match="contracts do not match"):
        merge_nonlinear_heat_flux_segments((first, changed))
    with pytest.raises(ValueError, match="contiguous"):
        merge_nonlinear_heat_flux_segments((first, gap))


def test_merge_nonlinear_segments_rejects_broken_lineage_and_trace_metadata(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    broken_lineage = _segment(tmp_path / "broken.json", 40.0, 60.0)
    with pytest.raises(ValueError, match="lineage schedules"):
        merge_nonlinear_heat_flux_segments((first, broken_lineage))

    wrong_endpoint = _segment(tmp_path / "wrong-endpoint.json", 40.0, 60.0, schedule=[40.0, 60.0])
    payload = json.loads(wrong_endpoint.read_text())
    payload["start_time"] = 39.0
    wrong_endpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trace endpoints"):
        merge_nonlinear_heat_flux_segments((first, wrong_endpoint))


def test_merge_nonlinear_segments_rejects_restart_diagnostic_discontinuity(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    second = _segment(tmp_path / "second.json", 40.0, 60.0, schedule=[40.0, 60.0])
    payload = json.loads(second.read_text())
    payload["heat_flux"][0] += 0.5
    second.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="heat flux is discontinuous"):
        merge_nonlinear_heat_flux_segments((first, second))


def test_merge_rejects_untrusted_producers_and_nonfinite_controls(tmp_path):
    segment = _segment(tmp_path / "segment.json", 20.0, 40.0)
    payload = json.loads(segment.read_text())
    payload["producer"] = "external-or-forged"
    segment.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="local producer"):
        merge_nonlinear_heat_flux_segments((segment,))

    local = _segment(tmp_path / "local.json", 20.0, 40.0)
    with pytest.raises(ValueError, match="finite and positive"):
        merge_nonlinear_heat_flux_segments((local,), max_relative_drift=float("nan"))


def test_merge_applies_amplitude_ratio_to_candidate_window(tmp_path):
    first = _segment(tmp_path / "first.json", 20.0, 40.0)
    second = _segment(tmp_path / "second.json", 40.0, 60.0, schedule=[40.0, 60.0])
    first_payload = json.loads(first.read_text())
    second_payload = json.loads(second.read_text())
    first_payload["nonzonal_phi_rms"] = np.linspace(1.0, 10.0, 101).tolist()
    second_payload["nonzonal_phi_rms"] = np.linspace(10.0, 7.0, 101).tolist()
    first.write_text(json.dumps(first_payload))
    second.write_text(json.dumps(second_payload))

    report = merge_nonlinear_heat_flux_segments((first, second), min_blocks=4)

    assert report["nonzonal_phi_rms_ratio"] > 0.8
    assert report["candidate_nonzonal_phi_rms_ratio"] == pytest.approx(0.7)
    assert abs(report["candidate_nonzonal_phi_growth_rate"]) < 0.02
    assert report["stationary"] is False
