from pathlib import Path

import pytest

from scripts.run_stella_collision_channel_traces import CHANNELS, summarize_channel_traces


def _write_trace(path: Path, before: float, rhs: float) -> Path:
    path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_trace_v1\n"
        "# iv imu iky ikx iz tube species vpa mu before_re before_im rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 -1.0 0.5 {before} 0.0 {rhs} 0.0\n"
    )
    return path


def _write_components(path: Path, before: float, rhs: float) -> Path:
    path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_components_v1\n"
        "# iv imu iky ikx iz tube species l m j vpa mu before_re before_im rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 0 0 1 -1.0 0.5 {before} 0.0 {rhs} 0.0\n"
    )
    return path


def _write_factors(path: Path, rhs: float) -> Path:
    path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_factors_v1\n"
        "# iv imu iky ikx iz tube target background l m j vpa mu psi_re psi_im basis rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 1 0 0 1 -1.0 0.5 {rhs} 0.0 1.0 {rhs} 0.0\n"
    )
    return path


def _write_matrix(path: Path, value: float) -> Path:
    path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_test_particle_matrix_v1\n"
        "# iky ikx iz species row col matrix_re matrix_im kperp2 code_dt\n"
        f"1 1 0 1 1 1 {value} 0.0 0.0 0.01\n"
    )
    return path


def test_pair_resolved_summary_requires_common_input_and_reports_closure(tmp_path):
    rhs = {
        "all": 1.0,
        "ion_ion": 0.1,
        "ion_electron": 0.2,
        "electron_electron": 0.3,
        "electron_ion": 0.4,
    }
    paths = {
        name: _write_trace(tmp_path / f"{name}.dat", 2.0, rhs[name]) for name in CHANNELS
    }

    report = summarize_channel_traces(paths, expected_revision="564ca09")

    assert report["status"] == "pair_resolved_native_traces_passed"
    assert report["metrics"]["identical_input_state"]
    assert report["metrics"]["isolated_sum_to_full_relative_l2"] == pytest.approx(0.0)


def test_pair_resolved_summary_rejects_input_mismatch(tmp_path):
    paths = {
        name: _write_trace(tmp_path / f"{name}.dat", 2.0, 0.2) for name in CHANNELS
    }
    _write_trace(paths["electron_ion"], 3.0, 0.2)

    with pytest.raises(ValueError, match="identical input"):
        summarize_channel_traces(paths, expected_revision="564ca09")


def test_pair_resolved_summary_validates_each_component_trace(tmp_path):
    rhs = {
        "all": 1.0,
        "ion_ion": 0.1,
        "ion_electron": 0.2,
        "electron_electron": 0.3,
        "electron_ion": 0.4,
    }
    paths = {
        name: _write_trace(tmp_path / f"{name}.dat", 2.0, rhs[name]) for name in CHANNELS
    }
    components = {
        name: _write_components(tmp_path / f"{name}_components.dat", 2.0, rhs[name])
        for name in CHANNELS
    }
    factors = {
        name: _write_factors(tmp_path / f"{name}_factors.dat", rhs[name])
        for name in CHANNELS
    }

    report = summarize_channel_traces(
        paths,
        expected_revision="564ca09",
        component_paths=components,
        factor_paths=factors,
    )

    assert report["metrics"]["component_reconstruction_passed"]
    assert set(report["component_channels"]) == set(CHANNELS)
    assert report["metrics"]["local_jax_factor_replay_passed"]
    assert set(report["factor_channels"]) == set(CHANNELS)


def test_pair_resolved_summary_validates_test_particle_matrix_closure(tmp_path):
    effects = {
        "ion_ion": 0.1,
        "ion_electron": 0.2,
        "electron_electron": 0.3,
        "electron_ion": 0.4,
    }
    paths = {
        name: _write_trace(tmp_path / f"{name}.dat", 2.0, 0.25) for name in CHANNELS
    }
    matrices = {
        "all": _write_matrix(tmp_path / "all_matrix.dat", 1.0 + sum(effects.values())),
        **{
            name: _write_matrix(tmp_path / f"{name}_matrix.dat", 1.0 + effect)
            for name, effect in effects.items()
        },
    }

    report = summarize_channel_traces(
        paths,
        expected_revision="564ca09",
        matrix_paths=matrices,
    )

    assert report["metrics"]["test_particle_matrix_channel_decomposition_passed"]
    assert report["metrics"]["test_particle_matrix_isolated_sum_relative_l2"] == pytest.approx(
        0.0
    )
    assert set(report["test_particle_matrix_channels"]) == set(effects)

    _write_matrix(matrices["electron_ion"], 1.5)
    with pytest.raises(ValueError, match="do not reconstruct"):
        summarize_channel_traces(
            paths,
            expected_revision="564ca09",
            matrix_paths=matrices,
        )
