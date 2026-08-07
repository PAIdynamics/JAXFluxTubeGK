from pathlib import Path

import pytest

from scripts.summarize_stella_collision_field_particle_factors import (
    summarize_factor_trace,
)


def _write_factor(path: Path, *, rhs: float) -> Path:
    path.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_factors_v1\n"
        "# iv imu iky ikx iz tube target background l m j vpa mu psi_re psi_im basis rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 1 0 0 1 -1 0.5 0.5 0 2 {rhs} 0\n"
    )
    return path


def _write_aggregate(path: Path, *, rhs: float) -> Path:
    path.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_trace_v1\n"
        "# iv imu iky ikx iz tube species vpa mu before_re before_im rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 -1 0.5 3 0 {rhs} 0\n"
    )
    return path


def test_factor_summary_reconstructs_product_and_aggregate(tmp_path):
    report = summarize_factor_trace(
        _write_factor(tmp_path / "factors.dat", rhs=1.0),
        _write_aggregate(tmp_path / "aggregate.dat", rhs=1.0),
        expected_revision="564ca09",
    )

    assert report["status"] == "native_low_rank_factorization_passed"
    assert report["factors_per_row"] == 1
    assert report["metrics"]["psi_basis_to_factor_relative_l2"] == 0.0
    assert report["metrics"]["factor_sum_to_aggregate_relative_l2"] == 0.0
    assert report["metrics"]["local_jax_replay_to_native_relative_l2"] == 0.0


def test_factor_summary_rejects_invalid_product(tmp_path):
    with pytest.raises(ValueError, match="do not reconstruct factor"):
        summarize_factor_trace(
            _write_factor(tmp_path / "factors.dat", rhs=0.5),
            _write_aggregate(tmp_path / "aggregate.dat", rhs=0.5),
            expected_revision="564ca09",
        )
