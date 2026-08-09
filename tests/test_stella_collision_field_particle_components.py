from pathlib import Path

import pytest

from scripts.summarize_stella_collision_field_particle_components import (
    summarize_component_trace,
)


def _write_component_trace(path: Path, values: tuple[float, ...]) -> Path:
    rows = [
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_components_v1",
        "# iv imu iky ikx iz tube species l m j vpa mu before_re before_im rhs_re rhs_im",
    ]
    for index, value in enumerate(values):
        rows.append(f"1 1 1 1 0 1 1 0 0 {index} -1 0.5 2 0 {value} 0")
    path.write_text("\n".join(rows) + "\n")
    return path


def _write_aggregate_trace(path: Path, rhs: float) -> Path:
    path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_trace_v1\n"
        "# iv imu iky ikx iz tube species vpa mu before_re before_im rhs_re rhs_im\n"
        f"1 1 1 1 0 1 1 -1 0.5 2 0 {rhs} 0\n"
    )
    return path


def test_component_summary_reconstructs_aggregate_action(tmp_path):
    components = _write_component_trace(tmp_path / "components.dat", (0.25, 0.75))
    aggregate = _write_aggregate_trace(tmp_path / "aggregate.dat", 1.0)

    report = summarize_component_trace(
        components, aggregate, expected_revision="564ca09"
    )

    assert report["status"] == "native_component_reconstruction_passed"
    assert report["components_per_row"] == 2
    assert report["metrics"]["component_sum_to_aggregate_relative_l2"] == 0.0


def test_component_summary_rejects_failed_reconstruction(tmp_path):
    components = _write_component_trace(tmp_path / "components.dat", (0.25, 0.5))
    aggregate = _write_aggregate_trace(tmp_path / "aggregate.dat", 1.0)

    with pytest.raises(ValueError, match="do not reconstruct"):
        summarize_component_trace(
            components, aggregate, expected_revision="564ca09"
        )
