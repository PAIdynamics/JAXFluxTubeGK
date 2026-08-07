from pathlib import Path

import pytest

from scripts.summarize_stella_collision_field_particle_primitives import (
    summarize_primitive_trace,
)


def _write_primitive(path: Path, *, basis: float) -> Path:
    rows = "".join(
        f"{iv} {imu} 1 1 0 1 1 1 0 0 0 {vpa} {mu} 1.2 2 0.28209479177387814 1 1 1 1 1 {basis}\n"
        for iv, vpa in ((1, -1), (2, 1))
        for imu, mu in ((1, 0.5), (2, 1.0))
    )
    path.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_primitives_v1\n"
        "# iv imu iky ikx iz tube target background l m j vpa mu bmag "
        "frequency clm legendre gyroaverage mass_factor delta_j sign basis\n" + rows
    )
    return path


def test_primitive_summary_reconstructs_local_response(tmp_path):
    report = summarize_primitive_trace(
        _write_primitive(tmp_path / "primitives.dat", basis=0.5641895835477563),
        expected_revision="564ca09",
    )

    assert report["status"] == "local_response_construction_passed"
    assert report["metrics"]["primitive_product_to_native_relative_l2"] == 0.0
    assert report["metrics"]["local_builder_to_native_relative_l2"] == 0.0


def test_primitive_summary_rejects_invalid_product(tmp_path):
    with pytest.raises(ValueError, match="do not reconstruct native"):
        summarize_primitive_trace(
            _write_primitive(tmp_path / "primitives.dat", basis=0.5),
            expected_revision="564ca09",
        )
