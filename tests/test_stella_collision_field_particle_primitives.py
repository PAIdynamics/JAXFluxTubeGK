from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_stella_collision_field_particle_primitives import (
    summarize_primitive_trace,
)
from stellarator_gk import stella_laguerre_legendre_delta0


def _write_primitive(path: Path, *, basis_scale: float = 1.0) -> Path:
    rows = "".join(
        f"{iv} {imu} 1 1 0 1 1 1 0 0 0 {vpa} {mu} 1.2 "
        f"2 0.28209479177387814 1 1 1 {delta} 1 "
        f"{basis_scale * 2 * 0.28209479177387814 * delta}\n"
        for iv, vpa in ((1, -1), (2, 1))
        for imu, mu in ((1, 0.5), (2, 1.0))
        for delta in (
            float(
                stella_laguerre_legendre_delta0(
                    np.sqrt(vpa**2 + 2 * 1.2 * mu),
                    1.0,
                    1.0,
                    laguerre_degree=0,
                    legendre_degree=0,
                )
            ),
        )
    )
    path.write_text(
        "# schema=stellarator_gk_stella_collision_fieldpart_primitives_v1\n"
        "# iv imu iky ikx iz tube target background l m j vpa mu bmag "
        "frequency clm legendre gyroaverage mass_factor delta_j sign basis\n" + rows
    )
    return path


def _write_quadrature(path: Path) -> Path:
    rows = "".join(
        f"{iv} {imu} 0 {vpa} {mu} 1.2 0.5 1\n"
        for iv, vpa in ((1, -1), (2, 1))
        for imu, mu in ((1, 0.5), (2, 1.0))
    )
    path.write_text(
        "# schema=stellarator_gk_stella_collision_velocity_quadrature_v1\n"
        "# iv imu iz vpa mu bmag w_vpa w_mu\n" + rows
    )
    return path


def test_primitive_summary_reconstructs_local_response(tmp_path):
    report = summarize_primitive_trace(
        _write_primitive(tmp_path / "primitives.dat"),
        expected_revision="564ca09",
    )

    assert report["status"] == "local_response_and_delta0_construction_passed"
    assert report["metrics"]["primitive_product_to_native_relative_l2"] == 0.0
    assert report["metrics"]["local_builder_to_native_relative_l2"] == 0.0
    assert report["metrics"]["local_delta0_to_native_relative_l2"] < 1.0e-14


def test_primitive_summary_reconstructs_recursive_delta_with_quadrature(tmp_path):
    report = summarize_primitive_trace(
        _write_primitive(tmp_path / "primitives.dat"),
        quadrature_path=_write_quadrature(tmp_path / "quadrature.dat"),
        expected_revision="564ca09",
    )

    assert report["status"] == "local_response_and_recursive_delta_construction_passed"
    assert report["metrics"]["local_recursive_delta_to_native_scaled_l2"] < 1.0e-14


def test_primitive_summary_rejects_invalid_product(tmp_path):
    with pytest.raises(ValueError, match="do not reconstruct native"):
        summarize_primitive_trace(
            _write_primitive(tmp_path / "primitives.dat", basis_scale=0.5),
            expected_revision="564ca09",
        )
