from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_stella_collision_field_particle_drivers import (
    summarize_driver_trace,
)
from jax_fluxtube_gk import (
    SpeciesParams,
    build_stella_laguerre_legendre_delta,
    build_stella_laguerre_legendre_driver,
    build_velocity_grid_from_nodes,
)


def _write_driver_fixture(root: Path, *, driver_scale: float = 1.0):
    vpar = np.asarray((-1.0, 1.0))
    mu = np.asarray((0.5, 1.0))
    magnetic_field = np.asarray((1.2,))
    measure = np.full((2, 2, 1), 0.5)
    labels = ((1, 0, 0),)
    species = SpeciesParams(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    grid = build_velocity_grid_from_nodes(
        vpar=vpar,
        mu=mu,
        w_vpar=np.ones(2),
        w_mu=np.ones(2),
        backend="fixture",
    )
    delta = build_stella_laguerre_legendre_delta(
        grid,
        magnetic_field,
        species,
        measure,
        component_labels=labels,
    )
    gyroaverage = np.ones((1, 1, 2, 1, 1, 1))
    driver = build_stella_laguerre_legendre_driver(
        grid,
        magnetic_field,
        species,
        measure,
        delta,
        gyroaverage,
        component_labels=labels,
    )
    clm = np.sqrt(3.0 / (4.0 * np.pi))
    driver_rows = []
    primitive_rows = []
    quadrature_rows = []
    for iv, parallel_velocity in enumerate(vpar, start=1):
        for imu, magnetic_moment in enumerate(mu, start=1):
            speed = np.sqrt(parallel_velocity**2 + 2.0 * 1.2 * magnetic_moment)
            legendre = parallel_velocity / speed
            delta_value = float(delta[0, 0, 0, iv - 1, imu - 1, 0])
            driver_value = float(driver[0, 0, 0, iv - 1, imu - 1, 0, 0, 0])
            maxwellian = np.exp(-(speed**2))
            psijnorm = 0.5 * clm * legendre * delta_value / maxwellian / driver_value
            driver_rows.append(
                f"{iv} {imu} 1 1 0 1 1 1 1 0 0 {parallel_velocity} "
                f"{magnetic_moment} 0.5 {clm} {legendre} 1 {delta_value} "
                f"{maxwellian} {psijnorm} 1 {driver_scale * driver_value}\n"
            )
            primitive_rows.append(
                f"{iv} {imu} 1 1 0 1 1 1 1 0 0 {parallel_velocity} "
                f"{magnetic_moment} 1.2 1 1 1 1 1 1 1 1\n"
            )
            quadrature_rows.append(
                f"{iv} {imu} 0 {parallel_velocity} {magnetic_moment} 1.2 0.5 1\n"
            )
    driver_path = root / "drivers.dat"
    driver_path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_drivers_v1\n"
        "# iv imu iky ikx iz tube target background l m j vpa mu measure clm "
        "legendre gyroaverage delta_j maxwellian psijnorm sign driver\n" + "".join(driver_rows)
    )
    primitive_path = root / "primitives.dat"
    primitive_path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_fieldpart_primitives_v1\n"
        "# columns\n" + "".join(primitive_rows)
    )
    quadrature_path = root / "quadrature.dat"
    quadrature_path.write_text(
        "# schema=jax_fluxtube_gk_stella_collision_velocity_quadrature_v1\n"
        "# columns\n" + "".join(quadrature_rows)
    )
    return driver_path, primitive_path, quadrature_path


def test_driver_summary_reconstructs_normalized_coefficients(tmp_path):
    paths = _write_driver_fixture(tmp_path)
    report = summarize_driver_trace(*paths, expected_revision="564ca09")

    assert report["status"] == "local_normalized_driver_construction_passed"
    assert report["metrics"]["local_driver_to_native_scaled_l2"] < 1.0e-12


def test_driver_summary_rejects_invalid_primitive_product(tmp_path):
    paths = _write_driver_fixture(tmp_path, driver_scale=0.5)
    with pytest.raises(ValueError, match="do not reconstruct native"):
        summarize_driver_trace(*paths, expected_revision="564ca09")
