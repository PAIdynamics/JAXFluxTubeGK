"""Validate stella's normalized field-particle driver coefficients in JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_field_particle_components import _read_values
from scripts.summarize_stella_collision_field_particle_factors import _axis_lookup
from scripts.summarize_stella_collision_field_particle_primitives import (
    PRIMITIVE_COLUMNS,
    QUADRATURE_COLUMNS,
    _set_consistent,
)
from jax_fluxtube_gk import (
    SpeciesParams,
    build_stella_laguerre_legendre_delta,
    build_stella_laguerre_legendre_driver,
    build_velocity_grid_from_nodes,
)


DRIVER_COLUMNS = (
    "iv",
    "imu",
    "iky",
    "ikx",
    "iz",
    "tube",
    "target",
    "background",
    "l",
    "m",
    "j",
    "vpa",
    "mu",
    "measure",
    "clm",
    "legendre",
    "gyroaverage",
    "delta_j",
    "maxwellian",
    "psijnorm",
    "sign",
    "driver",
)


def reconstruct_driver_with_local_builder(
    drivers: np.ndarray,
    primitives: np.ndarray,
    quadrature: np.ndarray,
) -> np.ndarray:
    """Build driver coefficients from grids, masses, and local recursive Delta_j."""

    axes = {
        "iv": _axis_lookup(drivers[:, 0]),
        "imu": _axis_lookup(drivers[:, 1]),
        "iky": _axis_lookup(drivers[:, 2]),
        "ikx": _axis_lookup(drivers[:, 3]),
        "iz": _axis_lookup(drivers[:, 4]),
        "target": _axis_lookup(drivers[:, 6]),
        "background": _axis_lookup(drivers[:, 7]),
    }
    labels = tuple(
        tuple(int(value) for value in label) for label in np.unique(drivers[:, 8:11], axis=0)
    )
    label_lookup = {label: index for index, label in enumerate(labels)}
    n_species = axes["target"][0].size
    n_vpa = axes["iv"][0].size
    n_mu = axes["imu"][0].size
    n_z = axes["iz"][0].size
    n_kx = axes["ikx"][0].size
    n_ky = axes["iky"][0].size
    maximum_order = max(abs(label[1]) for label in labels)
    vpar = np.full(n_vpa, np.nan)
    mu = np.full(n_mu, np.nan)
    magnetic_field = np.full(n_z, np.nan)
    measure = np.full((n_vpa, n_mu, n_z), np.nan)
    for row in quadrature:
        iv = axes["iv"][1][int(row[0])]
        imu = axes["imu"][1][int(row[1])]
        iz = axes["iz"][1][int(row[2])]
        _set_consistent(vpar, (iv,), row[3])
        _set_consistent(mu, (imu,), row[4])
        _set_consistent(magnetic_field, (iz,), row[5])
        _set_consistent(measure, (iv, imu, iz), row[6] * row[7])
    mass_factor = np.full((n_species, n_species), np.nan)
    for row in primitives:
        target = axes["target"][1][int(row[6])]
        background = axes["background"][1][int(row[7])]
        _set_consistent(mass_factor, (target, background), row[18])
    gyroaverage = np.full((n_species, maximum_order + 1, n_mu, n_z, n_kx, n_ky), np.nan)
    row_indices = []
    for row in drivers:
        target = axes["target"][1][int(row[6])]
        background = axes["background"][1][int(row[7])]
        label = tuple(int(value) for value in row[8:11])
        component = label_lookup[label]
        iv = axes["iv"][1][int(row[0])]
        imu = axes["imu"][1][int(row[1])]
        iky = axes["iky"][1][int(row[2])]
        ikx = axes["ikx"][1][int(row[3])]
        iz = axes["iz"][1][int(row[4])]
        _set_consistent(
            gyroaverage,
            (background, abs(label[1]), imu, iz, ikx, iky),
            row[16],
        )
        row_indices.append((target, background, component, iv, imu, iz, ikx, iky))
    arrays = (vpar, mu, magnetic_field, measure, mass_factor, gyroaverage)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("driver traces do not span a dense coefficient grid")
    masses = mass_factor[:, 0] ** (-2.0 / 3.0)
    species = tuple(SpeciesParams(1.0, mass, 1.0, 1.0, 0.0, 0.0) for mass in masses)
    grid = build_velocity_grid_from_nodes(
        vpar=vpar,
        mu=mu,
        w_vpar=np.ones_like(vpar),
        w_mu=np.ones_like(mu),
        backend="stella_trace",
    )
    delta = build_stella_laguerre_legendre_delta(
        grid,
        magnetic_field,
        species,
        measure,
        component_labels=labels,
    )
    driver = build_stella_laguerre_legendre_driver(
        grid,
        magnetic_field,
        species,
        measure,
        delta,
        gyroaverage,
        component_labels=labels,
    )
    return np.asarray([driver[index] for index in row_indices])


def summarize_driver_trace(
    driver_path: Path,
    primitive_path: Path,
    quadrature_path: Path,
    *,
    expected_revision: str,
    reconstruction_tolerance: float = 1.0e-9,
) -> dict[str, object]:
    """Require traced factorization and independent local driver parity."""

    drivers = _read_values(
        driver_path,
        columns=len(DRIVER_COLUMNS),
        schema="jax_fluxtube_gk_stella_collision_fieldpart_drivers_v1",
    )
    primitives = _read_values(
        primitive_path,
        columns=len(PRIMITIVE_COLUMNS),
        schema="jax_fluxtube_gk_stella_collision_fieldpart_primitives_v1",
    )
    quadrature = _read_values(
        quadrature_path,
        columns=len(QUADRATURE_COLUMNS),
        schema="jax_fluxtube_gk_stella_collision_velocity_quadrature_v1",
    )
    if not np.array_equal(drivers[:, :11], np.rint(drivers[:, :11])):
        raise ValueError("driver trace indices must be integers")
    native_driver = drivers[:, 21]
    traced_product = (
        np.prod(drivers[:, 13:18], axis=1) / drivers[:, 18] / drivers[:, 19] * drivers[:, 20]
    )
    scale = max(float(np.linalg.norm(native_driver)), 1.0)
    product_error = float(np.linalg.norm(traced_product - native_driver) / scale)
    if product_error > reconstruction_tolerance:
        raise ValueError(
            f"driver primitives do not reconstruct native coefficients: scaled L2={product_error:.6g}"
        )
    local_driver = reconstruct_driver_with_local_builder(drivers, primitives, quadrature)
    local_error = float(np.linalg.norm(local_driver - native_driver) / scale)
    if local_error > reconstruction_tolerance:
        raise ValueError(
            f"local normalized driver does not match stella: scaled L2={local_error:.6g}"
        )
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_field_particle_normalized_driver",
        "status": "local_normalized_driver_construction_passed",
        "stella_source_revision": expected_revision,
        "rows": int(drivers.shape[0]),
        "metrics": {
            "primitive_product_to_native_scaled_l2": product_error,
            "local_driver_to_native_scaled_l2": local_error,
            "native_driver_l2": float(np.linalg.norm(native_driver)),
        },
        "scope": "direct normalized moment driver; implicit response-system solve is separate",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drivers", type=Path, required=True)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--quadrature", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_driver_trace(
        args.drivers,
        args.primitives,
        args.quadrature,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
