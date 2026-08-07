"""Validate native field-particle response primitives with the local JAX builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_field_particle_components import _read_values
from scripts.summarize_stella_collision_field_particle_factors import _axis_lookup
from stellarator_gk import (
    SpeciesParams,
    build_stella_laguerre_legendre_response,
    build_velocity_grid_from_nodes,
    stella_laguerre_legendre_delta0,
)


PRIMITIVE_COLUMNS = (
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
    "bmag",
    "frequency",
    "clm",
    "legendre",
    "gyroaverage",
    "mass_factor",
    "delta_j",
    "sign",
    "basis",
)


def _set_consistent(array: np.ndarray, index: tuple[int, ...], value: float) -> None:
    previous = array[index]
    if np.isfinite(previous) and not np.isclose(previous, value, rtol=2e-13, atol=2e-13):
        raise ValueError(f"primitive value varies at coefficient index {index}")
    array[index] = value


def reconstruct_response_with_local_builder(rows: np.ndarray) -> np.ndarray:
    """Reconstruct every traced response basis from physical primitive arrays."""

    axes = {
        "iv": _axis_lookup(rows[:, 0]),
        "imu": _axis_lookup(rows[:, 1]),
        "iky": _axis_lookup(rows[:, 2]),
        "ikx": _axis_lookup(rows[:, 3]),
        "iz": _axis_lookup(rows[:, 4]),
        "target": _axis_lookup(rows[:, 6]),
        "background": _axis_lookup(rows[:, 7]),
    }
    n_species = axes["target"][0].size
    if axes["background"][0].size != n_species:
        raise ValueError("target and background species axes differ")
    labels = tuple(
        tuple(int(value) for value in label) for label in np.unique(rows[:, 8:11], axis=0)
    )
    label_lookup = {label: index for index, label in enumerate(labels)}
    n_vpa = axes["iv"][0].size
    n_mu = axes["imu"][0].size
    n_z = axes["iz"][0].size
    n_kx = axes["ikx"][0].size
    n_ky = axes["iky"][0].size
    maximum_order = max(abs(label[1]) for label in labels)

    vpar = np.full(n_vpa, np.nan)
    mu = np.full(n_mu, np.nan)
    magnetic_field = np.full(n_z, np.nan)
    frequency = np.full((n_species, n_species), np.nan)
    mass_factor = np.full((n_species, n_species), np.nan)
    delta = np.full((n_species, n_species, len(labels), n_vpa, n_mu, n_z), np.nan)
    gyroaverage = np.full((n_species, maximum_order + 1, n_mu, n_z, n_kx, n_ky), np.nan)

    row_indices = []
    for row in rows:
        iv = axes["iv"][1][int(row[0])]
        imu = axes["imu"][1][int(row[1])]
        iky = axes["iky"][1][int(row[2])]
        ikx = axes["ikx"][1][int(row[3])]
        iz = axes["iz"][1][int(row[4])]
        target = axes["target"][1][int(row[6])]
        background = axes["background"][1][int(row[7])]
        label = tuple(int(value) for value in row[8:11])
        component = label_lookup[label]
        _set_consistent(vpar, (iv,), row[11])
        _set_consistent(mu, (imu,), row[12])
        _set_consistent(magnetic_field, (iz,), row[13])
        _set_consistent(frequency, (target, background), row[14])
        _set_consistent(mass_factor, (target, background), row[18])
        _set_consistent(delta, (target, background, component, iv, imu, iz), row[19])
        _set_consistent(
            gyroaverage,
            (target, abs(label[1]), imu, iz, ikx, iky),
            row[17],
        )
        row_indices.append((target, background, component, iv, imu, iz, ikx, iky))

    arrays = (vpar, mu, magnetic_field, frequency, mass_factor, delta, gyroaverage)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("primitive trace does not span a dense coefficient grid")
    masses = mass_factor[:, 0] ** (-2.0 / 3.0)
    species = tuple(SpeciesParams(1.0, mass, 1.0, 1.0, 0.0, 0.0) for mass in masses)
    grid = build_velocity_grid_from_nodes(
        vpar=vpar,
        mu=mu,
        w_vpar=np.ones_like(vpar),
        w_mu=np.ones_like(mu),
        backend="stella_trace",
    )
    response = build_stella_laguerre_legendre_response(
        grid,
        magnetic_field,
        species,
        frequency,
        delta,
        gyroaverage,
        component_labels=labels,
    )
    return np.asarray([response[index] for index in row_indices])


def summarize_primitive_trace(
    primitive_path: Path,
    *,
    expected_revision: str,
    reconstruction_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Require primitive products and local response construction to match stella."""

    rows = _read_values(
        primitive_path,
        columns=len(PRIMITIVE_COLUMNS),
        schema="stellarator_gk_stella_collision_fieldpart_primitives_v1",
    )
    if not np.array_equal(rows[:, :11], np.rint(rows[:, :11])):
        raise ValueError("primitive trace indices must be integers")
    if np.unique(rows[:, :11], axis=0).shape[0] != rows.shape[0]:
        raise ValueError("primitive trace contains duplicate coefficient rows")
    basis = rows[:, 21]
    primitive_product = np.prod(rows[:, 14:21], axis=1)
    scale = max(float(np.linalg.norm(basis)), np.finfo(float).tiny)
    product_error = float(np.linalg.norm(primitive_product - basis) / scale)
    if product_error > reconstruction_tolerance:
        raise ValueError(
            "primitive factors do not reconstruct native response basis: "
            f"relative L2={product_error:.6g}"
        )
    local_basis = reconstruct_response_with_local_builder(rows)
    local_error = float(np.linalg.norm(local_basis - basis) / scale)
    if local_error > reconstruction_tolerance:
        raise ValueError(
            "local response builder does not match native response basis: "
            f"relative L2={local_error:.6g}"
        )
    delta0_rows = rows[:, 10] == 0
    selected_rows = rows[delta0_rows]
    native_delta0 = selected_rows[:, 19]
    local_delta0 = np.empty_like(native_delta0)
    speed = np.sqrt(selected_rows[:, 11] ** 2 + 2.0 * selected_rows[:, 13] * selected_rows[:, 12])
    mass_ratio = selected_rows[:, 18] ** (-2.0 / 3.0)
    for degree in np.unique(selected_rows[:, 8]).astype(int):
        selected = selected_rows[:, 8] == degree
        local_delta0[selected] = np.asarray(
            stella_laguerre_legendre_delta0(
                speed[selected],
                mass_ratio[selected],
                np.ones(np.count_nonzero(selected)),
                laguerre_degree=0,
                legendre_degree=degree,
            )
        )
    delta0_scale = max(float(np.linalg.norm(native_delta0)), 1.0)
    delta0_error = float(np.linalg.norm(local_delta0 - native_delta0) / delta0_scale)
    if delta0_error > reconstruction_tolerance:
        raise ValueError(
            f"local analytic delta0 does not match native delta_j: relative L2={delta0_error:.6g}"
        )
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_field_particle_response_primitives",
        "status": "local_response_and_delta0_construction_passed",
        "stella_source_revision": expected_revision,
        "primitive_trace": str(Path(primitive_path).resolve()),
        "rows": int(rows.shape[0]),
        "metrics": {
            "primitive_product_to_native_relative_l2": product_error,
            "local_builder_to_native_relative_l2": local_error,
            "local_delta0_to_native_relative_l2": delta0_error,
            "native_basis_l2": float(np.linalg.norm(basis)),
        },
        "scope": "response and analytic delta0 construction; higher delta_j and gyroaverage are native inputs",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_primitive_trace(
        args.primitives,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
