"""Validate the local stella gyro-diffusion diagonal against native matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_test_particle_matrix import (
    COLUMNS as MATRIX_COLUMNS,
    SCHEMA as MATRIX_SCHEMA,
    _dense_matrices,
)
from scripts.summarize_stella_collision_test_particle_primitives import (
    COLUMNS as PRIMITIVE_COLUMNS,
    SCHEMA as PRIMITIVE_SCHEMA,
)
from stellarator_gk import (
    SpeciesParams,
    build_stella_test_particle_gyro_diagonal,
    build_stella_test_particle_primitives,
    build_velocity_grid_from_nodes,
)


def _read(path: Path, schema: str, columns: int) -> np.ndarray:
    path = Path(path)
    header = path.read_text(encoding="utf-8").splitlines()[:1]
    if not header or header[0].strip() != schema:
        raise ValueError(f"unsupported or missing trace schema in {path}")
    values = np.atleast_2d(np.loadtxt(path))
    if values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f"invalid trace values in {path}")
    return values


def summarize_gyro_diagonal(
    primitive_trace: Path,
    matrix_trace: Path,
    *,
    expected_revision: str,
    absolute_tolerance: float = 2.0e-12,
) -> dict[str, object]:
    """Compare local per-``kperp2`` diagonals with every native matrix delta."""

    primitive = _read(primitive_trace, PRIMITIVE_SCHEMA, len(PRIMITIVE_COLUMNS))
    matrix_values = _read(matrix_trace, MATRIX_SCHEMA, len(MATRIX_COLUMNS))
    matrices = _dense_matrices(matrix_values)
    vpar = np.unique(primitive[:, 5])
    mu = np.unique(primitive[:, 6])
    z_values = np.unique(primitive[:, 2].astype(int))
    species_values = np.unique(primitive[:, 3].astype(int))
    magnetic_field = np.full(z_values.size, np.nan)
    masses = np.full(species_values.size, np.nan)
    smz = np.full(species_values.size, np.nan)
    frequencies = np.full((species_values.size, species_values.size), np.nan)
    z_lookup = {value: index for index, value in enumerate(z_values)}
    species_lookup = {value: index for index, value in enumerate(species_values)}
    for row in primitive:
        iz = z_lookup[int(row[2])]
        target = species_lookup[int(row[3])]
        background = species_lookup[int(row[4])]
        magnetic_field[iz] = row[7]
        masses[target] = row[8]
        masses[background] = row[9]
        frequencies[target, background] = row[10]
        smz[target] = row[16]
    if not all(
        np.isfinite(array).all()
        for array in (magnetic_field, masses, smz, frequencies)
    ):
        raise ValueError("primitive trace does not span the gyro coefficient inputs")
    scalar_columns = {
        name: np.unique(primitive[:, column])
        for name, column in {
            "dt": 19,
            "deflection": 20,
            "electron_parallel": 21,
            "electron_deflection": 22,
            "mixed": 23,
            "gyro": 24,
        }.items()
    }
    if any(values.size != 1 for values in scalar_columns.values()):
        raise ValueError("collision knobs vary within the primitive trace")
    knobs = {name: float(values[0]) for name, values in scalar_columns.items()}
    grid = build_velocity_grid_from_nodes(
        vpar=vpar,
        mu=mu,
        w_vpar=np.ones_like(vpar),
        w_mu=np.ones_like(mu),
        backend="stella_trace",
    )
    species = tuple(
        SpeciesParams(
            np.sqrt(mass) / smz_value,
            mass,
            1.0,
            1.0,
            0.0,
            0.0,
        )
        for mass, smz_value in zip(masses, smz, strict=True)
    )
    primitives = build_stella_test_particle_primitives(
        grid,
        magnetic_field,
        species,
        frequencies,
        deflection_scale=knobs["deflection"],
        electron_parallel_scale=knobs["electron_parallel"],
        electron_deflection_scale=knobs["electron_deflection"],
        mixed_scale=knobs["mixed"],
    )
    local = np.asarray(
        build_stella_test_particle_gyro_diagonal(
            grid,
            magnetic_field,
            species,
            primitives,
            knobs["dt"],
            gyro_scale=knobs["gyro"],
            deflection_scale=knobs["deflection"],
            electron_parallel_scale=knobs["electron_parallel"],
            electron_deflection_scale=knobs["electron_deflection"],
        )
    )

    errors = []
    native_values = []
    compared = 0
    kperp_by_key = {}
    for key in matrices:
        selected = np.all(matrix_values[:, :4] == np.asarray(key), axis=1)
        values = np.unique(matrix_values[selected, 8])
        if values.size != 1:
            raise ValueError(f"inconsistent kperp2 for matrix {key}")
        kperp_by_key[key] = float(values[0])
    for iz_value in z_values:
        for species_value in species_values:
            keys = [
                key
                for key in matrices
                if key[2:] == (int(iz_value), int(species_value))
            ]
            base_key = min(keys, key=kperp_by_key.__getitem__)
            if kperp_by_key[base_key] != 0.0:
                raise ValueError("native matrix group lacks a zero-kperp base")
            expected = local[
                species_lookup[int(species_value)],
                :,
                :,
                z_lookup[int(iz_value)],
            ].reshape(-1)
            for key in keys:
                kperp = kperp_by_key[key]
                if kperp == 0.0:
                    continue
                native = np.diag(matrices[key] - matrices[base_key]).real / kperp
                errors.append(native - expected)
                native_values.append(native)
                compared += 1
    error = np.concatenate(errors)
    native = np.concatenate(native_values)
    scale = max(float(np.linalg.norm(native)), np.finfo(float).tiny)
    relative_l2 = float(np.linalg.norm(error) / scale)
    maximum = float(np.max(np.abs(error)))
    if maximum > absolute_tolerance:
        raise ValueError("local gyro diagonal reconstruction exceeds tolerance")
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_test_particle_gyro_diagonal",
        "status": "local_gyro_diagonal_passed",
        "scope": "gyro diagonal parity; zero-kperp differential assembly pending",
        "stella_source_revision": expected_revision,
        "primitive_trace": str(Path(primitive_trace).resolve()),
        "matrix_trace": str(Path(matrix_trace).resolve()),
        "matrices_compared": compared,
        "metrics": {
            "relative_l2": relative_l2,
            "max_abs": maximum,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_gyro_diagonal(
        args.primitives,
        args.matrix,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
