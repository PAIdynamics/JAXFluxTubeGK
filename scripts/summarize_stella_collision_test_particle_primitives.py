"""Validate stella test-particle frequency primitives against the JAX builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_field_particle_factors import _axis_lookup
from jax_fluxtube_gk import (
    SpeciesParams,
    build_stella_test_particle_primitives,
    build_velocity_grid_from_nodes,
)


SCHEMA = "# schema=jax_fluxtube_gk_stella_collision_test_particle_primitives_v1"
COLUMNS = (
    "iv",
    "imu",
    "iz",
    "target",
    "background",
    "vpa",
    "mu",
    "bmag",
    "target_mass",
    "background_mass",
    "frequency",
    "speed",
    "maxwell",
    "nupa",
    "nuD",
    "nux",
    "target_smz",
    "dvpa",
    "dmu",
    "code_dt",
    "deflknob",
    "eiediffknob",
    "eideflknob",
    "nuxfac",
    "cfac",
)


def _consistent_scalar(rows: np.ndarray, column: int, name: str) -> float:
    values = np.unique(rows[:, column])
    if values.size != 1:
        raise ValueError(f"{name} varies within the primitive trace")
    return float(values[0])


def summarize_test_particle_primitives(
    trace_path: Path,
    *,
    expected_revision: str,
    tolerance: float = 2.0e-11,
) -> dict[str, object]:
    """Reconstruct all analytic collision-frequency arrays on the native grid."""

    trace_path = Path(trace_path)
    header = trace_path.read_text(encoding="utf-8").splitlines()[:1]
    if not header or header[0].strip() != SCHEMA:
        raise ValueError("unsupported or missing test-particle primitive schema")
    rows = np.atleast_2d(np.loadtxt(trace_path))
    if rows.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, found {rows.shape[1]}")
    if not np.isfinite(rows).all():
        raise ValueError("test-particle primitive trace contains non-finite values")
    if not np.array_equal(rows[:, :5], np.rint(rows[:, :5])):
        raise ValueError("test-particle primitive indices must be integers")
    if np.unique(rows[:, :5], axis=0).shape[0] != rows.shape[0]:
        raise ValueError("test-particle primitive trace contains duplicate rows")

    axes = {
        "iv": _axis_lookup(rows[:, 0]),
        "imu": _axis_lookup(rows[:, 1]),
        "iz": _axis_lookup(rows[:, 2]),
        "target": _axis_lookup(rows[:, 3]),
        "background": _axis_lookup(rows[:, 4]),
    }
    n_species = axes["target"][0].size
    if axes["background"][0].size != n_species:
        raise ValueError("target and background axes differ")
    vpar = np.full(axes["iv"][0].size, np.nan)
    mu = np.full(axes["imu"][0].size, np.nan)
    magnetic_field = np.full(axes["iz"][0].size, np.nan)
    masses = np.full(n_species, np.nan)
    frequencies = np.full((n_species, n_species), np.nan)
    row_indices = []
    for row in rows:
        iv = axes["iv"][1][int(row[0])]
        imu = axes["imu"][1][int(row[1])]
        iz = axes["iz"][1][int(row[2])]
        target = axes["target"][1][int(row[3])]
        background = axes["background"][1][int(row[4])]
        for array, index, value, name in (
            (vpar, (iv,), row[5], "vpa"),
            (mu, (imu,), row[6], "mu"),
            (magnetic_field, (iz,), row[7], "bmag"),
            (masses, (target,), row[8], "target mass"),
            (masses, (background,), row[9], "background mass"),
            (frequencies, (target, background), row[10], "frequency"),
        ):
            previous = array[index]
            if np.isfinite(previous) and previous != value:
                raise ValueError(f"inconsistent {name} at index {index}")
            array[index] = value
        row_indices.append((target, background, iv, imu, iz))
    if not all(
        np.isfinite(array).all()
        for array in (vpar, mu, magnetic_field, masses, frequencies)
    ):
        raise ValueError("primitive trace does not span a dense coefficient grid")

    grid = build_velocity_grid_from_nodes(
        vpar=vpar,
        mu=mu,
        w_vpar=np.ones_like(vpar),
        w_mu=np.ones_like(mu),
        backend="stella_trace",
    )
    species = tuple(
        SpeciesParams(1.0, mass, 1.0, 1.0, 0.0, 0.0) for mass in masses
    )
    local = build_stella_test_particle_primitives(
        grid,
        magnetic_field,
        species,
        frequencies,
        deflection_scale=_consistent_scalar(rows, 20, "deflknob"),
        electron_parallel_scale=_consistent_scalar(rows, 21, "eiediffknob"),
        electron_deflection_scale=_consistent_scalar(rows, 22, "eideflknob"),
        mixed_scale=_consistent_scalar(rows, 23, "nuxfac"),
        electron_index=1 if n_species > 1 else None,
        ion_index=0 if n_species > 1 else None,
    )
    local_arrays = {
        "speed": np.asarray(local.speed),
        "maxwell": np.asarray(local.maxwellian),
        "nupa": np.asarray(local.parallel_diffusion),
        "nuD": np.asarray(local.deflection),
        "nux": np.asarray(local.mixed_diffusion),
    }
    native_columns = {"speed": 11, "maxwell": 12, "nupa": 13, "nuD": 14, "nux": 15}
    metrics = {}
    for name, local_array in local_arrays.items():
        if name in ("speed", "maxwell"):
            local_values = np.asarray(
                [
                    local_array[(index[2], index[3], index[4])]
                    if name == "speed"
                    else local_array[(index[0], index[2], index[3], index[4])]
                    for index in row_indices
                ]
            )
        else:
            local_values = np.asarray([local_array[index] for index in row_indices])
        native = rows[:, native_columns[name]]
        scale = max(float(np.linalg.norm(native)), np.finfo(float).tiny)
        relative_l2 = float(np.linalg.norm(local_values - native) / scale)
        maximum = float(np.max(np.abs(local_values - native)))
        if relative_l2 > tolerance:
            raise ValueError(f"local {name} reconstruction exceeds tolerance")
        metrics[f"{name}_relative_l2"] = relative_l2
        metrics[f"{name}_max_abs"] = maximum

    return {
        "schema_version": 1,
        "benchmark": "stella_collision_test_particle_primitives",
        "status": "local_test_particle_primitives_passed",
        "scope": "analytic coefficient parity; finite-difference matrix assembly pending",
        "stella_source_revision": expected_revision,
        "trace": str(trace_path.resolve()),
        "rows": int(rows.shape[0]),
        "grid": {
            "nvpa": vpar.size,
            "nmu": mu.size,
            "nz": magnetic_field.size,
            "nspecies": n_species,
        },
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_test_particle_primitives(
        args.trace,
        expected_revision=args.expected_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
