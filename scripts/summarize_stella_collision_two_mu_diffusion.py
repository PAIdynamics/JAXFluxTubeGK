"""Validate the local two-node pure-mu collision blocks against stella."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_stella_collision_test_particle_blocks import (
    COLUMNS as BLOCK_COLUMNS,
    SCHEMA as BLOCK_SCHEMA,
    _block_arrays,
    _read,
)
from scripts.summarize_stella_collision_test_particle_primitives import (
    COLUMNS as PRIMITIVE_COLUMNS,
    SCHEMA as PRIMITIVE_SCHEMA,
    _consistent_scalar,
)
from stellarator_gk import (
    SpeciesParams,
    build_stella_test_particle_primitives,
    build_stella_two_mu_diffusion_blocks,
    build_stella_two_mu_mixed_blocks,
    build_stella_two_mu_vpar_mixed_blocks,
    build_stella_vpar_diffusion_blocks,
    build_velocity_grid_from_nodes,
)


def summarize_two_mu_diffusion(
    primitive_trace: Path,
    no_mixed_full_trace: Path,
    no_mixed_vpar_trace: Path,
    vpar_trace: Path | None = None,
    full_trace: Path | None = None,
    *,
    expected_revision: str,
    tolerance: float = 2.0e-11,
) -> dict[str, object]:
    """Compare local pure-mu boundary blocks with the native factorial split."""

    primitive = _read(primitive_trace, PRIMITIVE_SCHEMA, len(PRIMITIVE_COLUMNS))
    full_values = _read(no_mixed_full_trace, BLOCK_SCHEMA, len(BLOCK_COLUMNS))
    vpar_values = _read(no_mixed_vpar_trace, BLOCK_SCHEMA, len(BLOCK_COLUMNS))
    full = _block_arrays(full_values)
    vpar_only = _block_arrays(vpar_values)
    if not np.array_equal(full_values[:, :6], vpar_values[:, :6]):
        raise ValueError("native pure-mu traces have different index grids")
    native = tuple(left - right for left, right in zip(full[:3], vpar_only[:3], strict=True))

    vpar = np.unique(primitive[:, 5])
    mu = np.unique(primitive[:, 6])
    z_values = np.unique(primitive[:, 2].astype(int))
    target_values = np.unique(primitive[:, 3].astype(int))
    background_values = np.unique(primitive[:, 4].astype(int))
    if mu.size != 2 or target_values.size != background_values.size:
        raise ValueError("trace does not contain a square two-mu species grid")
    z_lookup = {value: index for index, value in enumerate(z_values)}
    species_lookup = {value: index for index, value in enumerate(target_values)}
    magnetic_field = np.full(z_values.size, np.nan)
    masses = np.full(target_values.size, np.nan)
    frequencies = np.full((target_values.size, target_values.size), np.nan)
    smz = np.full(target_values.size, np.nan)
    for row in primitive:
        iz = z_lookup[int(row[2])]
        target = species_lookup[int(row[3])]
        background = species_lookup[int(row[4])]
        magnetic_field[iz] = row[7]
        masses[target] = row[8]
        masses[background] = row[9]
        frequencies[target, background] = row[10]
        smz[target] = row[16]
    if not all(np.isfinite(array).all() for array in (magnetic_field, masses, frequencies, smz)):
        raise ValueError("primitive trace does not span the local block inputs")
    grid = build_velocity_grid_from_nodes(
        vpar=vpar,
        mu=mu,
        w_vpar=np.ones_like(vpar),
        w_mu=np.ones_like(mu),
        backend="stella_trace",
    )
    species = tuple(
        SpeciesParams(np.sqrt(mass) / scale, mass, 1.0, 1.0, 0.0, 0.0)
        for mass, scale in zip(masses, smz, strict=True)
    )
    knobs = {
        "dt": _consistent_scalar(primitive, 19, "code_dt"),
        "deflection": _consistent_scalar(primitive, 20, "deflknob"),
        "electron_parallel": _consistent_scalar(primitive, 21, "eiediffknob"),
        "electron_deflection": _consistent_scalar(primitive, 22, "eideflknob"),
    }
    primitives = build_stella_test_particle_primitives(
        grid,
        magnetic_field,
        species,
        frequencies,
        deflection_scale=knobs["deflection"],
        electron_parallel_scale=knobs["electron_parallel"],
        electron_deflection_scale=knobs["electron_deflection"],
    )
    local = tuple(
        np.asarray(item)
        for item in build_stella_two_mu_diffusion_blocks(
            grid,
            magnetic_field,
            species,
            frequencies,
            primitives,
            knobs["dt"],
            deflection_scale=knobs["deflection"],
            electron_parallel_scale=knobs["electron_parallel"],
            electron_deflection_scale=knobs["electron_deflection"],
        )
    )
    error = np.concatenate(
        [(observed - expected).ravel() for observed, expected in zip(local, native, strict=True)]
    )
    native_flat = np.concatenate([item.ravel() for item in native])
    relative_l2 = float(
        np.linalg.norm(error) / max(np.linalg.norm(native_flat), np.finfo(float).tiny)
    )
    maximum = float(np.max(np.abs(error)))
    if maximum > tolerance:
        raise ValueError("local two-mu diffusion blocks exceed native tolerance")
    local_vpar = tuple(
        np.asarray(item)
        for item in build_stella_vpar_diffusion_blocks(
            grid,
            magnetic_field,
            species,
            frequencies,
            primitives,
            knobs["dt"],
            deflection_scale=knobs["deflection"],
            electron_parallel_scale=knobs["electron_parallel"],
            electron_deflection_scale=knobs["electron_deflection"],
        )
    )
    vpar_error = np.concatenate(
        [
            (observed - expected).ravel()
            for observed, expected in zip(local_vpar, vpar_only[:3], strict=True)
        ]
    )
    vpar_native = np.concatenate([item.ravel() for item in vpar_only[:3]])
    vpar_relative = float(
        np.linalg.norm(vpar_error) / max(np.linalg.norm(vpar_native), np.finfo(float).tiny)
    )
    vpar_maximum = float(np.max(np.abs(vpar_error)))
    if vpar_maximum > tolerance:
        raise ValueError(
            "local vpar diffusion blocks exceed native tolerance: "
            f"relative_l2={vpar_relative:.6e}, max_abs={vpar_maximum:.6e}"
        )
    metrics = {
        "pure_mu_relative_l2": relative_l2,
        "pure_mu_max_abs": maximum,
        "pure_vpar_relative_l2": vpar_relative,
        "pure_vpar_max_abs": vpar_maximum,
    }
    status = "local_two_mu_diffusion_blocks_passed"
    scope = "pure two-node mu-boundary parity; mixed and general-grid branches pending"
    if vpar_trace is not None:
        if full_trace is None:
            raise ValueError("full_trace is required when validating mixed components")
        mixed_values = _read(vpar_trace, BLOCK_SCHEMA, len(BLOCK_COLUMNS))
        if not np.array_equal(mixed_values[:, :6], vpar_values[:, :6]):
            raise ValueError("native mixed-vpar traces have different index grids")
        mixed_full = _block_arrays(mixed_values)
        mixed_native = tuple(
            left - right for left, right in zip(mixed_full[:3], vpar_only[:3], strict=True)
        )
        mixed_local = tuple(
            np.asarray(item)
            for item in build_stella_two_mu_vpar_mixed_blocks(grid, primitives, knobs["dt"])
        )
        mixed_error = np.concatenate(
            [
                (observed - expected).ravel()
                for observed, expected in zip(mixed_local, mixed_native, strict=True)
            ]
        )
        mixed_native_flat = np.concatenate([item.ravel() for item in mixed_native])
        mixed_relative = float(
            np.linalg.norm(mixed_error)
            / max(np.linalg.norm(mixed_native_flat), np.finfo(float).tiny)
        )
        mixed_maximum = float(np.max(np.abs(mixed_error)))
        if mixed_maximum > tolerance:
            block_errors = [
                float(np.max(np.abs(observed - expected)))
                for observed, expected in zip(mixed_local, mixed_native, strict=True)
            ]
            block_indices = [
                np.unravel_index(np.argmax(np.abs(observed - expected)), observed.shape)
                for observed, expected in zip(mixed_local, mixed_native, strict=True)
            ]
            raise ValueError(
                "local two-mu mixed-vpar blocks exceed native tolerance: "
                f"relative_l2={mixed_relative:.6e}, max_abs={mixed_maximum:.6e}, "
                f"block_max={block_errors}, block_indices={block_indices}"
            )
        metrics.update(
            {
                "mixed_vpar_relative_l2": mixed_relative,
                "mixed_vpar_max_abs": mixed_maximum,
            }
        )
        full_values = _read(full_trace, BLOCK_SCHEMA, len(BLOCK_COLUMNS))
        if not np.array_equal(full_values[:, :6], mixed_values[:, :6]):
            raise ValueError("native full and vpar traces have different index grids")
        full_native = _block_arrays(full_values)
        mu_path_native = tuple(
            left - right for left, right in zip(full_native[:3], mixed_full[:3], strict=True)
        )
        mixed_mu_native = tuple(
            left - right for left, right in zip(mu_path_native, native, strict=True)
        )
        mixed_mu_local = tuple(
            np.asarray(item)
            for item in build_stella_two_mu_mixed_blocks(grid, primitives, knobs["dt"])
        )
        mixed_mu_error = np.concatenate(
            [
                (observed - expected).ravel()
                for observed, expected in zip(
                    mixed_mu_local, mixed_mu_native, strict=True
                )
            ]
        )
        mixed_mu_native_flat = np.concatenate(
            [item.ravel() for item in mixed_mu_native]
        )
        mixed_mu_relative = float(
            np.linalg.norm(mixed_mu_error)
            / max(np.linalg.norm(mixed_mu_native_flat), np.finfo(float).tiny)
        )
        mixed_mu_maximum = float(np.max(np.abs(mixed_mu_error)))
        if mixed_mu_maximum > tolerance:
            block_errors = [
                float(np.max(np.abs(observed - expected)))
                for observed, expected in zip(
                    mixed_mu_local, mixed_mu_native, strict=True
                )
            ]
            block_indices = [
                np.unravel_index(
                    np.argmax(np.abs(observed - expected)), observed.shape
                )
                for observed, expected in zip(
                    mixed_mu_local, mixed_mu_native, strict=True
                )
            ]
            raise ValueError(
                "local two-mu mixed-mu blocks exceed native tolerance: "
                f"relative_l2={mixed_mu_relative:.6e}, "
                f"max_abs={mixed_mu_maximum:.6e}, block_max={block_errors}, "
                f"block_indices={block_indices}"
            )
        full_local = tuple(
            pure_mu + pure_vpar + mixed_vpar + mixed_mu
            for pure_mu, pure_vpar, mixed_vpar, mixed_mu in zip(
                local, local_vpar, mixed_local, mixed_mu_local, strict=True
            )
        )
        full_error = np.concatenate(
            [
                (observed - expected).ravel()
                for observed, expected in zip(full_local, full_native[:3], strict=True)
            ]
        )
        full_native_flat = np.concatenate([item.ravel() for item in full_native[:3]])
        full_relative = float(
            np.linalg.norm(full_error)
            / max(np.linalg.norm(full_native_flat), np.finfo(float).tiny)
        )
        full_maximum = float(np.max(np.abs(full_error)))
        if full_maximum > tolerance:
            raise ValueError(
                "complete local two-mu blocks exceed native tolerance: "
                f"relative_l2={full_relative:.6e}, max_abs={full_maximum:.6e}"
            )
        metrics.update(
            {
                "mixed_mu_relative_l2": mixed_mu_relative,
                "mixed_mu_max_abs": mixed_mu_maximum,
                "full_blocks_relative_l2": full_relative,
                "full_blocks_max_abs": full_maximum,
            }
        )
        status = "local_two_mu_collision_blocks_passed"
        scope = "complete two-node block parity; general-grid branches pending"
    return {
        "schema_version": 1,
        "benchmark": "stella_collision_two_mu_diffusion_blocks",
        "status": status,
        "stella_source_revision": expected_revision,
        "rows": int(full_values.shape[0]),
        "metrics": metrics,
        "scope": scope,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--no-mixed-full", type=Path, required=True)
    parser.add_argument("--no-mixed-vpar", type=Path, required=True)
    parser.add_argument("--vpar", type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_two_mu_diffusion(
        args.primitives,
        args.no_mixed_full,
        args.no_mixed_vpar,
        vpar_trace=args.vpar,
        full_trace=args.full,
        expected_revision=args.expected_revision,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
