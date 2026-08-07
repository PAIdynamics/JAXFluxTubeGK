import numpy as np
import pytest

from scripts.summarize_stella_collision_test_particle_primitives import (
    SCHEMA,
    summarize_test_particle_primitives,
)
from stellarator_gk import (
    SpeciesParams,
    build_stella_test_particle_primitives,
    build_velocity_grid_from_nodes,
)


def _write_trace(path, *, perturb_nupa=0.0):
    vpar = np.asarray((-0.5, 0.5))
    mu = np.asarray((0.2, 0.8))
    magnetic_field = np.asarray((1.1,))
    masses = np.asarray((1.0, 0.01))
    frequencies = np.asarray(((0.1, 0.2), (0.3, 0.4)))
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
    primitives = build_stella_test_particle_primitives(
        grid,
        magnetic_field,
        species,
        frequencies,
    )
    rows = []
    for target in range(2):
        for background in range(2):
            for iz in range(1):
                for iv in range(2):
                    for imu in range(2):
                        rows.append(
                            (
                                iv + 1,
                                imu + 1,
                                iz,
                                target + 1,
                                background + 1,
                                vpar[iv],
                                mu[imu],
                                magnetic_field[iz],
                                masses[target],
                                masses[background],
                                frequencies[target, background],
                                primitives.speed[iv, imu, iz],
                                primitives.maxwellian[target, iv, imu, iz],
                                primitives.parallel_diffusion[target, background, iv, imu, iz]
                                + perturb_nupa,
                                primitives.deflection[target, background, iv, imu, iz],
                                primitives.mixed_diffusion[target, background, iv, imu, iz],
                                1.0,
                                1.0,
                                0.6,
                                0.01,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                                1.0,
                            )
                        )
    path.write_text(SCHEMA + "\n# columns\n")
    with path.open("a") as stream:
        np.savetxt(stream, np.asarray(rows, dtype=float))


def test_reconstructs_native_test_particle_primitives(tmp_path):
    trace = tmp_path / "primitives.dat"
    _write_trace(trace)

    report = summarize_test_particle_primitives(trace, expected_revision="abc123")

    assert report["status"] == "local_test_particle_primitives_passed"
    assert report["rows"] == 16
    assert report["metrics"]["nupa_relative_l2"] < 1.0e-15


def test_rejects_incorrect_test_particle_primitive(tmp_path):
    trace = tmp_path / "primitives.dat"
    _write_trace(trace, perturb_nupa=1.0e-3)

    with pytest.raises(ValueError, match="local nupa reconstruction"):
        summarize_test_particle_primitives(trace, expected_revision="abc123")
