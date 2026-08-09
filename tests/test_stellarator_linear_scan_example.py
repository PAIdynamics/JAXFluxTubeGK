import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from jax_fluxtube_gk import SyntheticGeometryProvider


ROOT = Path(__file__).resolve().parents[1]


def _load_scan_module():
    path = ROOT / "examples/run_stellarator_linear_scan.py"
    spec = importlib.util.spec_from_file_location("run_stellarator_linear_scan", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stellarator_linear_scan_example_writes_machine_readable_outputs(tmp_path):
    output_dir = tmp_path / "stellarator_scan"
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "examples/run_stellarator_linear_scan.py",
            "--output-dir",
            str(output_dir),
            "--steps-per-window",
            "1",
            "--n-windows",
            "1",
            "--dt",
            "0.002",
            "--ky-values",
            "0.0,0.35",
            "--n-vpar",
            "3",
            "--n-mu",
            "3",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PASS: stellarator linear scan" in result.stdout
    expected = {
        "geometry_audit.json",
        "geometry_audit.csv",
        "ky_growth.csv",
        "mode_structures.csv",
        "convergence_metadata.json",
        "convergence_history.csv",
        "quasilinear_proxy.json",
        "run_config.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    audit = json.loads((output_dir / "geometry_audit.json").read_text())
    assert audit["passed"]
    assert audit["checks"]["gx_eik_export_contract"]
    assert audit["geometry"]["geometry_source"] == "fixture"

    with (output_dir / "ky_growth.csv").open(newline="") as handle:
        ky_rows = tuple(csv.DictReader(handle))
    assert [float(row["ky"]) for row in ky_rows] == [0.0, 0.35]
    assert all(np.isfinite(float(row["growth_rate"])) for row in ky_rows)
    assert float(ky_rows[0]["quasilinear_contribution"]) == 0.0

    with (output_dir / "mode_structures.csv").open(newline="") as handle:
        mode_rows = tuple(csv.DictReader(handle))
    assert len(mode_rows) == 2 * int(audit["z"]["n_z"])
    assert {"ky", "z", "phi_real", "phi_imag", "growth_rate", "frequency"}.issubset(
        mode_rows[0]
    )

    metadata = json.loads((output_dir / "convergence_metadata.json").read_text())
    assert metadata["finite_growth"]
    assert metadata["finite_frequency"]
    assert metadata["steps_per_window"] == 1


def test_stellarator_linear_scan_loads_stella_geometry_with_normalized_z():
    module = _load_scan_module()

    args = module._parse_args(
        [
            "--geometry-source",
            "stella-geometry",
            "--n-kx",
            "1",
            "--kx-max",
            "0.0",
            "--ky-values",
            "0.1,0.2,0.3",
        ]
    )
    geometry, parallel, metadata = module._load_geometry(args)

    z = np.asarray(geometry.z, dtype=float)
    assert metadata["geometry_source"] == "stella-geometry"
    assert metadata["dropped_periodic_endpoint"]
    assert metadata["n_z"] == 256
    assert parallel.z.shape[0] == 256
    assert z[0] == -0.5
    assert np.isclose(z[-1], 0.5 - 1.0 / 256.0)
    assert metadata["field_line_periods"] > 6.9
    assert metadata["b_dot_grad_z_scaling"] == "F = stella b.Gz / (2*pi)"
    assert metadata["equilibrium_drive_scaling"] == "E_y = stella geometry-header flux_fac"
    assert metadata["schema_version"] == 2
    assert metadata["provider"] == "stella-geometry"
    assert geometry.source == "stella-geometry"
    assert np.all(np.isfinite(np.asarray(geometry.B)))
    assert np.all(np.isfinite(np.asarray(geometry.F)))
    assert np.all(np.isfinite(np.asarray(geometry.G)))
    assert np.max(np.abs(np.asarray(geometry.E_y, dtype=float))) > 0.0
    np.testing.assert_allclose(geometry.E_y, 0.5)


def test_stellarator_linear_scan_accepts_named_vmecpp_provider(monkeypatch):
    module = _load_scan_module()
    monkeypatch.setattr(
        module,
        "VmecppGeometryProvider",
        lambda **_kwargs: SyntheticGeometryProvider(nfp=5),
    )
    args = module._parse_args(
        [
            "--geometry-provider",
            "vmecpp",
            "--configuration",
            "w7x-standard",
            "--rho",
            "0.8",
            "--n-z",
            "16",
        ]
    )

    geometry, parallel, metadata = module._load_geometry(args)

    assert metadata["geometry_source"] == "vmecpp"
    assert metadata["configuration"] == "w7x-standard"
    assert metadata["nfp"] == 5
    assert parallel.z.shape == (16,)
    assert geometry.B.shape == (16,)


def test_late_window_frequency_uses_unaliased_short_phase_increments():
    module = _load_scan_module()
    omega = 0.12
    times = np.linspace(0.0, 200.0, 2001)
    base = np.ones((2, 1, 1), dtype=np.complex128)
    phi_samples = [base * np.exp(-1j * omega * time) for time in times]
    weights = np.full((2,), 0.5)
    late_start = 1000

    frequency = module._frequency_from_phi_samples(
        times,
        phi_samples,
        late_start,
        w_z=weights,
        connectivity=None,
    )
    endpoint_frequency = np.asarray(
        module.real_frequency(
            phi_samples[late_start],
            phi_samples[-1],
            times[late_start],
            times[-1],
            w_z=weights,
            connectivity=None,
        )
    )

    np.testing.assert_allclose(frequency, [omega], atol=1.0e-12)
    assert abs(float(endpoint_frequency[0]) - omega) > 0.05


def test_stella_initial_condition_matches_default_maxwellian_perturbation():
    module = _load_scan_module()
    from jax_fluxtube_gk import (
        FourierGridSpec,
        ParallelGridSpec,
        VelocityGridSpec,
        build_fourier_grid,
        build_parallel_grid,
        build_velocity_grid,
    )

    velocity = build_velocity_grid(
        VelocityGridSpec(n_vpar=4, n_mu=3, vpar_max=2.0, mu_max=1.5)
    )
    parallel = build_parallel_grid(ParallelGridSpec(n_z=5, z_min=-0.5, z_max=0.5))
    fourier = build_fourier_grid(
        FourierGridSpec(n_kx=1, n_ky=2, kx_max=0.0, ky_values=(0.1, 0.3))
    )
    geometry = type("Geometry", (), {"B": np.linspace(0.9, 1.2, 5)})()

    state = np.asarray(
        module._initial_state(
            velocity,
            parallel,
            fourier,
            0.01,
            geometry=geometry,
            initial_condition="stella_maxwellian",
        )
    )
    expected = (
        0.01
        * (1.0 + 1.0j)
        * np.exp(-np.asarray(velocity.vpar)[:, None, None] ** 2)
        * np.exp(
            -2.0
            * np.asarray(velocity.mu)[None, :, None]
            * np.asarray(geometry.B)[None, None, :]
        )
        * np.exp(-(2.0 * np.pi * np.asarray(parallel.z))[None, None, :] ** 2)
    )
    np.testing.assert_allclose(state[..., 0, 0], expected)
    np.testing.assert_allclose(state[..., 0, 1], expected)
