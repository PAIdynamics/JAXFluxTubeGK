from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stellarator_gk import PerKyModeStructureFixture


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_mode_structure_example_filters_requested_ky_values():
    module = _load_module(
        ROOT / "examples/compare_mode_structure_fixtures.py",
        "compare_mode_structure_fixtures",
    )
    fixture = PerKyModeStructureFixture(
        ky=np.asarray([0.0, 0.1, 0.2, 0.3]),
        z=np.asarray([-0.5, 0.0, 0.5]),
        phi=np.ones((4, 3), dtype=np.complex128),
        growth_rate=np.asarray([-1.0, 0.1, 0.2, 0.3]),
        frequency=np.asarray([0.0, 1.1, 1.2, 1.3]),
        source="synthetic-w7x",
    )

    filtered = module._select_ky_values(fixture, (0.1, 0.3), 1.0e-12)

    np.testing.assert_allclose(filtered.ky, np.asarray([0.1, 0.3]))
    np.testing.assert_allclose(filtered.growth_rate, np.asarray([0.1, 0.3]))
    np.testing.assert_allclose(filtered.frequency, np.asarray([1.1, 1.3]))
    assert filtered.phi.shape == (2, 3)
    assert ("ky_filter", "0.1,0.3") in filtered.metadata

    with pytest.raises(ValueError, match="requested ky=0.4"):
        module._select_ky_values(fixture, (0.4,), 1.0e-12)


def test_compare_mode_structure_example_resamples_reference_to_observed_z():
    module = _load_module(
        ROOT / "examples/compare_mode_structure_fixtures.py",
        "compare_mode_structure_fixtures",
    )
    observed = PerKyModeStructureFixture(
        ky=np.asarray([0.1]),
        z=np.asarray([0.0, 0.5, 1.0]),
        phi=np.asarray([[1.0, 1.5, 2.0]], dtype=np.complex128),
        growth_rate=np.asarray([0.1]),
        frequency=np.asarray([0.0]),
        source="observed",
    )
    reference = PerKyModeStructureFixture(
        ky=np.asarray([0.1]),
        z=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        phi=np.asarray([[1.0, 1.25, 1.5, 1.75, 2.0]], dtype=np.complex128),
        growth_rate=np.asarray([0.1]),
        frequency=np.asarray([0.0]),
        source="reference",
    )
    args = SimpleNamespace(
        resample_reference_to_observed_z=True,
        resample_observed_to_reference_z=False,
        periodic_z=False,
        z_period=None,
    )

    resampled_observed, resampled_reference = module._apply_z_resampling(
        observed,
        reference,
        args,
    )

    np.testing.assert_allclose(resampled_observed.z, observed.z)
    np.testing.assert_allclose(resampled_reference.z, observed.z)
    np.testing.assert_allclose(resampled_reference.phi, observed.phi)

    args.resample_observed_to_reference_z = True
    with pytest.raises(ValueError, match="at most one"):
        module._apply_z_resampling(observed, reference, args)


def test_prepare_gx_w7x_mode_structure_run_writes_external_workflow(tmp_path):
    module = _load_module(
        ROOT / "scripts/prepare_gx_w7x_mode_structure_run.py",
        "prepare_gx_w7x_mode_structure_run",
    )
    source = tmp_path / "itg_w7x_adiabatic_electrons.in"
    vmec = tmp_path / "wout_w7x.nc"
    source.write_text(
        """[Geometry]
 vmec_file = "wout_w7x.nc"

[Diagnostics]
 nwrite = 100
 omega = false
 fields = false
 moments = false
"""
    )
    vmec.write_bytes(b"small vmec placeholder")
    output_dir = tmp_path / "gx_w7x_run"
    external_fixture = tmp_path / "w7x_external.csv"
    observed_fixture = tmp_path / "observed.csv"
    comparison_output = tmp_path / "comparison.csv"

    metadata = module.prepare_gx_w7x_mode_structure_run(
        source,
        output_dir,
        vmec_file=vmec,
        nwrite_big=13,
        gx_executable="/opt/gx/bin/gx",
        ky_values="0.1,0.2",
        gx_z_coordinate="theta_over_2pi",
        external_fixture=external_fixture,
        observed_fixture=observed_fixture,
        comparison_output=comparison_output,
        copy_vmec=True,
    )

    prepared_input = output_dir / "itg_w7x_adiabatic_electrons.in"
    diagnostics = tomllib.loads(prepared_input.read_text())["Diagnostics"]
    assert diagnostics["nwrite_big"] == 13
    assert diagnostics["omega"] is True
    assert diagnostics["fields"] is True
    assert diagnostics["moments"] is True
    assert (output_dir / "wout_w7x.nc").read_bytes() == b"small vmec placeholder"
    assert metadata["status"] == "pending_external_gx_run"
    assert "--ky-values 0.1,0.2" in metadata["export_command"]
    assert "--ky-values 0.1,0.2" in metadata["comparison_command"]
    assert "--observed" in metadata["comparison_command"]
    assert "Diagnostics/Phi" in (output_dir / "README.md").read_text()


def test_committed_gx_w7x_external_run_prep_metadata_is_portable():
    metadata_path = ROOT / "fixtures/gx_w7x_mode_structure_run/mode_structure_run_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    assert metadata["status"] == "pending_external_gx_run"
    assert metadata["prepared_input"] == (
        "fixtures/gx_w7x_mode_structure_run/itg_w7x_adiabatic_electrons.in"
    )
    assert metadata["vmec_copied"] is False
    assert metadata["ky_values"] == "0.1,0.2,0.3"
    assert metadata["copy_vmec_command"].startswith(
        "cp dependency://gx/benchmarks/linear/ITG_w7x/wout_w7x.nc"
    )
    assert "--ky-values 0.1,0.2,0.3" in metadata["comparison_command"]
    assert "path/to/gx" in metadata["run_command"]
