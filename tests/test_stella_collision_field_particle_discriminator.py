from pathlib import Path

import netCDF4
import pytest

from scripts.run_stella_collision_field_particle_discriminator import (
    stella_collision_input,
    summarize_outputs,
)


def _write_output(path: Path, final_scale: float) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("t", 2)
        dataset.createDimension("species", 2)
        dataset.createDimension("vpa", 3)
        dataset.software_version = "misleading-parent-revision"
        for name in ("phi2", "g2_vs_vpamus", "h2_vs_vpamus", "f2_vs_vpamus"):
            variable = dataset.createVariable(name, "f8", ("t", "species", "vpa"))
            variable[0] = 1.0
            variable[1] = final_scale


def test_input_switches_only_requested_field_particle_setting():
    enabled = stella_collision_input(field_particle=True)
    disabled = stella_collision_input(field_particle=False)

    assert "fieldpart = .true." in enabled
    assert "fieldpart = .false." in disabled
    assert enabled.replace("fieldpart = .true.", "fieldpart = .false.") == disabled
    assert "include_parallel_streaming = .false." in enabled
    assert "collisions_implicit = .true." in enabled


def test_summary_requires_matched_initial_state_and_records_sensitive_effect(tmp_path):
    enabled = tmp_path / "enabled.nc"
    disabled = tmp_path / "disabled.nc"
    _write_output(enabled, 1.3)
    _write_output(disabled, 1.0)

    report = summarize_outputs(
        enabled,
        disabled,
        provenance={"revision": "abc", "dirty": False},
    )

    assert report["status"] == "native_discriminator_passed"
    assert report["source_provenance"]["revision"] == "abc"
    assert report["native_netcdf_software_version_informational"] == (
        "misleading-parent-revision"
    )
    assert report["metrics"]["h2_vs_vpamus"]["final_relative_l2_difference"] == pytest.approx(
        0.3
    )


def test_summary_rejects_different_initial_states(tmp_path):
    enabled = tmp_path / "enabled.nc"
    disabled = tmp_path / "disabled.nc"
    _write_output(enabled, 1.3)
    _write_output(disabled, 1.0)
    with netCDF4.Dataset(enabled, "a") as dataset:
        dataset.variables["phi2"][0, 0, 0] = 1.1

    with pytest.raises(ValueError, match="initial states differ"):
        summarize_outputs(
            enabled,
            disabled,
            provenance={"revision": "abc", "dirty": False},
        )
