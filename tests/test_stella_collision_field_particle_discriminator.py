from pathlib import Path

import netCDF4
import pytest

from scripts.run_stella_collision_field_particle_discriminator import (
    stella_collision_input,
    summarize_outputs,
)
from scripts.run_stella_collision_trace_state import run_trace_state


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


def test_input_exposes_native_collision_channel_knobs():
    text = stella_collision_input(
        field_particle=True,
        collision_knobs=(0.0, 1.0, 0.0, 0.0),
    )

    assert "iiknob = 0" in text
    assert "ieknob = 1" in text
    assert "eeknob = 0" in text
    assert "eiknob = 0" in text


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
    assert report["native_netcdf_software_version_informational"] == ("misleading-parent-revision")
    assert report["metrics"]["h2_vs_vpamus"]["final_relative_l2_difference"] == pytest.approx(0.3)


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


def test_collision_input_parameterizes_distinct_initial_state():
    text = stella_collision_input(
        field_particle=True,
        initial_amplitude=0.017,
        initial_width=0.7,
    )

    assert "phiinit = 0.017000000000000001" in text
    assert "width0 = 0.69999999999999996" in text


def test_parameterized_trace_runner_writes_distinct_state(tmp_path, monkeypatch):
    executable = tmp_path / "stella"
    executable.write_text("fixture")
    calls = []
    monkeypatch.setattr(
        "scripts.run_stella_collision_trace_state.subprocess.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    metadata = run_trace_state(
        executable,
        tmp_path / "run",
        initial_amplitude=0.017,
        initial_width=0.7,
    )

    assert metadata.is_file()
    assert (
        "phiinit = 0.017000000000000001" in (tmp_path / "run/collision_trace_state.in").read_text()
    )
    assert len(calls) == 1
