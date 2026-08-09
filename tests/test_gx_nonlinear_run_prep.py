from __future__ import annotations

import subprocess

import pytest

from scripts.prepare_gx_nonlinear_heat_flux_run import (
    PINNED_GX_REVISION,
    patch_gx_nonlinear_input,
    prepare_gx_nonlinear_heat_flux_run,
)


_INPUT = """[Dimensions]
 ntheta = 8
 nx = 4
 ny = 4
 nhermite = 2
 nlaguerre = 2
 nspecies = 1

[Domain]
 y0 = 15.0
 boundary = "linked"

[Physics]
 beta = 1e-3
 nonlinear_mode = true

[Time]
 t_max = 10.0

[Initialization]
 init_amp = 1e-2

[Geometry]
 geo_option = "s-alpha"

[species]
 vnewk = [0.1, 0.2]

[Dissipation]
 hyper = true

[Restart]
 restart = false

[Diagnostics]
 nwrite = 100
"""


def test_patch_gx_nonlinear_input_matches_local_case() -> None:
    import tomllib

    parsed = tomllib.loads(patch_gx_nonlinear_input(_INPUT))
    assert parsed["Dimensions"] == {
        "ntheta": 24,
        "nx": 32,
        "ny": 16,
        "nhermite": 8,
        "nlaguerre": 4,
        "nspecies": 1,
        "nperiod": 1,
    }
    assert parsed["Domain"]["y0"] == pytest.approx(10.0)
    assert parsed["Domain"]["jtwist"] == 1
    assert parsed["Physics"]["beta"] == 0.0
    assert parsed["Time"]["t_max"] == 500.0
    assert parsed["Time"]["nstep"] == 100000000
    assert parsed["Geometry"]["eps"] == pytest.approx(0.18)
    assert parsed["Dissipation"]["D_hyper"] == pytest.approx(0.05)
    assert parsed["species"]["vnewk"] == [0.0, 0.0]
    assert parsed["Diagnostics"]["fluxes"] is True


@pytest.mark.parametrize(
    "controls",
    ({"y0": float("nan")}, {"final_time": float("inf")}, {"nx": 4.5}),
)
def test_patch_gx_nonlinear_input_rejects_invalid_controls(controls) -> None:
    with pytest.raises(ValueError):
        patch_gx_nonlinear_input(_INPUT, **controls)


def test_prepare_gx_nonlinear_run_is_revision_pinned_and_external(tmp_path) -> None:
    gx_root = tmp_path / "gx"
    source = gx_root / "unit_tests/inputs"
    source.mkdir(parents=True)
    (source / "cyc_nl.in").write_text(_INPUT)
    subprocess.run(("git", "init", "-q"), cwd=gx_root, check=True)
    subprocess.run(("git", "add", "unit_tests/inputs/cyc_nl.in"), cwd=gx_root, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=gx_root,
        check=True,
    )
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=gx_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    output = tmp_path / "scratch"
    manifest = prepare_gx_nonlinear_heat_flux_run(
        gx_root, output, expected_revision=revision, gx_executable="/opt/gx/bin/gx"
    )
    assert manifest["gx_revision"] == revision
    assert manifest["case_contract"]["ky_min"] == pytest.approx(0.1)
    expected_numerics = {
        "ntheta": 24,
        "nx": 32,
        "ny": 16,
        "nhermite": 8,
        "nlaguerre": 4,
        "final_time": 500.0,
        "random_seed": 19,
        "nwrite": 20,
    }
    assert {key: manifest["case_contract"][key] for key in expected_numerics} == expected_numerics
    assert "--expected-revision" in manifest["summary_command"]
    assert (output / "jax_fluxtube_gk_cyclone_nonlinear.in").exists()

    with pytest.raises(RuntimeError, match="revision mismatch"):
        prepare_gx_nonlinear_heat_flux_run(
            gx_root,
            tmp_path / "other",
            expected_revision=PINNED_GX_REVISION,
        )
