from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path


def _load_prepare_module():
    path = Path("scripts/prepare_gx_mode_structure_run.py")
    spec = importlib.util.spec_from_file_location("prepare_gx_mode_structure_run", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_gx_diagnostics_text_updates_existing_section() -> None:
    module = _load_prepare_module()
    text = """[Dimensions]
 ntheta = 32

[Diagnostics]
 nwrite = 1000
 omega = false
 fields = false # old
"""
    patched = module.patch_gx_diagnostics_text(text, nwrite_big=7)
    diagnostics = tomllib.loads(patched)["Diagnostics"]

    assert diagnostics["nwrite"] == 1000
    assert diagnostics["nwrite_big"] == 7
    assert diagnostics["omega"] is True
    assert diagnostics["fields"] is True
    assert diagnostics["moments"] is True
    assert "fields = true # old" in patched


def test_prepare_gx_mode_structure_run_writes_metadata(tmp_path: Path) -> None:
    module = _load_prepare_module()
    source = tmp_path / "itg_salpha_adiabatic_electrons.in"
    source.write_text(
        """[Diagnostics]
 nwrite = 1000
 omega = false
 fields = false
"""
    )
    output_dir = tmp_path / "prepared"
    fixture_output = tmp_path / "fixture.csv"

    metadata = module.prepare_gx_mode_structure_run(
        source,
        output_dir,
        nwrite_big=11,
        gx_executable="/opt/gx/bin/gx",
        ky_values="0.3,0.5",
        gx_z_coordinate="theta_over_2pi",
        fixture_output=fixture_output,
    )

    prepared_input = Path(metadata["prepared_input"])
    assert prepared_input.exists()
    diagnostics = tomllib.loads(prepared_input.read_text())["Diagnostics"]
    assert diagnostics["nwrite_big"] == 11
    assert diagnostics["omega"] is True
    assert diagnostics["fields"] is True
    assert diagnostics["moments"] is True

    metadata_path = output_dir / "mode_structure_run_metadata.json"
    readme_path = output_dir / "README.md"
    loaded = json.loads(metadata_path.read_text())
    assert loaded["gx_big_output"].endswith("itg_salpha_adiabatic_electrons.big.nc")
    assert "--gx-z-coordinate theta_over_2pi" in loaded["export_command"]
    assert "--reference-fixture" in loaded["gate_command"]
    assert "Diagnostics/Phi" in readme_path.read_text()
