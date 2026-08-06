from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_EIK = ROOT / "fixtures/gx_desc_dshape_rho05_alpha0.eik.out"


def _load_module():
    path = ROOT / "scripts/run_w7x_production_cpu_timing.py"
    spec = importlib.util.spec_from_file_location("run_w7x_production_cpu_timing", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_w7x_production_cpu_timing_blocks_without_external_parity(tmp_path):
    module = _load_module()
    output = tmp_path / "production_cpu_timing.json"

    exit_code = module.main(
        [
            "--output",
            str(output),
            "--readiness-gate",
            str(tmp_path / "missing_readiness_gate.json"),
            "--preset",
            "production-control",
        ]
    )

    payload = json.loads(output.read_text())
    assert exit_code == 0
    assert payload["status"] == "blocked_until_external_parity_passes"
    assert not payload["passed"]
    assert not payload["production_claim"]
    assert payload["controls"]["n_z"] == 256
    assert len(payload["controls"]["ky_values"]) == 28
    assert payload["timing"] is None


def test_w7x_production_cpu_timing_smoke_override_runs_residual_timing(tmp_path):
    module = _load_module()
    output = tmp_path / "smoke_cpu_timing.json"

    exit_code = module.main(
        [
            "--output",
            str(output),
            "--preset",
            "smoke",
            "--eik-reference",
            str(SMOKE_EIK),
            "--allow-pending-external-parity",
            "--repeats",
            "1",
        ]
    )

    payload = json.loads(output.read_text())
    assert exit_code == 0
    assert payload["status"] == "development_timing_only"
    assert not payload["passed"]
    assert not payload["production_claim"]
    assert payload["controls"]["n_z"] == 17
    assert payload["timing"]["repeats"] == 1
    assert payload["timing"]["best_execute_seconds_per_rhs"] >= 0.0
    assert payload["timing"]["estimated_rk4_rhs_calls"] == 24


def test_stella_production_timing_uses_validated_advance_controls(tmp_path):
    module = _load_module()
    geometry = tmp_path / "reference.geometry"
    args = module._parse_args(
        [
            "--preset",
            "stella-production",
            "--stella-geometry",
            str(geometry),
            "--n-windows",
            "7",
        ]
    )

    controls = module.timing_controls(args)
    scan_args = module._stella_scan_args(args, tmp_path / "scratch")

    assert controls["n_z"] == 256
    assert controls["n_vpar"] == 32
    assert controls["n_mu"] == 8
    assert controls["ky_values"] == (0.3,)
    assert scan_args[scan_args.index("--mirror-interpolation") + 1] == "stella_cubic"
    assert scan_args[scan_args.index("--parallel-advance") + 1] == "stella_implicit"
    assert scan_args[scan_args.index("--initial-condition") + 1] == (
        "stella_maxwellian"
    )
    assert scan_args[scan_args.index("--n-windows") + 1] == "7"
