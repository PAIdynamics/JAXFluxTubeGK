from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "examples/desc_fixture_optimization_loop.py"
    spec = importlib.util.spec_from_file_location("desc_fixture_optimization_loop", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_desc_optimization_readiness_status_reports_missing_gate(tmp_path):
    module = _load_module()

    status = module._load_readiness_status(tmp_path / "missing_readiness_gate.json")

    assert status["status"] == "missing_production_readiness_gate"
    assert not status["passed"]
    assert status["desc_optimization_status"] == "keep_reduced_until_readiness_gate_exists"


def test_desc_optimization_loop_require_production_ready_blocks_on_open_gate():
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "examples/desc_fixture_optimization_loop.py",
            "--iterations",
            "1",
            "--require-production-ready",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "production DESC optimization is blocked" in result.stderr
