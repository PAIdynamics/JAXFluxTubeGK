from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from stellarator_gk import PerKyModeStructureFixture


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "fixtures/w7x_itg_stella_matched_time_ladder/frequency_profile_convention_audit.json"


def _load_module():
    path = ROOT / "scripts/audit_w7x_stella_frequency_profile_conventions.py"
    spec = importlib.util.spec_from_file_location(
        "audit_w7x_stella_frequency_profile_conventions",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(z, phi):
    return PerKyModeStructureFixture(
        ky=np.asarray([0.1]),
        z=np.asarray(z),
        phi=np.asarray(phi),
        growth_rate=np.asarray([0.0]),
        frequency=np.asarray([0.0]),
        source="synthetic",
        normalization="unit",
    )


def test_frequency_variant_audit_detects_sign_flip():
    module = _load_module()
    observed = np.asarray([-1.0, -2.0, -3.0])
    reference = np.asarray([1.0, 2.0, 3.0])

    direct = module._frequency_error_summary(observed, reference)
    flipped = module._frequency_error_summary(observed, -reference)

    assert direct["max_abs_error"] == 6.0
    assert flipped["max_abs_error"] == 0.0


def test_profile_variant_audit_detects_conjugation():
    module = _load_module()
    z = np.linspace(-0.5, 0.5, 8, endpoint=False)
    observed = np.exp(-2.0j * np.pi * z)[None, :]
    reference = np.exp(2.0j * np.pi * z)[None, :]

    variants = module._profile_variant_errors(_fixture(z, observed), _fixture(z, reference))

    assert variants["conjugate_reference"]["max_phase_aligned_error"] < 1.0e-14
    assert variants["direct_reference"]["max_phase_aligned_error"] > 0.1


def test_profile_variant_audit_detects_circular_shift():
    module = _load_module()
    z = np.linspace(-0.5, 0.5, 8, endpoint=False)
    reference = np.exp(2.0j * np.pi * z)[None, :]
    observed = np.roll(reference, 2, axis=1)

    variants = module._profile_variant_errors(_fixture(z, observed), _fixture(z, reference))

    assert variants["best_common_circular_shift"]["shift_index"] == 2
    assert variants["best_common_circular_shift"]["max_phase_aligned_error"] < 1.0e-14


def test_committed_w7x_frequency_profile_convention_audit_remains_open():
    import json

    report = json.loads(AUDIT.read_text())
    summary = report["summary"]

    assert report["status"] == "open"
    assert summary["direct_frequency_max_abs_error"] > 0.16
    assert summary["best_frequency_variant"] == "least_squares_affine_reference"
    assert summary["best_frequency_max_abs_error"] > 0.03
    assert summary["best_profile_variant"] == "best_common_circular_shift"
    assert summary["best_profile_max_abs_error"] > 0.17
    assert not summary["frequency_sign_flip_improves"]
    assert not summary["profile_conjugation_improves"]
    assert "inspect velocity/RHS terms" in summary["next_action"]
