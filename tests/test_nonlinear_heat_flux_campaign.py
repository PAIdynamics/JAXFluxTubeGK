import json

import pytest

from scripts.validate_nonlinear_heat_flux_campaign import evaluate_campaign


def _write_report(path, mean, *, normalization="native", stationary=True, drift=0.01):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "test",
                "normalization": normalization,
                "stationary": stationary,
                "statistics": {
                    "mean": mean,
                    "standard_error": abs(mean) * 0.01,
                    "relative_window_drift": drift,
                    "n_samples": 200,
                },
            }
        )
    )
    return path


def test_campaign_requires_both_convergence_axes_and_independent_parity(tmp_path):
    resolution = (
        _write_report(tmp_path / "r0.json", 4.4),
        _write_report(tmp_path / "r1.json", 4.0),
    )
    domain = (
        _write_report(tmp_path / "d0.json", 3.7),
        _write_report(tmp_path / "d1.json", 4.0),
    )
    reference = _write_report(tmp_path / "gx.json", 8.0, normalization="gx")

    report = evaluate_campaign(
        resolution,
        domain,
        reference,
        local_to_reference_factor=2.0,
    )

    assert report["passed"]
    assert report["resolution_convergence"]["passed"]
    assert report["domain_convergence"]["passed"]
    assert report["independent_parity"]["passed"]


def test_campaign_fails_closed_when_a_producer_rejected_a_rung(tmp_path):
    rejected = _write_report(tmp_path / "rejected.json", 4.0, stationary=False)
    accepted = _write_report(tmp_path / "accepted.json", 4.0)
    reference = _write_report(tmp_path / "gx.json", 4.0)

    report = evaluate_campaign(
        (rejected, accepted),
        (accepted, accepted),
        reference,
        local_to_reference_factor=1.0,
    )

    assert not report["passed"]
    assert not report["resolution_convergence"]["all_stationary"]


def test_campaign_requires_two_rungs_per_axis(tmp_path):
    report = _write_report(tmp_path / "report.json", 4.0)
    with pytest.raises(ValueError, match="at least two reports"):
        evaluate_campaign((report,), (report, report), report, local_to_reference_factor=1.0)
