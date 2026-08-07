import json

import pytest

from scripts.validate_nonlinear_heat_flux_campaign import evaluate_campaign


def _write_report(
    path, mean, *, normalization="native", stationary=True, drift=0.01, seed=1
):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "test",
                "normalization": normalization,
                "stationary": stationary,
                "trajectory_lineage": {
                    "schema_version": 1,
                    "seed": seed,
                    "initial_amplitude": 1.0e-3,
                    "initial_zonal_fraction": 0.0,
                    "segment_end_times": [100.0],
                },
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
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed)
        for seed in (1, 2, 3)
    )

    report = evaluate_campaign(
        resolution,
        domain,
        reference,
        lineage_paths=lineages,
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
    lineages = (
        _write_report(tmp_path / "lineage1.json", 4.0, seed=1),
        _write_report(tmp_path / "lineage2.json", 4.0, seed=2),
        _write_report(tmp_path / "lineage3.json", 4.0, seed=3, stationary=False),
    )

    report = evaluate_campaign(
        (rejected, accepted),
        (accepted, accepted),
        reference,
        lineage_paths=lineages,
        local_to_reference_factor=1.0,
    )

    assert not report["passed"]
    assert not report["resolution_convergence"]["all_stationary"]


def test_campaign_requires_two_rungs_per_axis(tmp_path):
    report = _write_report(tmp_path / "report.json", 4.0)
    with pytest.raises(ValueError, match="at least two reports"):
        evaluate_campaign(
            (report,),
            (report, report),
            report,
            lineage_paths=(report, report, report),
            local_to_reference_factor=1.0,
        )


def test_campaign_needs_no_factor_for_source_matched_normalization(tmp_path):
    local = _write_report(tmp_path / "local.json", 4.0, normalization="gx_Q_over_Q_GB")
    reference = _write_report(tmp_path / "gx.json", 4.0, normalization="gx_Q_over_Q_GB")
    lineages = tuple(
        _write_report(
            tmp_path / f"lineage{seed}.json",
            4.0,
            normalization="gx_Q_over_Q_GB",
            seed=seed,
        )
        for seed in (1, 2, 3)
    )

    report = evaluate_campaign(
        (local, local), (local, local), reference, lineage_paths=lineages
    )

    assert report["passed"]
    assert report["independent_parity"]["local_to_reference_factor"] == pytest.approx(1.0)


def test_campaign_rejects_duplicate_or_inconsistent_lineages(tmp_path):
    local = _write_report(tmp_path / "local.json", 4.0, seed=1)
    duplicate = _write_report(tmp_path / "duplicate.json", 4.0, seed=1)
    outlier = _write_report(tmp_path / "outlier.json", 7.0, seed=3)

    duplicate_report = evaluate_campaign(
        (local, local),
        (local, local),
        local,
        lineage_paths=(local, duplicate, _write_report(tmp_path / "third.json", 4.0, seed=3)),
    )
    assert not duplicate_report["lineage_ensemble"]["all_lineages_unique"]
    assert not duplicate_report["passed"]

    spread_report = evaluate_campaign(
        (local, local),
        (local, local),
        local,
        lineage_paths=(
            local,
            _write_report(tmp_path / "second.json", 4.0, seed=2),
            outlier,
        ),
    )
    assert not spread_report["lineage_ensemble"]["passed"]
