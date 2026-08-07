import json

import pytest

from scripts.validate_nonlinear_heat_flux_campaign import evaluate_campaign


def _write_report(
    path,
    mean,
    *,
    normalization="native",
    stationary=True,
    drift=0.01,
    seed=1,
    resolution=(12, 12, 6),
    kx=(-2.0, -1.0, 0.0, 1.0, 2.0),
    ky=(0.0, 0.1, 0.2),
    producer="test",
    reference_case=None,
):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": producer,
                "normalization": normalization,
                "stationary": stationary,
                "case": reference_case if reference_case is not None else {
                    "n_z": resolution[0],
                    "n_vpar": resolution[1],
                    "n_mu": resolution[2],
                    "n_kx": len(kx),
                    "n_ky": len(ky),
                    "kx": list(kx),
                    "ky": list(ky),
                    "parallel_boundary_model": "twist_shift",
                    "parallel_recurrence_rate": 1.0,
                    "rmaj_over_lref": 2.77778,
                    "gx_fprim": 0.8,
                    "gx_tprim": 2.49,
                    "density_gradient_R_over_Ln": 2.222224,
                    "temperature_gradient_R_over_LT": 6.9166722,
                    "hyperdiffusion": 0.05,
                    "collision_frequency": 0.0,
                    "flux_moment": "gx_total_energy",
                    "ikxspace": 1,
                },
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
        _write_report(tmp_path / "r1.json", 4.0, resolution=(16, 16, 8)),
    )
    domain = (
        _write_report(tmp_path / "d0.json", 3.7),
        _write_report(
            tmp_path / "d1.json",
            4.0,
            kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
            ky=(0.0, 0.05, 0.1, 0.15, 0.2),
        ),
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
    assert report["resolution_ladder_contract"]["passed"]
    assert report["domain_ladder_contract"]["passed"]


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
        (rejected, _write_report(tmp_path / "fine.json", 4.0, resolution=(16, 16, 8))),
        (
            accepted,
            _write_report(
                tmp_path / "wide.json",
                4.0,
                kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
                ky=(0.0, 0.05, 0.1, 0.15, 0.2),
            ),
        ),
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
        (local, _write_report(tmp_path / "fine.json", 4.0, normalization="gx_Q_over_Q_GB", resolution=(16, 16, 8))),
        (
            local,
            _write_report(
                tmp_path / "wide.json",
                4.0,
                normalization="gx_Q_over_Q_GB",
                kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
                ky=(0.0, 0.05, 0.1, 0.15, 0.2),
            ),
        ),
        reference,
        lineage_paths=lineages,
    )

    assert report["passed"]
    assert report["independent_parity"]["local_to_reference_factor"] == pytest.approx(1.0)


def test_campaign_rejects_duplicate_or_inconsistent_lineages(tmp_path):
    local = _write_report(tmp_path / "local.json", 4.0, seed=1)
    duplicate = _write_report(tmp_path / "duplicate.json", 4.0, seed=1)
    outlier = _write_report(tmp_path / "outlier.json", 7.0, seed=3)

    fine = _write_report(tmp_path / "fine.json", 4.0, seed=4, resolution=(16, 16, 8))
    wide = _write_report(
        tmp_path / "wide.json",
        4.0,
        seed=4,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    duplicate_report = evaluate_campaign(
        (local, fine),
        (local, wide),
        local,
        lineage_paths=(local, duplicate, _write_report(tmp_path / "third.json", 4.0, seed=3)),
    )
    assert not duplicate_report["lineage_ensemble"]["all_lineages_unique"]
    assert not duplicate_report["passed"]

    spread_report = evaluate_campaign(
        (local, fine),
        (local, wide),
        local,
        lineage_paths=(
            local,
            _write_report(tmp_path / "second.json", 4.0, seed=2),
            outlier,
        ),
    )
    assert not spread_report["lineage_ensemble"]["passed"]


def test_campaign_rejects_fake_ladders_that_repeat_or_change_physics(tmp_path):
    coarse = _write_report(tmp_path / "coarse.json", 4.0)
    repeated = _write_report(tmp_path / "repeated.json", 4.0)
    changed_physics = _write_report(tmp_path / "changed.json", 4.0, resolution=(16, 16, 8))
    payload = json.loads(changed_physics.read_text())
    payload["case"]["hyperdiffusion"] = 0.1
    changed_physics.write_text(json.dumps(payload))
    wide = _write_report(
        tmp_path / "wide.json",
        4.0,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed)
        for seed in (1, 2, 3)
    )

    repeated_report = evaluate_campaign(
        (coarse, repeated), (coarse, wide), coarse, lineage_paths=lineages
    )
    assert not repeated_report["resolution_ladder_contract"]["passed"]
    assert not repeated_report["passed"]

    physics_report = evaluate_campaign(
        (coarse, changed_physics), (coarse, wide), coarse, lineage_paths=lineages
    )
    assert not physics_report["resolution_ladder_contract"]["fixed_physics"]
    assert not physics_report["passed"]


def test_campaign_requires_gx_reference_case_to_match_local_physics(tmp_path):
    coarse = _write_report(tmp_path / "coarse.json", 4.0)
    fine = _write_report(tmp_path / "fine.json", 4.0, resolution=(16, 16, 8))
    wide = _write_report(
        tmp_path / "wide.json",
        4.0,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed)
        for seed in (1, 2, 3)
    )
    gx_case = {
        "geometry": "s-alpha",
        "q": 1.4,
        "shat": 0.8,
        "eps": 0.18,
        "rmaj_over_lref": 2.77778,
        "fprim": 0.8,
        "tprim": 2.49,
        "ky_min": 0.2,
        "boundary": "linked",
        "electrostatic": True,
        "hyperdiffusion": 0.05,
        "hyperdiffusion_order": 4,
    }
    gx = _write_report(
        tmp_path / "gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=gx_case,
    )
    report = evaluate_campaign(
        (coarse, fine), (coarse, wide), gx, lineage_paths=lineages
    )
    assert not report["independent_parity_contract"]["passed"]
    assert not report["independent_parity_contract"]["checks"]["ky_min"]
    assert not report["passed"]
