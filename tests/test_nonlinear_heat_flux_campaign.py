import json

import pytest

from scripts.validate_nonlinear_heat_flux_campaign import _parse_args, evaluate_campaign


def _gx_case(ky_min=0.05):
    return {
        "ntheta": 24,
        "nx": 32,
        "ny": 16,
        "nhermite": 8,
        "nlaguerre": 4,
        "final_time": 500.0,
        "random_seed": 19,
        "nwrite": 20,
        "geometry": "s-alpha",
        "q": 1.4,
        "shat": 0.8,
        "eps": 0.18,
        "rmaj_over_lref": 2.77778,
        "fprim": 0.8,
        "tprim": 2.49,
        "ky_min": ky_min,
        "boundary": "linked",
        "electrostatic": True,
        "hyperdiffusion": 0.05,
        "hyperdiffusion_order": 4,
        "collision_frequency": 0.0,
    }


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
    producer="jax-fluxtube-gk/nonlinear-heat-flux",
    reference_case=None,
):
    case = (
        reference_case
        if reference_case is not None
        else {
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
            "max_relative_drift": 0.2,
            "max_relative_standard_error": 0.1,
            "min_stationary_samples": 100,
            "min_stationary_window_duration": 10.0,
            "min_stationary_blocks": 6,
            "min_phi_rms_ratio": 0.8,
            "max_absolute_phi_growth_rate": 0.02,
        }
    )
    payload = {
        "schema_version": 1,
        "producer": producer,
        "normalization": normalization,
        "stationary": stationary,
        "start_time": 0.0,
        "end_time": 100.0,
        "case": case,
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
            "n_blocks": 6,
        },
        "stationary_window_duration": 20.0,
        "nonzonal_phi_rms_ratio": 1.0,
        "candidate_nonzonal_phi_rms_ratio": 1.0,
        "candidate_nonzonal_phi_growth_rate": 0.0,
    }
    if producer == "gx-nonlinear-heat-flux":
        payload["revision"] = "bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5"
        payload["run_manifest"] = "/scratch/gx/gx_nonlinear_run.json"
        payload["source_netcdf"] = "/scratch/gx/jax_fluxtube_gk_cyclone_nonlinear.nc"
        payload["statistics"] |= {"window_start_time": 80.0, "window_end_time": 100.0}
        payload["stationarity_controls"] = {
            "max_relative_drift": 0.2,
            "max_relative_standard_error": 0.1,
            "min_stationary_samples": 100,
            "min_stationary_window_duration": 10.0,
            "min_stationary_blocks": 6,
        }
    path.write_text(json.dumps(payload))
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
    reference = _write_report(
        tmp_path / "gx.json",
        8.0,
        normalization="gx",
        producer="gx-nonlinear-heat-flux",
        reference_case=_gx_case(),
    )
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed) for seed in (1, 2, 3)
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
    assert report["cross_ladder_contract"]["passed"]
    assert report["lineage_producer_contract"]["fixed_case"]


def test_campaign_fails_closed_when_a_producer_rejected_a_rung(tmp_path):
    rejected = _write_report(tmp_path / "rejected.json", 4.0, stationary=False)
    accepted = _write_report(tmp_path / "accepted.json", 4.0)
    reference = _write_report(
        tmp_path / "gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=_gx_case(),
    )
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
    reference = _write_report(
        tmp_path / "gx.json",
        4.0,
        normalization="gx_Q_over_Q_GB",
        producer="gx-nonlinear-heat-flux",
        reference_case=_gx_case(),
    )
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
        (
            local,
            _write_report(
                tmp_path / "fine.json", 4.0, normalization="gx_Q_over_Q_GB", resolution=(16, 16, 8)
            ),
        ),
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
        _write_report(
            tmp_path / "gx-duplicate.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=(local, duplicate, _write_report(tmp_path / "third.json", 4.0, seed=3)),
    )
    assert not duplicate_report["lineage_ensemble"]["all_lineages_unique"]
    assert not duplicate_report["passed"]

    spread_report = evaluate_campaign(
        (local, fine),
        (local, wide),
        _write_report(
            tmp_path / "gx-spread.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=(
            local,
            _write_report(tmp_path / "second.json", 4.0, seed=2),
            outlier,
        ),
    )
    assert not spread_report["lineage_ensemble"]["passed"]

    changed_case = _write_report(tmp_path / "changed-lineage.json", 4.0, seed=2)
    payload = json.loads(changed_case.read_text())
    payload["case"]["hyperdiffusion"] = 0.2
    changed_case.write_text(json.dumps(payload))
    case_report = evaluate_campaign(
        (local, fine),
        (local, wide),
        _write_report(
            tmp_path / "gx-case.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=(
            local,
            changed_case,
            _write_report(tmp_path / "case-third.json", 4.0, seed=3),
        ),
    )
    assert not case_report["lineage_producer_contract"]["fixed_case"]
    assert not case_report["passed"]


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
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed) for seed in (1, 2, 3)
    )

    repeated_report = evaluate_campaign(
        (coarse, repeated),
        (coarse, wide),
        _write_report(
            tmp_path / "gx-repeated.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=lineages,
    )
    assert not repeated_report["resolution_ladder_contract"]["passed"]
    assert not repeated_report["passed"]

    physics_report = evaluate_campaign(
        (coarse, changed_physics),
        (coarse, wide),
        _write_report(
            tmp_path / "gx-physics.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=lineages,
    )
    assert not physics_report["resolution_ladder_contract"]["fixed_physics"]
    assert not physics_report["passed"]

    changed_domain_base = _write_report(tmp_path / "changed-domain-base.json", 4.0)
    changed_domain_wide = _write_report(
        tmp_path / "changed-domain-wide.json",
        4.0,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    for path in (changed_domain_base, changed_domain_wide):
        payload = json.loads(path.read_text())
        payload["case"]["collision_frequency"] = 0.1
        path.write_text(json.dumps(payload))
    cross_report = evaluate_campaign(
        (coarse, _write_report(tmp_path / "cross-fine.json", 4.0, resolution=(16, 16, 8))),
        (changed_domain_base, changed_domain_wide),
        _write_report(
            tmp_path / "gx-cross.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=lineages,
    )
    assert not cross_report["cross_ladder_contract"]["shared_physics"]
    assert not cross_report["passed"]

    mismatched_base = _write_report(
        tmp_path / "mismatched-domain-base.json", 4.0, resolution=(10, 10, 4)
    )
    mismatched_wide = _write_report(
        tmp_path / "mismatched-domain-wide.json",
        4.0,
        resolution=(10, 10, 4),
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    phase_report = evaluate_campaign(
        (coarse, _write_report(tmp_path / "phase-fine.json", 4.0, resolution=(16, 16, 8))),
        (mismatched_base, mismatched_wide),
        _write_report(
            tmp_path / "gx-phase.json",
            4.0,
            producer="gx-nonlinear-heat-flux",
            reference_case=_gx_case(),
        ),
        lineage_paths=lineages,
    )
    assert not phase_report["cross_ladder_contract"]["shared_phase_space_resolution"]
    assert not phase_report["passed"]


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
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed) for seed in (1, 2, 3)
    )
    gx_case = _gx_case(0.2)
    gx = _write_report(
        tmp_path / "gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=gx_case,
    )
    report = evaluate_campaign((coarse, fine), (coarse, wide), gx, lineage_paths=lineages)
    assert not report["independent_parity_contract"]["passed"]
    assert not report["independent_parity_contract"]["checks"]["ky_min"]
    assert not report["passed"]

    gx_case["ky_min"] = 0.05
    gx = _write_report(
        tmp_path / "wrong-revision-gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=gx_case,
    )
    payload = json.loads(gx.read_text())
    payload["revision"] = "0" * 40
    gx.write_text(json.dumps(payload))
    report = evaluate_campaign((coarse, fine), (coarse, wide), gx, lineage_paths=lineages)
    assert not report["independent_parity_contract"]["checks"]["revision"]
    assert not report["passed"]

    gx_case["nx"] = 16
    gx = _write_report(
        tmp_path / "underresolved-gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=gx_case,
    )
    report = evaluate_campaign((coarse, fine), (coarse, wide), gx, lineage_paths=lineages)
    assert not report["independent_parity_contract"]["checks"]["resolution_contract"]
    assert not report["passed"]


def test_campaign_parity_uses_finest_domain_rung(tmp_path):
    resolution = (
        _write_report(tmp_path / "r0.json", 4.0),
        _write_report(tmp_path / "r1.json", 4.0, resolution=(16, 16, 8)),
    )
    domain = (
        _write_report(tmp_path / "d0.json", 8.0),
        _write_report(
            tmp_path / "d1.json",
            8.0,
            kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
            ky=(0.0, 0.05, 0.1, 0.15, 0.2),
        ),
    )
    reference = _write_report(
        tmp_path / "gx.json",
        8.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=_gx_case(),
    )
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed) for seed in (1, 2, 3)
    )

    report = evaluate_campaign(resolution, domain, reference, lineage_paths=lineages)

    assert report["passed"]
    assert report["independent_parity"]["mean_relative_error"] == pytest.approx(0.0)


def test_campaign_rejects_unknown_reference_and_irregular_fourier_grid(tmp_path):
    coarse = _write_report(tmp_path / "coarse.json", 4.0)
    fine = _write_report(tmp_path / "fine.json", 4.0, resolution=(16, 16, 8))
    wide = _write_report(
        tmp_path / "wide.json",
        4.0,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed) for seed in (1, 2, 3)
    )
    unknown = _write_report(tmp_path / "unknown.json", 4.0, producer="unknown")

    report = evaluate_campaign((coarse, fine), (coarse, wide), unknown, lineage_paths=lineages)
    assert not report["independent_parity_contract"]["passed"]
    assert not report["passed"]

    irregular = _write_report(
        tmp_path / "irregular.json",
        4.0,
        resolution=(16, 16, 8),
        kx=(-2.0, -0.7, 0.0, 0.7, 2.0),
    )
    with pytest.raises(ValueError, match="invalid Fourier grid"):
        evaluate_campaign((coarse, irregular), (coarse, wide), unknown, lineage_paths=lineages)


def test_campaign_rejects_stationarity_declared_with_weakened_controls(tmp_path):
    coarse = _write_report(tmp_path / "coarse.json", 4.0)
    fine = _write_report(tmp_path / "fine.json", 4.0, resolution=(16, 16, 8))
    wide = _write_report(
        tmp_path / "wide.json",
        4.0,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    weak = _write_report(tmp_path / "weak.json", 4.0, seed=1)
    payload = json.loads(weak.read_text())
    payload["case"]["min_stationary_samples"] = 2
    weak.write_text(json.dumps(payload))
    lineages = (
        weak,
        _write_report(tmp_path / "lineage2.json", 4.0, seed=2),
        _write_report(tmp_path / "lineage3.json", 4.0, seed=3),
    )
    gx = _write_report(
        tmp_path / "gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=_gx_case(),
    )

    report = evaluate_campaign((coarse, fine), (coarse, wide), gx, lineage_paths=lineages)

    assert not report["stationarity_evidence_contract"]["passed"]
    weak_evidence = next(
        item
        for item in report["stationarity_evidence_contract"]["local"]
        if item["path"] == str(weak)
    )
    assert not weak_evidence["checks"]["declared_sample_minimum"]
    assert not report["passed"]


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--convergence-tolerance", "nan"),
        ("--mean-tolerance", "inf"),
        ("--local-to-reference-factor", "nan"),
    ),
)
def test_campaign_cli_rejects_nonfinite_acceptance_controls(option, value):
    argv = [
        "--resolution-report",
        "r0.json",
        "--resolution-report",
        "r1.json",
        "--domain-report",
        "d0.json",
        "--domain-report",
        "d1.json",
        "--reference-report",
        "gx.json",
        "--lineage-report",
        "s0.json",
        "--lineage-report",
        "s1.json",
        "--lineage-report",
        "s2.json",
        "--output",
        "campaign.json",
        option,
        value,
    ]
    with pytest.raises(SystemExit):
        _parse_args(argv)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", 1.0, "seed must be an integer"),
        ("initial_amplitude", float("nan"), "amplitude must be finite"),
        ("segment_end_times", [99.0], "terminate at report end time"),
    ),
)
def test_campaign_rejects_invalid_lineage_identity(tmp_path, field, value, message):
    coarse = _write_report(tmp_path / "coarse.json", 4.0)
    fine = _write_report(tmp_path / "fine.json", 4.0, resolution=(16, 16, 8))
    wide = _write_report(
        tmp_path / "wide.json",
        4.0,
        kx=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
        ky=(0.0, 0.05, 0.1, 0.15, 0.2),
    )
    lineages = tuple(
        _write_report(tmp_path / f"lineage{seed}.json", 4.0, seed=seed) for seed in (1, 2, 3)
    )
    payload = json.loads(lineages[0].read_text())
    payload["trajectory_lineage"][field] = value
    lineages[0].write_text(json.dumps(payload))
    gx = _write_report(
        tmp_path / "gx.json",
        4.0,
        producer="gx-nonlinear-heat-flux",
        reference_case=_gx_case(),
    )

    with pytest.raises(ValueError, match=message):
        evaluate_campaign((coarse, fine), (coarse, wide), gx, lineage_paths=lineages)
