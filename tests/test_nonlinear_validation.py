import json

import pytest

from stellarator_gk import (
    NonlinearHeatFluxRecord,
    compare_nonlinear_heat_flux,
    compare_nonlinear_heat_flux_convergence,
    load_nonlinear_heat_flux_record,
)


def _record(mean=4.0, error=0.1, drift=0.02, normalization="shared"):
    return NonlinearHeatFluxRecord("test", normalization, mean, error, drift, 20)


def test_nonlinear_parity_requires_stationarity_and_mean_agreement():
    report = compare_nonlinear_heat_flux(_record(4.1), _record(4.0))

    assert report.passed
    assert report.local_stationary
    assert report.reference_stationary
    assert report.mean_relative_error == pytest.approx(0.025)

    drifting = compare_nonlinear_heat_flux(_record(drift=0.4), _record())
    assert not drifting.passed
    assert not drifting.local_stationary


def test_nonlinear_parity_requires_explicit_normalization_conversion():
    local = _record(mean=2.0, normalization="local")
    reference = _record(mean=4.0, normalization="gx")

    with pytest.raises(ValueError, match="normalizations differ"):
        compare_nonlinear_heat_flux(local, reference)
    report = compare_nonlinear_heat_flux(local, reference, local_to_reference_factor=2.0)
    assert report.passed


def test_nonlinear_convergence_requires_stationary_finest_pair():
    report = compare_nonlinear_heat_flux_convergence(
        (_record(3.0), _record(4.2), _record(4.0))
    )
    assert report.passed
    assert report.finest_relative_change == pytest.approx(0.05)

    failed = compare_nonlinear_heat_flux_convergence(
        (_record(3.0), _record(4.0, drift=0.3))
    )
    assert not failed.passed
    assert not failed.all_stationary


def test_nonlinear_record_loader_validates_schema(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer": "gx-nonlinear-heat-flux",
                "normalization": "gx_Q_over_Q_GB",
                "statistics": {
                    "mean": 3.0,
                    "standard_error": 0.1,
                    "relative_window_drift": 0.02,
                    "n_samples": 10,
                },
            }
        )
    )

    record = load_nonlinear_heat_flux_record(path)
    assert record.mean == pytest.approx(3.0)
    assert record.normalization == "gx_Q_over_Q_GB"
