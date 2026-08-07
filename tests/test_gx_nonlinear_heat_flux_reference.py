import numpy as np
import pytest

from scripts.summarize_gx_nonlinear_heat_flux import (
    gx_flux_stationary,
    read_gx_heat_flux,
    summarize_heat_flux,
)


def test_gx_heat_flux_summary_matches_declared_stationary_window():
    times = np.arange(8.0)
    flux = np.asarray([0.0, 1.0, 3.0, 5.0, 4.0, 4.0, 4.0, 4.0])

    summary = summarize_heat_flux(times, flux, start_fraction=0.5)

    assert summary["mean"] == pytest.approx(4.0)
    assert summary["standard_deviation"] == pytest.approx(0.0)
    assert summary["standard_error"] == pytest.approx(0.0)
    assert summary["relative_window_drift"] == pytest.approx(0.0)
    assert summary["n_samples"] == 4
    assert summary["window_start_time"] == pytest.approx(4.0)
    assert gx_flux_stationary(summary, min_samples=4, min_window_duration=3.0)
    assert not gx_flux_stationary(summary)


def test_gx_heat_flux_reader_uses_documented_netcdf_groups(tmp_path):
    from netCDF4 import Dataset

    path = tmp_path / "gx.nc"
    with Dataset(path, mode="w") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("species", 2)
        grids = dataset.createGroup("Grids")
        diagnostics = dataset.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = [0.0, 1.0, 2.0]
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "species"))[:] = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]

    times, heat_flux = read_gx_heat_flux(path, species_index=1)

    np.testing.assert_allclose(times, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(heat_flux, [2.0, 4.0, 6.0])


@pytest.mark.parametrize(
    ("times", "flux", "message"),
    (
        ([0.0], [1.0], "matching one-dimensional"),
        ([0.0, 0.0], [1.0, 1.0], "strictly increasing"),
        ([0.0, 1.0], [1.0, np.nan], "finite"),
    ),
)
def test_gx_heat_flux_summary_rejects_invalid_traces(times, flux, message):
    with pytest.raises(ValueError, match=message):
        summarize_heat_flux(times, flux)
