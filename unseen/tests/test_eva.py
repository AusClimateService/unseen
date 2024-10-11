"""Test extreme value analysis functions."""

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from unseen.eva import fit_gev, get_return_period, get_return_level

rtol = 0.3  # relative tolerance for testing close values


def add_example_gev_trend(data):
    trend = np.arange(data.time.size) * 2.5 / data.time.size
    trend = xr.DataArray(trend, coords={"time": data.time})
    return data + trend


@pytest.mark.parametrize("example_da_gev", ["xarray", "numpy", "dask"], indirect=True)
def test_fit_gev_1d(example_da_gev):
    """Run stationary GEV fit using 1D array."""
    data, dparams_i = example_da_gev
    dparams = fit_gev(
        data, stationary=True, assert_good_fit=False, pick_best_model=False
    )
    # Check fitted params match params used to create data
    npt.assert_allclose(dparams, dparams_i, rtol=rtol)


@pytest.mark.parametrize("example_da_gev", ["xarray", "numpy", "dask"], indirect=True)
@pytest.mark.parametrize(
    "fitstart",
    [
        [1, -5, 1],
        "LMM",
        "scipy_fitstart",
        "scipy",
        "scipy_subset",
        "xclim_fitstart",
        "xclim",
    ],
)
def test_fit_gev_1d_fitstart(example_da_gev, fitstart):
    """Run stationary GEV fit using 1D array & fitstart method."""
    data, dparams_i = example_da_gev
    dparams = fit_gev(
        data,
        stationary=True,
        fitstart=fitstart,
        assert_good_fit=False,
        pick_best_model=False,
    )
    # Check fitted params match params used to create data
    npt.assert_allclose(dparams, dparams_i, rtol=rtol)


@pytest.mark.parametrize("example_da_gev", ["xarray", "numpy", "dask"], indirect=True)
def test_fit_gev_1d_assert_good_fit(example_da_gev):
    """Run stationary GEV fit using 1D array & fit_goodness_test."""
    data, dparams_i = example_da_gev
    dparams = fit_gev(data, stationary=True, assert_good_fit=True)
    # Check fitted params match params used to create data
    npt.assert_allclose(dparams, dparams_i, rtol=0.3)


# todo FAILED unseen/tests/test_eva.py::test_fit_gev_3d[xarray] - AssertionError:
@pytest.mark.parametrize("example_da_gev_3d", ["xarray", "dask"], indirect=True)
def test_fit_gev_3d(example_da_gev_3d):
    """Run stationary GEV fit using 3D array & check results."""
    data, dparams_i = example_da_gev_3d
    dparams = fit_gev(data, stationary=True, fitstart="LMM", core_dim="time")
    # Check fitted params match params used to create data
    npt.assert_allclose(dparams, dparams_i, rtol=0.4)


@pytest.mark.parametrize("example_da_gev", ["xarray", "dask"], indirect=True)
def test_fit_ns_gev_1d(example_da_gev):
    """Run non-stationary GEV fit using 1D array & check results."""
    data, _ = example_da_gev
    data = add_example_gev_trend(data)
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")

    dparams = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        covariate=covariate,
    )
    assert np.all(dparams[2] > 0)  # Positive trend in location


@pytest.mark.parametrize("example_da_gev", ["xarray", "dask"], indirect=True)
def test_fit_ns_gev_1d_loc_only(example_da_gev):
    """Run non-stationary GEV fit using 1D array (location parameter only)."""
    data, _ = example_da_gev
    data = add_example_gev_trend(data)
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")

    dparams = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        loc1=0,
        scale1=None,
        covariate=covariate,
    )
    assert np.all(dparams[2] > 0)  # Positive trend in location
    assert np.all(dparams[4] == 0)  # No trend in scale


@pytest.mark.parametrize("example_da_gev", ["xarray"], indirect=True)
def test_fit_ns_gev_1d_scale_only(example_da_gev):
    """Run non-stationary GEV fit using 1D array (scale parameter only)."""
    data, _ = example_da_gev
    data = add_example_gev_trend(data)
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")

    dparams = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        loc1=None,
        scale1=0,
        covariate=covariate,
    )
    assert np.all(dparams[2] == 0)  # No trend in location
    assert np.all(dparams[4] != 0)  # Nonzero trend in scale


@pytest.mark.parametrize("example_da_gev_3d", ["xarray", "dask"], indirect=True)
def test_fit_ns_gev_3d(example_da_gev_3d):
    """Run non-stationary GEV fit using 3D array & check results."""
    data, _ = example_da_gev_3d
    # Add a positive linear trend
    data = add_example_gev_trend(data)
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")
    dparams = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")
    assert np.all(dparams.isel(dparams=2) > 0)  # Positive trend in location


@pytest.mark.parametrize("example_da_gev", ["xarray"], indirect=True)
def test_fit_ns_gev_1d_pick_best_model_bic_trend(example_da_gev):
    """Run non-stationary GEV fit & check 'BIC' test returns nonstationary params."""
    data, _ = example_da_gev
    # Add a large positive linear trend
    data = add_example_gev_trend(data)
    data = add_example_gev_trend(data)
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")

    dparams = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        covariate=covariate,
        pick_best_model="bic",
    )
    assert np.all(dparams[2] > 0)  # Positive trend in location


@pytest.mark.parametrize("example_da_gev", ["xarray"], indirect=True)
def test_fit_ns_gev_1d_pick_best_model_bic_no_trend(example_da_gev):
    """Run non-stationary GEV fit & check 'BIC' test returns stationary params."""
    data, _ = example_da_gev
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")

    dparams = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        covariate=covariate,
        pick_best_model="bic",
    )
    assert np.all(dparams[2] == 0)  # No trend in location
    assert np.all(dparams[4] == 0)  # No trend in scale


@pytest.mark.parametrize("example_da_gev", ["xarray", "numpy", "dask"], indirect=True)
def test_get_return_period(example_da_gev):
    """Run get_return_period for a single event using 1d data."""
    data, _ = example_da_gev
    event = np.mean(data)
    ari = get_return_period(event, data=data)
    assert ari.size == 1
    assert np.all(np.isfinite(ari))


@pytest.mark.parametrize("example_da_gev_3d", ["xarray", "dask"], indirect=True)
def test_get_return_period_3d(example_da_gev_3d):
    """Run get_return_period for 3d array of events using 3d data."""
    data, dparams = example_da_gev_3d
    dparams = fit_gev(data, stationary=True)
    # Multiple events unique to each lat/lon
    event = data.quantile([0.25, 0.5, 0.75], dim="time")
    ari = get_return_period(event, dparams)
    assert ari.shape == event.shape
    assert np.all(np.isfinite(ari))


@pytest.mark.parametrize("example_da_gev_3d", ["xarray", "dask"], indirect=True)
def test_get_return_period_3d_nonstationary(example_da_gev_3d):
    """Run get_return_period for 3d events using 3d nonstationary data."""
    data, _ = example_da_gev_3d
    data = add_example_gev_trend(data)
    covariate = xr.DataArray(np.arange(data.time.size), dims="time")
    params = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")

    # Multiple events unique to each lat/lon
    event = data.quantile([0.25, 0.5, 0.75], dim="time")
    covariate_subset = xr.DataArray([0, covariate.size], dims="time")
    ari = get_return_period(event, params, covariate=covariate_subset)
    assert ari.shape == (*list(event.shape), covariate_subset.size)
    assert np.all(np.isfinite(ari))


@pytest.mark.parametrize("example_da_gev", ["xarray", "numpy", "dask"], indirect=True)
@pytest.mark.parametrize("ari", [100, np.array([10, 100, 1000])])
def test_get_return_level(example_da_gev, ari):
    """Run get_return_level for 1d array of periods using 1d data."""
    _, dparams = example_da_gev
    return_level = get_return_level(ari, dparams)
    if isinstance(ari, int):
        assert return_level.size == 1
    else:
        assert return_level.shape == ari.shape
    assert np.all(np.isfinite(return_level))


@pytest.mark.parametrize("example_da_gev_3d", ["xarray", "dask"], indirect=True)
def test_get_return_level_3d(example_da_gev_3d):
    """Run get_return_level for 3d array of periods using 3d data."""
    data, dparams = example_da_gev_3d
    dparams = fit_gev(data, stationary=True, core_dim="time")

    # Multiple events unique to each lat/lon
    dims = ("lat", "lon", "return_period")
    ari = np.array([10, 100, 1000] * 4).T
    ari = xr.DataArray(ari.reshape(dparams.shape), dims=dims)
    return_level = get_return_level(ari, dparams)

    assert return_level.shape == ari.shape
    assert np.all(np.isfinite(return_level))
