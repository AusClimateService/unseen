"""Test extreme value analysis functions."""

from matplotlib.dates import date2num
import numpy as np
import numpy.testing as npt
from scipy.stats import genextreme
import xarray as xr

from unseen.eva import fit_gev, check_gev_fit


rtol = 0.3  # relative tolerance
alpha = 0.05


def example_da_gev_1d():
    """An example 1D GEV DataArray and distribution parameters."""
    time = xr.cftime_range(start="2000-01-01", periods=1500, freq="D")

    # Shape, location and scale parameters.
    np.random.seed(0)
    shape = np.random.uniform()
    loc = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.1, 10)
    theta = shape, loc, scale

    rvs = genextreme.rvs(shape, loc=loc, scale=scale, size=(time.size), random_state=0)
    data = xr.DataArray(rvs, coords=[time], dims=["time"])
    return data, theta


def example_da_gev_1d_dask():
    """An example 1D GEV dask array and distribution parameters."""
    data, theta = example_da_gev_1d()
    data = data.chunk({"time": -1})
    return data, theta


def example_da_gev_3d():
    """An example 3D GEV DataArray and distribution parameters."""
    time = xr.cftime_range(start="2000-01-01", periods=1500, freq="D")
    lat = np.arange(2)
    lon = np.arange(2)

    # Shape, location and scale parameters.
    size = (len(lat), len(lon))
    np.random.seed(0)
    shape = np.random.uniform(size=size)
    loc = np.random.uniform(-10, 10, size=size)
    scale = np.random.uniform(0.1, 10, size=size)
    theta = np.stack([shape, loc, scale], axis=-1)

    rvs = genextreme.rvs(
        shape,
        loc=loc,
        scale=scale,
        size=(len(time), len(lat), len(lon)),
        random_state=0,
    )
    data = xr.DataArray(rvs, coords=[time, lat, lon], dims=["time", "lat", "lon"])
    return data, theta


def example_da_gev_3d_dask():
    """An example 3D GEV dask array and its distribution parameters."""
    data, theta = example_da_gev_3d()
    data = data.chunk({"time": -1, "lat": 1, "lon": 1})
    return data, theta


def add_example_gev_trend(data):
    trend = np.arange(data.time.size) * 2 / data.time.size
    trend = xr.DataArray(trend, coords={"time": data.time})
    return data + trend


def example_da_gev_forecast():
    """Create example stacked forecast dataArray."""
    ensemble = np.arange(3)
    lead_time = np.arange(5)
    init_date = xr.cftime_range(start="2000-01-01", periods=24, freq="MS")
    time = [
        init_date.shift(i, freq="MS")[: len(lead_time)] for i in range(len(init_date))
    ]

    # Generate shape, location and scale parameters.
    np.random.seed(2)
    shape = np.random.uniform()
    loc = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.1, 10)
    theta = shape, loc, scale

    rvs = genextreme.rvs(
        shape,
        loc=loc,
        scale=scale,
        size=(len(ensemble), len(init_date), len(lead_time)),
        random_state=0,
    )
    data = xr.DataArray(
        rvs,
        coords=[ensemble, init_date, lead_time],
        dims=["ensemble", "init_date", "lead_time"],
    )
    data = data.assign_coords({"time": (["init_date", "lead_time"], time)})
    data_stacked = data.stack({"sample": ["ensemble", "init_date", "lead_time"]})
    return data_stacked, theta


def test_fit_gev_1d():
    """Run stationary fit using 1D array & check results."""
    data, theta_i = example_da_gev_1d()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_numpy():
    """Run stationary fit using 1D np.ndarray & check results."""
    data, theta_i = example_da_gev_1d()
    data = data.values
    theta = fit_gev(data, stationary=True, core_dim=None)
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_dask():
    """Run stationary fit using 1D dask array & check results."""
    data, theta_i = example_da_gev_1d_dask()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d():
    """Run stationary fit using 3D array & check results."""
    data, theta_i = example_da_gev_3d()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d_dask():
    """Run stationary fit using 3D dask array & check results."""
    data, theta_i = example_da_gev_3d_dask()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_ns_gev_1d():
    """Run non-stationary fit using 1D array & check results."""
    data, _ = example_da_gev_1d()
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(data, stationary=False, core_dim="time", covariate=covariate)
    pvalue = check_gev_fit(data, theta, covariate=covariate, core_dim="time")
    assert np.all(pvalue > alpha)


def test_fit_ns_gev_1d_dask():
    """Run non-stationary fit using 1D dask array & check results."""
    data, theta_i = example_da_gev_1d_dask()
    # Add a positive linear trend.
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")

    pvalue = check_gev_fit(data, theta, core_dim="time")
    assert np.all(pvalue > alpha)


def test_fit_ns_gev_3d():
    """Run non-stationary fit using 3D array & check results."""
    data, _ = example_da_gev_3d()
    # Add a positive linear trend.
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")
    pvalue = check_gev_fit(data, theta, covariate=covariate, core_dim="time")
    assert np.all(pvalue > alpha)


def test_fit_ns_gev_3d_dask():
    """Run non-stationary fit using 3D dask array & check results."""
    data, _ = example_da_gev_3d_dask()
    # Add a positive linear trend.
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")
    pvalue = check_gev_fit(data, theta, covariate=covariate, core_dim="time")
    assert np.all(pvalue > alpha)


def test_fit_ns_gev_forecast():
    """Run non-stationary fit using stacked forecast dataArray & check results."""
    data, theta_i = example_da_gev_forecast()

    # Convert times to numerical timesteps.
    covariate = xr.DataArray(date2num(data.time), coords={"sample": data.sample})
    # Add a positive linear trend
    trend = covariate / 1e2
    data = data + trend

    data = data.sortby(data.time)
    covariate = covariate.sortby(data.time)

    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="sample")
    shape, loc, loc1, scale, scale1 = theta

    pvalue = check_gev_fit(data, theta, covariate=covariate, core_dim="sample")
    assert np.all(pvalue > alpha)
    assert np.all(loc1 > 0)  # Positive trend in mean
