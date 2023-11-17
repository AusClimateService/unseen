import numpy as np
import xarray as xr

import numpy.testing as npt
from scipy.stats import genextreme

from unseen.eva import fit_gev, check_gev_fit

rtol = 0.3  # relative tolerance
alpha = 0.05


def example_da_gev_1d():
    """An example 1D GEV DataArray and its distribution parameters."""
    time = np.arange(
        "2000-01-01", "2003-01-01", np.timedelta64(1, "D"), dtype="datetime64[ns]"
    )

    # Generate shape, location and scale parameters.
    np.random.seed(0)
    shape = np.random.uniform()
    loc = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.1, 10)
    theta = shape, loc, scale

    rvs = genextreme.rvs(shape, loc=loc, scale=scale, size=(time.size), random_state=0)
    data = xr.DataArray(rvs, coords=[time], dims=["time"])
    return data, theta


def example_da_gev_1d_dask():
    """An example 1D GEV dask array and its distribution parameters."""
    data, theta = example_da_gev_1d()
    data = data.chunk({"time": -1})
    return data, theta


def example_da_gev_3d():
    """An example 3D GEV DataArray and its distribution parameters."""
    time = np.arange(
        "2000-01-01", "2003-01-01", np.timedelta64(1, "D"), dtype="datetime64[ns]"
    )
    lats = np.arange(0, 2)
    lons = np.arange(0, 2)
    shape = (len(lats), len(lons))

    np.random.seed(0)
    c = np.random.rand(*shape)
    loc = np.random.rand(*shape) + np.random.randint(-10, 10, shape)
    scale = np.random.rand(*shape)
    theta = np.stack([c, loc, scale], axis=-1)

    rvs = genextreme.rvs(
        c,
        loc=loc,
        scale=scale,
        size=(time.size, *shape),
        random_state=0,
    )
    data = xr.DataArray(rvs, coords=[time, lats, lons], dims=["time", "lat", "lon"])
    return data, theta


def example_da_gev_3d_dask():
    """An example 3D GEV dask array and its distribution parameters."""
    data, theta = example_da_gev_3d()
    data = data.chunk({"time": -1, "lat": 1, "lon": 1})
    return data, theta


def add_example_gev_trend(data):
    trend = np.arange(data.time.size) / data.time.size
    trend = xr.DataArray(trend, coords={"time": data.time})
    return data + trend


def test_fit_gev_1d():
    """Run stationary fit using 1D array & check results."""
    data, theta_i = example_da_gev_1d()
    theta = fit_gev(data, stationary=True, check_fit=False)
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_dask():
    """Run stationary fit using 1D dask array & check results."""
    data, theta_i = example_da_gev_1d_dask()
    theta = fit_gev(data, stationary=True, check_fit=False)
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d():
    """Run stationary fit using 3D array & check results."""
    data, theta_i = example_da_gev_3d()
    theta = fit_gev(data, stationary=True, check_fit=False)
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d_dask():
    """Run stationary fit using 3D dask array & check results."""
    data, theta_i = example_da_gev_3d_dask()
    theta = fit_gev(data, stationary=True, check_fit=False)
    # Check fitted params match params used to create data.
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_ns_gev_1d():
    """Run non-stationary fit using 1D array & check results."""
    data, _ = example_da_gev_1d()
    data = add_example_gev_trend(data)

    # Check function runs.
    theta = fit_gev(data, stationary=False, check_fit=False)

    check_gev_fit(data, theta, time_dim="time")

    pvalue = check_gev_fit(data, theta, time_dim="time")
    assert np.all(pvalue) > alpha


def test_fit_ns_gev_1d_dask():
    """Run non-stationary fit using 1D dask array & check results."""
    data, _ = example_da_gev_1d_dask()

    # Add a positive linear trend.
    data = add_example_gev_trend(data)

    # Check function runs.
    theta = fit_gev(data, stationary=False, check_fit=False)

    pvalue = check_gev_fit(data, theta, time_dim="time")
    assert np.all(pvalue) > alpha


def test_fit_ns_gev_3d():
    """Run non-stationary fit using 3D array & check results."""
    data, _ = example_da_gev_3d()

    # Add a positive linear trend.
    data = add_example_gev_trend(data)

    # Check function runs.
    theta = fit_gev(data, stationary=False, check_fit=False)

    pvalue = check_gev_fit(data, theta, time_dim="time")
    assert np.all(pvalue) > alpha


def test_fit_ns_gev_3d_dask():
    """Run non-stationary fit using 3D dask array & check results."""
    data, _ = example_da_gev_3d_dask()

    # Add a positive linear trend.
    data = add_example_gev_trend(data)

    # Check function runs.
    theta = fit_gev(data, stationary=False, check_fit=False)

    pvalue = check_gev_fit(data, theta, time_dim="time")
    assert np.all(pvalue) > alpha
