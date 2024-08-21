"""Test extreme value analysis functions."""

from matplotlib.dates import date2num
import numpy as np
import numpy.testing as npt
from scipy.stats import genextreme
from xarray import cftime_range, DataArray

from unseen.eva import fit_gev, get_return_period, get_return_level

rtol = 0.3  # relative tolerance
alpha = 0.05


def example_da_gev_1d():
    """An example 1D GEV DataArray and distribution parameters."""
    time = cftime_range(start="2000-01-01", periods=1500, freq="D")

    # Shape, location and scale parameters
    np.random.seed(0)
    shape = np.random.uniform()
    loc = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.1, 10)
    theta = shape, loc, scale

    rvs = genextreme.rvs(shape, loc=loc, scale=scale, size=(time.size), random_state=0)
    data = DataArray(rvs, coords=[time], dims=["time"])
    return data, theta


def example_da_gev_1d_dask():
    """An example 1D GEV dask array and distribution parameters."""
    data, theta = example_da_gev_1d()
    data = data.chunk({"time": -1})
    return data, theta


def example_da_gev_3d():
    """An example 3D GEV DataArray and distribution parameters."""
    time = cftime_range(start="2000-01-01", periods=1500, freq="D")
    lat = np.arange(2)
    lon = np.arange(2)

    # Shape, location and scale parameters
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
    data = DataArray(rvs, coords=[time, lat, lon], dims=["time", "lat", "lon"])
    return data, theta


def example_da_gev_3d_dask():
    """An example 3D GEV dask array and its distribution parameters."""
    data, theta = example_da_gev_3d()
    data = data.chunk({"time": -1, "lat": 1, "lon": 1})
    return data, theta


def add_example_gev_trend(data):
    trend = np.arange(data.time.size) * 2.5 / data.time.size
    trend = DataArray(trend, coords={"time": data.time})
    return data + trend


def example_da_gev_forecast():
    """Create example stacked forecast dataArray."""
    ensemble = np.arange(3)
    lead_time = np.arange(5)
    init_date = cftime_range(start="2000-01-01", periods=24, freq="MS")
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
    data = DataArray(
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
    theta = fit_gev(data, stationary=True)
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_user_estimates():
    """Run stationary fit using 1D array & user_estimates."""
    data, theta_i = example_da_gev_1d()
    user_estimates = list(theta_i)
    theta = fit_gev(data, stationary=True, user_estimates=user_estimates)
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_goodness():
    """Run stationary fit using 1D array & fit_goodness_test."""
    data, theta_i = example_da_gev_1d()
    theta = fit_gev(data, stationary=True, test_fit_goodness=True)
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_numpy():
    """Run stationary fit using 1D np.ndarray & check results."""
    data, theta_i = example_da_gev_1d()
    data = data.values
    theta = fit_gev(data, stationary=True)
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_dask():
    """Run stationary fit using 1D dask array & check results."""
    data, theta_i = example_da_gev_1d_dask()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d():
    """Run stationary fit using 3D array & check results."""
    data, theta_i = example_da_gev_3d()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d_dask():
    """Run stationary fit using 3D dask array & check results."""
    data, theta_i = example_da_gev_3d_dask()
    theta = fit_gev(data, stationary=True, core_dim="time")
    # Check fitted params match params used to create data
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_ns_gev_1d():
    """Run non-stationary fit using 1D array & check results."""
    data, _ = example_da_gev_1d()
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        covariate=covariate,
    )
    assert np.all(theta[2] > 0)  # Positive trend in location


def test_fit_ns_gev_1d_loc_only():
    """Run non-stationary fit using 1D array (location parameter only)."""
    data, _ = example_da_gev_1d()
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        loc1=0,
        scale1=None,
        covariate=covariate,
    )
    assert np.all(theta[2] > 0)  # Positive trend in location
    assert np.all(theta[4] == 0)  # No trend in scale


def test_fit_ns_gev_1d_scale_only():
    """Run non-stationary fit using 1D array (scale parameter only)."""
    data, _ = example_da_gev_1d()
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        loc1=None,
        scale1=0,
        covariate=covariate,
    )
    assert np.all(theta[2] == 0)  # No trend in location
    assert np.all(theta[4] != 0)  # Nonzero trend in scale


def test_fit_ns_gev_1d_dask():
    """Run non-stationary fit using 1D dask array & check results."""
    data, _ = example_da_gev_1d_dask()
    # Add a positive linear trend
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)
    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")
    assert np.all(theta[2] > 0)  # Positive trend in location


def test_fit_ns_gev_3d():
    """Run non-stationary fit using 3D array & check results."""
    data, _ = example_da_gev_3d()
    # Add a positive linear trend
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)
    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")
    assert np.all(theta.isel(theta=2) > 0)  # Positive trend in location


def test_fit_ns_gev_1d_relative_fit_test_bic_trend():
    """Run non-stationary fit & check 'BIC' test returns nonstationary params."""
    data, _ = example_da_gev_1d()
    # Add a large positive linear trend
    data = add_example_gev_trend(data)
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        covariate=covariate,
        relative_fit_test="bic",
    )
    assert np.all(theta[2] > 0)  # Positive trend in location


def test_fit_ns_gev_1d_relative_fit_test_bic_no_trend():
    """Run non-stationary fit & check 'BIC' test returns stationary params."""
    data, _ = example_da_gev_1d()
    covariate = np.arange(data.time.size, dtype=int)

    theta = fit_gev(
        data,
        stationary=False,
        core_dim="time",
        covariate=covariate,
        relative_fit_test="bic",
    )
    assert np.all(theta[2] == 0)  # No trend in location
    assert np.all(theta[4] == 0)  # No trend in scale


def test_fit_ns_gev_3d_dask():
    """Run non-stationary fit using 3D dask array & check results."""
    data, _ = example_da_gev_3d_dask()
    # Add a positive linear trend
    data = add_example_gev_trend(data)
    covariate = np.arange(data.time.size, dtype=int)
    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")
    assert np.all(theta.isel(theta=2) > 0)  # Positive trend in location


def test_fit_ns_gev_forecast():
    """Run non-stationary fit using stacked forecast dataArray."""
    data, _ = example_da_gev_forecast()
    # Convert times to numerical timesteps
    covariate = DataArray(date2num(data.time), coords={"sample": data.sample})
    # Add a positive linear trend
    trend = covariate / 1e2
    data = data + trend
    data = data.sortby(data.time)
    covariate = covariate.sortby(data.time)
    theta = fit_gev(data, stationary=False, covariate=covariate, core_dim="sample")
    assert np.all(theta[2] > 0)  # Positive trend in location


def test_get_return_period():
    """Run get_return_period for a single event using 1d data."""
    data, _ = example_da_gev_1d()
    event = data.mean()
    rp = get_return_period(event, data=data)
    assert rp.size == 1
    assert np.all(np.isfinite(rp))


def test_get_return_period_1d():
    """Run get_return_period for 1d array of events using 1d data."""
    data, theta = example_da_gev_1d()
    event = data.quantile([0.25, 0.5, 0.75], dim="time")
    rp = get_return_period(event, theta)
    assert rp.shape == event.shape
    assert np.all(np.isfinite(rp))


def test_get_return_period_3d():
    """Run get_return_period for 3d array of events using 3d data."""
    data, theta = example_da_gev_3d()
    theta = fit_gev(data, stationary=True)
    # Multiple events unique to each lat/lon
    event = data.quantile([0.25, 0.5, 0.75], dim="time")
    rp = get_return_period(event, theta)
    assert rp.shape == event.shape
    assert np.all(np.isfinite(rp))


def test_get_return_period_3d_nonstationary():
    """Run get_return_period for 3d events using 3d nonstationary data."""
    data, _ = example_da_gev_3d()
    data = add_example_gev_trend(data)
    covariate = DataArray(np.arange(data.time.size), dims="time")
    params = fit_gev(data, stationary=False, covariate=covariate, core_dim="time")

    # Multiple events unique to each lat/lon
    event = data.quantile([0.25, 0.5, 0.75], dim="time")
    covariate_subset = DataArray([0, covariate.size], dims="time")
    rp = get_return_period(event, params, covariate=covariate_subset)
    assert rp.shape == (*list(event.shape), covariate_subset.size)
    assert np.all(np.isfinite(rp))


def test_get_return_level():
    """Run get_return_level for a single return_period using 1d data."""
    _, theta = example_da_gev_1d()
    rp = 100
    return_level = get_return_level(rp, theta)
    assert return_level.size == 1
    assert np.all(np.isfinite(return_level))


def test_get_return_level_1d():
    """Run get_return_level for 1d array of periods using 1d data."""
    _, theta = example_da_gev_1d()
    rp = np.array([10, 100, 1000])
    return_level = get_return_level(rp, theta)
    assert return_level.shape == rp.shape
    assert np.all(np.isfinite(return_level))


def test_get_return_level_3d():
    """Run get_return_level for 3d array of periods using 3d data."""
    data, theta = example_da_gev_3d()
    theta = fit_gev(data, stationary=True)
    # Multiple events unique to each lat/lon
    dims = ("return_period", "lat", "lon")
    rp = np.array([10, 100, 1000] * 4).T
    rp = DataArray(rp.reshape((3, 2, 2)), dims=dims)
    return_level = get_return_level(rp, theta)
    assert return_level.shape == rp.shape
    assert np.all(np.isfinite(return_level))
