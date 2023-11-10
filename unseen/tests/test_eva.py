import pytest

import numpy as np
import xarray as xr

import numpy.testing as npt
from scipy.stats import genextreme

from unseen.general_utils import fit_gev, check_gev_fit

rtol = 0.25  # relative tolerance

def example_da_gev_1d():
    """An example 1D timeseries DataArray"""
    time = np.arange("2000-01-01", "2002-01-01", dtype=np.datetime64)

    c = np.random.rand()
    loc = np.random.randint(-10, 10) + np.random.rand()
    scale = np.random.rand()
    theta = c, loc, scale

    rvs = genextreme.rvs(c, loc=loc, scale=scale, size=(time.size), random_state=0)
    data = xr.DataArray(rvs, coords=[time], dims=['time'])
    return data, theta


def example_da_gev_1d_dask():
    data, theta = example_da_gev_1d()
    data = data.chunk()
    return data, theta


def example_da_gev_3d():
    """An example multi-dim timeseries DataArray"""
    time = np.arange("2000-01-01", "2002-01-01", dtype=np.datetime64)
    lats = np.arange(0, 2)
    lons = np.arange(0, 2)
    shape = (len(lats), len(lons))

    c = np.random.rand(*shape)
    loc = np.random.rand(*shape) + np.random.randint(-10, 10, shape)
    scale = np.random.rand(*shape)
    theta = np.stack([c, loc, scale], axis=-1)

    rvs = genextreme.rvs(c, loc=loc, scale=scale, size=(time.size, *shape), random_state=0)
    data = xr.DataArray(rvs, coords=[time, lats, lons], dims=['time', 'lat', 'lon'])
    return data, theta


def example_da_gev_3d_dask():
    data, theta = example_da_gev_3d()
    data = data.chunk()
    return data, theta


def test_fit_gev_1d():
    # 1D data matches given parameters.
    data, theta_i = example_da_gev_1d()
    theta = fit_gev(data, stationary=True, check_fit=False)
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_1d_dask():
    # 1D chunked data matches given parameters.
    data, theta_i = example_da_gev_1d_dask()
    theta = fit_gev(data, stationary=True, check_fit=False)
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d():
    # 3D data matches given parameters.
    data, theta_i = example_da_gev_3d()
    theta = fit_gev(data, stationary=True, check_fit=True)
    npt.assert_allclose(theta, theta_i, rtol=rtol)


def test_fit_gev_3d_dask():
    # 3D chunked data matches given parameters.
    data, theta_i = example_da_gev_3d_dask()
    theta = fit_gev(data, stationary=True, check_fit=True)
    npt.assert_allclose(theta, theta_i, rtol=rtol)
