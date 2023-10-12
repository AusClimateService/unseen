import pytest

import numpy as np
import xarray as xr
import dask
import dask.array as dsa

import unseen.array_handling
import unseen.bias_correction
import unseen.bootstrap
import unseen.dask_setup
import unseen.fileio
import unseen.general_utils
import unseen.independence
import unseen.indices
import unseen.moments
import unseen.similarity
import unseen.spatial_selection
import unseen.stability
import unseen.time_utils


def pytest_configure():
    pytest.TIME_DIM = "time"
    pytest.INIT_DIM = "init"
    pytest.LEAD_DIM = "lead"
    pytest.LAT_DIM = "lat"
    pytest.LON_DIM = "lon"


def empty_dask_array(shape, dtype=float, chunks=None):
    """A dask array that errors if you try to compute it
    Stolen from https://github.com/xgcm/xhistogram/blob/master/xhistogram/test/fixtures.py
    """

    def raise_if_computed():
        raise ValueError("Triggered forbidden computation on dask array")

    a = dsa.from_delayed(dask.delayed(raise_if_computed)(), shape, dtype)
    if chunks is not None:
        a = a.rechunk(chunks)
    return a


@pytest.fixture()
def example_da_timeseries(request):
    """An example timeseries DataArray"""
    time = xr.cftime_range(start="2000-01-01", end="2001-12-31", freq="D")
    if request.param == "dask":
        data = empty_dask_array((len(time),))
    else:
        data = np.array([t.toordinal() for t in time])
        data -= data[0]
    return xr.DataArray(data, coords=[time], dims=[pytest.TIME_DIM])


@pytest.fixture()
def example_da_lon(request):
    """An example DataArray of longitude values"""
    maxlon = request.param
    lats = np.arange(-90, 91)
    lons = np.arange(maxlon - 360, maxlon)
    data = np.tile((lons + 360) % 360, (len(lats), 1))
    da = xr.DataArray(data, coords=[lats, lons], dims=[pytest.LAT_DIM, pytest.LON_DIM])
    return da


@pytest.fixture()
def example_da_forecast(request):
    """An example forecast DataArray"""
    N_INIT = 24  # Keep at least 6
    N_LEAD = 12  # Keep at least 6
    START = "2000-01-01"  # DO NOT CHANGE
    init = xr.cftime_range(start=START, periods=N_INIT, freq="MS")
    lead = range(N_LEAD)
    time = [init.shift(i, freq="MS")[:N_LEAD] for i in range(len(init))]
    if request.param == "dask":
        data = empty_dask_array(
            (
                len(init),
                len(lead),
            )
        )
    else:
        data = np.random.random(
            (
                len(init),
                len(lead),
            )
        )
    ds = xr.DataArray(
        data, coords=[init, lead], dims=[pytest.INIT_DIM, pytest.LEAD_DIM]
    )
    return ds.assign_coords(
        {pytest.TIME_DIM: ([pytest.INIT_DIM, pytest.LEAD_DIM], time)}
    )
