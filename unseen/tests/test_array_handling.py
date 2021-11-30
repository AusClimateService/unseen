import pytest

import dask
import dask.array as dsa

import numpy as np
import xarray as xr

import numpy.testing as npt

from unseen.array_handling import (
    stack_by_init_date,
)


TIME_DIM = "time"
INIT_DIM = "init"
LEAD_DIM = "lead"


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
    return xr.DataArray(data, coords=[time], dims=[TIME_DIM])


@pytest.mark.parametrize("example_da_timeseries", ["numpy"], indirect=True)
@pytest.mark.parametrize("offset", [0, 10])
@pytest.mark.parametrize("stride", [1, 10, "irregular"])
@pytest.mark.parametrize("n_lead_steps", [1, 10])
def test_stack_by_init_date(example_da_timeseries, offset, stride, n_lead_steps):
    """Test values returned by stack_by_init_date"""

    def _np_stack_by_init_date(data, indexes, n_lead_steps):
        """Stack timeseries by index and n_lead_steps"""
        ver = np.empty((len(indexes), n_lead_steps))
        ver[:] = np.nan
        for i, idx in enumerate(indexes):
            time_slice = data[idx : idx + n_lead_steps]
            ver[i, : len(time_slice)] = time_slice
        return ver

    data = example_da_timeseries

    if stride == "irregular":
        indexes = np.concatenate(
            ([offset], np.random.randint(1, 20, size=1000))
        ).cumsum()
        indexes = indexes[indexes < data.sizes[TIME_DIM]]
    else:
        indexes = range(offset, data.sizes[TIME_DIM], stride)

    init_dates = data[TIME_DIM][indexes]
    res = stack_by_init_date(
        data, init_dates, n_lead_steps, init_dim=INIT_DIM, lead_dim=LEAD_DIM
    )
    ver = _np_stack_by_init_date(data, indexes, n_lead_steps)

    # Check that values are correct
    npt.assert_allclose(res, ver)

    # Check that init dates are correct
    npt.assert_allclose(
        xr.CFTimeIndex(init_dates.values).asi8, res.get_index(INIT_DIM).asi8
    )

    # Check that times at lead zero match the init dates
    npt.assert_allclose(
        xr.CFTimeIndex(init_dates.values).asi8,
        xr.CFTimeIndex(res[TIME_DIM].isel({LEAD_DIM: 0}).values).asi8,
    )


@pytest.mark.parametrize("example_da_timeseries", ["dask"], indirect=True)
def test_stack_by_init_date_dask(example_da_timeseries):
    """Test values returned by stack_by_init_date

    For now just checks that doesn't trigger compute, but may want to add tests for
    chunking etc in the future
    """

    data = example_da_timeseries
    n_lead_steps = 10
    init_dates = data[TIME_DIM][::10]

    stack_by_init_date(
        data, init_dates, n_lead_steps, init_dim=INIT_DIM, lead_dim=LEAD_DIM
    )
