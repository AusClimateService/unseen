import pytest
import numpy as np
from xarray.coding.times import cftime_to_nptime

from unseen.time_utils import select_time_period, temporal_aggregation


@pytest.mark.parametrize("example_da_forecast", ["numpy"], indirect=True)
@pytest.mark.parametrize("add_nans", [False, True])
@pytest.mark.parametrize("data_object", ["DataArray", "Dataset"])
def test_select_time_period(example_da_forecast, add_nans, data_object):
    """Test values returned by select_time_period"""
    PERIOD = ["2000-06-01", "2001-06-01"]

    data = example_da_forecast
    if data_object == "Datatset":
        data = data.to_dataset(name="var")

    if add_nans:
        time_nans = data[pytest.TIME_DIM].where(data[pytest.LEAD_DIM] > 3)
        data = data.assign_coords({pytest.TIME_DIM: time_nans})

    masked = select_time_period(data, PERIOD)

    min_time = cftime_to_nptime(data[pytest.TIME_DIM].where(masked.notnull()).min())
    max_time = cftime_to_nptime(data[pytest.TIME_DIM].where(masked.notnull()).max())

    assert min_time >= np.datetime64(PERIOD[0])
    assert max_time <= np.datetime64(PERIOD[1])


@pytest.mark.parametrize("example_da_timeseries", ["numpy"], indirect=True)
def test_temporal_aggregation_agg_dates(example_da_timeseries):
    """Test temporal_aggregation using agg_dates."""
    # Monotonically increasing time series
    data = example_da_timeseries
    ds = data.to_dataset(name="var")

    ds_resampled = temporal_aggregation(
        ds,
        target_freq="ME",
        input_freq="D",
        agg_method="max",
        variables=["var"],
        season=None,
        reset_times=False,
        min_tsteps=None,
        agg_dates=True,
        time_dim="time",
    )
    event_time = ds_resampled["event_time"].astype(dtype="datetime64[ns]")
    # Monthly maximum should be the last day of each month
    assert np.all(event_time.dt.day >= 28)


@pytest.mark.parametrize("example_da_timeseries", ["numpy"], indirect=True)
def test_temporal_aggregation_reset_times(example_da_timeseries):
    """Test temporal_aggregation using reset_times."""
    data = example_da_timeseries
    ds = data.to_dataset(name="var")

    ds_resampled = temporal_aggregation(
        ds,
        target_freq="YE-DEC",
        input_freq="D",
        agg_method="max",
        variables=["var"],
        season=None,
        reset_times=True,
        min_tsteps=None,
        agg_dates=True,
        time_dim="time",
    )
    assert np.all(ds_resampled.time.dt.month.values == ds.time.dt.month[0].item())


@pytest.mark.parametrize("example_da_timeseries", ["numpy"], indirect=True)
def test_temporal_aggregation_min_tsteps(example_da_timeseries):
    """Test temporal_aggregation using min_tsteps."""
    data = example_da_timeseries
    ds = data.to_dataset(name="var")

    # Remove days from first & last month and test the months are removed
    ds = ds.isel(time=slice(5, -5))

    variables = ["var"]
    target_freq = "ME"
    time_dim = "time"
    min_tsteps = 28
    counts = ds[variables[0]].resample(time=target_freq).count(dim=time_dim).load()

    ds_resampled = temporal_aggregation(
        ds,
        target_freq,
        input_freq="D",
        agg_method="max",
        variables=variables,
        season=None,
        reset_times=False,
        min_tsteps=min_tsteps,
        agg_dates=False,
        time_dim=time_dim,
    )
    assert np.all((counts >= min_tsteps).sum(time_dim) == len(ds_resampled.time))
