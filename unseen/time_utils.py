"""Utilities for working with time axes and values"""

import re

import numpy as np
import cftime
from datetime import datetime
import xarray as xr

from . import array_handling


def check_date_format(date_list):
    """Check for YYYY-MM-DD format."""

    date_pattern = "([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})"
    for date in date_list:
        assert re.search(date_pattern, date), "Date format must be YYYY-MM-DD"


def check_cftime(time_dim):
    """Check that time dimension is cftime.

    Args:
      time_dim (xarray DataArray) : Time dimension
    """

    t0 = time_dim.values.flatten()[0]
    cftime_types = list(cftime._cftime.DATE_TYPES.values())
    cftime_types.append(cftime._cftime.datetime)
    assert type(t0) in cftime_types, "Time dimension must use cftime objects"


def str_to_cftime(datestring, calendar="standard"):
    """Convert a date string to cftime object"""

    dt = datetime.strptime(datestring, "%Y-%m-%d")
    # cfdt = cftime.datetime(dt.year, dt.month, dt.day, calendar=calendar)
    cfdt = cftime.DatetimeJulian(dt.year, dt.month, dt.day)

    return cfdt


def cftime_to_str(time_dim):
    """Convert cftime array to YYY-MM-DD strings."""

    check_cftime(time_dim)
    str_times = [time.strftime("%Y-%m-%d") for time in time_dim.values]

    return str_times


def datetime_to_cftime(datetime_array):
    """Convert a numpy datetime array to a cftime array"""

    str_array = np.datetime_as_string(datetime_array, unit="D")
    str_to_cftime_func = np.vectorize(str_to_cftime)
    cftime_array = str_to_cftime_func(str_array)

    return cftime_array


def update_rate(da, input_freq, target_freq):
    """Update a flow rate due to temporal aggregation"""

    current_units = da.units
    rates_dict = {"D": "d-1", "M": "month-1", "Q": "season-1", "A": "yr-1"}
    input_rate = rates_dict[input_freq]
    if input_rate in current_units:
        target_rate = rates_dict[target_freq[0]]
        new_units = current_units.replace(input_rate, target_rate)
    else:
        new_units = current_units

    return new_units


def crop_to_complete_time_periods(ds, counts, input_freq, output_freq):
    """Crop an aggregated xarray dataset to include only complete time periods.

    Args:
      ds (xarray Dataset) : Temporally aggregated Dataset
      counts (xarray Data Array) : Number of samples in each aggregation
      input_freq (str) : Time frequency before temporal aggregation
      output_freq (str) : Time frequency after temporal aggregation
    """

    assert input_freq in ["D", "M"]
    assert output_freq in ["A-DEC", "M", "Q-NOV", "A-NOV"]

    # to X from X
    count_dict = {
        ("A", "D"): 365,
        ("A", "M"): 12,
        ("M", "D"): 28,
        ("Q", "M"): 3,
        ("Q", "D"): 89,
    }
    min_sample = count_dict[(output_freq[0], input_freq)]
    ds = ds.where(counts >= min_sample)

    return ds


def temporal_aggregation(
    ds,
    target_freq,
    input_freq,
    agg_method,
    variables,
    reset_times=False,
    complete=False,
):
    """Temporal aggregation of data.

    Args:
      ds (xarray Dataset)
      target_freq (str) : Target frequency for the resampling (see options below)
      agg_method (str) :  Aggregation method ('mean', 'min', 'max' or 'sum')
      variables (list) :  Variables in the dataset
      input_freq (str) :  Temporal frequency of input data (daily 'D', monthly 'M', annual 'A')
      reset_time (bool) : Shift time values after resampling so months match initial date
                          (used mainly for forecast data)
      complete (bool) :   Keep only complete time units (e.g. complete years or months)

    Valid target frequencies:
      A-DEC (annual, with date label being last day of year)
      M (monthly, with date label being last day of month)
      Q-NOV (DJF, MAM, JJA, SON, with date label being last day of season)
      A-NOV (annual Dec-Nov, date label being last day of the year)
    """

    assert target_freq in ["A-DEC", "M", "Q-NOV", "A-NOV"]
    assert input_freq in ["D", "M", "Q", "A"]

    if "time" not in ds.dims:
        ds = array_handling.reindex_forecast(ds)
        reindexed = True
    else:
        reindexed = False

    start_time = ds["time"].values[0]
    counts = ds[variables[0]].resample(time=target_freq).count(dim="time")

    if input_freq == target_freq[0]:
        pass
    elif agg_method == "max":
        ds = ds.resample(time=target_freq).max(dim="time", keep_attrs=True)
    elif agg_method == "min":
        ds = ds.resample(time=target_freq).min(dim="time", keep_attrs=True)
    elif agg_method == "sum":
        ds = ds.resample(time=target_freq).sum(dim="time", keep_attrs=True)
        for var in variables:
            ds[var].attrs["units"] = update_rate(ds[var], input_freq, target_freq)
    elif agg_method == "mean":
        if input_freq == "D":
            ds = ds.resample(time=target_freq).mean(dim="time", keep_attrs=True)
        elif input_freq == "M":
            ds = monthly_downsample_mean(ds, target_freq, variables)
        else:
            raise ValueError(f"Unsupported input time frequency: {input_freq}")
    else:
        raise ValueError(f"Unsupported temporal aggregation method: {agg_method}")

    if reset_times:
        diff = ds["time"].values[0] - start_time
        ds["time"] = ds["time"] - diff
        assert ds["time"].values[0] == start_time

    if complete:
        ds = crop_to_complete_time_periods(ds, counts, input_freq, target_freq)

    if reindexed:
        ds = ds.compute()
        ds = array_handling.time_to_lead(ds, target_freq[0])

    return ds


def monthly_downsample_mean(ds, target_freq, variables):
    """Downsample monthly data.

    Accounts for the different number of days in each month.
    """

    days_in_month = ds["time"].dt.days_in_month
    weighted_mean = (ds * days_in_month).resample(time=target_freq).sum(
        dim="time", keep_attrs=True
    ) / days_in_month.resample(time=target_freq).sum(dim="time")
    weighted_mean.attrs = ds.attrs
    for var in variables:
        weighted_mean[var].attrs = ds[var].attrs

    return weighted_mean


def get_clim(ds, dims, time_period=None, groupby_init_month=False):
    """Calculate climatology.

    Args:
      ds (xarray DataSet or DataArray)
      dims (str or list) : Dimension/s over which to calculate climatology
      time_period (list) : Time period
      groupby_init_month (bool) : Calculate separate climatologies for each
                                  forecast initialisation month
    """

    if time_period is not None:
        ds = select_time_period(ds.copy(), time_period)
        ds.attrs["climatological_period"] = str(time_period)

    if groupby_init_month:
        clim = ds.groupby("init_date.month").mean(dims, keep_attrs=True)
    else:
        clim = ds.mean(dims, keep_attrs=True)

    return clim


def select_time_period(ds, period):
    """Select a period of time.

    Args:
      ds (xarray DataSet or DataArray)
      period (list) : Start and stop dates (in YYYY-MM-DD format)

    Only works for cftime objects.
    """

    check_date_format(period)
    start, stop = period

    if "time" in ds.dims:
        selection = ds.sel({"time": slice(start, stop)})
    elif "time" in ds.coords:
        try:
            calendar = ds["time"].calendar_type.lower()
        except AttributeError:
            calendar = "standard"
        time_bounds = xr.cftime_range(
            start=start, end=stop, periods=2, freq=None, calendar=calendar
        )
        time_values = ds["time"].compute()
        check_cftime(time_values)
        mask = (time_values >= time_bounds[0]) & (time_values <= time_bounds[1])
        selection = ds.where(mask)
    else:
        raise ValueError("No time axis for masking")
    selection.attrs = ds.attrs

    return selection
