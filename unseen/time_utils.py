"""Utilities for working with time axes and values"""

import re
from datetime import datetime

import numpy as np
import cftime
import xarray as xr

from . import array_handling


def temporal_aggregation(
    ds,
    target_freq,
    input_freq,
    agg_method,
    variables,
    season=None,
    reset_times=False,
    complete=False,
    time_dim="time",
):
    """Temporal aggregation of data.

    Parameters
    ----------
    ds : xarray Dataset
    target_freq : {'A-DEC', 'Q-NOV', 'M', 'A-NOV'}
        Target frequency for the resampling
    agg_method : {'mean', 'min', 'max', 'sum'}
        Aggregation method
    variables : list
        Variables in the dataset
    input_freq : {'D', 'M', 'Q', 'A'}
        Temporal frequency of input data (daily, monthly or annual)
    season : {'DJF', 'MAM', 'JJA', 'SON'}, optional
        Select a single season after Q-NOV resampling
    reset_times : bool, default False
        Shift time values after resampling so months match initial date
        (used mainly for forecast data)
    complete : bool, default False
        Keep only complete time units (e.g. complete years or months)
    time_dim: str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    ds : xarray Dataset

    Notes
    -----
    A-DEC = annual, with date label being last day of year
    M = monthly, with date label being last day of month
    Q-NOV = DJF, MAM, JJA, SON, with date label being last day of season
    A-NOV = annual Dec-Nov, date label being last day of the year
    A-AUG = annual Sep-Aug, date label being last day of the year
    """

    assert target_freq in ["A-DEC", "M", "Q-NOV", "A-NOV", "A-AUG"]
    assert input_freq in ["D", "M", "Q", "A"]

    if time_dim not in ds.dims:
        ds = array_handling.reindex_forecast(ds)
        reindexed = True
    else:
        reindexed = False

    start_time = ds[time_dim].values[0]
    counts = ds[variables[0]].resample(time=target_freq).count(dim=time_dim)

    if input_freq == target_freq[0]:
        pass
    elif agg_method == "max":
        ds = ds.resample(time=target_freq).max(dim=time_dim, keep_attrs=True)
    elif agg_method == "min":
        ds = ds.resample(time=target_freq).min(dim=time_dim, keep_attrs=True)
    elif agg_method == "sum":
        ds = ds.resample(time=target_freq).sum(dim=time_dim, keep_attrs=True)
        for var in variables:
            ds[var].attrs["units"] = _update_rate(ds[var], input_freq, target_freq)
    elif agg_method == "mean":
        if input_freq == "D":
            ds = ds.resample(time=target_freq).mean(dim=time_dim, keep_attrs=True)
        elif input_freq == "M":
            ds = _monthly_downsample_mean(ds, target_freq, variables, time_dim=time_dim)
        else:
            raise ValueError(f"Unsupported input time frequency: {input_freq}")
    else:
        raise ValueError(f"Unsupported temporal aggregation method: {agg_method}")

    if season:
        assert target_freq == 'Q-NOV'
        final_month = {'DJF': 2, 'MAM': 5, 'JJA': 8, 'SON': 11}
        season_month = final_month[season]
        ds = select_month(ds, season_month, time_dim=time_dim)

    if reset_times:
        diff = ds[time_dim].values[0] - start_time
        ds[time_dim] = ds[time_dim] - diff
        assert ds[time_dim].values[0] == start_time

    if complete:
        ds = _crop_to_complete_time_periods(ds, counts, input_freq, target_freq)

    if reindexed:
        ds = ds.compute()
        ds = array_handling.time_to_lead(ds, target_freq[0])

    return ds


def select_time_period(ds, period, time_name="time"):
    """Select a period of time.

    Works for forecast datasets where (for instance) the dimensions are
    initial date and lead time and corresponding time values are a dataset variable.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Array containing a time dimension or variable.
        The times should be cftime objects but can contain nans.
    period : list of str
        Start and stop dates (in YYYY-MM-DD format)
    time_name: str
        Name of the time dimension, coordinate or variable

    Returns
    -------
    selection : xarray DataArray or Dataset
        Array containing only times within provided period
    """

    def _inbounds(t, bnds):
        """Check if time in bounds, allowing for nans"""
        if t != t:
            return False
        else:
            return (t >= bnds[0]) & (t <= bnds[1])

    _vinbounds = np.vectorize(_inbounds)
    _vinbounds.excluded.add(1)
    _check_date_format(period)
    start, stop = period

    if time_name in ds.dims:
        selection = ds.sel({time_name: slice(start, stop)})
    elif time_name in ds.coords:
        time_values = ds[time_name].values
        try:
            calendar = time_values.flatten()[0].calendar
        except AttributeError:
            calendar = "standard"
        time_bounds = xr.cftime_range(
            start=start, end=stop, periods=2, freq=None, calendar=calendar
        )
        mask_values = _vinbounds(time_values, time_bounds)
        mask = ds[time_name].copy()
        mask.values = mask_values
        selection = ds.where(mask)
    else:
        raise ValueError("No time axis for masking")
    selection.attrs = ds.attrs

    return selection


def get_clim(
    ds,
    dims,
    time_period=None,
    groupby_init_month=False,
    init_dim="init_date",
    time_name="time",
):
    """Calculate climatology.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    dims : str or list
        Dimension/s over which to calculate climatology
    time_period : list, optional
        Time period (start date and end date in YYYY-MM-DD format)
    groupby_init_month : bool, default False
        Calculate separate climatologies for each forecast initialisation month
    init_dim: str, default 'init_date'
        Name of the initial date dimension in ds

    Returns
    -------
    clim : xarray DataArray or Dataset
        Climatology
    """

    if time_period is not None:
        ds = select_time_period(ds.copy(), time_period, time_name=time_name)
        ds.attrs["climatological_period"] = str(time_period)

    if groupby_init_month:
        clim = ds.groupby(f"{init_dim}.month").mean(dims, keep_attrs=True)
    else:
        clim = ds.mean(dims, keep_attrs=True)

    return clim


def cftime_to_str(time_dim, str_format="%Y-%m-%d"):
    """Convert cftime array to list of date strings.

    Parameters
    ----------
    time_dim : xarray DataArray
        Time dimension
    str_format : str, default '%Y-%m-%d'
        Output date string format (any format accepted by strftime)

    Returns
    -------
    str_times : list
        Date strings
    """

    _check_cftime(time_dim)
    str_times = [time.strftime(str_format) for time in time_dim.values]

    return str_times


def str_to_cftime(datestring):
    """Convert a date string to cftime object"""

    dt = datetime.strptime(datestring, "%Y-%m-%d")
    # cfdt = cftime.datetime(dt.year, dt.month, dt.day, calendar=calendar)
    cfdt = cftime.DatetimeJulian(dt.year, dt.month, dt.day)

    return cfdt


def switch_calendar(ds):
    """Change time axis calendar.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Dataset with cftime time axis

    Returns
    -------
    ds : xarray DataArray or Dataset
    """

    str_times = cftime_to_str(ds["time"])
    str_to_cftime_func = np.vectorize(str_to_cftime)
    new_times = str_to_cftime_func(str_times)
    time_attrs = ds["time"].attrs

    ds = ds.assign_coords({"time": new_times})
    ds["time"].attrs = time_attrs

    return ds


def datetime_to_cftime(datetime_array):
    """Convert a numpy datetime array to a cftime array.

    Parameters
    ----------
    datetime_array : numpy ndarray
        Array of numpy datetime objects

    Returns
    -------
    cftime_array : numpy ndarray
        Array of cftime objects
    """

    str_array = np.datetime_as_string(datetime_array, unit="D")
    str_to_cftime_func = np.vectorize(str_to_cftime)
    cftime_array = str_to_cftime_func(str_array)

    return cftime_array


def _monthly_downsample_mean(ds, target_freq, variables, time_dim="time"):
    """Downsample monthly data.

    Accounts for the different number of days in each month.
    """

    days_in_month = ds[time_dim].dt.days_in_month
    weighted_mean = (ds * days_in_month).resample(time=target_freq).sum(
        dim=time_dim, keep_attrs=True
    ) / days_in_month.resample(time=target_freq).sum(dim=time_dim)
    weighted_mean.attrs = ds.attrs
    for var in variables:
        weighted_mean[var].attrs = ds[var].attrs

    return weighted_mean


def _check_cftime(time_dim):
    """Check that time dimension is cftime.

    Parameters
    ----------
    time_dim : xarray DataArray
        Time dimension
    """

    t0 = time_dim.values.flatten()[0]
    cftime_types = list(cftime._cftime.DATE_TYPES.values())
    cftime_types.append(cftime._cftime.datetime)
    assert type(t0) in cftime_types, "Time dimension must use cftime objects"


def _check_date_format(date_list):
    """Check for YYYY-MM-DD format."""

    date_pattern = "([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})"
    for date in date_list:
        assert re.search(date_pattern, date), "Date format must be YYYY-MM-DD"


def _crop_to_complete_time_periods(ds, counts, input_freq, output_freq):
    """Crop an aggregated xarray dataset to include only complete time periods.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Temporally aggregated data
    counts : xarray DataArray
        Number of samples in each aggregation
    input_freq : {'D, 'M'}
        Time frequency before temporal aggregation
    output_freq : {'A-DEC', 'M', 'Q-NOV', 'A-NOV'}
        Time frequency after temporal aggregation

    Returns
    -------
    ds : xarray DataArray or Dataset
        Temporally aggregated data with only complete time periods retained
    """

    assert input_freq in ["D", "M"]
    assert output_freq in ["A-DEC", "M", "Q-NOV", "A-NOV", "A-AUG"]

    # to X from X
    count_dict = {
        ("A", "D"): 360,
        ("A", "M"): 12,
        ("M", "D"): 28,
        ("Q", "M"): 3,
        ("Q", "D"): 89,
    }
    min_sample = count_dict[(output_freq[0], input_freq)]
    ds = ds.where(counts.values >= min_sample)

    return ds


def _update_rate(da, input_freq, target_freq):
    """Update a flow rate due to temporal aggregation."""

    current_units = da.units
    like_units = {
        "/d": " day-1",
        "/day": " day-1",
        "/month": " month-1",
        "/season": " season-1",
        "/yr": " yr-1",
        "/year": "yr-1",
        " -1": "-1",
    }
    for wrong_unit, correct_unit in like_units.items():
        current_units = current_units.replace(wrong_unit, correct_unit)

    rates_dict = {"D": "day-1", "M": "month-1", "Q": "season-1", "A": "yr-1"}
    input_rate = rates_dict[input_freq]
    if input_rate in current_units:
        target_rate = rates_dict[target_freq[0]]
        new_units = current_units.replace(input_rate, target_rate)
    else:
        new_units = current_units

    return new_units


def select_month(ds, month, init_month=False, time_dim="time"):
    """Select month from dataset.

    Parameters
    ----------
    ds : xarray Dataset or DataArray
    month : int
        Month to select (1-12)
    init_month : bool, default False
        Set the month on the time axis to the initial month
    time_dim: str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    ds_selection : xarray Dataset or DataArray
        Input dataset with month extracted
    """

    month_idxs = ds.groupby("time.month").groups
    ds_selection = ds.isel({time_dim: month_idxs[month]})
    if init_month:
        initial_date = ds[time_dim].data[0]
        diff = ds_selection[time_dim].values[0] - initial_date
        ds_selection[time_dim] = ds_selection[time_dim] - diff

    return ds_selection
