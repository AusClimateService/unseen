"""Utilities for working with time axes and values"""

import re
import datetime
import numpy as np
import cftime
import xarray as xr

from . import array_handling


def get_agg_dates(ds, var, target_freq, agg_method, time_dim="time"):
    """Record the date of each time aggregated/resampled event (e.g. annual max).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be resampled
    var : str
        A variable in the dataset
    target_freq : str
        Target frequency for the resampling
    agg_method : {'min', 'max'}
        Aggregation method
    time_dim: str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    event_datetimes_str : xarray.DataArray
        Event dates (YYYY-MM-DD) for the resampled array

    todo: replace loop with
        ds.resample(index=target_freq, label="left").map(xr.DataArray.idxmax, dim=time_dim, keep_attrs=True)
    """

    da = ds[var] if isinstance(ds, xr.Dataset) else ds
    ds_arg = da.resample({time_dim: target_freq}, label="left")
    if agg_method == "max":
        dates = [da.idxmax(time_dim) for _, da in ds_arg]
    elif agg_method == "min":
        dates = [da.idxmin(time_dim) for _, da in ds_arg]

    dates = xr.concat(dates, dim=time_dim)
    event_datetimes_str = dates.load().dt.strftime("%Y-%m-%d")
    event_datetimes_str = event_datetimes_str.astype(dtype=str)
    return event_datetimes_str


def temporal_aggregation(
    ds,
    target_freq,
    input_freq,
    agg_method,
    variables,
    season=None,
    reset_times=False,
    min_tsteps=None,
    agg_dates=False,
    time_dim="time",
):
    """Temporal aggregation of data.

    Parameters
    ----------
    ds : xarray.Dataset
    target_freq : str
        Target frequency for the resampling
        Options: https://pandas.pydata.org/docs/user_guide/timeseries.html#anchored-offsets
    input_freq : {'D', 'M', 'Q', 'Y'}
        Temporal frequency of input data (daily, monthly, quarterly or yearly)
    agg_method : {'mean', 'min', 'max', 'sum'}
        Aggregation method
    variables : list
        Variables in the dataset
    season : {'DJF', 'MAM', 'JJA', 'SON'}, default None
        Select a single season after Q-NOV resampling
    reset_times : bool, default False
        Shift time values after resampling so months match initial date
    min_tsteps : int, optional
        Minimum number of time steps for temporal aggregation
        (e.g. 365 for annual aggregation of daily data to ensure full years only)
    agg_dates : bool, default False
        Record the date of each time aggregated event (e.g. annual max)
    time_dim: str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    ds : xarray.Dataset
        Resampled dataset

    Notes
    -----
    Example target_freq includes:
      YE-DEC = annual, with date label being last day of year
      ME = monthly, with date label being last day of month
      Q-NOV = DJF, MAM, JJA, SON, with date label being last day of season
      YE-NOV = annual Dec-Nov, date label being last day of the year
      YE-AUG = annual Sep-Aug, date label being last day of the year
      YE-JUN = annual Jul-Jun, date label being last day of the year

    """

    assert input_freq in ["D", "M", "Q", "Y"]

    if time_dim not in ds.dims:
        # Swap time and lead time dimension
        ds = array_handling.reindex_forecast(ds)
        reindexed = True
    else:
        reindexed = False

    if reset_times:
        start_time = ds[time_dim]

    if min_tsteps:
        # Count the number of time steps in each resampled group
        counts = ds[variables[0]].resample({time_dim: target_freq}).count(dim=time_dim)

    if input_freq == target_freq[0]:
        pass

    elif agg_method in ["max", "min"]:
        if agg_dates:
            # Record the date of each time aggregated event (e.g. annual max)
            agg_dates_var = get_agg_dates(
                ds, variables[0], target_freq, agg_method, time_dim=time_dim
            )
        if agg_method == "max":
            ds = ds.resample({time_dim: target_freq}).max(dim=time_dim, keep_attrs=True)
        else:
            ds = ds.resample({time_dim: target_freq}).min(dim=time_dim, keep_attrs=True)

        if agg_dates:
            # Add the event time to the resampled array
            ds = ds.assign(event_time=agg_dates_var)
    elif agg_method == "sum":
        ds = ds.resample({time_dim: target_freq}).sum(dim=time_dim, keep_attrs=True)
        for var in variables:
            ds[var].attrs["units"] = _update_rate(ds[var], input_freq, target_freq)
    elif agg_method == "mean":
        if input_freq == "D":
            ds = ds.resample({time_dim: target_freq}).mean(
                dim=time_dim, keep_attrs=True
            )
        elif input_freq == "M":
            ds = _monthly_downsample_mean(ds, target_freq, variables, time_dim=time_dim)
        else:
            raise ValueError(f"Unsupported input time frequency: {input_freq}")
    else:
        raise ValueError(f"Unsupported temporal aggregation method: {agg_method}")

    if season:
        assert target_freq == "Q-NOV"
        final_month = {"DJF": 2, "MAM": 5, "JJA": 8, "SON": 11}
        season_month = final_month[season]
        ds = select_months(
            ds,
            [
                season_month,
            ],
            time_dim=time_dim,
        )

    if reset_times and target_freq[0] == "Y":
        # Shift month/day values to match initialisation date
        month = start_time.dt.month.values[0]
        day = start_time.dt.day.values[0]
        times_new = ds[time_dim].dt.strftime(f"%Y-{month:02d}-{day:02d}")
        # Convert to cftime (original calendar)
        cftime_type = type(
            start_time.isel(dict([(k, 0) for k in start_time.dims])).item()
        )
        times_new = np.vectorize(str_to_cftime)(times_new, cftime_type)
        ds[time_dim] = (ds[time_dim].dims, times_new)
        ds[time_dim].attrs = start_time.attrs
        # Check that all time points are in the same month
        assert np.unique(ds[time_dim].dt.month).size == 1
        assert np.unique(ds[time_dim].dt.month)[0] == month

    if min_tsteps:
        # Find resampled groups with fewer than min_tsteps
        invalid_mask = counts < min_tsteps
        # Reduce the mask to the time dimension (if there are other dimensions)
        dims = [dim for dim in ds.dims if dim != time_dim]
        if len(dims) > 0:
            invalid_mask = invalid_mask.any(dims)
        # Convert boolean mask to array indexes
        invalid_inds = np.arange(counts[time_dim].size)[invalid_mask]
        # Drop the invalid time steps
        ds = ds.drop_isel({time_dim: invalid_inds})

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
    ds : Union[xarray.DataArray, xarray.Dataset]
        Array containing a time dimension or variable.
        The times should be cftime objects but can contain NaNs.
    period : list of str
        Start and stop dates (in YYYY-MM-DD format)
    time_name: str, default 'time'
        Name of the time dimension, coordinate or variable

    Returns
    -------
    selection : Union[xarray.DataArray, xarray.Dataset]
        Array containing only times within provided period
    """

    def _inbounds(t, bnds):
        """Check if time in bounds, allowing for NaNs."""
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
        mask = xr.DataArray(
            mask_values, dims=ds[time_name].dims, coords=ds[time_name].coords
        )
        selection = ds.where(mask, drop=True)
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
    ds : Union[xarray.DataArray, xarray.Dataset]
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
    clim : Union[xarray.DataArray, xarray.Dataset]
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
    time_dim : xarray.DataArray
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


def str_to_cftime(datestring, cftime_type=cftime.DatetimeJulian):
    """Convert a date string to cftime object."""
    try:
        dt = datetime.datetime.strptime(datestring, "%Y-%m-%d")
        year, month, day = dt.year, dt.month, dt.day
    except ValueError:
        # Manually separate "YYYY-MM-DD" for non-standard calendars
        year, month, day = [int(i) for i in datestring.split("-")]

    if isinstance(cftime_type, str):
        cfdt = cftime.datetime(year, month, day, calendar=cftime_type)
    else:
        cfdt = cftime_type(year, month, day)

    return cfdt


def switch_calendar(ds):
    """Change time axis calendar.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
        Dataset with cftime time axis

    Returns
    -------
    ds : Union[xarray.DataArray, xarray.Dataset]
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
    datetime_array : numpy.ndarray
        Array of numpy datetime objects

    Returns
    -------
    cftime_array : numpy.ndarray
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
    weighted_mean = (ds * days_in_month).resample({time_dim: target_freq}).sum(
        dim=time_dim, keep_attrs=True
    ) / days_in_month.resample({time_dim: target_freq}).sum(dim=time_dim)
    weighted_mean.attrs = ds.attrs
    for var in variables:
        weighted_mean[var].attrs = ds[var].attrs

    return weighted_mean


def _check_cftime(time_dim):
    """Check that time dimension is cftime.

    Parameters
    ----------
    time_dim : xarray.DataArray
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


def select_months(ds, months, init_month=False, time_dim="time"):
    """Select months from dataset.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
    months : list
        Months to select (1-12)
    init_month : bool, default False
        Set the month on the time axis to the initial month
    time_dim: str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    ds_selection : Union[xarray.DataArray, xarray.Dataset]
        Input data with month extracted
    """

    da_months = ds[time_dim].dt.month
    time_selection = da_months.isin(months)
    ds_selection = ds.sel({time_dim: time_selection})
    if init_month:
        initial_date = ds[time_dim].data[0]
        diff = ds_selection[time_dim].values[0] - initial_date
        ds_selection[time_dim] = ds_selection[time_dim] - diff

    return ds_selection


def _get_groupby_and_reduce_dims(
    ds, frequency, init_dim="init_date", ensemble_dim="ensemble", time_name="time"
):
    """Get groupby and reduction dimensions.

    For performing operations like calculating anomalies and percentile thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        The data to be grouped
    frequency : str
        The frequency at which to bin the data, e.g. per month
    init_dim : str, default 'init_date'
        Name of the initial date dimension in ds
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in ds
    time_name : str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    groupby : str
        The groupby string to use in the xarray groupby operation
    reduce_dim : str or list
        The dimension or dimensions to reduce along
    """

    def _same_group_per_lead(time, frequency):
        group_value = getattr(time.dt, frequency)
        return (group_value == group_value.isel({init_dim: 0})).all()

    if time_name in ds.dims:
        groupby = f"{time_name}.{frequency}" if (frequency is not None) else None
        reduce_dim = time_name
    elif init_dim in ds.dims:
        if frequency is not None:
            # In the case of forecast data, if frequency is not None, all that
            # is done is to check that all the group values are the same for each
            # lead
            time = ds[time_name].compute()
            same_group_per_lead = (
                time.groupby(f"{init_dim}.month")
                .map(_same_group_per_lead, frequency=frequency)
                .values
            )
            assert all(
                same_group_per_lead
            ), "All group values are not the same for each lead"
        groupby = f"{init_dim}.month"
        reduce_dim = init_dim
    else:
        raise ValueError("I can't work out how to apply groupby on this data")

    if ensemble_dim in ds.dims:
        reduce_dim = [reduce_dim, ensemble_dim]

    return groupby, reduce_dim


def anomalise(
    ds,
    clim_period,
    frequency=None,
    init_dim="init_date",
    ensemble_dim="ensemble",
    time_name="time",
):
    """Calculate anomaly.

    Uses a shortcut for calculating hindcast climatologies that will not work
    for hindcasts with initialisation frequencies more regular than monthly.

    Parameters
    ----------
    ds : xarray.Dataset
        The data to anomalise
    clim_period : iterable
        Size 2 iterable containing strings indicating the start and end dates
        of the climatological period
    frequency : str, optional
        The frequency at which to bin the climatology, e.g. per month. Must be
        an available attribute of the datetime accessor. Specify "None" to
        indicate no frequency (climatology calculated by averaging all times).
        Note, setting to "None" for hindcast data can be dangerous, since only
        certain times may be available at each lead.
    init_dim : str, default 'init_date'
        Name of the initial date dimension in ds
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in ds
    time_name : str, default 'time'
        Name of the time dimension in ds

    Returns
    -------
    xarray.Dataset
        The data with the climatology subtracted
    """
    ds_period = select_time_period(ds, clim_period, time_name=time_name)

    groupby, reduce_dim = _get_groupby_and_reduce_dims(
        ds,
        frequency,
        init_dim=init_dim,
        ensemble_dim=ensemble_dim,
        time_name=time_name,
    )

    if groupby is None:
        clim = ds_period.mean(reduce_dim)
        return ds - clim
    else:
        clim = ds_period.groupby(groupby).mean(reduce_dim)
        return (ds.groupby(groupby) - clim).drop(groupby.split(".")[-1])
