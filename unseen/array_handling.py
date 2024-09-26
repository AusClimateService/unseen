"""Functions for array handling and manipulation."""

import numpy as np
import xarray as xr
import cftime

from . import time_utils


def stack_by_init_date(
    ds,
    init_dates,
    n_lead_steps,
    time_dim="time",
    init_dim="init_date",
    lead_dim="lead_time",
    time_rounding="D",
):
    """Stack timeseries array in initial date / lead time format.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Input array containing a time dimension
    period : list
        List of initial dates of the same object type as the times in
        the time dimension of ds
    n_lead_steps: int
        Maximum number of lead time steps
    time_dim: str, default 'time'
        Name of the time dimension in ds
    init_dim: str, default 'init_date'
        Name of the initial date dimension to create in the output
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension to create in the output
    time_rounding :  {'A', 'M', 'D'}, default 'D'
        Match time axis and init dates by floor rounding to nearest day, month, or year

    Returns
    -------
    stacked : xarray DataArray or Dataset
        Array with data stacked by specified initial dates and lead steps

    Notes
    -----
    Only initial dates that fall within the time range of the input
    timeseries are retained. Thus, initial dates prior to the time range of
    the input timeseries that include data at longer lead times are not
    included in the output dataset. To include these data, prepend the input
    timeseries with NaNs so that the initial dates in question are present
    in the time dimension of the input timeseries.
    """
    # Only keep init dates that fall within available times
    times = ds[time_dim]
    init_dates = init_dates[
        np.logical_and(init_dates >= times.min(), init_dates <= times.max())
    ]

    # Initialise indexes of specified initial dates and time info for each initial date
    time2d = np.empty((len(init_dates), n_lead_steps), "object")
    time2d[:] = cftime.DatetimeGregorian(
        3000, 1, 1
    )  # Year 3000 where data do not exist
    init_date_indexes = []
    for ndate, init_date in enumerate(init_dates):
        start_index = _get_match_index(times, init_date.item(), time_rounding)
        end_index = start_index + n_lead_steps
        time_slice = ds[time_dim][start_index:end_index]
        time2d[ndate, : len(time_slice)] = time_slice
        init_date_indexes.append(start_index)

    # Use `rolling` to stack timeseries like forecasts
    # Note, rolling references each window to the RH edge of the window. Hence we reverse the timeseries
    # so that each window starts at the specified initial date and includes n_lead_steps to the right of
    # that element
    ds = ds.copy().sel({time_dim: slice(None, None, -1)})
    init_date_indexes = [ds.sizes[time_dim] - 1 - i for i in init_date_indexes]

    ds = ds.rolling({time_dim: n_lead_steps}, min_periods=1).construct(
        lead_dim, keep_attrs=True
    )
    ds = ds.isel({time_dim: init_date_indexes})

    # Account for reversal of timeseries
    ds = ds.sel({lead_dim: slice(None, None, -1)})

    ds = ds.rename({time_dim: init_dim})
    ds = ds.assign_coords({lead_dim: ds[lead_dim].values})
    ds = ds.assign_coords({time_dim: ([init_dim, lead_dim], time2d)})
    ds = ds.assign_coords({init_dim: init_dates.values})

    return ds


def reindex_forecast(ds, dropna=False):
    """Swap lead time dimension for time dimension (or vice versa) in a forecast dataset.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Forecast array containing an initial date and time or lead time dimension
    dropna : bool, default False
        Remove N/A values from array

    Returns
    -------
    swapped : xarray DataArray or Dataset
        Output array with time and lead time swapped

    Raises
    ------
    ValueError
        If a time or lead time dimension can't be found

    """

    if "lead_time" in ds.dims:
        index_dim = "lead_time"
        reindex_dim = "time"
    elif "time" in ds.dims:
        index_dim = "time"
        reindex_dim = "lead_time"
    else:
        raise ValueError("Neither a time nor lead_time dimension can be found")
    swap = {index_dim: reindex_dim}
    to_concat = []
    for init_date in ds["init_date"]:
        fcst = ds.sel({"init_date": init_date})
        fcst = fcst.where(fcst[reindex_dim].notnull(), drop=True)
        fcst = fcst.assign_coords({"lead_time": fcst["lead_time"].astype(int)})
        to_concat.append(fcst.swap_dims(swap))
    swapped = xr.concat(to_concat, dim="init_date")
    if dropna:
        swapped = swapped.where(swapped.notnull(), drop=True)

    return swapped


def time_to_lead(ds, freq):
    """Convert from time to (a newly created) lead time dimension.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Forecast array with initial date and time dimensions
    freq : str
        Time step frequency for lead time attributes

    Returns
    -------
    ds : xarray DataArray or Dataset
        Output array with initial date and lead time dimensions
    """

    time_values = []
    datasets = []
    time_attrs = ds["time"].attrs
    for init_date, ds_init_date in list(ds.groupby("init_date")):
        ds_init_date_cropped = ds_init_date.dropna("time")
        time_values.append(ds_init_date_cropped["time"].values)
        ds_init_date_cropped = to_init_lead(ds_init_date_cropped, init_date=init_date)
        datasets.append(ds_init_date_cropped)
    ds = xr.concat(datasets, dim="init_date")
    time_values = np.stack(time_values, axis=-1)
    time_dimension = xr.DataArray(
        time_values,
        attrs=time_attrs,
        dims={"lead_time": ds["lead_time"], "init_date": ds["init_date"]},
    )
    ds = ds.assign_coords({"time": time_dimension})
    ds["lead_time"].attrs["units"] = freq

    return ds


def to_init_lead(ds, init_date=None):
    """Swap time dimension for initial date and lead time dimensions.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Input array with time dimension
    init_date : cftime object, optional
        Initial date

    Returns
    -------
    ds : xarray DataArray or Dataset
        Output array with initial date and lead time dimensions
    """

    lead_time = range(len(ds["time"]))
    if not init_date:
        init_date = time_utils.str_to_cftime(ds["time"].values[0].strftime("%Y-%m-%d"))
    new_coords = {"lead_time": lead_time, "init_date": init_date}
    ds = ds.rename({"time": "lead_time"})
    ds = ds.assign_coords(new_coords)

    return ds


def _get_match_index(time_axis, target_date, time_rounding):
    """Find the index of a target date in a time axis.

    Parameters
    ----------
    time_axis : xarray DataArray (of cftime objects)
        Time dimension
    target_date : cftime object
        Target date
    time_rounding : {'A', 'M', 'D'}
        Time resolution (annual, monthly, or daily)

    Returns
    -------
    match_index : int
       Index of match

    Raises
    ------
    ValueError
       For invalid time_rounding value
    """

    if time_rounding == "A":
        str_format = "%Y"
    elif time_rounding == "M":
        str_format = "%Y-%m"
    elif time_rounding == "D":
        str_format = "%Y-%m-%d"
    else:
        raise ValueError("Time rounding must be A (annual), M (monthly) or D (daily)")

    time_values = time_utils.cftime_to_str(time_axis, str_format=str_format)
    init_value = target_date.strftime(str_format)
    match_index = time_values.index(init_value)

    return match_index
