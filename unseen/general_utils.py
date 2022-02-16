"""General utility functions."""

import argparse
import re

import numpy as np
import xclim


class store_dict(argparse.Action):
    """An argparse action for parsing a command line argument as a dictionary.

    Examples
    --------
    precip=mm/day becomes {'precip': 'mm/day'}
    ensemble=1:5 becomes {'ensemble': slice(1, 5)}
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, val = value.split("=")
            if ":" in val:
                start, end = val.split(":")
                try:
                    start = int(start)
                    end = int(end)
                except ValueError:
                    pass
                val = slice(start, end)
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = val


def convert_units(da, target_units):
    """Convert units.

    Parameters
    ----------
    da : xarray DataArray
        Input array containing a units attribute
    target_units : str
        Units to convert to

    Returns
    -------
    da : xarray DataArray
       Array with converted units
    """

    xclim_unit_check = {"deg_k": "degK"}

    if da.attrs["units"] in xclim_unit_check:
        da.attrs["units"] = xclim_unit_check[da.units]

    if da.attrs["units"] == "kg m-2 s-1 month-1" and target_units == "mm month-1":
        # special case for CAFE monthly rainfall data
        da.data = da.data * 86400
        da.attrs["units"] = "mm month-1"
    else:
        da = xclim.units.convert_units_to(da, target_units)

    return da


def date_pair_to_time_slice(date_list):
    """Convert two dates to a time slice object.

    Parameters
    ----------
    date_list : list or tuple
        Start and end date in YYYY-MM-DD format

    Returns
    -------
    time_slice : slice
        Slice from start to end date
    """

    assert len(date_list) == 2
    start_date, end_date = date_list

    date_pattern = "([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})"
    assert re.search(date_pattern, start_date), "Start date not in YYYY-MM-DD format"
    assert re.search(date_pattern, end_date), "End date not in YYYY-MM-DD format"

    time_slice = slice(start_date, end_date)

    return time_slice


def exceedance_curve(data, deceedance=False):
    """Calculate exceedance curve.

    Parameters
    ----------
    data : numpy ndarray
        Data array
    deceedance : bool
        Return deceedance curve instead

    Returns
    -------
    sorted_data : numpy ndarray
        Sorted data for plot x-axis values
    pct : numpy ndarray
        exceedance (or deceedance) data
    """

    sorted_data = np.sort(data, axis=None)
    exceedance = 1.0 - np.arange(1.0, len(data) + 1.0) / len(data)

    pct = exceedance * 100
    if deceedance:
        pct = 100 - pct

    return sorted_data, pct


def event_in_context(data, threshold, direction):
    """Put an event in context.

    Parameters
    ----------
    data : numpy ndarray
        Population data
    threshold : float
        Event threshold
    direction : {'above', 'below'}
        Provide statistics for above or below threshold

    Returns
    -------
    percentile : float
        Event percentile relative to population (%)
    return_period : float
        Return period for event
    """

    n_population = len(data)
    if direction == "below":
        n_events = np.sum(data < threshold)
    elif direction == "above":
        n_events = np.sum(data > threshold)
    else:
        raise ValueError("""direction must be 'below' or 'above'""")
    percentile = (np.sum(data < threshold) / n_population) * 100
    return_period = n_population / n_events

    return percentile, return_period
