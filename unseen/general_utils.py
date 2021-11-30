"""General utility functions."""

import argparse
import re

import yaml
import matplotlib as mpl
import xclim


class store_dict(argparse.Action):
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

    Args:
      da (xarray DataArray)
      target_units (str)
    """

    xclim_unit_check = {"deg_k": "degK"}

    if da.attrs["units"] in xclim_unit_check:
        da.attrs["units"] = xclim_unit_check[da.units]

    da = xclim.units.convert_units_to(da, target_units)

    return da


def date_pair_to_time_slice(date_list):
    """Convert two dates to a time slice object."""

    assert len(date_list) == 2
    start_date, end_date = date_list

    date_pattern = "([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})"
    assert re.search(date_pattern, start_date), "Start date not in YYYY-MM-DD format"
    assert re.search(date_pattern, end_date), "End date not in YYYY-MM-DD format"

    time_slice = slice(start_date, end_date)

    return time_slice


def set_plot_params(param_file):
    """Set the matplotlib parameters."""

    if param_file:
        with open(param_file, "r") as reader:
            param_dict = yaml.load(reader, Loader=yaml.BaseLoader)
    else:
        param_dict = {}
    for param, value in param_dict.items():
        mpl.rcParams[param] = value
