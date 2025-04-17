"""General utility functions."""

import argparse
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from xarray import Dataset
from xclim.core import units
from xesmf import Regridder


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

    xclim_unit_check = {
        "deg_k": "degK",
        "kg/m2/s": "kg m-2 s-1",
    }

    if da.attrs["units"] in xclim_unit_check:
        da.attrs["units"] = xclim_unit_check[da.units]

    try:
        da = units.convert_units_to(da, target_units)
    except Exception as e:
        in_precip_kg = da.attrs["units"] == "kg m-2 s-1"
        out_precip_mm = target_units in ["mm d-1", "mm day-1"]
        if in_precip_kg and out_precip_mm:
            da = da * 86400
            da.attrs["units"] = target_units
        else:
            raise e

    return da


def regrid(ds, ds_grid, method="conservative", **kwargs):
    """Regrid `ds` to the grid of `ds_grid` using xESMF.

    Parameters
    ----------
    ds : Union[xarray.DataArray, xarray.Dataset]
        Input data
    ds_grid : Union[xarray.DataArray, xarray.Dataset]
        Target grid.
    method : {"conservative", "bilinear", "nearest_s2d", "nearest_d2s"}, default "conservative"
        Regridding method
    **kwargs
        Additional keyword arguments for xESMF.Regridder

    Returns
    -------
    ds_regrid : Union[xarray.DataArray, xarray.Dataset]
        Regridded xarray.DataArray or xarray.Dataset

    Notes
    -----
    - The input and target grids should have the same coordinate names.
    - Recommended using the "conservative" method for regridding from fine to
    coarse and "bilinear" for the opposite.
    """

    # Copy attributes
    global_attrs = ds.attrs
    if isinstance(ds, Dataset):
        var_attrs = {var: ds[var].attrs for var in ds.data_vars}

    # Regrid data
    regridder = Regridder(ds, ds_grid, method, **kwargs)
    ds_regrid = regridder(ds)

    # Update regridded data attributes
    ds_regrid.attrs.update(global_attrs)
    if isinstance(ds_regrid, Dataset):
        for var in ds_regrid.data_vars:
            ds_regrid[var].attrs.update(var_attrs[var])

    return ds_regrid


def get_model_makefile_dict(cwd, project_details, model, model_details, obs_details):
    """Get dictionary of variables defined in config files and makefile.

    Parameters
    ----------
    cwd : str
        Directory of makefile
    project_details : str
        Project details file
    model : str
        Model name
    model_details : str
        Model details file
    obs_details : str
        Observed data details file

    Returns
    -------
    model_var_dict : dict
        Dictionary of model variables defined in the makefile and details files
    """

    args = [
        "make",
        "print_file_vars",
        f"PROJECT_DETAILS={project_details}",
        f"MODEL={model}",
        f"MODEL_DETAILS={model_details}",
        f"OBS_DETAILS={obs_details}",
    ]

    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)

    # Read stdout into dictionary
    model_var_dict = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            model_var_dict[key.lower()] = value

    # Sort dictionary by key
    model_var_dict = dict(sorted(model_var_dict.items()))
    return model_var_dict


def plot_timeseries_scatter(
    da,
    da_obs=None,
    ax=None,
    title=None,
    label=None,
    obs_label=None,
    units=None,
    time_dim="time",
    outfile=None,
):
    """Timeseries scatter plot of ensemble and observed data.

    Parameters
    ----------
    da : xarray.DataArray
        Stacked ensemble data
    da_obs : xarray.DataArray, optional
        Observed data
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, if None, a new figure is created
    title : str, optional
        Title of the plot
    label : str, optional
        Label for ensemble data
    obs_label : str, optional
        Label for observed data
    units : str, optional
        Units of the data. If None, the units attribute of da is used.
    time_dim : str, optional
        Name of the time dimension in da and da_obs
    outfile : str, optional
        Filename to save the plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object
    """

    if units is None:
        if "units" in da.attrs:
            units = da.attrs["units"]
        else:
            units = ""

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    if title is not None:
        ax.set_title(title, loc="left")

    # Plot observed data
    if da_obs is not None:
        ax.scatter(
            da_obs[time_dim],
            da_obs,
            s=20,
            c="k",
            marker="x",
            label=obs_label,
            zorder=10,
        )
    # Plot ensemble data
    ax.scatter(da[time_dim], da, s=5, c="deepskyblue", label=label)

    ax.set_ylabel(units)
    ax.set_xmargin(1e-2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    return ax


def plot_timeseries_box_plot(
    da,
    da_obs=None,
    ax=None,
    title=None,
    label=None,
    obs_label=None,
    units=None,
    time_dim="time",
    outfile=None,
):
    """Timeseries box and whisker plot of ensemble and observed data.

    Parameters
    ----------
    da : xarray.DataArray
        Stacked ensemble data (see Notes about the time dimension)
    da_obs : xarray.DataArray, optional
        Observed data
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, if None, a new figure is created
    title : str, optional
        Title of the plot
    label : str, optional
        Label for ensemble data
    obs_label : str, optional
        Label for observed data
    units : str, optional
        Units of the data. If None, the units attribute of da is used.
    time_dim : str, optional
        Name of the time dimension in da and da_obs
    outfile : str, optional
        Filename to save the plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object

    Notes
    -----
    Ensure all time dimensions are set to the correct frequency before calling
    this function.

    Examples
    --------
    - Input ensemble data grouped by year:
        da = da_orig.copy()
        da.coords["time"] = da.time.dt.year
        da["init_date"] = da.init_date.dt.year
        da = da.stack({"sample": ["ensemble", "init_date", "lead_time"]})
        plot_timeseries_box_plot(da, time_dim="time")
    """

    if units is None:
        if "units" in da.attrs:
            units = da.attrs["units"]
        else:
            units = ""
    # Group model data
    da_grps = [da.isel(sample=v) for k, v in da.groupby(da[time_dim]).groups.items()]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    if title is not None:
        ax.set_title(title, loc="left")

    # Plot model box and whiskers for each unique time
    ax.boxplot(da_grps, positions=np.unique(da[time_dim]), manage_ticks=False)

    # Plot observed data as blue crosses
    ax.scatter(da_obs[time_dim], da_obs, s=30, c="b", marker="x", label=obs_label)
    ax.set_ylabel(units)
    ax.set_xmargin(1e-2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend()

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
    return ax
