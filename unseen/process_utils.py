"""Utilities for process-based assessment"""

import glob
import re
import calendar
import datetime

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import xclim as xc
import cmocean

from . import fileio


def get_event_seasonality(event_time, core_dim="time"):
    """Count the number of events each month

    Parameters
    ----------
    event_time : xarray.DataArray
        Time of each event
    core_dim : list, default "time"
        Core dimensions of the input data array

    Returns
    -------
    counts : xarray.DataArray
        Number of events occuring in each month
    """

    def count_months(event_months):
        """Count the number of events in each month."""
        months_unique, month_counts = np.unique(event_months, return_counts=True)
        counts = np.zeros(12)
        counts[months_unique - 1] = month_counts
        return counts

    if not isinstance(core_dim, list):
        core_dim = [core_dim]

    # Event not a value or datetime64
    if event_time.dtype in ["<U10", "str", np.dtype(object)]:
        # Convert time strings to datetime64
        event_time = event_time.astype("datetime64[ns]")

    event_months = event_time.dt.month
    counts = xr.apply_ufunc(
        count_months,
        event_months,
        input_core_dims=[core_dim],
        output_core_dims=[["month"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs=dict(output_sizes={"month": 12}),
    )
    return counts


def plot_event_seasonality(ds, ax=None, core_dim="time", outfile=None):
    """Plot event seasonality

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a event_time variable
    ax : matplotlib.axes, optional
        Axes on which to plot. By default, create a new figure
    core_dim : list, default None
        Core dimensions of the input data array
    outfile : str, optional
        Path for output image file
    """
    months = np.arange(1, 13)
    counts = get_event_seasonality(ds["event_time"], core_dim=core_dim)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    ax.bar(months, counts)
    # ax.hist(event_months, bins=np.arange(1, 14))
    ax.set_ylabel("Number of events")
    ax.set_xlabel("Month")
    ax.set_xticks(months, labels=[calendar.month_abbr[i] for i in months])
    plt.tight_layout()

    if outfile:
        plt.savefig(
            outfile,
            bbox_inches="tight",
            facecolor="white",
            dpi=200,
        )
    else:
        plt.show()


def _get_run(file_list):
    """Get the model run information from a CMIP6 file path"""

    match = re.search("r.i.p.f.", file_list)
    try:
        run = match.group()
    except AttributeError:
        match = re.search("r..i.p.f.", file_list)
        run = match.group()

    return run


def _get_model_name(file_path):
    """Get the model name from an NPCC file path on NCI"""

    return file_path.split("/")[8]


def _get_dcpp_da(
    df_row, var, plot_units, time_agg, event_duration, infile_list, init_year_offset=0
):
    """Get DCPP data for an atmospheric circulation plot"""

    init_year = int(df_row["init_date"].strftime("%Y")) + init_year_offset
    ensemble_index = int(df_row["ensemble"])
    end_date = df_row["event_time"]
    start_datetime = datetime.datetime.strptime(
        end_date, "%Y-%m-%d"
    ) - datetime.timedelta(days=event_duration)
    start_date = start_datetime.strftime("%Y-%m-%d")
    with open(infile_list) as f:
        infiles = f.read().splitlines()
    runs = list(map(_get_run, infiles))
    ensemble_labels = []
    for run in runs:
        if run not in ensemble_labels:
            ensemble_labels.append(run)
    target_text = f"s{init_year}-{ensemble_labels[ensemble_index]}"
    model_name = _get_model_name(infiles[0])
    target_files = glob.glob(
        f"/g/data/oi10/replicas/CMIP6/DCPP/*/{model_name}/dcppA-hindcast/{target_text}/day/{var}/*/*/*.nc"
    )
    target_files = sorted(target_files)
    ds = xr.open_mfdataset(target_files)
    da = ds.sel({"time": slice(start_date, end_date)})[var]
    da = xc.units.convert_units_to(da, plot_units)
    if time_agg == "sum":
        da_agg = da.sum("time", keep_attrs=True)
    else:
        da_agg = da.mean("time", keep_attrs=True)
    title = f"{start_date} to {end_date} ({target_text})"

    return da_agg, title


def _get_cafe_da(df_row, var, plot_units, time_agg, event_duration):
    """Get CAFE data for an atmospheric circulation plot"""

    init_date = df_row["init_date"].strftime("%Y%m%d")
    ensemble = df_row["ensemble"]
    end_date = df_row["event_time"]
    start_datetime = datetime.datetime.strptime(
        end_date, "%Y-%m-%d"
    ) - datetime.timedelta(days=event_duration)
    start_date = start_datetime.strftime("%Y-%m-%d")
    cafe_vars = {"pr": "pr", "psl": "slp", "z500": "h500", "ua300": "ucomp"}
    cafe_var = cafe_vars[var]
    ds = fileio.open_dataset(
        f"/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-{init_date}/atmos_isobaric_daily.zarr.zip",
        metadata_file="/home/599/dbi599/unseen/config/dataset_cafe_daily.yml",
        variables=[cafe_var],
    )
    selection = {"ensemble": ensemble, "time": slice(start_date, end_date)}
    if var == "ua300":
        selection["level"] = 300
    da = ds.sel(selection)[cafe_var]
    da = xc.units.convert_units_to(da, plot_units)
    if time_agg == "sum":
        da_agg = da.sum("time", keep_attrs=True)
    else:
        da_agg = da.mean("time", keep_attrs=True)
    title = f"{start_date} to {end_date} (CAFE, c5-d60-pX-f6-{init_date})"

    return da_agg, title


def _get_barra_da(df_row, var, plot_units, time_agg, event_duration):
    """Get BARRA2 data for an atmospheric circulation plot"""

    end_date = df_row["event_time"]
    start_datetime = datetime.datetime.strptime(
        end_date, "%Y-%m-%d"
    ) - datetime.timedelta(days=event_duration)
    start_date = start_datetime.strftime("%Y-%m-%d")
    home_dir = "/g/data/yb19/australian-climate-service/release/ACS-BARRA2"
    day_dir = "output/AUS-11/BOM/ECMWF-ERA5/historical/hres/BOM-BARRA-R2/v1/day"
    barra2_files = glob.glob(
        f"{home_dir}/{day_dir}/{var}/{var}_AUS-11_ECMWF-ERA5_historical_hres_BOM-BARRA-R2_v1_day_*.nc"
    )
    ds = xr.open_mfdataset(barra2_files)
    da = ds.sel({"time": slice(start_date, end_date)})[var]
    da = xc.units.convert_units_to(da, plot_units)
    if time_agg == "sum":
        da_agg = da.sum("time", keep_attrs=True)
    else:
        da_agg = da.mean("time", keep_attrs=True)
    title = f"{start_date} to {end_date} (BARRA2)"

    return da_agg, title


def plot_circulation(
    df,
    event_var,
    top_n_events,
    event_duration,
    dataset,
    infile_list=None,
    color_var=None,
    contour_var=None,
    color_levels=None,
    init_year_offset=0,
    outfile=None,
):
    """Plot the mean circulation for the n most extreme events.

    Parameters
    ----------
    df : pandas dataframe
    event_var : str
        Variable (df column label)
    top_n_events : int
        Plot the top N events
    event_duration: int
        Duration (in days) of each event (e.g. Rx5day = 5)
    dataset : {'DCPP', 'CAFE', 'BARRA2'}
        Dataset to plot
    infile_list : str, optional (required for DCPP dataset)
        Input file list used to calculate the metric of interest
    color_var : {'pr', 'ua300'}, optional
        Variable for color plot
    contour_var : {'z500', 'psl', 'ua300'}, optional
        Variable for contour plot
    init_year_offset : optional, default=0
        Offset for initial year labelling (needed for some DCPP models)
    outfile : str, optional
        Path for output file
    """

    ranked_events = df.sort_values(by=[event_var], ascending=False)
    data_func = {
        "DCPP": _get_dcpp_da,
        "CAFE": _get_cafe_da,
        "BARRA2": _get_barra_da,
    }
    data_kwargs = {}
    if dataset == "DCPP":
        data_kwargs["init_year_offset"] = init_year_offset
        data_kwargs["infile_list"] = infile_list

    fig = plt.figure(figsize=[10, top_n_events * 6])
    map_proj = ccrs.PlateCarree(central_longitude=180)
    plotnum = 1
    for index, event_df_row in ranked_events.head(n=top_n_events).iterrows():
        ax = fig.add_subplot(top_n_events, 1, plotnum, projection=map_proj)
        if color_var:
            if color_var == "pr":
                label = "total precipitation (mm)"
                cmap = cmocean.cm.rain
                extend = "max"
                color_units = "mm d-1"
                color_time_agg = "sum"
            elif color_var == "ua300":
                label = "300hPa zonal wind"
                cmap = "RdBu_r"
                extend = "both"
                color_units = "m s-1"
                color_time_agg = "mean"
            else:
                raise ValueError("Invalid color variable")
            color_da, title = data_func[dataset](
                event_df_row,
                color_var,
                color_units,
                color_time_agg,
                event_duration,
                **data_kwargs,
            )
            color_da.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                levels=color_levels,
                extend=extend,
                cbar_kwargs={"label": label},
            )
        if contour_var:
            if contour_var == "z500":
                levels = np.arange(5000, 6300, 50)
                contour_units = "m"
                contour_time_agg = "mean"
            elif contour_var == "psl":
                levels = np.arange(900, 1100, 2.5)
                contour_units = "hPa"
                contour_time_agg = "mean"
            elif contour_var == "ua300":
                levels = np.arange(15, 60, 5)
                contour_units = "m s-1"
                contour_time_agg = "mean"
            else:
                raise ValueError("Invalid contour variable")
            contour_da, title = data_func[dataset](
                event_df_row,
                contour_var,
                contour_units,
                contour_time_agg,
                event_duration,
                **data_kwargs,
            )
            lines = contour_da.plot.contour(
                ax=ax, transform=ccrs.PlateCarree(), levels=levels, colors=["0.1"]
            )
            ax.clabel(lines, colors=["0.1"], manual=False, inline=True)
        ax.coastlines()
        ax.set_extent([90, 205, -55, 10], crs=ccrs.PlateCarree())
        ax.gridlines(linestyle="--", draw_labels=True)
        if contour_var:
            ax.set_title(f"Average {contour_var} ({contour_da.units}), {title}")
        else:
            ax.set_title(title)
        plotnum += 1

    if outfile:
        plt.savefig(
            outfile,
            bbox_inches="tight",
            facecolor="white",
            dpi=200,
        )
    else:
        plt.show()
