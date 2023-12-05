"""Utilities for process-based assessment"""

import glob
import re
from collections import Counter
import calendar
import datetime

import numpy as np
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import xarray as xr
import xclim as xc
import cmocean


def plot_event_seasonality(df, outfile=None):
    """Plot event seasonality

    Parameters
    ----------
    df : pandas dataframe
    outfile : str, optional
        Path for output image file
    """

    event_months = [int(date[5:7]) for date in df['event_time'].values]
    event_years = [int(date[0:4]) for date in df['event_time'].values]

    month_counts = Counter(event_months)
    months = np.arange(1, 13)
    counts = [month_counts[month] for month in months]

    plt.bar(months, counts)
    plt.ylabel('number of events')
    plt.xlabel('month')
    xlabels = [calendar.month_abbr[i] for i in months]
    plt.xticks(months, xlabels)

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

    match = re.search('r.i.p.f.', file_list)
    try:
        run = match.group()
    except AttributeError:
        match = re.search('r..i.p.f.', file_list)
        run = match.group()
        
    return run


def _get_model_name(file_path):
    """Get the model name from an NPCC file path on NCI"""

    return file_path.split('/')[8]


def _get_dcpp_da(df_row, infile_list, var, plot_units, time_agg, event_duration, init_year_offset=0):
    """Get DCPP data for an atmospheric circulation plot"""

    init_year = int(df_row['init_date'].strftime('%Y')) + init_year_offset
    ensemble_index = int(df_row['ensemble'])
    end_date = df_row['event_time']
    start_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=event_duration)
    start_date = start_datetime.strftime("%Y-%m-%d")
    with open(infile_list) as f:
        infiles = f.read().splitlines()
    runs = list(map(_get_run, infiles))
    ensemble_labels = []
    for run in runs:
        if not run in ensemble_labels:
            ensemble_labels.append(run)
    target_text = f's{init_year}-{ensemble_labels[ensemble_index]}'
    model_name = _get_model_name(infiles[0])
    target_files = glob.glob(f'/g/data/oi10/replicas/CMIP6/DCPP/*/{model_name}/dcppA-hindcast/{target_text}/day/{var}/*/*/*.nc')
    target_files = sorted(target_files)
    ds = xr.open_mfdataset(target_files)
    da = ds.sel({'time': slice(start_date, end_date)})[var]
    da = xc.units.convert_units_to(da, plot_units)
    if time_agg == 'sum':
        da_agg = da.sum('time', keep_attrs=True)
    else:
        da_agg = da.mean('time', keep_attrs=True)
    title = f'{start_date} to {end_date} ({target_text})'

    return da_agg, title


def plot_circulation(
    df,
    event_var,
    top_n_events,
    event_duration,
    infile_list,
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
    infile_list : str
        Input file list used to calculate the metric of interest
    color_var : str, optional
        Variable for color plot
    contour_var : str, optional
        Variable for contour plot

    """

    ranked_events = df.sort_values(by=[event_var], ascending=False)

    fig = plt.figure(figsize=[10, top_n_events*6])
    map_proj=ccrs.PlateCarree(central_longitude=180)
    plotnum = 1
    for index, event_df_row in ranked_events.head(n=top_n_events).iterrows():
        ax = fig.add_subplot(top_n_events, 1, plotnum, projection=map_proj)
        if color_var:
            if color_var == 'pr':
                label = 'total precipitation (mm)'
                cmap = cmocean.cm.rain
                extend = 'max'
                color_units = 'mm d-1'
                color_time_agg = 'sum'
            elif color_var == 'ua300':
                label = '300hPa zonal wind'
                cmap='RdBu_r'
                extend = 'both'
                color_units = 'm s-1'
                color_time_agg = 'mean'
            else:
                raise ValueError('Invalid color variable')
            color_da, title = _get_dcpp_da(
                event_df_row,
                infile_list,
                color_var,
                color_units,
                color_time_agg,
                event_duration,
                init_year_offset=0
            )
            color_da.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                levels=color_levels,
                extend=extend,
                cbar_kwargs={'label': label},
            )
        if contour_var:
            if contour_var == 'z500':
                levels = np.arange(5000, 6300, 50)
                contour_units = 'm'
                contour_time_agg = 'mean'
            elif contour_var == 'psl':
                levels = np.arange(900, 1100, 2.5)
                contour_units = 'hPa'
                contour_time_agg = 'mean'
            elif contour_var == 'ua300':
                levels = np.arange(15, 60, 5)
                contour_units = 'm s-1'
                contour_time_agg = 'mean'
            else:
                raise ValueError('Invalid contour variable')
            contour_da, title = _get_dcpp_da(
                event_df_row,
                infile_list,
                contour_var,
                contour_units,
                contour_time_agg,
                event_duration,
                init_year_offset=0
            )
            lines = contour_da.plot.contour(
                ax=ax,
                transform=ccrs.PlateCarree(),
                levels=levels,
                colors=['0.1']
            )
            ax.clabel(lines, colors=['0.1'], manual=False, inline=True)
        ax.coastlines()
        ax.set_extent([90, 205, -55, 10], crs=ccrs.PlateCarree())
        ax.gridlines(linestyle='--', draw_labels=True)
        if contour_var:
            ax.set_title(f'Average {contour_var} ({contour_da.units}), {title}')
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

