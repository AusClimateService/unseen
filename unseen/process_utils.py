"""Utilities for process-based assessment"""

import re
from collections import Counter
import calendar
import datetime

import numpy as np
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import xclim as xc


def plot_event_seasonality(df):
    """Plot event seasonality

    Parameters
    ----------
    df : pandas dataframe
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


def _get_run(file_list):
    """Get the model run information from a CMIP6 file path"""

    match = re.search('r.i.p.f.', file_list)
    try:
        run = match.group()
    except AttributeError:
        match = re.search('r..i.p.f.', file_list)
        run = match.group()
        
    return run


def _get_model_name(file_list):
    """Get the model name from an NPCC file path on NCI"""

    with open(file_list) as f:
        first_file = f.readline()

    return first_file.split('/')[8]


def _get_dcpp_da(df_row, infile_list, var, plot_units, time_agg, event_duration, init_year_offset=0):
    """Get DCPP data for an atmospheric circulation plot"""

    init_year = int(df_row['init_date'].strftime('%Y')) + init_year_offset
    ensemble_index = int(df_row['ensemble']) + 1
    end_date = df_row['event_time']
    start_datetime = datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=event_duration)
    start_date = start_datetime.strftime("%Y-%m-%d")
    runs = list(map(_get_run, infile_list))
    ensemble_labels = []
    for run in runs:
        if not run in ensemble_labels:
            ensemble_labels.append(run)
    target_text = f's{init_date}-{ensmeble_labels[ensemble_index]}'
#    target_files = list(filter(lambda x: target_text in x, infile_list))
    model_name = _get_model_name(infile_list)
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
    top_n_events,
    event_duration,
    infile_list,
    color_var=None,
    contour_var=None,
    init_year_offset=0
):
    """Plot the mean circulation for the n most extreme events.

    Parameters
    ----------
    df : pandas dataframe
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

    ranked_events = df.sort_values(by=[var], ascending=False)

    fig = plt.figure(figsize=[10, 17])
    map_proj=ccrs.PlateCarree(central_longitude=180)
    plotnum = 1
    for index, event_df_row in ranked_events.head(n=top_n_events).iterrows():
        ax = fig.add_subplot(top_n_events, 1, plotnum, projection=map_proj)
        if color_var:
            color_time_agg = 'sum' if color_var == 'pr' else 'mean'
            color_da, title = _get_dcpp_da(
                event_df_row,
                infile_list,
                color_var,
                color_units,
                color_time_agg,
                event_duration,
                init_year_offset=0
            )
            if color_var == 'pr':
                levels = [0, 100, 200, 300, 400, 500, 600, 700, 800]
                label = 'total precipitation (mm)'
                cmap = cmocean.cm.rain
                extend = 'max'
            elif color_var == 'ua300':
                levels = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
                label = '300hPa zonal wind'
                cmap='RdBu_r'
                extend = 'both'
            else:
                raise ValueError('Invalid color variable')
            color_da.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                levels=levels,
                extend=extend,
                cbar_kwargs={'label': label},
            )
        if contour_var:
            contour_time_agg = 'sum' if contour_var == 'pr' else 'mean'
            contour_da, title = _get_dcpp_da(
                event_df_row,
                infile_list,
                contour_var,
                contour_units,
                contour_time_agg,
                event_duration,
                init_year_offset=0
            )
            if contour_var == 'z500':
                levels = np.arange(5000, 6300, 50)
            elif contour_var == 'psl':
                levels = np.arange(900, 1100, 2.5)
            elif contour_var == 'ua300':
                levels = np.arange(15, 60, 5)
            else:
                raise ValueError('Invalid contour variable')
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

