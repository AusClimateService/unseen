"""Functions for array handling and manipulation."""

import pdb

import numpy as np
import xarray as xr

import time_utils


def stack_by_init_date(ds, init_dates, n_lead_steps, freq='D'):
    """Stack timeseries array in inital date / lead time format.

    Args:
      ds (xarray Dataset)
      init_dates (list) : Initial dates in YYYY-MM-DD format
      n_lead_steps (int) : Maximum lead time
      freq (str) : Time-step frequency
    """

    time_utils.check_date_format(init_dates)
    time_utils.check_cftime(ds['time'])

    rounded_times = ds['time'].dt.floor(freq).values
    ref_time = init_dates[0]
    ref_calendar = rounded_times[0].calendar
    ref_var = list(ds.keys())[0]
    ref_array = ds[ref_var].sel(time=ref_time).values    

    time2d = np.empty((len(init_dates), n_lead_steps), 'object')
    init_date_indexes = []
    offset = n_lead_steps - 1
    for ndate, date in enumerate(init_dates):
        date_cf = time_utils.str_to_cftime(date, ref_calendar)
        start_index = np.where(rounded_times == date_cf)[0][0]
        end_index = start_index + n_lead_steps
        time2d[ndate, :] = ds['time'][start_index:end_index].values
        init_date_indexes.append(start_index + offset)

    ds = ds.rolling(time=n_lead_steps, min_periods=1).construct("lead_time")
    ds = ds.assign_coords({'lead_time': ds['lead_time'].values})
    ds = ds.rename({'time': 'init_date'})
    ds = ds.isel(init_date=init_date_indexes)
    ds = ds.assign_coords({'init_date': time2d[:, 0]})
    ds = ds.assign_coords({'time': (['init_date', 'lead_time'], time2d)})
    ds['lead_time'].attrs['units'] = freq

    actual_array = ds[ref_var].sel({'init_date': ref_time, 'lead_time': 0}).values
    np.testing.assert_allclose(actual_array[0], ref_array[0])
    
    # TODO: Return nans if requested times lie outside of the available range
    
    return ds


def reindex_forecast(ds, dropna=False):
    """Switch out lead_time axis for time axis (or vice versa) in a forecast dataset."""
    
    if 'lead_time' in ds.dims:
        index_dim = 'lead_time'
        reindex_dim = 'time'
    elif 'time' in ds.dims:
        index_dim = 'time'
        reindex_dim = 'lead_time'
    else:
        raise ValueError("Neither a time nor lead_time dimension can be found")
    swap = {index_dim: reindex_dim}
    to_concat = []
    for init_date in ds['init_date']:
        fcst = ds.sel({'init_date': init_date})
        fcst = fcst.where(fcst[reindex_dim].notnull(), drop=True)
        fcst = fcst.assign_coords({'lead_time': fcst['lead_time'].astype(int)})
        to_concat.append(fcst.swap_dims(swap))
    concat = xr.concat(to_concat, dim='init_date')
    if dropna:
        concat = concat.where(concat.notnull(), drop=True)
    
    return concat


def to_init_lead(ds):
    """Switch out time axis for init_date and lead_time."""

    lead_time = range(len(ds['time']))
    init_date = time_utils.str_to_cftime(ds['time'].values[0].strftime('%Y-%m-%d'))
    new_coords = {'lead_time': lead_time, 'init_date': init_date}
    ds = ds.rename({'time': 'lead_time'})
    ds = ds.assign_coords(new_coords)

    return ds

