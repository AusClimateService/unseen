"""Functions for array handling and manipulation."""

import pdb

import numpy as np
import xarray as xr

import time_utils


def stack_by_init_date(ds, init_dates, n_lead_steps,
                       time_dim='time', init_dim='init', lead_dim='lead'):
    """Stack timeseries array in inital date / lead time format.

    Args:
      ds (xarray Dataset)
      init_dates (list) : Initial dates in YYYY-MM-DD format
      n_lead_steps (int) : Maximum lead time
      time_dim (str) : The name of the time dimension on ds
      init_dim (str) : The name of the initial date dimension on the output array
      lead_dim (str) : The name of the lead time dimension on the output array
      
    Note, only initial dates that fall within the time range of the input
    timeseries are retained. Thus, inital dates prior to the time range of
    the input timeseries that include data at longer lead times are not 
    included in the output dataset. To include these data, prepend the input
    timeseries with nans so that the initial dates in question are present
    in the time dimension of the input timeseries.
    """
    
    # Only keep init dates that fall within available times
    times = ds[time_dim]  
    init_dates = init_dates[np.logical_and(init_dates>=times.min(), init_dates<=times.max())]
    
    # Initialise indices of specified inital dates and time info for each initial date
    time2d = np.empty((len(init_dates), n_lead_steps), 'object')
    time2d[:] = np.nan # Nans where data do not exist
    init_date_indexes = []
    for ndate, init_date in enumerate(init_dates):
        start_index = np.where(times == init_date)[0][0]
        end_index = start_index + n_lead_steps
        time_slice = ds[time_dim][start_index:end_index]
        time2d[ndate, :len(time_slice)] = time_slice
        init_date_indexes.append(start_index)
        
    # Use `rolling` to stack timeseries like forecasts
    # Note, rolling references each window to the RH edge of the window. Hence we reverse the timeseries
    # so that each window starts at the specified initial date and includes n_lead_steps to the right of
    # that element
    ds = ds.copy().sel({time_dim: slice(None, None, -1)})
    init_date_indexes = [ds.sizes[time_dim] - 1 - i for i in init_date_indexes]
    init_date_indexes.reverse()
    
    strides = np.diff(init_date_indexes)
    # If stride is regular, use `stride` in rolling construct to reduce memory usage, otherwise construct
    # windowed object and then index out desired initial dates
    if np.all(strides == strides[0]):
        stride = strides[0]
        remainder = init_date_indexes[0] % stride # Where to start so that strides fall in right place
        ds = ds.isel({time_dim: slice(remainder, init_date_indexes[-1]+stride)})
        ds = ds.rolling({time_dim: n_lead_steps}, min_periods=1).construct(
            lead_dim, stride=stride, keep_attrs=True)
        ds = ds.isel({time_dim: slice(int(init_date_indexes[0] / stride), None)})
    else:
        ds = ds.rolling({time_dim: n_lead_steps}, min_periods=1).construct(
            lead_dim, keep_attrs=True)
        ds = ds.isel({time_dim: init_date_indexes})

    # Account for reversal of timeseries
    ds = ds.sel({time_dim: slice(None, None, -1), lead_dim: slice(None, None, -1)})
        
    ds = ds.rename({time_dim: init_dim})
    ds = ds.assign_coords({lead_dim: ds[lead_dim].values})
    ds = ds.assign_coords({time_dim: ([init_dim, lead_dim], time2d)})
    
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

