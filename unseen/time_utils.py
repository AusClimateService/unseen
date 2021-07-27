"""Utilities for working with time axes and values"""

import pdb
import re

import cftime
from datetime import datetime
import xarray as xr


def check_date_format(date_list):
    """Check for YYYY-MM-DD format."""

    date_pattern = '([0-9]{4})-([0-9]{1,2})-([0-9]{1,2})'
    for date in date_list:
        assert re.search(date_pattern, date), \
            'Date format must be YYYY-MM-DD'


def check_cftime(time_dim):
    """Check that time dimension is cftime.

    Args:
      time_dim (xarray DataArray) : Time dimension
    """

    t0 = time_dim.values[0]
    assert type(t0) in cftime._cftime.DATE_TYPES.values(), \
        'Time dimension must use cftime objects'


def str_to_cftime(datestring, calendar='standard'):
    """Convert a date string to cftime object"""
    
    dt = datetime.strptime(datestring, '%Y-%m-%d')
    cfdt = cftime.datetime(dt.year, dt.month, dt.day, calendar=calendar)
     
    return cfdt


def cftime_to_str(time_dim):
    """Convert cftime array to YYY-MM-DD strings."""

    check_cftime(time_dim)
    str_times = [time.strftime('%Y-%m-%d') for time in time_dim.values]

    return str_times


def update_rate(da, input_freq, target_freq):
    """Update a flow rate due to temporal aggregation"""
    
    current_units = da.units
    rates_dict = {'D': 'd-1',
                  'M': 'month-1',
                  'Q': 'season-1',
                  'A': 'yr-1'}
    input_rate = rates_dict[input_freq]
    if input_rate in current_units:
        target_rate = rates_dict[target_freq[0]]
        new_units = current_units.replace(input_rate, target_rate)
    else:
        new_units = current_units
        
    return new_units

    
def temporal_aggregation(ds, target_freq, agg_method, variables, input_freq=None):
    """Temporal aggregation of data.

    Args:
      ds (xarray Dataset)
      target_freq (str) : Target frequency for the resampling (see options below)
      agg_method (str) : Aggregation method ('mean' or 'sum')
      variabless (list) : Variables in the dataset
      input_freq (str) : Temporal frequency of input data (daily 'D' or monthly 'M')

    Valid target frequencies:
      A-DEC (annual, with date label being last day of year) 
      M (monthly, with date label being last day of month)
      Q-NOV (DJF, MAM, JJA, SON, with date label being last day of season)
      A-NOV (annual Dec-Nov, date label being last day of the year)
    """

    assert target_freq in ['A-DEC', 'M', 'Q-NOV', 'A-NOV']
    
    start_time = ds['time'].values[0]
    if not input_freq:
        input_freq = xr.infer_freq(ds.indexes['time'][0:3])

    if input_freq == target_freq:
        pass
    elif agg_method == 'sum':
        ds = ds.resample(time=target_freq).sum(keep_attrs=True)
        for var in variables:
            ds[var].attrs['units'] = update_rate(ds[var], input_freq, target_freq)
    elif agg_method == 'mean':
        if input_freq == 'D':
            ds = ds.resample(time=target_freq).mean(keep_attrs=True)
        elif input_freq == 'M':
            ds = monthly_downsample_mean(ds, target_freq)
        else:
            raise ValueError(f'Unsupported input time frequency: {input_freq}')    
    else:
        raise ValueError(f'Unsupported temporal aggregation method: {agg_method}') 

    diff = ds['time'].values[0] - start_time
    ds['time'] = ds['time'] - diff
    assert ds['time'].values[0] == start_time

    return ds


def monthly_downsample_mean(ds, target_freq):
    """Downsample monthly data.

    Accounts for the different number of days in each month.
    """

    days_in_month = ds['time'].dt.days_in_month
    weighted_mean = ( (ds * days_in_month).resample(time=target_freq).sum(keep_attrs=True) / days_in_month.resample(time=target_freq).sum() )

    return weighted_mean


def get_monthly_clim(da, dim, time_period=None):
    """Calculate monthly climatology

    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate climatology
        (e.g. init_date)
      time_period (list) : Time period
    """

    if time_period is not None:
        da = mask_time_period(da.copy(), period)
        da.attrs['climatological_period'] = str(period)
    
    clim = da.groupby(f'{dim}.month').mean(dim, keep_attrs=True)

    return clim


def mask_time_period(da, period):
    """Mask a period of time.

    Args:
      da (xarray DataArray)
      period (list) : Start and stop dates (in YYYY-MM-DD format)

    Only works for cftime objects.
    """

    check_date_format(period)
    start, stop = period

    if 'time' in da.dims:
        masked = da.sel({'time': slice(start, stop)})
    elif 'time' in da.coords:
        calendar = da['time'].calendar_type.lower()
        time_bounds = xr.cftime_range(start=start, end=stop,
                                      periods=2, freq=None,
                                      calendar=calendar)
        time_values = da['time'].compute()
        mask = (time_values >= time_bounds[0]) & (time_values <= time_bounds[1])
        masked = da.where(mask)
    else:
        raise ValueError(f'No time axis for masking')
    masked.attrs = da.attrs

    return masked

