import pdb

import numpy as np
import pandas as pd
import xarray as xr
from skimage.util.shape import view_as_windows


#AUS_BOX = [-46, -9, 111, 157]
AUS_BOX = [-44, -11, 113, 154]


def get_region(da, box):
    """Select grid points that fall within a lat/lon box.
    
    Args:
      da (xarray DataArray)
      box (array-like) : [south bound, north bound, east bound, west bound]
    
    """

    lat_south_bound, lat_north_bound, lon_east_bound, lon_west_bound = box
    assert -90 <= lat_south_bound <= 90, "Valid latitude range is [-90, 90]"
    assert -90 <= lat_north_bound <= 90, "Valid latitude range is [-90, 90]"
    assert lat_south_bound < lat_north_bound, "South bound greater than north bound"
    assert 0 <= lon_east_bound < 360, "Valid longitude range is [0, 360)"
    assert 0 <= lon_west_bound < 360, "Valid longitude range is [0, 360)"
    
    da = da.assign_coords({'lon': (da['lon'] + 360)  % 360})
        
    mask_lat = (da['lat'] > lat_south_bound) & (da['lat'] < lat_north_bound)
    if lon_east_bound < lon_west_bound:
        mask_lon = (da['lon'] > lon_east_bound) & (da['lon'] < lon_west_bound)
    else:
        mask_lon = (da['lon'] > lon_east_bound) | (da['lon'] < lon_west_bound)
    
    da = da.where(mask_lat & mask_lon, drop=True) 
        
    #if sort:
    #    da = da.sortby(lat_name).sortby(lon_name)
    #da.sel({'lat': slice(box[0], box[1]), 'lon': slice(box[2], box[3])})

    return da


def stack_by_init_date_old(da, init_dates, N_lead_steps, freq='D'):
    """Stack timeseries array in inital date / lead time format. 
    
    Returns nans if requested times lie outside of the available range.
    
    """

    if xr.core.common.contains_cftime_datetimes(da['time']):
        times_np = xr.coding.times.cftime_to_nptime(da['time'])
    else:
        times_np = da['time']
    
    times_np = times_np.astype(f'datetime64[{freq}]')
    init_dates_np = init_dates.astype(f'datetime64[{freq}]')
    
    init_list = []
    for i in range(len(init_dates)):
        start_index = np.where(times_np == init_dates_np[i])[0]
        start_index = start_index.item()
        end_index = min([start_index + N_lead_steps, len(times_np)])
        lead_times = np.arange(0, end_index - start_index)
        
        da_slice = da.isel({'time': range(start_index, end_index)})
        time_slice = da_slice['time'].expand_dims({'init_date': [init_dates[i]]})
        da_slice = da_slice.expand_dims({'init_date': [init_dates[i]]})
        da_slice = da_slice.assign_coords({'time_new': time_slice})
        da_slice = da_slice.assign_coords({'time': lead_times})
        da_slice = da_slice.rename({'time': 'lead_time'})
        da_slice = da_slice.rename({'time_new': 'time'})
        
        init_list.append(da_slice)
            
    stacked = xr.concat(init_list, dim='init_date')
    stacked['lead_time'].attrs['units'] = freq
    
    return stacked


def stack_by_init_date_skimage(da, init_dates, N_lead_steps, freq='D'):
    """Stack timeseries array in inital date / lead time format."""
    
    if xr.core.common.contains_cftime_datetimes(da['time']):
        times_np = xr.coding.times.cftime_to_nptime(da['time'])
    else:
        times_np = da['time']
    
    times_np = times_np.astype(f'datetime64[{freq}]')
    init_dates_np = init_dates.astype(f'datetime64[{freq}]')
    
    start_index = np.where(times_np == init_dates_np[0])[0][0]
    end_index = start_index + N_lead_steps + (365 * (len(init_dates) - 1))
    array = da.values[start_index:end_index, ::]
    target_shape = list(array.shape)
    target_shape[0] = N_lead_steps
    stacked_data = view_as_windows(da.values[start_index:end_index, ::],
                                   target_shape, step=365).squeeze()
                                   # (init_date, lead_time, lat, lon)

        
def stack_by_init_date_xr(da, init_dates, N_lead_steps, freq='D'):
    """Stack timeseries array in inital date / lead time format. 
    
    
    
    """
    
    da = da.sel(time=~((da['time'].dt.month == 2) & (da['time'].dt.day == 29)))
    
    rounded_times = da['time'].dt.floor(freq).values
    
    time2d = np.empty((len(init_dates), N_lead_steps), 'datetime64[ns]')
    init_date_indexes = []
    offset = N_lead_steps - 1  # xarray rolling puts nans at the front
                               # and labels each window according to last value
                               # so an offset is needed
    for ndate, date in enumerate(init_dates):
        start_index = np.where(rounded_times == date)[0][0]
        end_index = start_index + N_lead_steps
        time2d[ndate, :] = da['time'][start_index:end_index].values
        init_date_indexes.append(start_index + offset)

    da = da.rolling(time=N_lead_steps, min_periods=1).construct("lead_time")
    da = da.assign_coords({'lead_time': da['lead_time'].values})
    da = da.rename({'time': 'init_date'})
    da = da[init_date_indexes, ::]
    da = da.assign_coords({'init_date': time2d[:, 0]})
    da = da.assign_coords({'time': (['init_date', 'lead_time'], time2d)})
    da['lead_time'].attrs['units'] = freq
    
    # TODO: Return nans if requested times lie outside of the available range
    
    return da
