import pdb

import numpy as np
import pandas as pd
import xarray as xr


box_regions = {'aus': [-44, -11, 113, 154]}


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
        return concat.where(concat.notnull(), drop=True)
    else:
        return concat
