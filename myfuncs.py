import pdb

import numpy as np
import pandas as pd
import xarray as xr

regions = {'AUS-BOX': [-44, -11, 113, 154],
           'MEL-POINT': (-37.81, 144.96),
           'TAS-POINT': (-42, 146.5),
           }


def select_region(da, region):
    """Select region."""
    
    ## TODO: Add shapefile region selection
    if len(region) == 4:
        da = select_box_region(da, region)
    elif len(region) == 2:
        da = select_point_region(da, region)
    else:
        raise ValueError('region is not a box (4 values) or point (2 values)')
    
    return da


def select_point_region(da, point):
    """Select a single grid point.
    
    Args:
      da (xarray DataArray)
      point (array-like) : [lat, lon]
    
    """
    
    lat, lon = point
    da = da.sel({'lat': lat, 'lon': lon}, method='nearest', drop=True)
    
    return da
    
    
def select_box_region(da, box):
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


def convert_pr_units(da):
    """Convert kg m-2 s-1 to mm day-1.
    
    Args:
      da (xarray.DataArray): Precipitation data
   
    """
   
    if da.units in ['kg m-2 s-1', 'kg/m2/s']:
        da = da * 86400
        da.attrs['units'] = 'mm/day'
    
    if da.units == 'mm':
        da.attrs['units'] = 'mm/day'
    
    assert da.units == 'mm/day'

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
