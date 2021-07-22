"""Functions for spatial selection."""

import pdb

import numpy as np
import xarray as xr
import geopandas as gp
import regionmask


def select_region(ds, region=None, agg=None, header=None):
    """Select region.
    
    Args:
      ds (xarray Dataset or DataArray)
      region (str or list) : shapefile name, or
                             list length 2 (point selection), or
                             list length 4 (box selection).
      agg (str) : Aggregation method (spatial 'mean' or 'sum')
      header (str) : Name of the shapefile column containing the region names 
    """

    if type(region) == str:
        ds = select_shapefile_regions(ds, region, agg=agg, header=header)
    elif len(region) == 4:
        ds = select_box_region(ds, region)
    elif len(region) == 2:
        ds = select_point_region(ds, region)
    elif region == None:
        pass
    else:
        msg = 'region must be None, shapefile, box (list of 4 floats) or point (list of 2 floats)'
        raise ValueError(msg)
    
    if (agg == 'sum') and not type(region) == str:
        ds = ds.sum(dim=('lat', 'lon'))
    elif (agg == 'mean') and not type(region) == str:
        ds = ds.mean(dim=('lat', 'lon'))
    elif (agg == None) or type(region) == str:
        pass 
    else:
         raise ValueError("""agg must be None, 'sum' or 'mean'""")   

    return ds


def select_shapefile_regions(ds, shapefile, agg=None, header=None):
    """Select region using a shapefile.

    Args:
      ds (xarray Dataset or DataArray)
      shapefile (str) : Shapefile
      agg(str) : Aggregation method (spatial 'mean' or 'sum')
      header (str) : Name of the shapefile column containing the region names 
    """

    lons = ds['lon'].values
    lats = ds['lat'].values

    shapes = gp.read_file(shapefile)

    if agg == None:
        mask = regionmask.mask_geopandas(shapes, lons, lats)
        mask = xr.where(mask.notnull(), True, False)
        ds = ds.where(mask)
    elif agg == 'sum':
        mask = regionmask.mask_geopandas(shapes, lons, lats)
        ds = ds.groupby(mask).sum(keep_attrs=True)
    elif agg == 'mean':
        mask = regionmask.mask_3D_geopandas(shapes, lons, lats)
        weights = np.cos(np.deg2rad(ds['lat']))
        ds = ds.weighted(mask * weights).mean(dim=('lat', 'lon'), keep_attrs=True)

    if header:
        ds = ds.assign_coords(region=shapes[header].values)

    return ds


def select_box_region(ds, box):
    """Select grid points that fall within a lat/lon box.
    
    Args:
      ds (xarray Dataset or DataArray)
      box (list) : [south bound, north bound, east bound, west bound]
    """

    lat_south_bound, lat_north_bound, lon_east_bound, lon_west_bound = box
    assert -90 <= lat_south_bound <= 90, "Valid latitude range is [-90, 90]"
    assert -90 <= lat_north_bound <= 90, "Valid latitude range is [-90, 90]"
    assert lat_south_bound < lat_north_bound, "South bound greater than north bound"
    assert 0 <= lon_east_bound < 360, "Valid longitude range is [0, 360)"
    assert 0 <= lon_west_bound < 360, "Valid longitude range is [0, 360)"
    
    ds = ds.assign_coords({'lon': (da['lon'] + 360)  % 360})
        
    mask_lat = (ds['lat'] > lat_south_bound) & (ds['lat'] < lat_north_bound)
    if lon_east_bound < lon_west_bound:
        mask_lon = (ds['lon'] > lon_east_bound) & (ds['lon'] < lon_west_bound)
    else:
        mask_lon = (ds['lon'] > lon_east_bound) | (ds['lon'] < lon_west_bound)
    
    ds = ds.where(mask_lat & mask_lon, drop=True) 

    return ds


def select_point_region(ds, point):
    """Select a single grid point.
    
    Args:
      ds (xarray Dataset or DataArray)
      point (list) : [lat, lon]
    """
    
    lat, lon = point
    ds = ds.sel({'lat': lat, 'lon': lon}, method='nearest', drop=True)
    
    return ds

