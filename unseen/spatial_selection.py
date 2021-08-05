"""Functions for spatial selection."""

import pdb

import numpy as np
import xarray as xr
import geopandas as gp
import regionmask


def select_region(ds,
                  coords=None,
                  shapefile=None,
                  header=None,
                  combine_shapes=False,
                  agg=None):
    """Select region.
    
    Args:
      ds (xarray Dataset or DataArray)
      coords (list) : List of length 2 [lat, lon] or 4 [south bound, north bound, east bound, west bound]
      shapefile (str) : Shapefile for spatial subseting
      header (str) : Name of the shapefile column containing the region names 
      combine_shapes (bool) : Add region that combines all shapes in shapefile
      agg (str) : Aggregation method (spatial 'mean' or 'sum')
    """

    if coords == None:
        pass
    elif len(coords) == 4:
        ds = select_box_region(ds, coords)
    elif len(coords) == 2:
        ds = select_point_region(ds, coords)
    else:
        msg = 'coordinate selection must be None, a box (list of 4 floats) or a point (list of 2 floats)'
        raise ValueError(msg)

    if shapefile:
        ds = select_shapefile_regions(ds, shapefile, agg=agg, header=header,
                                      combine_shapes=combine_shapes)
        
    if (agg == 'sum') and not shapefile:
        ds = ds.sum(dim=('lat', 'lon'))
    elif (agg == 'mean') and not shapefile:
        ds = ds.mean(dim=('lat', 'lon'))
    elif (agg == None) or shapefile:
        pass 
    else:
         raise ValueError("""agg must be None, 'sum' or 'mean'""")   

    return ds


def add_combined_shape(mask):
    """Add new region to mask that combines all other regions."""

    new_region_number = int(mask['region'].max()) + 1
    mask_combined = mask.max(dim='region')
    mask_combined = mask_combined.assign_coords(region=new_region_number).expand_dims('region')
    mask = xr.concat([mask, mask_combined], 'region')

    return mask


def select_shapefile_regions(ds, shapefile, agg=None, header=None, combine_shapes=False):
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
        if combine_shapes:
            mask = add_combined_shape(mask)
        weights = np.cos(np.deg2rad(ds['lat']))
        ds = ds.weighted(mask * weights).mean(dim=('lat', 'lon'), keep_attrs=True)

    if header:
        shape_names = shapes[header].to_list()
        shape_names.append('all')
        ds = ds.assign_coords(region=shape_names)

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
    
    ds = ds.assign_coords({'lon': (ds['lon'] + 360)  % 360})
        
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

