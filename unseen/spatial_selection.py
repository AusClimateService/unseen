"""Functions for spatial selection."""

import pdb

import xarray as xr
import geopandas as gp
import regionmask


regions = {'AUS-BOX': [-44, -11, 113, 154],
           'AUS-SHAPE': 'NRM_regions_2020.zip',
           'MEL-POINT': (-37.81, 144.96),
           'TAS-POINT': (-42, 146.5),
           }


def select_region(ds, region_name):
    """Select region.
    
    Args:
      ds (xarray Dataset or DataArray)
      region_name (str) : Region name
    """
    
    region = regions[region_name]

    if type(region) == str:
        ds = select_shapefile_region(ds, region)
    elif len(region) == 4:
        ds = select_box_region(ds, region)
    elif len(region) == 2:
        ds = select_point_region(ds, region)
    else:
        raise ValueError('region is not a box (4 values) or point (2 values)')
    
    return ds


def select_shapefile_region(ds, shapefile):
    """Select region using a shapefile"""

    lon = ds['lon'].values
    lat = ds['lat'].values

    regions_gp = gp.read_file(shapefile)
    regions_xr = regionmask.mask_geopandas(regions_gp, lon, lat)

    mask = xr.where(regions_xr.notnull(), True, False)
    ds = ds.where(mask)

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
        
    #if sort:
    #    da = da.sortby(lat_name).sortby(lon_name)
    #da.sel({'lat': slice(box[0], box[1]), 'lon': slice(box[2], box[3])})

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

