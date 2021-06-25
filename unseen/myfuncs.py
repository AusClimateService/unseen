import sys
repo_dir = sys.path[0]
import os
import re
import argparse
import pdb

import git
import yaml
import shutil
import zipfile
import cftime
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import dask
import geopandas as gp
import regionmask
import cmdline_provenance as cmdprov
import xclim


## Miscellanous utilities


class store_dict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, val = value.split('=')
            if ':' in val:
                start, end = val.split(':')
                try:
                    start = int(start)
                    end = int(end)
                except ValueError:
                    pass
                val = slice(start, end)
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = val


def convert_units(da, target_units):
    """Convert units.
    
    Args:
      da (xarray DataArray)
      target_units (str)
    """

    xclim_unit_check = {'deg_k': 'degK',
                       }
    if da.units in xclim_unit_check:
        da.attrs['units'] = xclim_unit_check[da.units]

    da = xclim.units.convert_units_to(da, target_units)

    return da


## Time utilities


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


def str_to_cftime(datestring, calendar):
    """Convert a date string to cftime object"""
    
    dt = datetime.strptime(datestring, '%Y-%m-%d')
    cfdt = cftime.datetime(dt.year, dt.month, dt.day, calendar=calendar)
     
    return cfdt


def cftime_to_str(time_dim):
    """Convert cftime array to YYY-MM-DD strings."""

    check_cftime(time_dim)
    str_times = [time.strftime('%Y-%m-%d') for time in time_dim.values]

    return str_times


def temporal_aggregation(ds, target_freq, agg_method):
    """Temporal aggregation of data.

    resample frequencies:

      A-DEC (annual, with date label being last day of year) 
      M (monthly, with date label being last day of month)
      Q-NOV (DJF, MAM, JJA, SON, with date label being last day of season)
      A-NOV (annual Dec-Nov, date label being last day of the year)
    """

    assert target_freq in ['A-DEC', 'M', 'Q-NOV', 'A-NOV']

    input_freq = xr.infer_freq(ds.indexes['time'][0:3])
    if input_freq == target_freq:
        pass
    elif input_freq == 'D':
        ds = ds.resample(time=target_freq)
    elif input_freq == 'M':
        # TODO: monthly downsampling accounting for number of days
        raise ValueError('Monhtly downsampling not implemented yet')
    else:
        raise ValueError(f'Unsupported input time frequency: {input_freq}')    

    return ds


## File I/O


def open_file(infile,
              metadata_file=None,
              no_leap_days=False,
              region=None,
              time_freq=None,
              units={},
              variables=[],
              isel={},
              sel={},
              chunks='auto'):
    """Create an xarray Dataset from an input zarr file.

    Args:
      infile (str) : Input file path
      metadata_file (str) : YAML file specifying required file metadata changes
      no_leap_days (bool) : Remove leap days from data
      region (str) : Spatial subset (extract this region)
      units (dict) : Variable/s (keys) and desired units (values)
      variables (list) : Variables of interest
      isel (dict) : Selection using xarray.Dataset.isel
      sel (dict) : Selection using xarray.Dataset.sel
      chunks (dict) : Chunks for xarray.open_zarr 
    """

    ds = xr.open_zarr(infile, consolidated=True, use_cftime=True, chunks=chunks)

    #if chunks:
    #    ds = ds.chunk(input_chunks)
    
    # Metadata
    if metadata_file:
        ds = fix_metadata(ds, metadata_file, variables)

    # Variable selection
    if variables:
        ds = ds[variables]

    # Spatial subsetting and aggregation
    if region:
        ds = select_region(ds, regions[region])

    # Temporal aggregation
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    if no_leap_days:
        ds = ds.sel(time=~((ds['time'].dt.month == 2) & (ds['time'].dt.day == 29)))
    if time_freq:
        ds = temporal_aggregation(ds, time_freq)

    # General selection/subsetting
    if isel:
        ds = ds.isel(isel)
    if sel:
        ds = ds.sel(sel)

    # Units
    for var, target_units in units.items():
        ds[var] = convert_units(ds[var], target_units)

    assert type(ds) == xr.core.dataset.Dataset

    return ds


def open_mfforecast(infiles, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    for infile in infiles:
        ds = open_file(infile, **kwargs)
        ds = to_init_lead(ds)
        datasets.append(ds)
    ds = xr.concat(datasets, dim='init_date')

    time_values = [ds.get_index('init_date').shift(int(lead), 'D') for lead in ds['lead_time']]
    time_dimension = xr.DataArray(time_values,
                                  dims={'lead_time': ds['lead_time'],
                                        'init_date': ds['init_date']})
    ds = ds.assign_coords({'time': time_dimension})
    ds['lead_time'].attrs['units'] = 'D'

    return ds


def fix_metadata(ds, metadata_file, variables):
    """Edit the attributes of an xarray Dataset.
    
    ds (xarray Dataset or DataArray)
    metadata_file (str) : YAML file specifying required file metadata changes
    variables (list): Variables to rename (provide target name)
    """
 
    with open(metadata_file, 'r') as reader:
        metadata_dict = yaml.load(reader, Loader=yaml.BaseLoader)

    valid_keys = ['rename', 'drop_coords', 'units']
    for key in metadata_dict.keys():
        if not key in valid_keys:
            raise KeyError(f'Invalid metadata key: {key}')

    if 'rename' in metadata_dict:
        valid_vars = variables + ['time']
        for orig_var, target_var in metadata_dict['rename'].items():
            if target_var in valid_vars:
                ds = ds.rename({orig_var: target_var})

    if 'drop_coords' in metadata_dict:
        for drop_coord in metadata_dict['drop_coords']:
            if drop_coord in ds.coords:
                ds = ds.drop(drop_coord)

    if 'units' in metadata_dict:
        for var, units in metadata_dict['units'].items():
            ds[var].attrs['units'] = units

    return ds


def get_new_log(infile_logs=None):
    """Generate command log for output file.

    Args:
      infile_logs (dict) : keys are file names,
        values are the command log
    """

    try:
        repo = git.Repo(repo_dir)
        repo_url = repo.remotes[0].url.split('.git')[0]
    except git.exc.InvalidGitRepositoryError:
        repo_url = None
    new_log = cmdprov.new_log(code_url=repo_url,
                              infile_logs=infile_logs)

    return new_log


def zip_zarr(zarr_filename, zip_filename):
    """Zip a zarr collection"""
    
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as fh:
        for root, _, filenames in os.walk(zarr_filename):
            for each_filename in filenames:
                each_filename = os.path.join(root, each_filename)
                fh.write(each_filename, os.path.relpath(each_filename, zarr_filename))


def to_zarr(ds, filename):
    """Write to zarr file"""
                
    for var in ds.variables:
        ds[var].encoding = {}

    if filename[-4:] == '.zip':
        zarr_filename = filename[:-4]
    else:
        zarr_filename = filename

    ds.to_zarr(zarr_filename, mode='w', consolidated=True)
    if filename[-4:] == '.zip':
        zip_zarr(zarr_filename, filename)
        shutil.rmtree(zarr_filename)


## Array handling


def stack_by_init_date(ds, init_dates, n_lead_steps, freq='D'):
    """Stack timeseries array in inital date / lead time format.

    Args:
      ds (xarray Dataset)
      init_dates (list) : Initial dates in YYYY-MM-DD format
      n_lead_steps (int) : Maximum lead time
      freq (str) : Time-step frequency
    """

    check_date_format(init_dates)
    check_cftime(ds['time'])

    rounded_times = ds['time'].dt.floor(freq).values
    ref_time = init_dates[0]
    ref_calendar = rounded_times[0].calendar
    ref_var = list(ds.keys())[0]
    ref_array = ds[ref_var].sel(time=ref_time).values    

    time2d = np.empty((len(init_dates), n_lead_steps), 'object')
    init_date_indexes = []
    offset = n_lead_steps - 1
    for ndate, date in enumerate(init_dates):
        date_cf = str_to_cftime(date, ref_calendar)
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
    np.testing.assert_allclose(actual_array, ref_array)
    
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
    init_date = np.datetime64(ds['time'].values[0].strftime('%Y-%m-%d'))
    new_coords = {'lead_time': lead_time, 'init_date': init_date}
    ds = ds.rename({'time': 'lead_time'})
    ds = ds.assign_coords(new_coords)

    return ds


## Region selection


regions = {'AUS-BOX': [-44, -11, 113, 154],
           'AUS-SHAPE': 'NRM_regions_2020.zip',
           'MEL-POINT': (-37.81, 144.96),
           'TAS-POINT': (-42, 146.5),
           }


def select_region(ds, region):
    """Select region.
    
    Args:
      ds (xarray Dataset or DataArray)
      region (str) : Region name
    """
    
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

