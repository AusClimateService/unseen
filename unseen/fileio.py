"""Functions for file I/O"""

import pdb
import os
import git
import yaml
import shutil
import zipfile
import pandas as pd
import xarray as xr
import cmdline_provenance as cmdprov

import general_utils
import spatial_selection
import time_utils
import array_handling


image_metadata_keys = {'png': 'History',
                       'pdf': 'Title',
                       'eps': 'Creator',
                       'ps' : 'Creator'} 


def open_file(infile,
              chunks='auto',
              metadata_file=None,
              variables=[],
              spatial_coords=None,
              shapefile=None,
              shape_label_header=None,
              combine_shapes=False,
              spatial_agg=None,
              no_leap_days=False,
              time_freq=None,
              time_agg=None,
              reset_times=False,
              complete_time_agg_periods=False,
              input_freq=None,
              isel={},
              sel={},
              units={},
              ):
    """Create an xarray Dataset from an input zarr file.

    Args:
      infile (str) : Input file path
      chunks (dict) : Chunks for xarray.open_zarr 
      metadata_file (str) : YAML file specifying required file metadata changes
      variables (list) : Variables of interest
      spatial_coords (list) : List of length 2 [lat, lon], 4 [south bound, north bound, east bound, west bound].
      shapefile (str) : Shapefile for spatial subseting
      shape_label_header (str) : Name of the shapefile column containing the region names
      combine_shapes (bool) : Add a region that combines all shapes
      spatial_agg (str) : Spatial aggregation method ('mean' or 'sum') 
      no_leap_days (bool) : Remove leap days from data
      time_freq (str) : Target temporal frequency for resampling
      time_agg (str) : Temporal aggregation method ('mean', 'sum', 'min' or 'max')
      reset_times (bool) : Shift time values after resampling so months match initial date
      complete_time_agg_periods (bool) : Limit temporal aggregation output to complete years/months
      input_freq (str) : Input time frequency for resampling (estimated if not provided) 
      isel (dict) : Selection using xarray.Dataset.isel
      sel (dict) : Selection using xarray.Dataset.sel
      units (dict) : Variable/s (keys) and desired units (values)
    """

    if infile[-3:] == '.nc':
        ds = xr.open_dataset(infile, use_cftime=True)
    elif 'zarr' in infile[-9:]:
        ds = xr.open_zarr(infile, consolidated=True, use_cftime=True)  #, chunks=chunks)
    else:
        ValueError(f'File must end in .nc, .zarr or .zarr.zip')

    if not chunks == 'auto':
        ds = ds.chunk(chunks)

    # Metadata
    if metadata_file:
        ds = fix_metadata(ds, metadata_file, variables)

    # Variable selection
    if variables:
        ds = ds[variables]

    # Units
    for var, target_units in units.items():
        ds[var] = general_utils.convert_units(ds[var], target_units)
        
    # Spatial subsetting and aggregation
    if spatial_coords or shapefile or spatial_agg:
        ds = spatial_selection.select_region(ds,
                                             coords=spatial_coords,
                                             shapefile=shapefile,
                                             header=shape_label_header,
                                             combine_shapes=combine_shapes,
                                             agg=spatial_agg)

    # Temporal aggregation
    if no_leap_days:
        ds = ds.sel(time=~((ds['time'].dt.month == 2) & (ds['time'].dt.day == 29)))
    if time_freq:
        assert time_agg, """Provide a time_agg"""
        if not input_freq:
            input_freq = xr.infer_freq(ds.indexes['time'][0:3])[0] 
        ds = time_utils.temporal_aggregation(ds, time_freq, input_freq, time_agg,
                                             variables, reset_times=reset_times)
        if complete_time_agg_periods:
            ds = time_utils.select_complete_time_periods(ds, time_freq)

    output_freq = time_freq[0] if time_freq else input_freq
    if output_freq:
        ds['time'].attrs['frequency'] = output_freq    

    # General selection/subsetting
    if isel:
        ds = ds.isel(isel)
    if sel:
        ds = ds.sel(sel)

    assert type(ds) == xr.core.dataset.Dataset

    return ds


def times_from_init_lead(ds, time_freq):
    """Get time values from init dates and lead times"""

    step_units = {'D': 'days',
                  'M': 'months',
                  'Q': 'months',
                  'A': 'years',
                  'Y': 'years'}
    
    step_unit = step_units[time_freq]
    scale_factor = 3 if time_freq == 'Q' else 1

    time_values = [ds.get_index('init_date') + pd.offsets.DateOffset(**{step_unit: lead * scale_factor}) for lead in ds['lead_time']]

    return time_values


def open_mfzarr(infiles, **kwargs):
    """Open multiple zarr files"""

    datasets = []
    for infile in infiles:
        ds = open_file(infile, **kwargs)
        datasets.append(ds) 
    ds = xr.concat(datasets, dim='time')

    return ds


def open_mfforecast(infiles, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    for infile in infiles:
        ds = open_file(infile, **kwargs)
        time_attrs = ds['time'].attrs
        ds = array_handling.to_init_lead(ds)
        datasets.append(ds)
    ds = xr.concat(datasets, dim='init_date')

    time_values = times_from_init_lead(ds, time_attrs['frequency'])
    time_dimension = xr.DataArray(time_values, attrs=time_attrs,
                                  dims={'lead_time': ds['lead_time'],
                                        'init_date': ds['init_date']})
    ds = ds.assign_coords({'time': time_dimension})
    ds['lead_time'].attrs['units'] = time_attrs['frequency']

    return ds


def fix_metadata(ds, metadata_file, variables):
    """Edit the attributes of an xarray Dataset.
    
    Args:
      ds (xarray Dataset or DataArray)
      metadata_file (str) : YAML file specifying required file metadata changes
      variables (list): Variables to rename (provide target name)
    """
 
    with open(metadata_file, 'r') as reader:
        metadata_dict = yaml.load(reader, Loader=yaml.BaseLoader)

    valid_keys = ['rename', 'drop_coords', 'round_coords', 'units']
    for key in metadata_dict.keys():
        if not key in valid_keys:
            raise KeyError(f'Invalid metadata key: {key}')

    if 'rename' in metadata_dict:
        for orig_var, target_var in metadata_dict['rename'].items():
            try:
                ds = ds.rename({orig_var: target_var})
            except ValueError:
                pass

    if 'drop_coords' in metadata_dict:
        for drop_coord in metadata_dict['drop_coords']:
            if drop_coord in ds.coords:
                ds = ds.drop(drop_coord)

    if 'round_coords' in metadata_dict:
        for coord in metadata_dict['round_coords']:
            ds = ds.assign_coords({coord: ds[coord].round(decimals=10)})

    if 'units' in metadata_dict:
        for var, units in metadata_dict['units'].items():
            ds[var].attrs['units'] = units

    return ds


def get_new_log(infile_logs=None, repo_dir=None):
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

