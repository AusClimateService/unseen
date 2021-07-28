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


def open_file(infile,
              chunks='auto',
              metadata_file=None,
              variables=[],
              region=None,
              shape_label_header=None,
              spatial_agg=None,
              no_leap_days=False,
              time_freq=None,
              time_agg=None,
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
      region (str or list) : Spatial subset. Can be:
                             shapefile name, or
                             list length 2 (point selection), or
                             list length 4 (box selection).
      shape_label_header (str) : Name of the shapefile column containing the region names 
      no_leap_days (bool) : Remove leap days from data
      time_freq (str) : Target temporal frequency for resampling
      time_agg (str) : Temporal aggregation method ('mean' or 'sum')
      complete_time_agg_periods (bool) : Limit temporal aggregation output to complete years/months
      input_freq (str) : Input time frequency for resampling (estimated if not provided) 
      isel (dict) : Selection using xarray.Dataset.isel
      sel (dict) : Selection using xarray.Dataset.sel
      units (dict) : Variable/s (keys) and desired units (values)
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

    # Units
    for var, target_units in units.items():
        ds[var] = general_utils.convert_units(ds[var], target_units)
        
    # Spatial subsetting and aggregation
    if region or spatial_agg:
        ds = spatial_selection.select_region(ds, region=region, agg=spatial_agg,
                                             header=shape_label_header)

    # Temporal aggregation
    if not input_freq:
        input_freq = xr.infer_freq(ds.indexes['time'][0:3])

    if no_leap_days:
        ds = ds.sel(time=~((ds['time'].dt.month == 2) & (ds['time'].dt.day == 29)))
    if time_freq:
        assert time_agg, """Provide a time_agg ('mean' or 'sum')"""
        ds = time_utils.temporal_aggregation(ds, time_freq, time_agg, variables, input_freq=input_freq)
        if complete_time_agg_periods:
            ds = time_utils.select_complete_time_periods(ds, time_freq)

    output_freq = time_freq[0] if time_freq else input_freq
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


def open_mfforecast(infiles, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    for infile in infiles:
        ds = open_file(infile, **kwargs)
        time_freq = ds['time'].attrs['frequency']
        ds = array_handling.to_init_lead(ds)
        datasets.append(ds)
    ds = xr.concat(datasets, dim='init_date')

    time_values = times_from_init_lead(ds, time_freq)
    time_dimension = xr.DataArray(time_values,
                                  dims={'lead_time': ds['lead_time'],
                                        'init_date': ds['init_date']})
    ds = ds.assign_coords({'time': time_dimension})
    ds['lead_time'].attrs['units'] = time_freq

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
        for orig_var, target_var in metadata_dict['rename'].items():
            try:
                ds = ds.rename({orig_var: target_var})
            except ValueError:
                pass

    if 'drop_coords' in metadata_dict:
        for drop_coord in metadata_dict['drop_coords']:
            if drop_coord in ds.coords:
                ds = ds.drop(drop_coord)

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

