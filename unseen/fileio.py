"""Functions and command line program for file I/O"""

import os
import zipfile
import shutil
import argparse

import git
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import cmdline_provenance as cmdprov

from . import general_utils
from . import spatial_selection
from . import time_utils
from . import array_handling
from . import dask_setup
from . import indices


image_metadata_keys = {
    "png": "History",
    "pdf": "Title",
    "eps": "Creator",
    "ps": "Creator",
}


def guess_file_format(filenames):
    """Guess file format from file name."""

    if type(filenames) == list:
        filename = filenames[0]
    else:
        filename = filenames

    if ".nc" in filename:
        file_format = "netcdf4"
    elif ".zarr" in filename:
        file_format = "zarr"
    else:
        ValueError("File must contain .nc or .zarr")

    return file_format


def open_dataset(
    infiles,
    file_format=None,
    chunks="auto",
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
    """Create an xarray Dataset from one or more data files.

    Args:
      infiles (str or list) : Input file path/s
      file_format (str) : File format of input files (e.g. netcdf4, zarr)
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

    engine = file_format if file_format else guess_file_format(infiles)
    ds = xr.open_mfdataset(infiles, engine=engine, use_cftime=True)

    if not chunks == "auto":
        ds = ds.chunk(chunks)

    # Metadata
    if metadata_file:
        ds = fix_metadata(ds, metadata_file, variables)

    # Variable selection
    if variables:
        ds = ds[variables]

    # General selection/subsetting
    if isel:
        ds = ds.isel(isel)
    if sel:
        ds = ds.sel(sel)

    # Spatial subsetting and aggregation
    if spatial_coords or shapefile or spatial_agg:
        ds = spatial_selection.select_region(
            ds,
            coords=spatial_coords,
            shapefile=shapefile,
            header=shape_label_header,
            combine_shapes=combine_shapes,
            agg=spatial_agg,
        )

    # Temporal aggregation
    if no_leap_days:
        ds = ds.sel(time=~((ds["time"].dt.month == 2) & (ds["time"].dt.day == 29)))
    if time_freq:
        assert time_agg, "Provide a time_agg"
        assert variables, "Variables argument is required for temporal aggregation"
        if not input_freq:
            input_freq = xr.infer_freq(ds.indexes["time"][0:3])[0]
        ds = time_utils.temporal_aggregation(
            ds,
            time_freq,
            input_freq,
            time_agg,
            variables,
            reset_times=reset_times,
            complete=complete_time_agg_periods,
        )
    output_freq = time_freq[0] if time_freq else input_freq
    if output_freq:
        ds["time"].attrs["frequency"] = output_freq

    # Units
    for var, target_units in units.items():
        ds[var] = general_utils.convert_units(ds[var], target_units)

    assert type(ds) == xr.core.dataset.Dataset

    return ds


def times_from_init_lead(ds, time_freq):
    """Get time values from init dates and lead times"""

    step_units = {"D": "days", "M": "months", "Q": "months", "A": "years", "Y": "years"}

    step_unit = step_units[time_freq]
    scale_factor = 3 if time_freq == "Q" else 1

    init_dates_cftime = ds["init_date"]
    init_dates_str = time_utils.cftime_to_str(init_dates_cftime)
    init_dates_datetime = pd.to_datetime(init_dates_str)

    times_datetime = [
        init_dates_datetime + pd.offsets.DateOffset(**{step_unit: lead * scale_factor})
        for lead in ds["lead_time"].values
    ]
    times_cftime = time_utils.datetime_to_cftime(times_datetime)

    return times_cftime


def open_mfforecast(infiles, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    time_values = []
    for infile in infiles:
        ds = open_dataset(infile, **kwargs)
        time_attrs = ds["time"].attrs
        time_values.append(ds["time"].values)
        ds = array_handling.to_init_lead(ds)
        datasets.append(ds)
    ds = xr.concat(datasets, dim="init_date")
    time_values = np.stack(time_values, axis=-1)
    time_dimension = xr.DataArray(
        time_values,
        attrs=time_attrs,
        dims={"lead_time": ds["lead_time"], "init_date": ds["init_date"]},
    )
    ds = ds.assign_coords({"time": time_dimension})
    ds["lead_time"].attrs["units"] = time_attrs["frequency"]

    return ds


def fix_metadata(ds, metadata_file, variables):
    """Edit the attributes of an xarray Dataset.

    Args:
      ds (xarray Dataset or DataArray)
      metadata_file (str) : YAML file specifying required file metadata changes
      variables (list): Variables to rename (provide target name)
    """

    with open(metadata_file, "r") as reader:
        metadata_dict = yaml.load(reader, Loader=yaml.BaseLoader)

    valid_keys = ["rename", "drop_coords", "round_coords", "units"]
    for key in metadata_dict.keys():
        if key not in valid_keys:
            raise KeyError(f"Invalid metadata key: {key}")

    if "rename" in metadata_dict:
        for orig_var, target_var in metadata_dict["rename"].items():
            try:
                ds = ds.rename({orig_var: target_var})
            except ValueError:
                pass

    if "drop_coords" in metadata_dict:
        for drop_coord in metadata_dict["drop_coords"]:
            if drop_coord in ds.coords:
                ds = ds.drop(drop_coord)

    if "round_coords" in metadata_dict:
        for coord in metadata_dict["round_coords"]:
            ds = ds.assign_coords({coord: ds[coord].round(decimals=10)})

    if "units" in metadata_dict:
        for var, units in metadata_dict["units"].items():
            ds[var].attrs["units"] = units

    return ds


def get_new_log(infile_logs=None, repo_dir=None):
    """Generate command log for output file.

    Args:
      infile_logs (dict) : keys are file names,
        values are the command log
      repo_dir (str) : Path for git repository
    """

    try:
        repo = git.Repo(repo_dir)
        repo_url = repo.remotes[0].url.split(".git")[0]
    except (git.exc.InvalidGitRepositoryError, NameError):
        repo_url = None
    new_log = cmdprov.new_log(code_url=repo_url, infile_logs=infile_logs)

    return new_log


def zip_zarr(zarr_filename, zip_filename):
    """Zip a zarr collection"""

    with zipfile.ZipFile(
        zip_filename, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as fh:
        for root, _, filenames in os.walk(zarr_filename):
            for each_filename in filenames:
                each_filename = os.path.join(root, each_filename)
                fh.write(each_filename, os.path.relpath(each_filename, zarr_filename))


def to_zarr(ds, filename):
    """Write to zarr file"""

    for var in ds.variables:
        ds[var].encoding = {}

    if filename[-4:] == ".zip":
        zarr_filename = filename[:-4]
    else:
        zarr_filename = filename

    ds.to_zarr(zarr_filename, mode="w", consolidated=True)
    if filename[-4:] == ".zip":
        zip_zarr(zarr_filename, filename)
        shutil.rmtree(zarr_filename)


def _indices_setup(kwargs, variables):
    """Set variables and units for index calculation."""

    index = ""
    if "ffdi" in variables:
        kwargs["variables"] = ["pr", "hur", "tasmax", "uas", "vas"]
        kwargs["units"] = {
            "pr": "mm/day",
            "tasmax": "C",
            "uas": "km/h",
            "vas": "km/h",
            "hur": "%",
        }
        index = "ffdi"

    return kwargs, index


def _parse_command_line():
    """Parse the command line for input agruments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("infiles", type=str, nargs="*", help="Input files")
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument(
        "--forecast",
        action="store_true",
        default=False,
        help="Input files are a forecast dataset [default=False]",
    )

    parser.add_argument(
        "--dask_config", type=str, help="YAML file specifying dask client configuration"
    )
    parser.add_argument(
        "--input_chunks",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        default="auto",
        help="Chunks for reading data (e.g. time=-1)",
    )
    parser.add_argument(
        "--output_chunks",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        default={},
        help="Chunks for writing data to file (e.g. lead_time=50)",
    )

    parser.add_argument(
        "--metadata_file",
        type=str,
        help="YAML file specifying required file metadata changes",
    )

    parser.add_argument(
        "--no_leap_days",
        action="store_true",
        default=False,
        help="Remove leap days from time series [default=False]",
    )
    parser.add_argument(
        "--time_freq",
        type=str,
        choices=("A-DEC", "M", "Q-NOV", "A-NOV"),
        default=None,
        help="Target frequency for temporal aggregation",
    )
    parser.add_argument(
        "--time_agg",
        type=str,
        choices=("mean", "max", "min", "sum"),
        default=None,
        help="Temporal aggregation method",
    )
    parser.add_argument(
        "--reset_times",
        action="store_true",
        default=False,
        help="Shift time values after resampling so months match initial date [default=False]",
    )
    parser.add_argument(
        "--complete_time_agg_periods",
        action="store_true",
        default=False,
        help="Limit temporal aggregation output to complete years/months [default=False]",
    )
    parser.add_argument(
        "--input_freq",
        type=str,
        choices=("M", "D", "Q", "A"),
        default=None,
        help="Time frequency of input data",
    )

    parser.add_argument(
        "--spatial_coords",
        type=float,
        nargs="*",
        default=None,
        help="Point [lat, lon] or box [south bound, north bound, east bound, west bound] for spatial subsetting",
    )
    parser.add_argument(
        "--shapefile", type=str, default=None, help="Shapefile for region selection"
    )
    parser.add_argument(
        "--shp_header",
        type=str,
        default=None,
        help="Shapefile column header for region names",
    )
    parser.add_argument(
        "--combine_shapes",
        action="store_true",
        default=False,
        help="Add a region that combines all shapes [default=False]",
    )
    parser.add_argument(
        "--spatial_agg",
        type=str,
        choices=("mean", "sum"),
        default=None,
        help="Spatial aggregation method",
    )

    parser.add_argument(
        "--units",
        type=str,
        nargs="*",
        default={},
        action=general_utils.store_dict,
        help="Variable / new unit pairs (e.g. precip=mm/day)",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="*",
        help="Variables to select (or index to calculate)",
    )
    parser.add_argument(
        "--isel",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        help="Index selection along dimensions (e.g. ensemble=1:5)",
    )
    parser.add_argument(
        "--sel",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        help="Selection along dimensions (e.g. ensemble=1:5)",
    )

    args = parser.parse_args()

    return args


def _main():
    """Run the command line program."""

    args = _parse_command_line()

    if args.dask_config:
        client = dask_setup.launch_client(args.dask_config)
        print(client)

    kwargs = {
        "chunks": args.input_chunks,
        "metadata_file": args.metadata_file,
        "variables": args.variables,
        "spatial_coords": args.spatial_coords,
        "shapefile": args.shapefile,
        "shape_label_header": args.shp_header,
        "combine_shapes": args.combine_shapes,
        "spatial_agg": args.spatial_agg,
        "no_leap_days": args.no_leap_days,
        "time_freq": args.time_freq,
        "time_agg": args.time_agg,
        "reset_times": args.reset_times,
        "complete_time_agg_periods": args.complete_time_agg_periods,
        "input_freq": args.input_freq,
        "isel": args.isel,
        "sel": args.sel,
        "units": args.units,
    }

    kwargs, index = _indices_setup(kwargs, args.variables)

    if args.forecast:
        ds = open_mfforecast(args.infiles, **kwargs)
        temporal_dim = "lead_time"
    else:
        ds = open_dataset(args.infiles, **kwargs)
        temporal_dim = "time"

    if index == "ffdi":
        ds["ffdi"] = indices.calc_FFDI(
            ds, time_dim=temporal_dim, scale_dims=[temporal_dim]
        )

    if args.output_chunks:
        ds = ds.chunk(args.output_chunks)
    ds = ds[args.variables]

    ds.attrs["history"] = get_new_log()
    to_zarr(ds, args.outfile)


if __name__ == "__main__":
    _main()
