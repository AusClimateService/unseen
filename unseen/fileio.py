"""Functions and command line program for file I/O"""

import os
import zipfile
import shutil
import argparse
import logging

import git
import yaml
import numpy as np
import pandas as pd
import geopandas as gp
import xarray as xr
import cmdline_provenance as cmdprov

from . import general_utils
from . import spatial_selection
from . import time_utils
from . import array_handling
from . import dask_setup
from . import indices


def open_dataset(
    infiles,
    file_format=None,
    chunks=None,
    metadata_file=None,
    variables=[],
    point_selection=None,
    lat_bnds=None,
    lon_bnds=None,
    shapefile=None,
    shapefile_label_header=None,
    shape_overlap=None,
    combine_shapes=False,
    spatial_agg="none",
    lat_dim="lat",
    lon_dim="lon",
    agg_y_dim=None,
    agg_x_dim=None,
    standard_calendar=False,
    no_leap_days=False,
    rolling_sum_window=None,
    time_freq=None,
    time_agg=None,
    time_agg_dates=False,
    months=None,
    season=None,
    reset_times=False,
    time_agg_min_tsteps=None,
    input_freq=None,
    time_dim="time",
    isel={},
    sel={},
    scale_factors={},
    units={},
    units_timing="end",
):
    """Create an xarray Dataset from one or more data files.

    Parameters
    ----------
    infiles : str or list
        Input file path/s
    file_format : str, optional
        Formats/engines accepted by xarray.open_dataset (e.g. netcdf4, zarr, cfgrid).
        Estimated if not provided.
    chunks : dict, optional
        Chunks for xarray.open_mfdataset
    metadata_file : str
        YAML file path specifying required file metadata changes
    variables : list, optional
        Subset of variables of interest
    point_selection : list, optional
        Point coordinates: [lat, lon]
    lat_bnds : list, optional
        Latitude bounds: [south bound, north bound]
    lon_bnds : list, optional
        Longitude bounds: [west bound, east bound]
    shapefile : str, optional
        Shapefile for spatial subseting
    shapefile_label_header : str
        Name of the shapefile column containing the region names
    shape_overlap : float, optional
        Fraction that a grid cell must overlap with a shape to be included.
        If no fraction is provided, grid cells are selected if their centre
        point falls within the shape.
    combine_shapes : bool, default False
        Add a region that combines all shapes
    spatial_agg : {'mean', 'sum', 'weighted_mean'}, optional
        Spatial aggregation method
    lat_dim: str, default 'lat'
        Name of the latitude dimension in infiles
    lon_dim: str, default 'lon'
        Name of the longitude dimension in infiles
    agg_y_dim : str, optional
        Name of Y dimension for spatial aggregation (default = lat_dim)
    agg_x_dim : str, optional
        Name of X dimension for spatial aggregation (default = lon_dim)
    no_leap_days : bool, default False
        Remove leap days from data
    rolling_sum_window : int, default None
        Apply a rolling sum with this window width
    time_freq : str, optional
        Target temporal frequency for resampling
        Options: https://pandas.pydata.org/docs/user_guide/timeseries.html#anchored-offsets
    time_agg : {'mean', 'sum', 'min', 'max'}, optional
        Temporal aggregation method
    time_agg_dates : bool, default False
        Record the date of each time aggregated event (e.g. annual max)
    standard_calendar : bool, default False
        Force a common calendar on all input files
    months : list, optional
        Select months from the dataset
    season : {'DJF', 'MAM', 'JJA', 'SON'}, optional
        Select a single season after Q-NOV temporal resampling
    reset_times : bool, default False
        Shift time values after resampling so months match initial date
    time_agg_min_tsteps : int, optional
        Minimum number of timesteps for temporal aggregation
    input_freq : {'Y', 'Q', 'M', 'D'}, optional
        Input time frequency for resampling (estimated if not provided)
    time_dim: str, default 'time'
        Name of the time dimension in infiles
    isel : dict, optional
        Selection using xarray.Dataset.isel
    sel : dict, optional
        Selection using xarray.Dataset.sel
    scale_factors : dict, optional
        Divide input data by this value.
        Variable/s (keys) and scale factor (values).
        Scale factors can be a float or "days_in_month"
    units : dict, optional
        Variable/s (keys) and desired units (values)
    units_timing : str, {'start', 'middle', 'end'}, default 'end'
        When to perform the unit conversions in units
        Middle is after the spatial aggregation but before the temporal aggregation

    Returns
    -------
    ds : xarray Dataset
    """

    preprocess = time_utils.switch_calendar if standard_calendar else None
    engine = file_format if file_format else _guess_file_format(infiles)
    ds = xr.open_mfdataset(
        infiles, engine=engine, preprocess=preprocess, use_cftime=True, chunks=chunks
    )

    # Metadata
    if metadata_file:
        ds = _fix_metadata(ds, metadata_file)

    # Variable selection
    if variables:
        if not isinstance(variables, list):
            variables = list(variables)
        ds = ds[variables]

    # General selection/subsetting
    if isel:
        ds = ds.isel(isel)
    if sel:
        ds = ds.sel(sel)
    if months:
        ds = time_utils.select_months(
            ds, months, init_month=reset_times, time_dim=time_dim
        )

    # Scale factors
    if scale_factors:
        with xr.set_options(keep_attrs=True):
            for var, scale_factor in scale_factors.items():
                if scale_factor == "days_in_month":
                    ds[var] = ds[var] / ds[time_dim].dt.days_in_month
                else:
                    ds[var] = ds[var] / scale_factor

    # Unit conversion (at start)
    if units and (units_timing == "start"):
        for var, target_units in units.items():
            ds[var] = general_utils.convert_units(ds[var], target_units)

    # Spatial subsetting and aggregation
    if point_selection:
        ds = spatial_selection.select_point(
            ds, point_selection, lat_dim=lat_dim, lon_dim=lon_dim
        )
    if lat_bnds:
        ds = spatial_selection.subset_lat(ds, lat_bnds, lat_dim=lat_dim)
    if lon_bnds:
        ds = spatial_selection.subset_lon(ds, lon_bnds, lon_dim=lon_dim)
    if shapefile:
        shapes = gp.read_file(shapefile)
        ds = spatial_selection.select_shapefile_regions(
            ds,
            shapes,
            agg=spatial_agg,
            overlap_fraction=shape_overlap,
            header=shapefile_label_header,
            combine_shapes=combine_shapes,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )
    elif spatial_agg != "none":
        agg_y_dim = agg_y_dim if agg_y_dim else lat_dim
        agg_x_dim = agg_x_dim if agg_x_dim else lon_dim
        ds = spatial_selection.aggregate(
            ds, spatial_agg, lat_dim=agg_y_dim, lon_dim=agg_x_dim
        )

    # Unit conversion (at middle)
    if units and (units_timing == "middle"):
        for var, target_units in units.items():
            ds[var] = general_utils.convert_units(ds[var], target_units)

    # Temporal aggregation
    if no_leap_days:
        ds = ds.sel(time=~((ds[time_dim].dt.month == 2) & (ds[time_dim].dt.day == 29)))
    if rolling_sum_window:
        ds = ds.rolling({time_dim: rolling_sum_window}).sum()
    if time_freq:
        assert time_agg, "Provide a time_agg"
        assert variables, "Variables argument is required for temporal aggregation"
        if not input_freq:
            input_freq = xr.infer_freq(ds.indexes[time_dim][0:3])[0]
        ds = time_utils.temporal_aggregation(
            ds,
            time_freq,
            input_freq,
            time_agg,
            variables,
            season=season,
            reset_times=reset_times,
            min_tsteps=time_agg_min_tsteps,
            agg_dates=time_agg_dates,
        )

    output_freq = time_freq[0] if time_freq else input_freq
    if output_freq:
        ds[time_dim].attrs["frequency"] = output_freq

    # Unit conversion (at end)
    if units and (units_timing == "end"):
        for var, target_units in units.items():
            ds[var] = general_utils.convert_units(ds[var], target_units)

    assert isinstance(ds, xr.core.dataset.Dataset)
    ds = ds.squeeze(drop=True)

    return ds


def _chunks(lst, n):
    """Split a list into n sub-lists"""

    new_lst = [lst[i : i + n] for i in range(0, len(lst), n)]

    return new_lst


def _process_mfilelist(file_list, n_time_files, n_ensemble_files):
    """Read and chunk an input file list"""

    if isinstance(file_list, str) or (len(file_list) == 1):
        if len(file_list) == 1:
            file_list = file_list[0]
        with open(file_list) as f:
            input_files = f.read().splitlines()
    else:
        input_files = file_list
    input_files_chunked = _chunks(_chunks(input_files, n_time_files), n_ensemble_files)

    return input_files_chunked


def open_mfforecast(
    file_list,
    n_time_files=1,
    n_ensemble_files=1,
    verbose=False,
    time_dim="time",
    ensemble_dim="ensemble",
    init_dim="init_date",
    lead_dim="lead_time",
    **kwargs,
):
    """Open multi-file forecast.

    Parameters
    ----------
    file_list: str or list
        List (or name of text file) containing input file paths (one file per line if text file).
        The list should be ordered by initialisation date (i), ensemble member (e) then time chunk (t).
          e.g. 'i1e1t1', 'i1e1t2', 'i1e1t3', 'i1e2t1', 'i1e2t2', 'i1e2t3', 'i2e1f1', ...
    n_time_files: int, default 1
        Number of consecutive files that span the time period (for a given initialisation date).
    n_ensemble_files: int, default 1
        Number of consecutive files (or n_time_file groupings) that form a complete ensemble.
    verbose: bool, default False
        Print file names as they are being processed
    time_dim: str, default 'time'
        Name of the time dimension in the input files
    ensemble_dim: str, default 'ensemble'
        Name of the ensemble dimension
        (May or may not be in the infiles already.)
    init_dim: str, default 'init_date'
        Name of the initial date dimension for output ds
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension for output ds
    **kwargs : dict, optional
        Extra arguments to `open_dataset`

    Returns
    -------
    ds : xarray Dataset
    """
    log_lev = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_lev)

    infiles = _process_mfilelist(file_list, n_time_files, n_ensemble_files)
    init_datasets = []
    time_values = []
    for init_file_group in infiles:
        init_ds_group = []
        for init_ensemble_file_group in init_file_group:
            logging.info(f"Processing file group: {init_ensemble_file_group}...")
            init_ensemble_ds = open_dataset(init_ensemble_file_group, **kwargs)
            shape = init_ensemble_ds[kwargs["variables"][0]].shape
            logging.info(f"Ensemble member shape: {shape}")
            init_ds_group.append(init_ensemble_ds)
        if len(init_ds_group) == 1:
            init_ds = init_ds_group[0]
            assert ensemble_dim in init_ds.dims
        else:
            n_ensemble_members = len(init_ds_group)
            init_ds = xr.concat(
                init_ds_group,
                pd.Index(np.arange(n_ensemble_members), name=ensemble_dim),
            )
        shape = init_ds[kwargs["variables"][0]].shape
        logging.info(f"Ensemble shape: {shape}")
        time_attrs = init_ds[time_dim].attrs
        time_values.append(init_ds[time_dim].values)
        init_ds = array_handling.to_init_lead(init_ds)
        init_datasets.append(init_ds)
    ds = xr.concat(init_datasets, dim=init_dim)
    time_values = np.stack(time_values, axis=-1)
    time_dimension = xr.DataArray(
        time_values,
        attrs=time_attrs,
        dims={lead_dim: init_ds[lead_dim], init_dim: init_ds[init_dim]},
    )
    ds = ds.assign_coords({time_dim: time_dimension})
    try:
        ds[lead_dim].attrs["units"] = time_attrs["frequency"]
    except KeyError:
        pass

    return ds


def get_new_log(infile_logs=None, repo_dir=None):
    """Generate command log for output file.

    Parameters
    ----------
    infile_logs : dict, optional
        keys are file names, values are the command log
    repo_dir : str, optional
        Path for git repository

    Returns
    -------
    new_log : str
        New command log
    """

    try:
        repo = git.Repo(repo_dir)
        repo_url = repo.remotes[0].url.split(".git")[0]
    except (git.exc.InvalidGitRepositoryError, NameError):
        repo_url = None
    new_log = cmdprov.new_log(code_url=repo_url, infile_logs=infile_logs)

    return new_log


def to_zarr(ds, file_name):
    """Write to zarr file.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Input dataset
    file_name : str
        Output file path
    """

    for var in ds.variables:
        ds[var].encoding = {}

    if file_name[-4:] == ".zip":
        zarr_filename = file_name[:-4]
    else:
        zarr_filename = file_name

    ds.to_zarr(zarr_filename, mode="w", consolidated=True)
    if file_name[-4:] == ".zip":
        zip_zarr(zarr_filename, file_name)
        shutil.rmtree(zarr_filename)


def zip_zarr(zarr_filename, zip_filename):
    """Zip a zarr collection.

    Parameters
    ----------
    zarr_filename : str
        Path to (unzipped) zarr collection
    zip_filename : str
        Path to output zipped zarr collection
    """

    with zipfile.ZipFile(
        zip_filename, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as fh:
        for root, _, filenames in os.walk(zarr_filename):
            for each_filename in filenames:
                each_filename = os.path.join(root, each_filename)
                fh.write(each_filename, os.path.relpath(each_filename, zarr_filename))


def _fix_metadata(ds, metadata_file):
    """Edit the attributes of an xarray Dataset.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Input dataset
    metadata_file : str
        YAML file specifying required file metadata changes

    Returns
    -------
    ds : xarray DataArray or Dataset
    """

    with open(metadata_file, "r") as reader:
        metadata_dict = yaml.load(reader, Loader=yaml.BaseLoader)

    valid_keys = ["rename", "drop_coords", "round_coords", "units"]
    for key in metadata_dict.keys():
        if key not in valid_keys:
            raise KeyError(f"Invalid metadata key: {key}")

    if "rename" in metadata_dict:
        for orig_var, target_var in metadata_dict["rename"].items():
            if orig_var in ds:
                ds = ds.rename({orig_var: target_var})
                ds[target_var].attrs["original_name"] = orig_var

    if "drop_coords" in metadata_dict:
        for drop_coord in metadata_dict["drop_coords"]:
            if drop_coord in ds.coords:
                ds = ds.drop(drop_coord)

    if "round_coords" in metadata_dict:
        for coord in metadata_dict["round_coords"]:
            ds = ds.assign_coords({coord: ds[coord].round(decimals=6)})

    if "units" in metadata_dict:
        for var, units in metadata_dict["units"].items():
            if var in ds.data_vars:
                ds[var].attrs["units"] = units

    return ds


def _guess_file_format(file_names):
    """Guess file format from file name.

    Parameters
    ----------
    file_names : str or list
        File name/s

    Returns
    -------
    file_format : {'netcdf4', 'zarr'}

    Raises
    ------
    ValueError
        If file name doesn't contain .nc or zarr

    """

    if isinstance(file_names, list):
        file_name = file_names[0]
    else:
        file_name = file_names

    if ".nc" in file_name:
        file_format = "netcdf4"
    elif ".zarr" in file_name:
        file_format = "zarr"
    else:
        ValueError("File must contain .nc or .zarr")

    return file_format


def _indices_setup(kwargs, variables):
    """Set variables and units for index calculation.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments (for passing to `open_dataset`)
    variables : list
        Variables list (for passing to `open_dataset`)

    Returns
    -------
    kwargs : dict
        Keyword arguments (for passing to `open_dataset`)
    index : str
        Name of the index to be calculated
    """

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
    """Parse the command line for input arguments"""

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
        "--n_ensemble_files",
        type=int,
        default=1,
        help="Number of consecutive files (or n_time_file groupings) that form a complete ensemble [default=1]",
    )
    parser.add_argument(
        "--n_time_files",
        type=int,
        default=1,
        help="Number of consecutive files that span the time period (for a given initialisation date) [default=1]",
    )
    parser.add_argument(
        "--dask_config", type=str, help="YAML file specifying dask client configuration"
    )
    parser.add_argument(
        "--input_chunks",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        default=None,
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
        "--rolling_sum_window",
        type=int,
        default=None,
        help="Apply a rolling sum with this window width",
    )
    parser.add_argument(
        "--time_freq",
        type=str,
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
        "--time_agg_dates",
        action="store_true",
        default=False,
        help="Record the date of each time aggregated event (e.g. annual max) [default=False]",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="Select months from the dataset",
    )
    parser.add_argument(
        "--season",
        type=str,
        choices=("DJF", "MAM", "JJA", "SON"),
        default=None,
        help="Select a single season after Q-NOV temporal resampling",
    )
    parser.add_argument(
        "--reset_times",
        action="store_true",
        default=False,
        help="Shift time values after resampling so months match initial date [default=False]",
    )
    parser.add_argument(
        "--time_agg_min_tsteps",
        type=int,
        default=None,
        help="Minimum number of time steps for temporal aggregation [default=None]",
    )
    parser.add_argument(
        "--input_freq",
        type=str,
        choices=("M", "D", "Q", "Y"),
        default=None,
        help="Time frequency of input data",
    )
    parser.add_argument(
        "--time_dim",
        type=str,
        default="time",
        help="Name of time dimension",
    )
    parser.add_argument(
        "--anomaly",
        type=str,
        nargs=2,
        default=None,
        help="Calculate anomaly with this base period: (base_start_date, base_end_date)",
    )
    parser.add_argument(
        "--anomaly_freq",
        type=str,
        default=None,
        choices=["month"],
        help="Anomaly can monthly (month) or all times (none)",
    )
    parser.add_argument(
        "--point_selection",
        type=float,
        nargs=2,
        default=None,
        help="Point coordinates: [lat, lon]",
    )
    parser.add_argument(
        "--lat_bnds",
        type=float,
        nargs=2,
        default=None,
        help="Latitude bounds: (south_bound, north_bound)",
    )
    parser.add_argument(
        "--lon_bnds",
        type=float,
        nargs=2,
        default=None,
        help="Longitude bounds: (west_bound, east_bound)",
    )
    parser.add_argument(
        "--shapefile", type=str, default=None, help="Shapefile for region selection"
    )
    parser.add_argument(
        "--shp_overlap",
        type=float,
        default=None,
        help="Fraction that a grid cell must overlap with a shape to be included",
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
        choices=("mean", "sum", "weighted_mean"),
        default="none",
        help="Spatial aggregation method [default is none]",
    )
    parser.add_argument(
        "--lat_dim",
        type=str,
        default="lat",
        help="Name of latitude dimension",
    )
    parser.add_argument(
        "--lon_dim",
        type=str,
        default="lon",
        help="Name of longitude dimension",
    )
    parser.add_argument(
        "--agg_y_dim",
        type=str,
        default=None,
        help="Name of Y dimension for spatial aggregation",
    )
    parser.add_argument(
        "--agg_x_dim",
        type=str,
        default=None,
        help="Name of X dimension for spatial aggregation",
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
        "--units_timing",
        type=str,
        choices=("start", "middle", "end"),
        default="end",
        help="When to perform the unit conversions in units",
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
    parser.add_argument(
        "--scale_factors",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        help="Divide input data by this value. Can be a float or days_in_month (e.g. pr=days_in_month)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Have open_mfforecast print file names as they are processed",
    )
    parser.add_argument(
        "--standard_calendar",
        action="store_true",
        default=False,
        help="Force a standard calendar when opening each file",
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
        "point_selection": args.point_selection,
        "lat_bnds": args.lat_bnds,
        "lon_bnds": args.lon_bnds,
        "shapefile": args.shapefile,
        "shapefile_label_header": args.shp_header,
        "shape_overlap": args.shp_overlap,
        "combine_shapes": args.combine_shapes,
        "spatial_agg": args.spatial_agg,
        "lat_dim": args.lat_dim,
        "lon_dim": args.lon_dim,
        "agg_y_dim": args.agg_y_dim,
        "agg_x_dim": args.agg_x_dim,
        "standard_calendar": args.standard_calendar,
        "no_leap_days": args.no_leap_days,
        "rolling_sum_window": args.rolling_sum_window,
        "time_freq": args.time_freq,
        "time_agg": args.time_agg,
        "time_agg_dates": args.time_agg_dates,
        "months": args.months,
        "season": args.season,
        "reset_times": args.reset_times,
        "time_agg_min_tsteps": args.time_agg_min_tsteps,
        "input_freq": args.input_freq,
        "time_dim": args.time_dim,
        "isel": args.isel,
        "sel": args.sel,
        "units": args.units,
        "scale_factors": args.scale_factors,
        "units_timing": args.units_timing,
    }

    kwargs, index = _indices_setup(kwargs, args.variables)

    if args.forecast:
        ds = open_mfforecast(
            args.infiles,
            n_time_files=args.n_time_files,
            n_ensemble_files=args.n_ensemble_files,
            verbose=args.verbose,
            **kwargs,
        )
        temporal_dim = "lead_time"
    else:
        ds = open_dataset(args.infiles, **kwargs)
        temporal_dim = args.time_dim

    if index == "ffdi":
        ds["ffdi"] = indices.calc_FFDI(
            ds, time_dim=temporal_dim, scale_dims=[temporal_dim]
        )

    if args.anomaly:
        ds = time_utils.anomalise(
            ds, args.anomaly, frequency=args.anomaly_freq, time_name=args.time_dim
        )

    if args.output_chunks:
        ds = ds.chunk(args.output_chunks)
    if args.time_agg_dates:
        ds = ds.set_coords(("event_time"))
    ds = ds[kwargs["variables"]]

    ds.attrs["history"] = get_new_log()
    if "zarr" in args.outfile:
        to_zarr(ds, args.outfile)
    else:
        ds.to_netcdf(args.outfile)


if __name__ == "__main__":
    _main()
