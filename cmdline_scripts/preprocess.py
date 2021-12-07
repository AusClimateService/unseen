"""Command line program for preprocessing data."""

import sys

script_dir = sys.path[0]
repo_dir = "/".join(script_dir.split("/")[:-1])
module_dir = repo_dir + "/unseen"
sys.path.insert(1, module_dir)

import argparse

import fileio
import indices
import dask_setup
import general_utils


def indices_setup(kwargs, variables):
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


def _main(args):
    """Run the command line program."""

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
        "units": args.units,
    }

    kwargs, index = indices_setup(kwargs, args.variables)

    if args.data_type == "obs":
        assert len(args.infiles) == 1
        ds = fileio.open_file(args.infiles[0], **kwargs)
        temporal_dim = "time"
    elif args.data_type == "forecast":
        ds = fileio.open_mfforecast(args.infiles, **kwargs)
        temporal_dim = "lead_time"
    else:
        raise ValueError(f"Unrecognised data type: {args.data_type}")

    if index == "ffdi":
        ds["ffdi"] = indices.calc_FFDI(
            ds, time_dim=temporal_dim, scale_dims=[temporal_dim]
        )

    if args.output_chunks:
        ds = ds.chunk(args.output_chunks)
    ds = ds[args.variables]

    ds.attrs["history"] = fileio.get_new_log(repo_dir=repo_dir)
    fileio.to_zarr(ds, args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("infiles", type=str, nargs="*", help="Input files")
    parser.add_argument(
        "data_type", type=str, choices=("forecast", "obs"), help="Data type"
    )
    parser.add_argument("outfile", type=str, help="Output file")

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

    args = parser.parse_args()
    _main(args)
