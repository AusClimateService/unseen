"""Functions and command line program for bias correction."""

import argparse
import operator
import xarray as xr

from . import array_handling
from . import time_utils
from . import fileio
from . import general_utils


def get_bias(
    fcst,
    obs,
    method,
    time_period=None,
    time_rounding="D",
    ensemble_dim="ensemble",
    init_dim="init_date",
    lead_dim="lead_time",
    lat_dim="lat",
    lon_dim="lon",
    by_lead=False,
    regrid="obs",
    regrid_method="conservative",
):
    """Calculate forecast bias.

    Parameters
    ----------
    fcst : xarray.DataArray
        Forecast array with ensemble, initial date and lead time dimensions
    obs : xarray.DataArray
        Observational array with time dimension
    method : {'additive', 'multiplicative'}
        Bias removal method
    time_period : array_like, optional
        Start and end dates (in YYYY-MM-DD format)
    time_rounding : str, default 'D'
        Rounding (floor) frequency for time matching
    ensemble_name: str, default 'ensemble'
        Name of the ensemble member dimension in fcst
    init_dim: str, default 'init_date'
        Name of the initial date dimension in fcst
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in fcst
    lat_dim: str, default 'lat'
        Name of the latitude dimension in fcst and obs (if regridding)
    lon_dim: str, default 'lon'
        Name of the longitude dimension in fcst and obs (if regridding)
    by_lead: bool, default False
        Calculate bias for each lead time separately
    regrid: {None, 'obs', 'fcst'}, default 'obs'
        Regrid observational data to model grid (or vice versa)
    regrid_method: {'conservative', 'bilinear', 'nearest_s2d', 'nearest_d2s'}, default 'conservative'
        Regridding method (see xesmf.Regridder)


    Returns
    -------
    bias : xarray.DataArray

    Raises
    ------
    ValueError
        For invalid method
    """

    fcst_ave_dims = [ensemble_dim, init_dim]
    obs_ave_dims = [init_dim]
    if not by_lead:
        fcst_ave_dims = fcst_ave_dims + [lead_dim]
        obs_ave_dims = obs_ave_dims + [lead_dim]

    fcst_clim = time_utils.get_clim(
        fcst,
        dims=fcst_ave_dims,
        time_period=time_period,
        groupby_init_month=True,
        time_name="time",
    )

    obs_stacked = array_handling.stack_by_init_date(
        obs,
        init_dates=fcst[init_dim],
        n_lead_steps=fcst.sizes[lead_dim],
        time_rounding=time_rounding,
    )
    obs_clim = time_utils.get_clim(
        obs_stacked,
        obs_ave_dims,
        time_period=time_period,
        groupby_init_month=True,
    )

    # Check fcst dimensions include lat_dim and lon_dim
    if set({lat_dim, lon_dim}).issubset(fcst_clim.dims):
        # Check lat and lon coordinates match
        if not all(
            [fcst_clim[dim].equals(obs_clim[dim]) for dim in [lat_dim, lon_dim]]
        ):
            if regrid == "obs":
                obs_clim = general_utils.regrid(
                    obs_clim, fcst_clim, method=regrid_method
                )
            elif regrid == "fcst":
                fcst_clim = general_utils.regrid(
                    fcst_clim, obs_clim, method=regrid_method
                )
            else:
                raise ValueError(
                    "Forecast and observational data coordinates do not match. Consider regridding."
                )

    with xr.set_options(keep_attrs=True):
        if method == "additive":
            bias = fcst_clim - obs_clim
        elif method == "multiplicative":
            bias = fcst_clim / obs_clim
        else:
            raise ValueError(f"Unrecognised bias removal method {method}")

    bias.attrs["bias_correction_method"] = method
    if time_period:
        bias.attrs["bias_correction_period"] = "-".join(time_period)

    return bias


def remove_bias(fcst, bias, method, init_dim="init_date"):
    """Remove model bias.

    Parameters
    ----------
    fcst : xarray.DataArray
        Forecast array with initial date and lead time dimensions
    bias : xarray.DataArray
        Bias array
    method : {'additive', 'multiplicative'}
        Bias removal method
    init_dim: str, default 'init_date'
        Name of the initial date dimension in fcst

    Returns
    -------
    fcst_bc : xarray.DataArray
        Bias corrected forecast array

    Raises
    ------
    ValueError
        For invalid method
    """

    if method == "additive":
        op = operator.sub
    elif method == "multiplicative":
        op = operator.truediv
    else:
        raise ValueError(f"Unrecognised bias removal method {method}")

    with xr.set_options(keep_attrs=True):
        fcst_bc = op(fcst.groupby(f"{init_dim}.month"), bias).drop_vars("month")

    fcst_bc.attrs["bias_correction_method"] = bias.attrs["bias_correction_method"]
    try:
        fcst_bc.attrs["bias_correction_period"] = bias.attrs["bias_correction_period"]
    except KeyError:
        pass

    return fcst_bc


def _parse_command_line():
    """Parse the command line for input arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument(
        "method",
        type=str,
        choices=("multiplicative", "additive"),
        help="Bias correction method",
    )
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument(
        "--base_period",
        type=str,
        nargs=2,
        help="Start and end date for baseline (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--output_chunks",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        default={},
        help="Chunks for writing data to file (e.g. init_date=-1 lead_time=-1)",
    )
    parser.add_argument(
        "--rounding_freq",
        type=str,
        choices=("M", "D", "A"),
        default="D",
        help="Floor rounding to nearest day, month or year for time matching",
    )
    parser.add_argument(
        "--min_lead",
        default=None,
        help="Minimum lead time to include in analysis (int or filename)",
    )
    parser.add_argument(
        "--min_lead_kwargs",
        nargs="*",
        action=general_utils.store_dict,
        help="Optional fileio.open_dataset kwargs for lead independence (e.g., spatial_agg=median)",
    )
    parser.add_argument(
        "--by_lead",
        action="store_true",
        default=False,
        help="Remove bias for each lead time separately [default=False]",
    )
    parser.add_argument(
        "--regrid",
        choices=("obs", "fcst"),
        default="obs",
        help="Regrid observational or forecast data if they are on different grids[default=obs]",
    )
    parser.add_argument(
        "--regrid_method",
        choices=("conservative", "bilinear", "nearest_s2d", "nearest_d2s"),
        default="conservative",
        help="Regridding method for observational or forecast data [default=conservative]",
    )

    args = parser.parse_args()

    return args


def _main():
    """Run the command line program."""

    args = _parse_command_line()

    ds_obs = fileio.open_dataset(args.obs_file, variables=[args.var])
    da_obs = ds_obs[args.var]

    ds_fcst = fileio.open_dataset(args.fcst_file, variables=[args.var])
    da_fcst = ds_fcst[args.var].load()

    # Mask lead times below min_lead
    if args.min_lead:
        if isinstance(args.min_lead, str):
            # Load min_lead from file
            ds_min_lead = fileio.open_dataset(args.min_lead, **args.min_lead_kwargs)
            min_lead = ds_min_lead["min_lead"].load()
            # Assumes min_lead has only one init month
            assert min_lead.month.size == 1, "Not implemented for multiple init months"
            min_lead = min_lead.drop_vars("month")
        else:
            min_lead = args.min_lead
        da_fcst = da_fcst.where(da_fcst >= min_lead)

    # Calculate bias
    bias = get_bias(
        da_fcst,
        da_obs,
        args.method,
        time_period=args.base_period,
        time_rounding=args.rounding_freq,
        by_lead=args.by_lead,
        regrid=args.regrid,
        regrid_method=args.regrid_method,
    )
    # Remove bias
    da_fcst_bc = remove_bias(da_fcst, bias, args.method)

    # Save to file
    ds_fcst_bc = da_fcst_bc.to_dataset(name=args.var)
    ds_fcst_bc.attrs.update(ds_fcst.attrs)

    infile_logs = {
        args.fcst_file: ds_fcst.attrs["history"],
        args.obs_file: ds_obs.attrs["history"],
    }
    if isinstance(args.min_lead, str):
        infile_logs[args.min_lead] = ds_min_lead.attrs["history"]
    ds_fcst_bc.attrs["history"] = fileio.get_new_log(infile_logs=infile_logs)

    if args.output_chunks:
        ds_fcst_bc = ds_fcst_bc.chunk(args.output_chunks)

    if "zarr" in args.outfile:
        fileio.to_zarr(ds_fcst_bc, args.outfile)
    else:
        ds_fcst_bc.to_netcdf(args.outfile)


if __name__ == "__main__":
    _main()
