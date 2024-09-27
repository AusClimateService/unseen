"""Functions and command line program for similarity testing."""

import argparse
import xarray as xr
import xstatstests

from . import dask_setup
from . import fileio
from . import general_utils


def ks_test(obs_ds, fcst_ds):
    """Calculate KS test statistic and p-value.

    Parameters
    ----------
    obs_ds, fcst_ds : xarray.Dataset
        Observational and forecast dataset with 'sample' dimension

    Returns
    -------
    ks : xarray.Dataset
        Dataset with KS statistic and p-value variables
    """

    ks = xstatstests.ks_2samp_1d(obs_ds, fcst_ds, dim="sample")
    ks = ks.rename({"statistic": "ks_statistic"})
    ks = ks.rename({"pvalue": "ks_pval"})

    return ks


def anderson_darling_test(obs_ds, fcst_ds):
    """Calculate Anderson Darling test statistic and p-value.

    Parameters
    ----------
    obs_ds, fcst_ds : xarray.Dataset
        Observational and forecast dataset with 'sample' dimension

    Returns
    -------
    ad : xarray.Dataset
        Dataset with Anderson Darling statistic and p-value variables
    """

    ad = xstatstests.anderson_ksamp(obs_ds, fcst_ds, dim="sample")
    ad = ad.rename({"statistic": "ad_statistic"})
    ad = ad.rename({"pvalue": "ad_pval"})

    return ad


def similarity_tests(
    fcst,
    obs,
    min_lead=None,
    lead_dim="lead_time",
    init_dim="init_date",
    ensemble_dim="ensemble",
    time_dim="time",
    lat_dim="lat",
    lon_dim="lon",
    by_lead=False,
    regrid="obs",
    regrid_method="conservative",
):
    """Perform a series of similarity tests.

    Parameters
    ----------
    fcst : xarray Dataset or DataArray
        Forecast dataset with initial date, lead time and ensemble member dimensions.
    obs : xarray DataArray or DataArray
        Observational/comparison dataset with a time dimension.
    var : str
        Variable from the datasets to process.
    min_lead : int, optional
        Minimum lead time
    init_dim: str, default 'init_date'
        Name of the initial date dimension in fcst
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in fcst
    ensemble_dim: str, default 'ensemble'
        Name of the ensemble member dimension in fcst
    time_dim: str, default 'time'
        Name of the time dimension in obs
    lat_dim: str, default 'lat'
        Name of the latitude dimension in fcst and obs (if regridding)
    lon_dim: str, default 'lon'
        Name of the longitude dimension in fcst and obs (if regridding)
    by_lead: bool, default False
        Test each lead time separately
    regrid: {None, 'obs', 'fcst'}, default 'obs'
        Regrid observational data to model grid (or vice versa)
    regrid_method: {'conservative', 'bilinear', 'nearest_s2d', 'nearest_d2s'}, default 'conservative'
        Regriding method (see xesmf.Regridder)

    Returns
    -------
    ds : xarray Dataset
        Dataset with AD and KS statistic and p-value variables

    Notes
    -----
    If p > 0.05 you can't reject the null hypothesis
    that the two samples are from the same population.
    """

    if min_lead is not None:
        fcst = fcst.where(fcst[lead_dim] >= min_lead)

    if isinstance(fcst, xr.DataArray):
        fcst = fcst.to_dataset()

    if isinstance(obs, xr.DataArray):
        obs = obs.to_dataset()

    # Regrid fcst or obs if needed
    if set({lat_dim, lon_dim}).issubset(fcst.dims) and not all(
        [fcst[dim].equals(obs[dim]) for dim in [lat_dim, lon_dim]]
    ):
        if regrid == "obs":
            obs = general_utils.regrid(obs, fcst, method=regrid_method)
        elif regrid == "fcst":
            fcst = general_utils.regrid(fcst, obs, method=regrid_method)

    stack_dims = [ensemble_dim, init_dim]
    if not by_lead:
        stack_dims = stack_dims + [lead_dim]
    fcst = fcst.dropna(dim=lead_dim, how="all")
    fcst_stacked = fcst.stack({"sample": stack_dims})
    fcst_stacked = fcst_stacked.chunk({"sample": -1})

    obs_stacked = obs.rename({time_dim: "sample"})
    obs_stacked = obs_stacked.chunk({"sample": -1})

    ks = ks_test(obs_stacked, fcst_stacked)
    ad = anderson_darling_test(obs_stacked, fcst_stacked)
    ds = xr.merge([ks, ad])

    ds["ks_statistic"].attrs = {"long_name": "kolmogorov_smirnov_statistic"}
    ds["ks_pval"].attrs = {
        "long_name": "p_value",
        "note": "If p > 0.05 cannot reject the null hypothesis that the samples are drawn from the same population",
    }
    ds["ad_statistic"].attrs = {"long_name": "anderson_darling_statistic"}
    ds["ad_pval"].attrs = {
        "long_name": "p_value",
        "note": "If p > 0.05 cannot reject the null hypothesis that the samples are drawn from the same population",
    }

    return ds


def _parse_command_line():
    """Parse the command line for input arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument(
        "--dask_config", type=str, help="YAML file specifying dask client configuration"
    )

    parser.add_argument(
        "--reference_time_period",
        type=str,
        nargs=2,
        default=None,
        help="Start and end date (YYYY-MM-DD format)",
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
        "--time_dim",
        type=str,
        default="time",
        help="Name of time dimension",
    )
    parser.add_argument(
        "--ensemble_dim",
        type=str,
        default="ensemble",
        help="Name of ensemble member dimension",
    )
    parser.add_argument(
        "--init_dim",
        type=str,
        default="init_date",
        help="Name of initial date dimension",
    )
    parser.add_argument(
        "--lead_dim",
        type=str,
        default="lead_time",
        help="Name of lead time dimension",
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
        help="Similarity test each lead time separately [default=False]",
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

    if args.dask_config:
        client = dask_setup.launch_client(args.dask_config)
        print(client)

    ds_fcst = fileio.open_dataset(args.fcst_file, variables=[args.var])
    ds_obs = fileio.open_dataset(args.obs_file, variables=[args.var])
    if args.reference_time_period:
        start_date, end_date = args.reference_time_period
        ds_obs = ds_obs.sel({args.time_dim: slice(start_date, end_date)})

    if args.min_lead:
        if isinstance(args.min_lead, str):
            # Load min_lead from file
            ds_min_lead = fileio.open_dataset(args.min_lead, **args.min_lead_kwargs)
            min_lead = ds_min_lead["min_lead"].load()
            # Assumes min_lead has only one init month
            assert min_lead.month.size == 1, "Not implemented for multiple init months"
            min_lead = min_lead.drop_vars("month")
            if min_lead.size == 1:
                min_lead = min_lead.item()
        else:
            min_lead = args.min_lead

    ds_similarity = similarity_tests(
        ds_fcst,
        ds_obs,
        args.var,
        min_lead=min_lead,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
        ensemble_dim=args.ensemble_dim,
        time_dim=args.time_dim,
        by_lead=args.by_lead,
    )

    infile_logs = {
        args.fcst_file: ds_fcst.attrs["history"],
        args.obs_file: ds_obs.attrs["history"],
    }
    if isinstance(args.min_lead, str):
        infile_logs[args.min_lead] = ds_min_lead.attrs["history"]
    ds_similarity.attrs["history"] = fileio.get_new_log(infile_logs=infile_logs)

    if args.output_chunks:
        ds_similarity = ds_similarity.chunk(args.output_chunks)

    if "zarr" in args.outfile:
        fileio.to_zarr(ds_similarity, args.outfile)
    else:
        ds_similarity.to_netcdf(args.outfile)


if __name__ == "__main__":
    _main()
