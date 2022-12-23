"""Funcitons and command line program for similarity testing."""

import argparse

import numpy as np
import xarray as xr
import xstatstests

from . import dask_setup
from . import fileio
from . import general_utils


def similarity_tests(
    fcst,
    obs,
    var,
    lead_dim="lead_time",
    init_dim="init_date",
    ensemble_dim="ensemble",
    time_dim="time",
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
    init_dim: str, default 'init_date'
        Name of the initial date dimension in fcst
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in fcst
    ensemble_dim: str, default 'ensemble'
        Name of the ensemble member dimension in fcst
    time_dim: str, default 'time'
        Name of the time dimension in obs

    Returns
    -------
    ds : xarray Dataset
        Dataset with KS statistic and p-value variables

    Notes
    -----
    If p > 0.05 you can't reject the null hypothesis
    that the two samples are from the same population.
    """

    if isinstance(fcst, xr.DataArray):
        fcst = fcst.to_dataset()
    if isinstance(obs, xr.DataArray):
        obs = obs.to_dataset()

    fcst_stacked = fcst.stack({"sample": [ensemble_dim, init_dim]})
    fcst_stacked = fcst_stacked.chunk({"sample": -1})

    obs_stacked = obs.rename({time_dim: "sample"})
    obs_stacked = obs_stacked.chunk({"sample": -1})

    ks_statistics = []
    ks_pvals = []
    ad_statistics = []
    ad_pvals = []
    for lead_time in fcst_stacked[lead_dim].values:
        fcst_data = fcst_stacked.sel({lead_dim: lead_time})
        if not np.isnan(fcst_data[var].values).all():
            ks = xstatstests.ks_2samp_1d(obs_stacked, fcst_data, dim="sample")
            ks = ks.rename({"statistic": "ks_statistic"})
            ks = ks.rename({"pvalue": "ks_pval"})
            ks_statistics.append(ks["ks_statistic"])
            ks_pvals.append(ks["ks_pval"])
            ad = xstatstests.anderson_ksamp(obs_stacked, fcst_data, dim="sample")
            ad = ad.rename({"statistic": "ad_statistic"})
            ad = ad.rename({"pvalue": "ad_pval"})
            ad_statistics.append(ad["ad_statistic"])
            ad_pvals.append(ad["ad_pval"])

    ks_statistics = xr.concat(ks_statistics, lead_dim)
    ks_pvals = xr.concat(ks_pvals, lead_dim)
    ad_statistics = xr.concat(ad_statistics, lead_dim)
    ad_pvals = xr.concat(ad_pvals, lead_dim)

    ds = xr.merge([ks_statistics, ks_pvals, ad_statistics, ad_pvals])

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
    """Parse the command line for input agruments"""

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
        time_slice = general_utils.date_pair_to_time_slice(args.reference_time_period)
        ds_obs = ds_obs.sel({args.time_dim: time_slice})

    ds_similarity = similarity_tests(
        ds_fcst,
        ds_obs,
        args.var,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
        ensemble_dim=args.ensemble_dim,
        time_dim=args.time_dim,
    )

    infile_logs = {
        args.fcst_file: ds_fcst.attrs["history"],
        args.obs_file: ds_obs.attrs["history"],
    }
    ds_similarity.attrs["history"] = fileio.get_new_log(infile_logs=infile_logs)

    if args.output_chunks:
        ds_similarity = ds_similarity.chunk(args.output_chunks)
    fileio.to_zarr(ds_similarity, args.outfile)


if __name__ == "__main__":
    _main()
