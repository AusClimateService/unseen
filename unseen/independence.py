"""Functions and command line program for independence testing."""

import argparse
import calendar
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
import matplotlib.pyplot as plt

from . import dask_setup
from . import fileio
from . import general_utils


def remove_ensemble_mean_trend(da, dim="init_date"):
    """Remove ensemble mean trend along given dimension

    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate and remove trend
    """

    ensmean_trend = da.mean("ensemble").polyfit(dim=dim, deg=1)
    ensmean_trend_line = xr.polyval(da[dim], ensmean_trend["polyfit_coefficients"])
    ensmean_trend_line_anomaly = ensmean_trend_line - ensmean_trend_line.isel({dim: 0})
    da_detrended = da - ensmean_trend_line_anomaly

    return da_detrended


def mean_ensemble_correlation(da, dim="init_date"):
    """Mean correlation between all ensemble members.

    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate correlation
    """

    n_ensemble_members = len(da["ensemble"])
    combinations = np.array(list(itertools.combinations(range(n_ensemble_members), 2)))

    new_ensemble_coord = {"ensemble": range(combinations.shape[0])}
    e1 = da.isel(ensemble=combinations[:, 0]).assign_coords(new_ensemble_coord)
    e2 = da.isel(ensemble=combinations[:, 1]).assign_coords(new_ensemble_coord)

    e1 = e1.chunk({dim: -1})
    e2 = e2.chunk({dim: -1})

    corr_combinations = xs.spearman_r(e1, e2, dim=dim, skipna=True)
    mean_corr = corr_combinations.mean("ensemble")

    return mean_corr


def random_sample(ds, sample_dim, sample_size):
    """Take random sample along a given dimension.

    Args:
      ds (xarray Dataset or DataArray)
      sample_dim (str) : Dimension along which to sample
      sample_size (int) : Number of points to sample along sample_dim
    """

    n_population = len(ds[sample_dim])
    random_indexes = np.random.choice(n_population, size=sample_size, replace=False)
    # random_indexes.sort()
    ds_random_sample = ds.isel({sample_dim: random_indexes})

    return ds_random_sample


def random_mean_ensemble_correlation(ds, n_init_dates, n_ensembles):
    """Mean correlation between a random selection of samples"""

    sample_size = n_init_dates * n_ensembles
    ds_random_sample = random_sample(ds, "sample", sample_size)
    index = pd.MultiIndex.from_product(
        [range(n_init_dates), range(n_ensembles)], names=["init_date", "ensemble"]
    )
    ds_random_sample = ds_random_sample.assign_coords({"sample": index}).unstack()
    mean_corr = mean_ensemble_correlation(ds_random_sample, dim="init_date")

    return mean_corr


def get_null_correlation_bounds(da):
    """Get the uncertainty bounds on zero correlation.

    Performs bootstrapping via a simple loop.
    """

    n_init_dates = len(da["init_date"])
    n_ensembles = len(da["ensemble"])
    da_stacked = da.stack(sample=("init_date", "lead_time", "ensemble"))

    null_correlations = []
    n_repeats = 100
    for repeat in range(n_repeats):
        mean_corr = random_mean_ensemble_correlation(
            da_stacked, n_init_dates, n_ensembles
        )
        null_correlations.append(mean_corr)
    null_correlations = xr.concat(null_correlations, "k")
    null_correlations = null_correlations.chunk({"k": -1})

    lower_bound = float(null_correlations.quantile(0.025).values)
    upper_bound = float(null_correlations.quantile(0.975).values)

    return lower_bound, upper_bound


def create_plot(mean_correlations, null_correlation_bounds, max_lead_times, outfile):
    """Create plot."""

    fig, ax = plt.subplots()

    months = list(mean_correlations.keys())
    months.sort()
    for month in months:
        color = next(ax._get_lines.prop_cycler)["color"]

        month_abbr = calendar.month_abbr[month]
        mean_correlations[month].plot(
            color=color, marker="o", linestyle="None", label=f"{month_abbr} starts"
        )

        lower_bound, upper_bound = null_correlation_bounds[month]
        lead_time_bounds = [1, max_lead_times[month]]

        plt.plot(
            lead_time_bounds, [lower_bound, lower_bound], color=color, linestyle="--"
        )
        plt.plot(
            lead_time_bounds, [upper_bound, upper_bound], color=color, linestyle="--"
        )

    plt.ylabel("correlation")
    plt.legend()
    plt.savefig(outfile)


def _parse_command_line():
    """Parse the command line for input agruments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument(
        "--dask_config", type=str, help="YAML file specifying dask client configuration"
    )
    parser.add_argument(
        "--lead_time_increment",
        type=int,
        default=None,
        help="Increment the lead time (e.g. to account for exclusion of non-complete years)",
    )
    parser.add_argument(
        "--spatial_selection",
        type=str,
        nargs="*",
        default={},
        action=general_utils.store_dict,
        help="Spatial variable / selection pair (e.g. region=all)",
    )

    args = parser.parse_args()

    return args


def _main():
    """Run the command line program."""

    args = _parse_command_line()

    if args.dask_config:
        client = dask_setup.launch_client(args.dask_config)
        print(client)

    ds_fcst = fileio.open_file(
        args.fcst_file, variables=[args.var], sel=args.spatial_selection
    )
    da_fcst = ds_fcst[args.var]
    if args.lead_time_increment:
        da_fcst = da_fcst.assign_coords(
            {"lead_time": da_fcst["lead_time"] + args.lead_time_increment}
        )
    months = np.unique(da_fcst["init_date"].dt.month.values)
    mean_correlations = {}
    null_correlation_bounds = {}
    max_lead_times = {}
    for month in months:
        da_fcst_month = da_fcst.where(da_fcst["init_date"].dt.month == month, drop=True)
        da_fcst_month_detrended = remove_ensemble_mean_trend(
            da_fcst_month, dim="init_date"
        )
        mean_correlations[month] = mean_ensemble_correlation(
            da_fcst_month_detrended, dim="init_date"
        )
        null_correlation_bounds[month] = get_null_correlation_bounds(
            da_fcst_month_detrended
        )
        max_lead_times[month] = da_fcst_month["lead_time"].max()

    create_plot(
        mean_correlations, null_correlation_bounds, max_lead_times, args.outfile
    )


if __name__ == "__main__":
    _main()
