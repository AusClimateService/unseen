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


def run_tests(
    fcst,
    comparison_fcst=None,
    init_dim="init_date",
    lead_dim="lead_time",
    ensemble_dim="ensemble",
):
    """Perform independence tests for each lead time and initial month.

    Parameters
    ----------
    fcst : xarray DataArray
        Forecast data
    comparison_fcst : xarray DataArray, optional
        Forecast data to compare against
        If None, the ensemble members of fcst are compared against each other
    init_dim: str, default 'init_date'
        Name of the initial date dimension in fcst
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in fcst
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in fcst

    Returns
    -------
    mean_correlations : dict of
        Mean correlation between all ensemble members for each lead time and initial month
    null_correlation_bounds : dict of
        Bounds on zero correlation for each lead time and initial month

    """

    months = np.unique(fcst[init_dim].dt.month.values)
    mean_correlations = {}
    null_correlation_bounds = {}
    for month in months:
        fcst_month = fcst.where(fcst[init_dim].dt.month == month, drop=True)
        fcst_month_detrended = _remove_ensemble_mean_trend(
            fcst_month, dim=init_dim, ensemble_dim=ensemble_dim
        )
        if not isinstance(comparison_fcst, type(None)):
            comparison_fcst_month = comparison_fcst.where(
                comparison_fcst[init_dim].dt.month == month, drop=True
            )
            comparison_fcst_month_detrended = _remove_ensemble_mean_trend(
                comparison_fcst_month, dim=init_dim, ensemble_dim=ensemble_dim
            )
            mean_correlations[month] = _mean_ensemble_correlation(
                fcst_month_detrended,
                comparison_da=comparison_fcst_month_detrended,
                dim=init_dim,
                ensemble_dim=ensemble_dim,
            )
        else:
            mean_correlations[month] = _mean_ensemble_correlation(
                fcst_month_detrended, dim=init_dim, ensemble_dim=ensemble_dim
            )

        null_correlation_bounds[month] = _get_null_correlation_bounds(
            fcst_month_detrended,
            init_dim=init_dim,
            lead_dim=lead_dim,
            ensemble_dim=ensemble_dim,
        )

    return mean_correlations, null_correlation_bounds


def create_plot(
    mean_correlations, null_correlation_bounds, outfile, lead_dim="lead_time"
):
    """Create independence plot.

    Parameters
    ----------
    mean_correlations : xarray Dataset
        Mean correlation (for each lead time) data
    null_correlation_bounds : list
        Bounds on zero correlation [lower_bound, upper_bound]
    outfile : str
        Path for output image file
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in mean_correlations
    """

    fig, ax = plt.subplots()

    months = list(mean_correlations.keys())
    months.sort()
    for month in months:
        color = next(ax._get_lines.prop_cycler)["color"]
        mean_corr = mean_correlations[month].dropna(lead_dim)
        month_abbr = calendar.month_abbr[month]
        mean_corr.plot(
            color=color, marker="o", linestyle="None", label=f"{month_abbr} starts"
        )
        lower_bound, upper_bound = null_correlation_bounds[month]
        lead_time_bounds = [mean_corr[lead_dim].min(), mean_corr[lead_dim].max()]
        plt.plot(
            lead_time_bounds, [lower_bound, lower_bound], color=color, linestyle="--"
        )
        plt.plot(
            lead_time_bounds, [upper_bound, upper_bound], color=color, linestyle="--"
        )
    plt.ylabel("correlation")
    plt.legend()
    plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)


def _remove_ensemble_mean_trend(da, dim="init_date", ensemble_dim="ensemble"):
    """Remove ensemble mean trend along given dimension.

    Parameters
    ----------
    da : xarray DataArray
        Input data array
    dim : str, default init_date
        Dimension over which to calculate and remove trend
    init_dim: str, default 'init_date'
        Name of the initial date dimension to create in the output
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in da

    Returns
    -------
    da_detrended : xarray DataArray
        Detrended data array
    """

    ensmean_trend = da.mean(ensemble_dim).polyfit(dim=dim, deg=1)
    ensmean_trend_line = xr.polyval(da[dim], ensmean_trend["polyfit_coefficients"])
    ensmean_trend_line_anomaly = ensmean_trend_line - ensmean_trend_line.isel({dim: 0})
    da_detrended = da - ensmean_trend_line_anomaly

    return da_detrended


def _mean_ensemble_correlation(
    da, comparison_da=None, dim="init_date", ensemble_dim="ensemble"
):
    """Mean correlation between all ensemble members.

    Parameters
    ----------
    da : xarray DataArray
        Input data array
    comparison_da : xarray DataArray
        Input data array to compare da against
        If None, the ensemble members of da are compared against each other
    dim : str, default init_date
        Dimension over which to calculate correlation
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in da

    Returns
    -------
    mean_corr : xarray DataArray
        Mean correlations
    """

    n_ensemble_members = len(da[ensemble_dim])
    if isinstance(comparison_da, type(None)):
        combinations = np.array(
            list(itertools.combinations(range(n_ensemble_members), 2))
        )
    else:
        combinations = np.array(
            list(itertools.combinations_with_replacement(range(n_ensemble_members), 2))
        )

    new_ensemble_coord = {ensemble_dim: range(combinations.shape[0])}
    e1 = da.isel(ensemble=combinations[:, 0]).assign_coords(new_ensemble_coord)
    if isinstance(comparison_da, type(None)):
        e2 = da.isel(ensemble=combinations[:, 1]).assign_coords(new_ensemble_coord)
    else:
        e2 = comparison_da.isel(ensemble=combinations[:, 1]).assign_coords(
            new_ensemble_coord
        )

    e1 = e1.chunk({dim: -1})
    e2 = e2.chunk({dim: -1})

    corr_combinations = xs.spearman_r(e1, e2, dim=dim, skipna=True)
    mean_corr = corr_combinations.mean(ensemble_dim)

    return mean_corr


def _random_sample(ds, sample_dim, sample_size):
    """Take random sample along a given dimension.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Input data
    sample_dim : str
        Dimension along which to sample
    sample_size : int
        Number of points to sample along sample_dim

    Returns
    -------
    ds_random_sample : xarray DataArray or Dataset
        Random sample of the input data
    """

    n_population = len(ds[sample_dim])
    random_indexes = np.random.choice(n_population, size=sample_size, replace=False)
    # random_indexes.sort()
    ds_random_sample = ds.isel({sample_dim: random_indexes})

    return ds_random_sample


def _random_mean_ensemble_correlation(
    ds, n_init_dates, n_ensembles, init_dim="init_date", ensemble_dim="ensemble"
):
    """Mean correlation between a random selection of samples.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
        Input data
    n_init_dates : int
        Number of initial dates
    n_ensembles : int
        Number of ensemble members
    init_dim: str, default 'init_date'
        Name of the initial date dimension to create in the output
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in ds

    Returns
    -------
    mean_corr : xarray DataArray or Dataset
        Mean correlations
    """

    sample_size = n_init_dates * n_ensembles
    ds_random_sample = _random_sample(ds, "sample", sample_size)
    index = pd.MultiIndex.from_product(
        [range(n_init_dates), range(n_ensembles)], names=[init_dim, ensemble_dim]
    )
    ds_random_sample = ds_random_sample.assign_coords({"sample": index}).unstack()
    mean_corr = _mean_ensemble_correlation(ds_random_sample, dim=init_dim)

    return mean_corr


def _get_null_correlation_bounds(
    da, init_dim="init_time", lead_dim="lead_dim", ensemble_dim="ensemble"
):
    """Get the uncertainty bounds on zero correlation.

    Parameters
    ----------
    da : xarray DataArray
    init_dim: str, default 'init_date'
        Name of the initial date dimension in da
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in da
    ensemble_dim: str, default 'ensemble'
        Name of the ensemble member dimension in da

    Returns
    -------
    lower_bound : float
        Lower uncertainly bound
    upper_bound : float
        Upper uncertainly bound

    Notes
    -----
    Performs bootstrapping via a simple loop.
    """

    n_init_dates = len(da["init_date"])
    n_ensembles = len(da["ensemble"])
    da_stacked = da.stack(sample=("init_date", "lead_time", "ensemble"))

    null_correlations = []
    n_repeats = 100
    for repeat in range(n_repeats):
        mean_corr = _random_mean_ensemble_correlation(
            da_stacked, n_init_dates, n_ensembles
        )
        null_correlations.append(mean_corr)
    null_correlations = xr.concat(null_correlations, "k")
    null_correlations = null_correlations.chunk({"k": -1})

    lower_bound = float(null_correlations.quantile(0.025).values)
    upper_bound = float(null_correlations.quantile(0.975).values)

    return lower_bound, upper_bound


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
        "--spatial_selection",
        type=str,
        nargs="*",
        default={},
        action=general_utils.store_dict,
        help="Spatial variable / selection pair (e.g. region=all)",
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

    ds_fcst = fileio.open_dataset(
        args.fcst_file, variables=[args.var], sel=args.spatial_selection
    )
    da_fcst = ds_fcst[args.var]

    mean_correlations, null_correlation_bounds = run_tests(
        da_fcst,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
        ensemble_dim=args.ensemble_dim,
    )

    create_plot(mean_correlations, null_correlation_bounds, args.outfile)


if __name__ == "__main__":
    _main()
