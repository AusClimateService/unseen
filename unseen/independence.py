"""Functions and command line program for independence testing."""

import argparse
import calendar
from cartopy.crs import PlateCarree
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs

from . import dask_setup
from . import fileio
from . import general_utils

rng = np.random.default_rng(seed=0)


def run_tests(
    fcst,
    comparison_fcst=None,
    init_dim="init_date",
    lead_dim="lead_time",
    ensemble_dim="ensemble",
    confidence_interval=0.95,
    n_resamples=100,
):
    """Perform independence tests for each lead time and initial month.

    Parameters
    ----------
    fcst : xarray.DataArray
        Forecast data
    comparison_fcst : xarray.DataArray, optional
        Forecast data to compare against
        If None, the ensemble members of fcst are compared against each other
    init_dim: str, default 'init_date'
        Name of the initial date dimension in fcst
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in fcst
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in fcst
    confidence_interval : float, default 0.95
        Confidence interval for the uncertainty bounds
    n_resamples : int, default 100
        Number of confidence interval bootstrap resamples

    Returns
    -------
    ds : xarray.Dataset
        Correlation ensemble mean and confidence interval bounds
    """

    months = np.unique(fcst[init_dim].dt.month.values)
    r_mean = {}
    ci = {}
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
            r_mean[month] = _mean_ensemble_correlation(
                fcst_month_detrended,
                comparison_da=comparison_fcst_month_detrended,
                dim=init_dim,
                ensemble_dim=ensemble_dim,
            )
        else:
            r_mean[month] = _mean_ensemble_correlation(
                fcst_month_detrended, dim=init_dim, ensemble_dim=ensemble_dim
            )

        ci[month] = _get_null_correlation_bounds(
            fcst_month_detrended,
            init_dim=init_dim,
            lead_dim=lead_dim,
            ensemble_dim=ensemble_dim,
            confidence_interval=confidence_interval,
            n_resamples=n_resamples,
        )

    ds = xr.Dataset(
        {
            "r": xr.concat(
                [da.assign_coords({"month": k}) for k, da in r_mean.items()],
                dim="month",
            ),
            "ci": xr.concat(
                [da.assign_coords({"month": k}) for k, da in ci.items()], dim="month"
            ),
        },
        coords=fcst.coords,
    )
    # Add minimum independent lead as variable
    ds["min_lead"] = min_independent_lead(ds, lead_dim=lead_dim)
    return ds


def min_independent_lead(ds, lead_dim="lead_time"):
    """Get the first lead time within confidence interval.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of correlation coefficient and confidence interval
    lead_dim : str, default 'lead_time'
        Name of the lead time dimension in ds

    Returns
    -------
    lead : int
        Index of the first lead time within the confidence interval
    """
    mask = (ds["r"] >= ds["ci"].isel(quantile=0)) & (
        ds["r"] >= ds["ci"].isel(quantile=-1)
    )

    min_lead = mask.rank(lead_dim).argmin(lead_dim)
    min_lead.attrs["long_name"] = "Minimum lead time"
    min_lead.attrs["description"] = (
        "First lead within the independence confidence interval"
    )
    return min_lead


def point_plot(
    ds,
    outfile=None,
    dataset_name=None,
    lead_dim="lead_time",
    confidence_interval=0.95,
    **kwargs,
):
    """Scatter plot of lead time dependence (for each init month).

    Parameters
    ----------
    ds : xarray.Dataset
        Mean correlation (for each lead time) and confidence interval bounds
    outfile : str, optional
        Path for output image file
    dataset_name : str, optional
        Name of the dataset for plot suptitle
    lead_dim : str, default 'lead_time'
        Name of the lead time dimension
    confidence_interval : float, default 0.95
        Confidence interval for the uncertainty bounds (for plot legend)
    kwargs : dict
        Additional keyword arguments for plt.subplots
    """

    fig, ax = plt.subplots(**kwargs)
    colors = iter(plt.cm.Set1(np.linspace(0, 1, 9)))

    months = list(ds["r"].month.values)
    months.sort()
    for i, month in enumerate(months):
        color = next(colors)
        month_abbr = calendar.month_abbr[month]

        # Plot the ensemble mean correlation
        mean_corr = ds["r"].isel(month=i).dropna(lead_dim)
        mean_corr.plot.scatter(
            ax=ax,
            x=lead_dim,
            color=color,
            marker="o",
            linestyle="None",
            label=f"{month_abbr} starts",
        )
        # Plot the null correlation bounds as dashed lines
        bounds = ds["ci"].isel(month=i).values
        ax.axhline(bounds[0], c=color, ls="--")
        ax.axhline(
            bounds[1], c=color, ls="--", label=f"{confidence_interval * 100: g}% CI"
        )

    ax.set_xlabel("Lead")
    ax.set_ylabel(ds["r"].attrs["long_name"])
    ax.legend()

    if dataset_name:
        fig.suptitle(dataset_name, x=0.6, y=1.02)

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()


def spatial_plot(ds, dataset_name=None, outfile=None, kwargs=dict(figsize=[8, 5])):
    """Contour plot of the first independent lead time (for each init month).

    Parameters
    ----------
    ds : xarray.Dataset
        Index of the first independent lead time (for each init month)
    dataset_name : str, optional
        Name of the dataset for plot titles
    outfile : str, optional
        Path for output image file
    kwargs : dict, optional
        Additional keyword arguments for xarray.DataArray.plot
    """
    # Discrete colour bar based on lead times
    cbar_ticks = np.arange(ds.min_lead.min(), ds.min_lead.max() + 2)
    # Convert integer to month names for plot titles
    titles = [f"{calendar.month_name[m]} starts" for m in ds.month.values]

    cm = ds.min_lead.plot.pcolormesh(
        col="month",
        col_wrap=min(3, len(ds.month)),
        transform=PlateCarree(),
        subplot_kws=dict(projection=PlateCarree()),
        levels=cbar_ticks,
        cbar_kwargs=dict(ticks=cbar_ticks[:-1]),
        add_colorbar=False,
        **kwargs,
    )
    # Fix hidden axis ticks and labels
    for i, ax in enumerate(cm.axs.flat):
        subplotspec = ax.get_subplotspec()
        if subplotspec.is_last_row():
            ax.xaxis.set_visible(True)
        if subplotspec.is_first_col():
            ax.yaxis.set_visible(True)
        ax.coastlines()
        ax.set_title(titles[i])
    cm.fig.set_constrained_layout(True)
    cm.fig.get_layout_engine().set(h_pad=0.2)
    cm.add_colorbar()

    if dataset_name:
        cm.fig.suptitle(dataset_name, x=0.6, y=1.02)

    # Fix lat/lon axis labels
    if all([dim in ds.dims for dim in ["lat", "lon"]]):
        cm.set_xlabels(f"{ds.lon.attrs['long_name']} [{ds.lon.attrs['units']}]")
        cm.set_ylabels(f"{ds.lat.attrs['long_name']} [{ds.lat.attrs['units']}]")
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()


def _remove_ensemble_mean_trend(da, dim="init_date", ensemble_dim="ensemble"):
    """Remove ensemble mean trend along given dimension.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    dim : str, default init_date
        Dimension over which to calculate and remove trend
    init_dim : str, default 'init_date'
        Name of the initial date dimension to create in the output
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in da

    Returns
    -------
    da_detrended : xarray.DataArray
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
    da : xarray.DataArray
        Input data array
    comparison_da : xarray.DataArray
        Input data array to compare da against
        If None, the ensemble members of da are compared against each other
    dim : str, default 'init_date'
        Dimension over which to calculate correlation
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in da

    Returns
    -------
    mean_corr : xarray.DataArray
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
    mean_corr.attrs = {
        "long_name": "Correlation coefficient",
        "description": "Ensemble mean Spearman rank correlation coefficient",
    }
    return mean_corr


def _random_sample(ds, sample_dim, sample_size):
    """Take random sample along a given dimension.

    Parameters
    ----------
    ds : xarray.DataArray or Dataset
        Input data
    sample_dim : str
        Dimension along which to sample
    sample_size : int
        Number of points to sample along sample_dim

    Returns
    -------
    ds_random_sample : xarray.DataArray or Dataset
        Random sample of the input data
    """

    n_population = len(ds[sample_dim])
    random_indexes = rng.choice(n_population, size=sample_size, replace=False)
    ds_random_sample = ds.isel({sample_dim: random_indexes})

    return ds_random_sample


def _random_mean_ensemble_correlation(
    ds, n_init_dates, n_ensembles, init_dim="init_date", ensemble_dim="ensemble"
):
    """Mean correlation between a random selection of samples.

    Parameters
    ----------
    ds : xarray.DataArray or Dataset
        Input data
    n_init_dates : int
        Number of initial dates
    n_ensembles : int
        Number of ensemble members
    init_dim : str, default 'init_date'
        Name of the initial date dimension to create in the output
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in ds

    Returns
    -------
    mean_corr : xarray.DataArray or Dataset
        Mean correlations
    """

    sample_size = n_init_dates * n_ensembles
    ds_random_sample = _random_sample(ds, "sample", sample_size)

    # Combine init_dim and ensemble_dim
    index = pd.MultiIndex.from_product(
        [range(n_init_dates), range(n_ensembles)], names=[init_dim, ensemble_dim]
    )
    index_coords = xr.Coordinates.from_pandas_multiindex(index, "sample")
    ds_random_sample = ds_random_sample.drop_vars(["sample"])
    ds_random_sample = ds_random_sample.assign_coords(index_coords).unstack()

    mean_corr = _mean_ensemble_correlation(ds_random_sample, dim=init_dim)
    return mean_corr


def _get_null_correlation_bounds(
    da,
    init_dim="init_date",
    lead_dim="lead_time",
    ensemble_dim="ensemble",
    confidence_interval=0.95,
    n_resamples=100,
):
    """Get the uncertainty bounds on zero correlation.

    Parameters
    ----------
    da : xarray.DataArray
    init_dim : str, default 'init_date'
        Name of the initial date dimension in da
    lead_dim : str, default 'lead_time'
        Name of the lead time dimension in da
    ensemble_dim : str, default 'ensemble'
        Name of the ensemble member dimension in da
    confidence_interval : float, default 0.95
        Confidence interval for the uncertainty bounds (between 0 and 1)
    n_resamples : int, default 100
        Number of confidence interval bootstrap resamples

    Returns
    -------
    bounds : xarray.DataArray
        Lower and upper bounds of the confidence interval

    Notes
    -----
    Performs bootstrapping via a simple loop.
    """
    assert 0 < confidence_interval < 1, "Confidence interval must be between 0 and 1"

    n_init_dates = len(da[init_dim])
    n_ensembles = len(da[ensemble_dim])
    da_stacked = da.stack(sample=(init_dim, lead_dim, ensemble_dim))

    null_correlations = []
    for _ in range(n_resamples):
        mean_corr = _random_mean_ensemble_correlation(
            da_stacked, n_init_dates, n_ensembles
        )
        null_correlations.append(mean_corr)
    null_correlations = xr.concat(null_correlations, "k")
    null_correlations = null_correlations.chunk({"k": -1})

    # Confidence interval bounds (half of the remaining probability on each side)
    q = (100 - confidence_interval * 100) * 0.5 / 100
    bounds = null_correlations.quantile([q, 1 - q], dim="k")

    # Ensure upper and lower bounds are not the same
    assert np.all(bounds.diff("quantile") > 0), "CI bounds are the same"
    bounds.attrs = {
        "long_name": "Confidence interval",
        "description": f"Correlation {confidence_interval * 100:g}% confidence interval ({n_resamples} resamples)",
    }
    return bounds


def _parse_command_line():
    """Parse the command line for input arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Data filename")
    parser.add_argument("--plot_outfile", type=str, default=None, help="Plot filename")
    parser.add_argument(
        "--confidence_interval",
        type=float,
        default=95,
        help="Confidence interval e.g., --confidence_interval 0.95",
    )
    parser.add_argument(
        "--n_resamples",
        type=int,
        default=100,
        help="Number of resamples for confidence interval test",
    )
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
    parser.add_argument(
        "--output_chunks",
        type=str,
        nargs="*",
        action=general_utils.store_dict,
        default={},
        help="Chunks for writing data to file (e.g. init_date=-1 lead_time=-1)",
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

    ds = run_tests(
        da_fcst,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
        ensemble_dim=args.ensemble_dim,
        confidence_interval=args.confidence_interval,
        n_resamples=args.n_resamples,
    )
    ds.attrs = ds_fcst.attrs  # Add forecast dataset attributes
    # Save correlation coefficients, confidence intervals and minimum lead
    infile_logs = {args.fcst_file: ds_fcst.attrs["history"]}
    ds.attrs["history"] = fileio.get_new_log(infile_logs=infile_logs)

    if args.output_chunks:
        ds = ds.chunk(args.output_chunks)
    if "zarr" in args.outfile:
        fileio.to_zarr(ds, args.outfile)
    else:
        ds.to_netcdf(args.outfile, compute=True)

    if args.plot_outfile and len(da_fcst.dims) <= 3:
        # Scatter plot of correlation vs lead (if there are no extra dimensions)
        point_plot(ds, args.plot_outfile, args.lead_dim, args.confidence_interval)

    elif args.plot_outfile:
        # Spatial plot of minimum correlation
        spatial_plot(
            ds,
            args.plot_outfile,
        )


if __name__ == "__main__":
    _main()
