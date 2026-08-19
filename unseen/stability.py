"""Functions and command line program for stability testing."""

import argparse
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme, _resampling
import seaborn as sns
import xarray as xr

from . import fileio
from . import eva
from . import time_utils


def plot_dist_by_lead(ax, sample_da, metric, units=None, lead_dim="lead_time"):
    """Plot distribution curves for each lead time.

    Parameters
    ----------
    ax : matplotlib Axes
        Plot axes
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension
    metric : str
        Metric name for plot title
    units : str, optional
        units for plot axis labels
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in sample_da
    """

    lead_times = np.unique(sample_da[lead_dim].values)
    colors = iter(plt.cm.BuPu(np.linspace(0, 1, len(lead_times))))
    for lead in lead_times:
        selection_da = sample_da.sel({lead_dim: lead})
        selection_da = selection_da.dropna("sample")
        color = next(colors)
        lead_df = pd.DataFrame(selection_da.values)
        n_values = len(selection_da)
        sns.kdeplot(
            lead_df[0],
            ax=ax,
            color=color,
            label=f"lead time: {lead} year ({n_values} samples)",
            linewidth=2.0,
        )
    ax.grid(True)
    ax.set_title(f"(a) {metric} distribution by lead time")
    units_label = units if units else sample_da.attrs["units"]
    ax.set_xlabel(units_label)
    ax.legend()


def plot_dist_by_time(ax, sample_da, metric, start_years, units=None):
    """Plot distribution curves for each time slice (e.g. decade).

    Parameters
    ----------
    ax : matplotlib Axes
        Plot axes
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension
    metric : str
        Metric name for plot title
    start_years : list
        Equally spaced list of start years
    units : str, optional
        units for plot axis labels
    """

    step = start_years[1] - start_years[0] - 1
    colors = iter(plt.cm.hot_r(np.linspace(0.3, 1, len(start_years))))
    for start_year in start_years:
        end_year = start_year + step
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-25"
        selection_da = time_utils.select_time_period(sample_da, [start_date, end_date])
        selection_da = selection_da.dropna("sample")
        color = next(colors)
        decade_df = pd.DataFrame(selection_da.values)
        n_values = len(selection_da)
        sns.kdeplot(
            decade_df[0],
            ax=ax,
            color=color,
            label=f"{start_year}-{end_year} ({n_values} samples)",
            linewidth=2.0,
        )
    ax.grid(True)
    ax.set_title(f"(c) {metric} distribution by year")
    units_label = units if units else sample_da.attrs["units"]
    ax.set_xlabel(units_label)
    ax.legend()


def return_curve(data, method, params=[], **kwargs):
    """Return x and y data for a return period curve.

    Parameters
    ----------
    data : xarray DataArray
    method : {'gev', 'empirical'}
        Fit a GEV or not to data
    params : list, default None
        shape, location and scale parameters (calculated if None)
    kwargs : dict, optional
        kwargs passed to eva.fit_gev
    """

    if method == "empirical":
        return_values = np.sort(data, axis=None)[::-1]
        return_periods = len(data) / np.arange(1.0, len(data) + 1.0)
    else:
        return_periods = np.logspace(0, 4, num=10000)
        probabilities = 1.0 / return_periods
        if params:
            shape, loc, scale = params
        else:
            shape, loc, scale = eva.fit_gev(data, **kwargs)
        return_values = genextreme.isf(probabilities, shape, loc, scale)

    return return_periods, return_values


def plot_return_by_lead(
    ax,
    sample_da,
    metric,
    method,
    uncertainty=False,
    units=None,
    ylim=None,
    lead_dim="lead_time",
):
    """Plot return period curves for each lead time.

    Parameters
    ----------
    ax : matplotlib Axes
        Plot axes
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension
    metric : str
        Metric name for plot title
    method : str {'empirical', 'gev'}
        Method for producing return period curve
    uncertainty: bool, default False
        Plot 95% confidence interval
    units : str, optional
        units for plot axis labels
    ylim : list, optional
        y axis limits for return curve [min, max]
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in sample_da
    """

    lead_times = np.unique(sample_da["lead_time"].values)
    colors = iter(plt.cm.BuPu(np.linspace(0, 1, len(lead_times))))
    for lead in lead_times:
        selection_da = sample_da.sel({"lead_time": lead})
        selection_da = selection_da.dropna("sample")
        return_periods, return_values = return_curve(
            selection_da, method, core_dim="sample"
        )
        n_values = len(selection_da)
        label = f"lead time {lead} ({n_values} samples)"
        color = next(colors)
        ax.plot(return_periods, return_values, label=label, color=color, linewidth=2.0)

    if uncertainty:
        random_return_values = []
        for i in range(1000):
            random_sample = np.random.choice(sample_da, n_values)
            return_periods, return_values = return_curve(
                random_sample, method, core_dim=None
            )
            random_return_values.append(return_values)
        random_return_values_stacked = np.stack(random_return_values)
        upper_ci = np.percentile(random_return_values_stacked, 97.5, axis=0)
        lower_ci = np.percentile(random_return_values_stacked, 2.5, axis=0)
        ax.fill_between(
            return_periods,
            upper_ci,
            lower_ci,
            label="95% confidence interval",
            color="0.5",
            alpha=0.3,
        )

    ax.set_title(f"(b) {metric} return period by lead time")
    ax.set_xscale("log")
    ax.grid(True, which="both")
    ax.set_xlabel("return period (years)")
    units_label = units if units else sample_da.attrs["units"]
    ax.set_ylabel(units_label)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ymin, ymax = ax.get_ylim()
        if ymin < 0:
            ymin = 0
        ax.set_ylim([ymin, ymax])
    ax.legend()


def plot_return_by_time(
    ax, sample_da, metric, start_years, method, uncertainty=False, units=None, ylim=None
):
    """Plot return period curves for each time slice (e.g. decade).

    Parameters
    ----------
    ax : matplotlib Axes
        Plot axes
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension
    metric : str
        Metric name for plot title
    start_years : list
        Equally spaced list of start years
    method : str {'empirical', 'gev'}
        Method for producing return period curve
    uncertainty: bool, default False
        Plot 95% confidence interval
    units : str, optional
        units for plot axis labels
    ylim : tuple, optional
        y axis limits for return curve [min, max]
    """

    step = start_years[1] - start_years[0] - 1
    colors = iter(plt.cm.hot_r(np.linspace(0.3, 1, len(start_years))))
    for start_year in start_years:
        end_year = start_year + step
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-25"
        selection_da = time_utils.select_time_period(sample_da, [start_date, end_date])
        selection_da = selection_da.dropna("sample")
        return_periods, return_values = return_curve(
            selection_da, method, core_dim="sample"
        )
        n_years = len(selection_da)
        label = f"{start_year}-{end_year} ({n_years} samples)"
        color = next(colors)
        ax.plot(return_periods, return_values, label=label, color=color, linewidth=2.0)

    if uncertainty:
        random_return_values = []
        for i in range(1000):
            random_sample = np.random.choice(sample_da, n_years)
            return_periods, return_values = return_curve(
                random_sample, method, core_dim=None
            )
            random_return_values.append(return_values)
        random_return_values_stacked = np.stack(random_return_values)
        upper_ci = np.percentile(random_return_values_stacked, 97.5, axis=0)
        lower_ci = np.percentile(random_return_values_stacked, 2.5, axis=0)
        ax.fill_between(
            return_periods,
            upper_ci,
            lower_ci,
            label="95% confidence interval",
            color="0.5",
            alpha=0.3,
        )

    ax.set_title(f"(d) {metric} return period by year")
    ax.set_xscale("log")
    ax.grid(True, which="both")
    ax.set_xlabel("return period (years)")
    units_label = units if units else sample_da.attrs["units"]
    ax.set_ylabel(units_label)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ymin, ymax = ax.get_ylim()
        if ymin < 0:
            ymin = 0
        ax.set_ylim([ymin, ymax])
    ax.legend()


def statistic_by_lead_confidence_interval(
    da,
    statistic,
    sample_size=None,
    n_resamples=9999,
    confidence_level=0.95,
    method="percentile",
    rng=np.random.default_rng(0),
    ensemble_dim="ensemble",
    init_dim="init_date",
    lead_dim="lead_time",
    **kwargs,
):
    """Estimate confidence intervals for a statistic using bootstrapping.

    Similar to `scipy.stats.bootstrap`, with optional sample size of resamples.

    Parameters
    ----------
    da : xarray.DataArray
        Data with dimensions (ensemble_dim, init_dim, lead_dim, ...)
    statistic : callable
        Function to calculate the statistic (e.g. np.median)
    sample_size : int, optional
        Size of the resample. If None, based on ensemble * init dim sizes
    n_resamples : int, default 1000
        Number of bootstrap resamples
    confidence_level : float, default 0.95
        Confidence level for the confidence intervals
    method : {"percentile", "bca"}, default "percentile"
        Method for calculating the confidence intervals
    rng : numpy.random.Generator, optional
    ensemble_dim : str, default "ensemble"
    init_dim : str, default "init_date"
    lead_dim : str, default "lead_time"
    kwargs : dict, optional
        Additional keyword arguments passed to the statistic function

    Returns
    -------
    ci : xarray.DataArray
        Confidence intervals stacked along dimension "bounds" (lower and upper).

    Notes
    -----
    The statistic function should accept a 1D array and return a scalar.
    If method is "bca", the statistic function should also accept kw `axis`.
    """

    def bootstrap_1d(
        data, statistic, sample_size, n_resamples, confidence_level, method
    ):
        """Bootstrap confidence interval for a statistic function."""

        # Resample the data
        theta_hat_b = []
        for _ in range(n_resamples):
            resample = rng.choice(data, size=sample_size, replace=True)
            theta_hat_b.append(statistic(resample))
        theta_hat_b = np.array(theta_hat_b)

        alpha = (1 - confidence_level) / 2

        if method == "percentile":
            interval = (alpha, 1 - alpha)
            percentile_func = np.percentile

        elif method.lower() == "bca":
            interval = _resampling._bca_interval(
                (data,),
                statistic,
                axis=-1,
                alpha=alpha,
                theta_hat_b=theta_hat_b,
                batch=None,
            )[:2]
            percentile_func = _resampling._percentile_along_axis

        ci_l = percentile_func(theta_hat_b, interval[0] * 100)
        ci_u = percentile_func(theta_hat_b, interval[1] * 100)
        return np.array([ci_l, ci_u])

    if sample_size is None:
        # Calculate the size of the resample (# samples per lead)
        sample_size = da[ensemble_dim].size * da[init_dim].size

    # Pass kwargs to the statistic function
    if kwargs:
        statistic = functools.partial(statistic, **kwargs)

    # Stack the data along the sample dimension
    da_stacked = da.stack(
        sample=[ensemble_dim, init_dim, lead_dim], create_index=False
    ).dropna("sample", how="all")

    # Ensure sample dim is on axis -1 for vectorization
    da_stacked = da_stacked.transpose(..., "sample")

    ci = xr.apply_ufunc(
        bootstrap_1d,
        da_stacked,
        input_core_dims=[["sample"]],
        output_core_dims=[["bounds"]],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(
            statistic=statistic,
            sample_size=sample_size,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method=method,
        ),
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"bounds": 2}},
    )

    # Assign the "bounds" dimension labels
    ci = ci.assign_coords(bounds=["lower", "upper"])
    ci.attrs["long_name"] = f"{confidence_level:%} Confidence Interval"
    return ci


def create_plot(
    da_fcst,
    metric,
    start_years,
    outfile=None,
    uncertainty=False,
    ylim=None,
    units=None,
    return_method="empirical",
    ensemble_dim="ensemble",
    init_dim="init_date",
    lead_dim="lead_time",
):
    """Create a stability assessment plot.

    Parameters
    ----------
    da_fcst : xarray Data Array
        Forecast data
    metric: str
        Metric name (for plot title)
    start_years : list
        Equally spaced list of start years
    outfile : str, default None
        Path for output image file
    uncertainty: bool, default False
        Plot the 95% confidence interval
    ylim : tuple, optional
        y axis limits for return curve plots [min, max]
    units : str, optional
        units for plot axis labels
    return_method : {'empirical', 'gev'}, default empirial
        Method for fitting the return period curve
    ensemble_dim : str, default ensemble
        Name of ensemble member dimension
    init_dim : str, default init_date
        Name of initial date dimension
    lead_dim : str, default lead_time
        Name of lead time dimension
    """

    dims = [ensemble_dim, init_dim, lead_dim]
    da_fcst_stacked = da_fcst.dropna(lead_dim).stack({"sample": dims})

    fig = plt.figure(figsize=[20, 17])
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    plot_dist_by_lead(ax1, da_fcst_stacked, metric, units=units, lead_dim=lead_dim)
    plot_return_by_lead(
        ax2,
        da_fcst_stacked,
        metric,
        return_method,
        uncertainty=uncertainty,
        ylim=ylim,
        units=units,
        lead_dim=lead_dim,
    )
    plot_dist_by_time(ax3, da_fcst_stacked, metric, start_years, units=units)
    plot_return_by_time(
        ax4,
        da_fcst_stacked,
        metric,
        start_years,
        return_method,
        units=units,
        uncertainty=uncertainty,
        ylim=ylim,
    )

    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()


def _parse_command_line():
    """Parse the command line for input agruments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "fcst_file", type=str, help="Forecast file containing metric of interest"
    )
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("metric", type=str, help="Metric name")

    parser.add_argument("--outfile", type=str, default=None, help="Output file name")
    parser.add_argument(
        "--start_years",
        type=int,
        nargs="*",
        help="Start years for time plots",
        required=True,
    )
    parser.add_argument(
        "--uncertainty",
        default=False,
        action="store_true",
        help="Plot the 95 percent confidence interval [default: False]",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=None,
        nargs=2,
        help="y axis limits for return curve plots [min, max]",
    )
    parser.add_argument(
        "--return_method",
        type=str,
        default="empirical",
        choices=("empirical", "gev"),
        help="Method for fitting the return period curve",
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
        "--units",
        type=str,
        default=None,
        help="Units label for the plot axes",
    )
    args = parser.parse_args()

    return args


def _main():
    """Run the command line program."""

    args = _parse_command_line()

    ds_fcst = fileio.open_dataset(args.fcst_file)
    da_fcst = ds_fcst[args.var]

    create_plot(
        da_fcst.compute(),
        args.metric,
        args.start_years,
        outfile=args.outfile,
        return_method=args.return_method,
        uncertainty=args.uncertainty,
        ylim=args.ylim,
        units=args.units,
        ensemble_dim=args.ensemble_dim,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
    )


if __name__ == "__main__":
    _main()
