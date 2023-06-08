"""Functions and command line program for stability testing."""

import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev

from . import fileio
from . import indices
from . import time_utils


def plot_dist_by_lead(sample_da, outfile=None, lead_dim="lead_time"):
    """Plot distribution curves for each lead time.

    Parameters
    ----------
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension    
    outfile : str, default None
        Path for output image file
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in sample_da
    """

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot()
    lead_times = np.unique(sample_da[lead_dim].values)
    colors = iter(plt.cm.BuPu(np.linspace(0.3, 1, len(lead_times))))

    for lead in lead_times:
        selection_da = sample_da.sel({lead_dim: lead})
        selection_da = selection_da.dropna('sample')
        color = next(colors)
        lead_df = pd.DataFrame(selection_da.values)
        n_values = len(selection_da)
        sns.kdeplot(lead_df[0], ax=ax, color=color, label=f"lead time {lead} ({n_values} samples)")

    ax.grid(True)    
    ax.set_xlabel(sample_da.attrs["units"])
    ax.legend()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()


def plot_dist_by_time(sample_da, start_years, outfile=None):
    """Plot distribution curves for each time slice (e.g. decade).

    Parameters
    ----------
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension
    start_years : list
        Equally spaced list of start years    
    outfile : str, default None
        Path for output image file
    """

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot()
    step = start_years[1] - start_years[0] - 1    
    colors = iter(plt.cm.hot_r(np.linspace(0.3, 1, len(start_years))))

    for start_year in start_years:
        end_year = start_year + step
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-25'
        selection_da = time_utils.select_time_period(sample_da, [start_date, end_date])
        selection_da = selection_da.dropna('sample')
        color = next(colors)
        decade_df = pd.DataFrame(selection_da.values)
        n_values = len(selection_da) 
        sns.kdeplot(decade_df[0], ax=ax, color=color, label=f'{start_year}-{end_year} ({n_values} samples)')

    ax.grid(True)
    ax.set_xlabel(sample_da.attrs["units"])
    ax.legend()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()
    

def return_curve(data, method):
    """Return x and y data for a return period curve.
    
    Parameters
    ----------
    data : xarray DataArray
    method : {'gev', 'empirical'}
        Fit a GEV or not to data
    """
    
    if method == 'gev':
        return_periods = np.logspace(0, 4, num=10000)
        probabilities = 1. / return_periods
        shape, loc, scale = indices.fit_gev(data, generate_estimates=True)
        return_values = gev.isf(probabilities, shape, loc, scale)
    elif method == 'empirical':
        return_values = np.sort(data, axis=None)[::-1]
        return_periods = len(data) / np.arange(1.0, len(data) + 1.0)
        
    return return_periods, return_values


def plot_return(data, method, outfile=None):
    """Plot a single return period curve.
    
    Parameters
    ----------
    data : xarray DataArray
    method : {'gev', 'empirical'}
        Fit a GEV or not to data 
    """
    
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot()
    return_periods, return_values = return_curve(data, method)
    ax.plot(return_periods, return_values)
    ax.set_xscale('log')
    ax.set_xlabel('return period (years)')
    ax.set_ylabel(data.attrs['units'])
    ax.grid()
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', facecolor='white', dpi=dpi)
        print(outfile)
    else:
        plt.show()


def plot_return_by_lead(sample_da, outfile=None, uncertainty=False, lead_dim="lead_time"):
    """Plot return period curves for each lead time.

    Parameters
    ----------
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension    
    outfile : str, default None
        Path for output image file
    uncertainty: bool, default False
        Plot 95% confidence interval
    lead_dim: str, default 'lead_time'
        Name of the lead time dimension in sample_da
    """

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot()
    lead_times = np.unique(sample_da['lead_time'].values)
    colors = iter(plt.cm.BuPu(np.linspace(0.3, 1, len(lead_times))))

    for lead in lead_times:
        selection_da = sample_da.sel({'lead_time': lead})
        selection_da = selection_da.dropna('sample')
        return_periods, return_values = return_curve(selection_da, method='empirical')
        n_values = len(selection_da)
        label = f'lead time {lead} ({n_values} samples)'
        color = next(colors)
        ax.plot(return_periods, return_values, label=label, color=color)
    
    if uncertainty:
        random_return_values = []
        for i in range(1000):
            random_sample = np.random.choice(sample_da, n_values)
            return_periods, return_values = return_curve(random_sample, method='empirical')
            random_return_values.append(return_values)
        random_return_values_stacked = np.stack(random_return_values)
        upper_ci = np.percentile(random_return_values_stacked, 97.5, axis=0)
        lower_ci = np.percentile(random_return_values_stacked, 2.5, axis=0)
        ax.fill_between(return_periods, upper_ci, lower_ci, label='95% confidence interval', color='0.5', alpha=0.1)
    
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('return period (years)')
    ax.set_ylabel(sample_da.attrs["units"])
    ax.legend()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()


def plot_return_by_time(sample_da, start_years, method, outfile=None, uncertainty=False):
    """Plot return period curves for each time slice (e.g. decade).
    
    Parameters
    ----------
    sample_da : xarray DataArray
        Stacked forecast array with a sample dimension 
    start_years : list
        Equally spaced list of start years
    method : str {'empirical', 'gev'}
        Method for producing return period curve   
    outfile : str, default None
        Path for output image file
    uncertainty: bool, default False
        Plot 95% confidence interval
    """

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot()
    step = start_years[1] - start_years[0] - 1
    colors = iter(plt.cm.hot_r(np.linspace(0.3, 1, len(start_years))))

    for start_year in start_years:
        end_year = start_year + step
        start_date = f'{start_year}-01-01'
        end_date = f'{end_year}-12-25'
        selection_da = time_utils.select_time_period(sample_da, [start_date, end_date])
        selection_da = selection_da.dropna('sample')
        return_periods, return_values = return_curve(selection_da, method)
        n_years = len(selection_da)
        label = f'{start_year}-{end_year} ({n_years} samples)'
        color = next(colors)
        ax.plot(return_periods, return_values, label=label, color=color)
    
    if uncertainty:
        random_return_values = []
        for i in range(1000):
            random_sample = np.random.choice(sample_da, n_years)
            return_periods, return_values = return_curve(random_sample, method)
            random_return_values.append(return_values)
        random_return_values_stacked = np.stack(random_return_values)
        upper_ci = np.percentile(random_return_values_stacked, 97.5, axis=0)
        lower_ci = np.percentile(random_return_values_stacked, 2.5, axis=0)
        ax.fill_between(return_periods, upper_ci, lower_ci, label='95% confidence interval', color='0.5', alpha=0.2)
    
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('return period (years)')
    ax.set_ylabel(sample_da.attrs["units"])
    ax.legend()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
    else:
        plt.show()


def _parse_command_line():
    """Parse the command line for input agruments"""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("fcst_file", type=str, help="Forecast file containing metric of interest")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Output file")
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
        type=int,
        default=None,
        help="Minimum lead time",
    )
    args = parser.parse_args()

    return args


def _main():
    """Run the command line program."""

    args = _parse_command_line()

    ds_fcst = fileio.open_dataset(args.fcst_file)
    da_fcst = ds_fcst[args.var]
    if min_lead is not None:
        da_fcst = da_fcst.where(ds_fcst[args.lead_dim] >= args.min_lead)
    dims = [args.ensemble_dim, args.init_dim, args.lead_dim]
    da_fcst_stacked = da_fcst.dropna(args.lead_dim).stack({'sample': dims})

    plot_dist_by_lead(da_fcst_stacked, model_name)
    plot_return_by_lead(da_fcst_stacked, model_name, uncertainty=True)
    plot_dist_by_time(model_da_stacked, model_name)
    plot_return_by_time(model_da_stacked, model_name, method='empirical', uncertainty=True)


if __name__ == "__main__":
    _main()
