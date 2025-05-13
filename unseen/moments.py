"""Functions and command line program for moments testing."""

import argparse
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import eva
from . import fileio
from . import general_utils

logging.basicConfig(level=logging.INFO)
mpl.rcParams["axes.labelsize"] = "x-large"
mpl.rcParams["axes.titlesize"] = "xx-large"
mpl.rcParams["xtick.labelsize"] = "x-large"
mpl.rcParams["ytick.labelsize"] = "x-large"
mpl.rcParams["legend.fontsize"] = "x-large"


def calc_ci(data):
    """Calculate the 95% confidence interval

    Parameters
    ----------
    data : list
        List of data values

    Returns
    -------
    lower_ci, upper_ci : float
        Lower and upper confidence interval bounds
    """

    lower_ci = np.percentile(np.array(data), 2.5, axis=0)
    upper_ci = np.percentile(np.array(data), 97.5, axis=0)

    return lower_ci, upper_ci


def calc_moments(sample_da, **kwargs):
    """Calculate all the moments for a given sample.

    Parameters
    ----------
    sample_da : xarray.DataArray
        Sample data array
    kwargs : dict
        Keyword arguments for the GEV fit

    Returns
    -------
    moments : dict
        Dictionary of moments
    """

    moments = {}
    moments["mean"] = float(np.mean(sample_da))
    moments["standard deviation"] = float(np.std(sample_da))
    moments["skew"] = float(scipy.stats.skew(sample_da))
    moments["kurtosis"] = float(scipy.stats.kurtosis(sample_da))
    gev_shape, gev_loc, gev_scale = eva.fit_gev(sample_da, **kwargs)
    moments["GEV shape"] = gev_shape
    moments["GEV location"] = gev_loc
    moments["GEV scale"] = gev_scale

    return moments


def log_results(moments_obs, model_lower_cis, model_upper_cis, bias_corrected=False):
    """Log the moments test results.

    Parameters
    ----------
    moments_obs : dict
        Dictionary of observed moments
    model_lower_cis : dict
        Dictionary of model lower confidence intervals
    model_upper_cis : dict
        Dictionary of model upper confidence intervals
    bias_corrected : bool, default False
        Flag for bias corrected model

    Returns
    -------
    metadata : dict
        Dictionary of logged metadata
    """

    if bias_corrected:
        text_insert = "Bias corrected model"
    else:
        text_insert = "Model"

    metadata = {}
    for moment, obs_value in moments_obs.items():
        lower_ci = model_lower_cis[moment]
        upper_ci = model_upper_cis[moment]
        text = f"Obs = {obs_value}, {text_insert} 95% CI ={lower_ci} to {upper_ci}"
        if bias_corrected:
            metadata["bias corrected " + moment] = text
        else:
            metadata[moment] = text
        logging.info(f"{moment}: {text}")

    return metadata


def create_plot(
    da_fcst,
    da_obs,
    da_bc_fcst=None,
    outfile=None,
    units=None,
    ensemble_dim="ensemble",
    init_dim="init_date",
    lead_dim="lead_time",
    infile_logs=None,
    **kwargs,
):
    """Create a stability assessment plot.

    Parameters
    ----------
    da_fcst : xarray Data Array
        Forecast data for metric of interest
    da_obs : xarray Data Array
        Observations data for metric of interest
    da_bc_fcst : xarray Data Array, optional
        Bias corrected forecast data for metric of interest
    outfile : str, optional
        Path for output image file
    units : str, optional
        Units for plot axis labels
    ensemble_dim : str, default 'ensemble'
        Name of ensemble member dimension
    init_dim : str, default 'init_date'
        Name of initial date dimension
    lead_dim : str, default 'lead_time'
        Name of lead time dimension
    infile_logs : dict, optional
        File names (keys) and history attributes (values) of input data files
        (For outfile image metadata)
    """

    sample_size = len(da_obs)
    moments_obs = calc_moments(da_obs)

    dims = [ensemble_dim, init_dim, lead_dim]
    da_fcst_stacked = da_fcst.dropna(lead_dim).stack({"sample": dims})
    moments_fcst = calc_moments(da_fcst_stacked, core_dim="sample", **kwargs)

    if da_bc_fcst is not None:
        da_bc_fcst_stacked = da_bc_fcst.dropna(lead_dim).stack({"sample": dims})
        moments_bc_fcst = calc_moments(da_bc_fcst_stacked, core_dim="sample", **kwargs)

    bootstrap_values = {}
    bootstrap_lower_ci = {}
    bootstrap_upper_ci = {}
    if da_bc_fcst is not None:
        bc_bootstrap_values = {}
        bc_bootstrap_lower_ci = {}
        bc_bootstrap_upper_ci = {}
    moments = [
        "mean",
        "standard deviation",
        "skew",
        "kurtosis",
        "GEV shape",
        "GEV location",
        "GEV scale",
    ]
    for moment in moments:
        bootstrap_values[moment] = []
        bootstrap_lower_ci[moment] = []
        bootstrap_upper_ci[moment] = []
        if da_bc_fcst is not None:
            bc_bootstrap_values[moment] = []
            bc_bootstrap_lower_ci[moment] = []
            bc_bootstrap_upper_ci[moment] = []

    for i in range(1000):
        random_sample = np.random.choice(da_fcst_stacked, sample_size)
        sample_moments = calc_moments(
            random_sample,
            fitstart=[
                moments_fcst["GEV shape"],
                moments_fcst["GEV location"],
                moments_fcst["GEV scale"],
            ],
        )
        for moment in moments:
            bootstrap_values[moment].append(sample_moments[moment])

        if da_bc_fcst is not None:
            bc_random_sample = np.random.choice(da_bc_fcst_stacked, sample_size)
            bc_sample_moments = calc_moments(
                bc_random_sample,
                fitstart=[
                    moments_fcst["GEV shape"],
                    moments_fcst["GEV location"],
                    moments_fcst["GEV scale"],
                ],
            )
            for moment in moments:
                bc_bootstrap_values[moment].append(bc_sample_moments[moment])

    for moment in moments:
        lower_ci, upper_ci = calc_ci(bootstrap_values[moment])
        bootstrap_lower_ci[moment] = lower_ci
        bootstrap_upper_ci[moment] = upper_ci
        if da_bc_fcst is not None:
            bc_lower_ci, bc_upper_ci = calc_ci(bc_bootstrap_values[moment])
            bc_bootstrap_lower_ci[moment] = bc_lower_ci
            bc_bootstrap_upper_ci[moment] = bc_upper_ci

    letters = "abcdefg"
    units_label = units if units else da_fcst.attrs["units"]
    units = {
        "mean": f"mean ({units_label})",
        "standard deviation": f"standard deviation ({units_label})",
        "skew": "skew",
        "kurtosis": "kurtosis",
        "GEV shape": "shape parameter",
        "GEV scale": "scale parameter",
        "GEV location": "location parameter",
    }
    fig = plt.figure(figsize=[15, 28])
    for plotnum, moment in enumerate(moments):
        ax = fig.add_subplot(4, 2, plotnum + 1)
        ax.hist(
            bootstrap_values[moment],
            rwidth=0.8,
            color="tab:blue",
            alpha=0.7,
            label="model",
        )
        ax.axvline(
            bootstrap_lower_ci[moment], color="tab:blue", linestyle="--", linewidth=3.0
        )
        ax.axvline(
            bootstrap_upper_ci[moment], color="tab:blue", linestyle="--", linewidth=3.0
        )
        ax.axvline(
            moments_obs[moment], linewidth=4.0, color="tab:gray", label="observations"
        )
        ax.axvline(moments_fcst[moment], linewidth=4.0, color="tab:blue")
        if da_bc_fcst is not None:
            ax.hist(
                bc_bootstrap_values[moment],
                rwidth=0.8,
                color="tab:orange",
                alpha=0.7,
                label="model (corrected)",
            )
            ax.axvline(
                bc_bootstrap_lower_ci[moment],
                color="tab:orange",
                linestyle="--",
                linewidth=3.0,
            )
            ax.axvline(
                bc_bootstrap_upper_ci[moment],
                color="tab:orange",
                linestyle="--",
                linewidth=3.0,
            )
            ax.axvline(moments_bc_fcst[moment], linewidth=4.0, color="tab:orange")
        ax.set_ylabel("count", fontsize="large")
        ax.set_xlabel(units[moment], fontsize="large")
        letter = letters[plotnum]
        ax.set_title(f"({letter}) {moment}")
        if letter == "a":
            ax.legend()

    metadata = log_results(moments_obs, bootstrap_lower_ci, bootstrap_upper_ci)
    if da_bc_fcst is not None:
        bc_metadata = log_results(
            moments_obs,
            bc_bootstrap_lower_ci,
            bc_bootstrap_upper_ci,
            bias_corrected=True,
        )
        metadata = metadata | bc_metadata

    if outfile:
        metadata["history"] = fileio.get_new_log(infile_logs=infile_logs)
        plt.savefig(
            outfile,
            bbox_inches="tight",
            facecolor="white",
            dpi=200,
            metadata=metadata,
        )
    else:
        plt.show()


def _parse_command_line():
    """Parse the command line for input arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")

    parser.add_argument("--outfile", type=str, default=None, help="Output file name")
    parser.add_argument("--bias_file", type=str, help="Bias corrected forecast file")
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

    # Mask lead times below min_lead
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
        da_fcst = da_fcst.where(da_fcst[args.lead_dim] >= min_lead)

    ds_obs = fileio.open_dataset(args.obs_file)
    da_obs = ds_obs[args.var].dropna("time")

    if args.bias_file:
        ds_bc_fcst = fileio.open_dataset(args.bias_file)
        da_bc_fcst = ds_bc_fcst[args.var]
        if args.min_lead is not None:
            da_bc_fcst = da_bc_fcst.where(ds_bc_fcst[args.lead_dim] >= min_lead)
        da_bc_fcst = da_bc_fcst.compute()
    else:
        da_bc_fcst = None

    if args.outfile:
        infile_logs = {args.obs_file: ds_obs.attrs["history"]}
        if args.bias_file:
            infile_logs[args.bias_file] = ds_bc_fcst.attrs["history"]
        else:
            infile_logs[args.fcst_file] = ds_fcst.attrs["history"]
        if isinstance(args.min_lead, str):
            infile_logs[args.min_lead] = ds_min_lead.attrs["history"]
    else:
        infile_logs = None

    create_plot(
        da_fcst.compute(),
        da_obs.compute(),
        da_bc_fcst=da_bc_fcst,
        outfile=args.outfile,
        units=args.units,
        ensemble_dim=args.ensemble_dim,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
        infile_logs=infile_logs,
    )


if __name__ == "__main__":
    _main()
