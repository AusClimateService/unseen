"""Functions and command line program for moments testing."""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import fileio
from . import eva


logging.basicConfig(level=logging.INFO)


def calc_ci(data):
    """Calculate the 95% confidence interval"""

    lower_ci = np.percentile(np.array(data), 2.5, axis=0)
    upper_ci = np.percentile(np.array(data), 97.5, axis=0)

    return lower_ci, upper_ci


def calc_moments(sample_da, **kwargs):
    """Calculate all the moments for a given sample."""

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
    """Log the results"""

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
        units for plot axis labels
    ensemble_dim : str, default ensemble
        Name of ensemble member dimension
    init_dim : str, default init_date
        Name of initial date dimension
    lead_dim : str, default lead_time
        Name of lead time dimension
    infile_logs : dict, optional
        File names (keys) and history attributes (values) of input data files
        (For outfile image metadata)
    """

    sample_size = len(da_obs)
    moments_obs = calc_moments(da_obs)

    dims = [ensemble_dim, init_dim, lead_dim]
    da_fcst_stacked = da_fcst.dropna(lead_dim).stack({"sample": dims})
    moments_fcst = calc_moments(da_fcst_stacked)

    if da_bc_fcst is not None:
        da_bc_fcst_stacked = da_bc_fcst.dropna(lead_dim).stack({"sample": dims})

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
            gev_estimates=[moments_fcst["GEV location"], moments_fcst["GEV scale"]],
        )
        for moment in moments:
            bootstrap_values[moment].append(sample_moments[moment])

        if da_bc_fcst is not None:
            bc_random_sample = np.random.choice(da_bc_fcst_stacked, sample_size)
            bc_sample_moments = calc_moments(
                bc_random_sample,
                gev_estimates=[moments_fcst["GEV location"], moments_fcst["GEV scale"]],
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
    fig = plt.figure(figsize=[15, 22])
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
        ax.set_ylabel("count")
        ax.set_xlabel(units[moment])
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
    """Parse the command line for input agruments"""

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
        type=int,
        default=None,
        help="Minimum lead time",
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
    if args.min_lead is not None:
        da_fcst = da_fcst.where(ds_fcst[args.lead_dim] >= args.min_lead)

    ds_obs = fileio.open_dataset(args.obs_file)
    da_obs = ds_obs[args.var].dropna("time")

    if args.bias_file:
        ds_bc_fcst = fileio.open_dataset(args.bias_file)
        da_bc_fcst = ds_bc_fcst[args.var]
        if args.min_lead is not None:
            da_bc_fcst = da_bc_fcst.where(ds_bc_fcst[args.lead_dim] >= args.min_lead)
    else:
        da_bc_fcst = None

    if args.outfile:
        infile_logs = {args.obs_file: ds_obs.attrs["history"]}
        if args.bias_file:
            infile_logs[args.bias_file] = ds_bc_fcst.attrs["history"]
        else:
            infile_logs[args.fcst_file] = ds_fcst.attrs["history"]
    else:
        infile_logs = None

    create_plot(
        da_fcst,
        da_obs,
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
