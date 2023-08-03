"""Functions and command line program for moments testing."""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy

from . import fileio


def calc_ci(data):
    """Calculate the 95% confidence interval"""

    lower_ci = np.percentile(np.array(data), 2.5, axis=0)
    upper_ci = np.percentile(np.array(data), 97.5, axis=0)

    return lower_ci, upper_ci


def create_plot(
    fcst_file,
    obs_file,
    var,
    outfile=None,
    min_lead=None,
    ensemble_dim="ensemble",
    init_dim="init_date",
    lead_dim="lead_time",
):
    """Create a stability assessment plot.

    Parameters
    ----------
    fcst_file : str
        Forecast file containing metric of interest
    obs_file : str
        Observations file containing metric of interest
    var : str
        Variable name (in fcst_file)
    outfile : str, default None
        Path for output image file
    min_lead : int, optional
        Minimum lead time
    ensemble_dim : str, default ensemble
        Name of ensemble member dimension
    init_dim : str, default init_date
        Name of initial date dimension
    lead_dim : str, default lead_time
        Name of lead time dimension
    """

    ds_obs = fileio.open_dataset(obs_file)
    da_obs = ds_obs[var].dropna("time")
    sample_size = len(da_obs)
    mean_obs = float(np.mean(da_obs))
    std_obs = float(np.std(da_obs))
    skewness_obs = float(scipy.stats.skew(da_obs))
    kurtosis_obs = float(scipy.stats.kurtosis(da_obs))

    ds_fcst = fileio.open_dataset(fcst_file)
    da_fcst = ds_fcst[var]
    if min_lead is not None:
        da_fcst = da_fcst.where(ds_fcst[lead_dim] >= min_lead)
    dims = [ensemble_dim, init_dim, lead_dim]
    da_fcst_stacked = da_fcst.dropna(lead_dim).stack({"sample": dims})

    mean_values = []
    std_values = []
    skewness_values = []
    kurtosis_values = []
    for i in range(1000):
        random_sample = np.random.choice(da_fcst_stacked, sample_size)
        mean = float(np.mean(random_sample))
        std = float(np.std(random_sample))
        skewness = float(scipy.stats.skew(random_sample))
        kurtosis = float(scipy.stats.kurtosis(random_sample))
        mean_values.append(mean)
        std_values.append(std)
        skewness_values.append(skewness)
        kurtosis_values.append(kurtosis)

    mean_lower_ci, mean_upper_ci = calc_ci(mean_values)
    std_lower_ci, std_upper_ci = calc_ci(std_values)
    skewness_lower_ci, skewness_upper_ci = calc_ci(skewness_values)
    kurtosis_lower_ci, kurtosis_upper_ci = calc_ci(kurtosis_values)

    fig = plt.figure(figsize=[15, 12])
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.hist(mean_values, rwidth=0.8, color="0.5")
    ax1.set_title("(a) mean")
    ax1.axvline(mean_lower_ci, color="0.2", linestyle="--")
    ax1.axvline(mean_upper_ci, color="0.2", linestyle="--")
    ax1.axvline(mean_obs)
    ax1.set_ylabel("count")

    ax2.hist(std_values, rwidth=0.8, color="0.5")
    ax2.set_title("(b) standard deviation")
    ax2.axvline(std_lower_ci, color="0.2", linestyle="--")
    ax2.axvline(std_upper_ci, color="0.2", linestyle="--")
    ax2.axvline(std_obs)
    ax2.set_ylabel("count")

    ax3.hist(skewness_values, rwidth=0.8, color="0.5")
    ax3.set_title("(c) skewness")
    ax3.set_ylabel("count")
    ax3.axvline(skewness_lower_ci, color="0.2", linestyle="--")
    ax3.axvline(skewness_upper_ci, color="0.2", linestyle="--")
    ax3.axvline(skewness_obs)

    ax4.hist(kurtosis_values, rwidth=0.8, color="0.5")
    ax4.set_title("(d) kurtosis")
    ax4.set_ylabel("count")
    ax4.axvline(kurtosis_lower_ci, color="0.2", linestyle="--")
    ax4.axvline(kurtosis_upper_ci, color="0.2", linestyle="--")
    ax4.axvline(kurtosis_obs)

    if outfile:
        plt.savefig(outfile, bbox_inches="tight", facecolor="white", dpi=200)
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

    create_plot(
        args.fcst_file,
        args.obs_file,
        args.var,
        outfile=args.outfile,
        min_lead=args.min_lead,
        ensemble_dim=args.ensemble_dim,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
    )


if __name__ == "__main__":
    _main()
