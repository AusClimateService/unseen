"""Functions and command line program for moments testing."""

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy

from . import fileio
from . import indices


logging.basicConfig(level=logging.INFO)


def calc_ci(data):
    """Calculate the 95% confidence interval"""

    lower_ci = np.percentile(np.array(data), 2.5, axis=0)
    upper_ci = np.percentile(np.array(data), 97.5, axis=0)

    return lower_ci, upper_ci


def calc_moments(sample_da, gev_estimates=[]):
    """Calculate all the moments for a given sample."""

    mean = float(np.mean(sample_da))
    std = float(np.std(sample_da))
    skew = float(scipy.stats.skew(sample_da))
    kurtosis = float(scipy.stats.kurtosis(sample_da))
    gev_shape, gev_loc, gev_scale = indices.fit_gev(sample_da.values, user_estimates=gev_estimates)

    return mean, std, skew, kurtosis, gev_shape, gev_loc, gev_scale


def log_results(moments_obs, model_lower_cis, model_upper_cis):
    """Log the results"""

    mean_text = f"Obs = {mean_obs}, Model 95% CI ={mean_lower_ci} to {mean_upper_ci}"
    std_text = f"Obs = {std_obs}, Model 95% CI ={std_lower_ci} to {std_upper_ci}"
    skew_text = f"Obs = {skew_obs}, Model 95% CI ={skew_lower_ci} to {skew_upper_ci}"
    kurtosis_text = f"Obs = {kurtosis_obs}, Model 95% CI ={kurtosis_lower_ci} to {kurtosis_upper_ci}"
    gev_shape_text = f"Obs = {gev_shape_obs}, Model 95% CI ={gev_shape_lower_ci} to {gev_shape_upper_ci}"
    gev_loc_text = f"Obs = {gev_loc_obs}, Model 95% CI ={gev_loc_lower_ci} to {gev_loc_upper_ci}"
    gev_scale_text = f"Obs = {gev_scale_obs}, Model 95% CI ={gev_scale_lower_ci} to {gev_scale_upper_ci}"
    logging.info(f"Mean: {mean_text}")
    logging.info(f"Standard deviation: {std_text}")
    logging.info(f"Skewness: {skew_text}")
    logging.info(f"Kurtosis: {kurtosis_text}")
    logging.info(f"GEV shape: {gev_shape_text}")
    logging.info(f"GEV location: {gev_loc_text}")
    logging.info(f"GEV scale: {gev_scale_text}") 


def create_plot(
    fcst_file,
    obs_file,
    var,
    outfile=None,
    bc_fcst_file=None,
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
    outfile : str, optional
        Path for output image file
    bc_fcst_file : str, optional
        Forecast file containing bias corrected metric of interest
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
    moments_obs = calc_moments(da_obs)
    mean_obs, std_obs, skew_obs, kurtosis_obs = moments_obs[:4]
    gev_shape_obs, gev_loc_obs, gev_scale_obs = moments_obs[4:]

    ds_fcst = fileio.open_dataset(fcst_file)
    da_fcst = ds_fcst[var]
    if min_lead is not None:
        da_fcst = da_fcst.where(ds_fcst[lead_dim] >= min_lead)
    dims = [ensemble_dim, init_dim, lead_dim]
    da_fcst_stacked = da_fcst.dropna(lead_dim).stack({"sample": dims})
    moments_fcst = calc_moments(da_fcst_stacked)
    mean_fcst, std_fcst, skew_fcst, kurtosis_fcst = moments_fcst[:4]
    gev_shape_fcst, gev_loc_fcst, gev_scale_fcst = moments_fcst[4:]

    if bc_fcst_file:
        ds_bc_fcst = fileio.open_dataset(bc_fcst_file)
        da_bc_fcst = ds_bc_fcst[var]
        if min_lead is not None:
            da_bc_fcst = da_bc_fcst.where(ds_bc_fcst[lead_dim] >= min_lead)
        da_bc_fcst_stacked = da_bc_fcst.dropna(lead_dim).stack({"sample": dims})
        moments_bc_fcst = calc_moments(da_bc_fcst_stacked)
        mean_bc_fcst, std_bc_fcst, skew_bc_fcst, kurtosis_bc_fcst = moments_bc_fcst[:4]
        gev_shape_bc_fcst, gev_loc_bc_fcst, gev_scale_bc_fcst = moments_bc_fcst[4:]

    mean_values = []
    std_values = []
    skew_values = []
    kurtosis_values = []
    gev_shape_values = []
    gev_loc_values = []
    gev_scale_values = []
    if bc_fcst_file:
        bc_mean_values = []
        bc_std_values = []
        bc_skew_values = []
        bc_kurtosis_values = []
        bc_gev_shape_values = []
        bc_gev_loc_values = []
        bc_gev_scale_values = []
    for i in range(1000):
        random_sample = np.random.choice(da_fcst_stacked, sample_size)
        moments = calc_moments(random_sample, gev_estimates=[gev_loc_fcst, gev_scale_fcst])
        mean_values.append(moments[0])
        std_values.append(moments[1])
        skew_values.append(moments[2])
        kurtosis_values.append(moments[3])
        gev_shape_values.append(moments[4])
        gev_loc_values.append(moments[5])
        gev_scale_values.append(moments[6])
        if bc_fcst_file:
            bc_random_sample = np.random.choice(da_bc_fcst_stacked, sample_size)
            bc_moments = calc_moments(bc_random_sample, gev_estimates=[gev_loc_fcst, gev_scale_fcst])
            bc_mean_values.append(bc_moments[0])
            bc_std_values.append(bc_moments[1])
            bc_skew_values.append(bc_moments[2])
            bc_kurtosis_values.append(bc_moments[3])
            bc_gev_shape_values.append(bc_moments[4])
            bc_gev_loc_values.append(bc_moments[5])
            bc_gev_scale_values.append(bc_moments[6])

    mean_lower_ci, mean_upper_ci = calc_ci(mean_values)
    std_lower_ci, std_upper_ci = calc_ci(std_values)
    skew_lower_ci, skew_upper_ci = calc_ci(skew_values)
    kurtosis_lower_ci, kurtosis_upper_ci = calc_ci(kurtosis_values)
    gev_shape_lower_ci, gev_shape_upper_ci = calc_ci(gev_shape_values)
    gev_loc_lower_ci, gev_loc_upper_ci = calc_ci(gev_loc_values)
    gev_scale_lower_ci, gev_scale_upper_ci = calc_ci(gev_scale_values)
    if bc_fcst_file:
        bc_mean_lower_ci, bc_mean_upper_ci = calc_ci(bc_mean_values)
        bc_std_lower_ci, bc_std_upper_ci = calc_ci(bc_std_values)
        bc_skew_lower_ci, bc_skew_upper_ci = calc_ci(bc_skew_values)
        bc_kurtosis_lower_ci, bc_kurtosis_upper_ci = calc_ci(bc_kurtosis_values)
        bc_gev_shape_lower_ci, bc_gev_shape_upper_ci = calc_ci(bc_gev_shape_values)
        bc_gev_loc_lower_ci, bc_gev_loc_upper_ci = calc_ci(bc_gev_loc_values)
        bc_gev_scale_lower_ci, bc_gev_scale_upper_ci = calc_ci(bc_gev_scale_values)

    fig = plt.figure(figsize=[15, 20])
    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)

    ax1.hist(mean_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax1.set_title("(a) mean")
    ax1.axvline(mean_lower_ci, color="tab:blue", linestyle="--")
    ax1.axvline(mean_upper_ci, color="tab:blue", linestyle="--")
    ax1.axvline(mean_obs, linewidth=2.0, color="tab:gray")
    ax1.axvline(mean_fcst, linewidth=2.0, color="tab:blue")
    ax1.set_ylabel("count")
    if bc_fcst_file:
        ax1.hist(bc_mean_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax1.axvline(bc_mean_lower_ci, color="tab:orange", linestyle="--")
        ax1.axvline(bc_mean_upper_ci, color="tab:orange", linestyle="--")
        ax1.axvline(bc_mean_fcst, linewidth=2.0, color="tab:orange")

    ax2.hist(std_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax2.set_title("(b) standard deviation")
    ax2.axvline(std_lower_ci, color="tab:blue", linestyle="--")
    ax2.axvline(std_upper_ci, color="tab:blue", linestyle="--")
    ax2.axvline(std_obs, linewidth=2.0, color="tab:gray")
    ax2.axvline(std_fcst, linewidth=2.0, color="tab:blue")
    ax2.set_ylabel("count")
    if bc_fcst_file:
        ax2.hist(bc_std_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax2.axvline(bc_std_lower_ci, color="tab:orange", linestyle="--")
        ax2.axvline(bc_std_upper_ci, color="tab:orange", linestyle="--")
        ax2.axvline(bc_std_fcst, linewidth=2.0, color="tab:orange")

    ax3.hist(skew_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax3.set_title("(c) skewness")
    ax3.set_ylabel("count")
    ax3.axvline(skew_lower_ci, color="tab:blue", linestyle="--")
    ax3.axvline(skew_upper_ci, color="tab:blue", linestyle="--")
    ax3.axvline(skew_obs, linewidth=2.0, color="tab:gray")
    ax3.axvline(skew_fcst, linewidth=2.0, color="tab:blue")
    if bc_fcst_file:
        ax3.hist(bc_skew_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax3.axvline(bc_skew_lower_ci, color="tab:orange", linestyle="--")
        ax3.axvline(bc_skew_upper_ci, color="tab:orange", linestyle="--")
        ax3.axvline(bc_skew_fcst, linewidth=2.0, color="tab:orange")

    ax4.hist(kurtosis_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax4.set_title("(d) kurtosis")
    ax4.set_ylabel("count")
    ax4.axvline(kurtosis_lower_ci, color="tab:blue", linestyle="--")
    ax4.axvline(kurtosis_upper_ci, color="tab:blue", linestyle="--")
    ax4.axvline(kurtosis_obs, linewidth=2.0, color="tab:gray")
    ax4.axvline(kurtosis_fcst, linewidth=2.0, color="tab:blue")
    if bc_fcst_file:
        ax4.hist(bc_kurtosis_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax4.axvline(bc_kurtosis_lower_ci, color="tab:orange", linestyle="--")
        ax4.axvline(bc_kurtosis_upper_ci, color="tab:orange", linestyle="--")
        ax4.axvline(bc_kurtosis_fcst, linewidth=2.0, color="tab:orange")

    ax5.hist(gev_shape_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax5.set_title("(e) GEV shape")
    ax5.set_ylabel("count")
    ax5.axvline(gev_shape_lower_ci, color="tab:blue", linestyle="--")
    ax5.axvline(gev_shape_upper_ci, color="tab:blue", linestyle="--")
    ax5.axvline(gev_shape_obs, linewidth=2.0, color="tab:gray")
    ax5.axvline(gev_shape_fcst, linewidth=2.0, color="tab:blue")
    if bc_fcst_file:
        ax5.hist(bc_gev_shape_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax5.axvline(bc_gev_shape_lower_ci, color="tab:orange", linestyle="--")
        ax5.axvline(bc_gev_shape_upper_ci, color="tab:orange", linestyle="--")
        ax5.axvline(bc_gev_shape_fcst, linewidth=2.0, color="tab:orange")

    ax6.hist(gev_loc_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax6.set_title("(f) GEV location")
    ax6.set_ylabel("count")
    ax6.axvline(gev_loc_lower_ci, color="tab:blue", linestyle="--")
    ax6.axvline(gev_loc_upper_ci, color="tab:blue", linestyle="--")
    ax6.axvline(gev_loc_obs, linewidth=2.0, color="tab:gray")
    ax6.axvline(gev_loc_fcst, linewidth=2.0, color="tab:blue")
    if bc_fcst_file:
        ax6.hist(bc_gev_loc_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax6.axvline(bc_gev_loc_lower_ci, color="tab:orange", linestyle="--")
        ax6.axvline(bc_gev_loc_upper_ci, color="tab:orange", linestyle="--")
        ax6.axvline(bc_gev_loc_fcst, linewidth=2.0, color="tab:orange")

    ax7.hist(gev_scale_values, rwidth=0.8, color="tab:blue", alpha=0.7)
    ax7.set_title("(g) GEV scale")
    ax7.set_ylabel("count")
    ax7.axvline(gev_scale_lower_ci, color="tab:blue", linestyle="--")
    ax7.axvline(gev_scale_upper_ci, color="tab:blue", linestyle="--")
    ax7.axvline(gev_scale_obs, linewidth=2.0, color="tab:gray")
    ax7.axvline(gev_scale_fcst, linewidth=2.0, color="tab:blue")
    if bc_fcst_file:
        ax7.hist(bc_gev_scale_values, rwidth=0.8, color="tab:orange", alpha=0.7)
        ax7.axvline(bc_gev_scale_lower_ci, color="tab:orange", linestyle="--")
        ax7.axvline(bc_gev_scale_upper_ci, color="tab:orange", linestyle="--")
        ax7.axvline(bc_gev_scale_fcst, linewidth=2.0, color="tab:orange")

    log_results(moments_obs, 

    if outfile:
        infile_logs = {
            fcst_file: ds_fcst.attrs["history"],
            obs_file: ds_obs.attrs["history"],
        }
        command_history = fileio.get_new_log(infile_logs=infile_logs)
        metadata = {
            "mean": mean_text,
            "standard deviation": std_text,
            "skewness": skew_text,
            "kurtosis": kurtosis_text,
            "GEV shape": gev_shape_text,
            "GEV location": gev_loc_text,
            "GEV scale": gev_scale_text,
            "history": command_history,
        }
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
        bc_fcst_file=args.bias_file,
        min_lead=args.min_lead,
        ensemble_dim=args.ensemble_dim,
        init_dim=args.init_dim,
        lead_dim=args.lead_dim,
    )


if __name__ == "__main__":
    _main()
