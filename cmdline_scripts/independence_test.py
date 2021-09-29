"""Command line program for similarity testing."""

import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-1])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import pdb
import argparse

import xarray as xr

import fileio
import general_utils


def remove_ensemble_mean_trend(da, dim='init_date'):
    """Remove ensemble mean trend along given dimension
    
    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate and remove trend
    """
    
    ensmean_trend = da.mean('ensemble').polyfit(dim=dim, deg=1)
    ensmean_trend_line = xr.polyval(da[dim], ensmean_trend['polyfit_coefficients'])
    ensmean_trend_line_anomaly = ensmean_trend_line - ensmean_trend_line.isel({dim:0})
    da_detrended = da - ensmean_trend_line_anomaly
                            
    return da_detrended


def mean_ensemble_correlation(da, dim='init_date'):
    """Mean correlation between all ensemble members.
    
    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate correlation
    """

    n_ensemble_members = len(da['ensemble'])
    combinations = np.array(list(itertools.combinations(range(n_ensemble_members), 2)))
    
    new_ensemble_coord = {'ensemble': range(combinations.shape[0])}                 
    e1 = da.isel(ensemble=combinations[:,0]).assign_coords(new_ensemble_coord)
    e2 = da.isel(ensemble=combinations[:,1]).assign_coords(new_ensemble_coord)

    e1 = e1.chunk({dim: -1})
    e2 = e2.chunk({dim: -1})
                            
    corr_combinations = xs.spearman_r(e1, e2, dim=dim, skipna=True)
    mean_corr = corr_combinations.mean('ensemble')
    
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
    #random_indexes.sort()
    ds_random_sample = ds.isel({sample_dim: random_indexes})

    return ds_random_sample


def random_mean_ensemble_correlation(ds, n_init_dates, n_ensembles):
    """Mean correlation between a random selection of samples"""
    
    sample_size = n_init_dates * n_ensembles
    ds_random_sample = random_sample(ds, 'sample', sample_size)
    index = pd.MultiIndex.from_product([range(n_init_dates), range(n_ensembles)],
                                       names=['init_date', 'ensemble'])
    ds_random_sample = ds_random_sample.assign_coords({'sample': index}).unstack()
    mean_corr = mean_ensemble_correlation(ds_random_sample, dim='init_date')
    
    return mean_corr


def _main(args):
    """Run the command line program."""

    if args.dask_config:
        client = dask_setup.launch_client(args.dask_config)
        print(client)

    ds_fcst = fileio.open_file(args.fcst_file, variables=[args.var])
    da_fcst = ds_fcst[args.var]

    ds_obs = fileio.open_file(args.obs_file, variables=[args.var])
    da_obs = ds_obs[args.var]
    if args.reference_time_period:
        time_slice = general_utils.date_pair_to_time_slice(args.reference_time_period)
        da_obs = da_obs.sel({'time': time_slice})

    # TODO: write a loop for each init_date month

    ds_fcst_may = ds_fcst.where(ds_fcst['init_date'].dt.month == 5, drop=True)
    ds_fcst_nov = ds_fcst.where(ds_fcst['init_date'].dt.month == 11, drop=True)

    n_init_dates = len(ds_fcst_may['init_date'])
    n_ensembles = len(ds_fcst_may['ensemble'])

    da_fcst_may_detrended = remove_ensemble_mean_trend(ds_fcst_may['pr'], dim='init_date')
    da_fcst_nov_detrended = remove_ensemble_mean_trend(ds_fcst_nov['pr'], dim='init_date')

    mean_corr_may = mean_ensemble_correlation(da_fcst_may_detrended, dim='init_date')
    mean_corr_nov = mean_ensemble_correlation(da_fcst_nov_detrended, dim='init_date')        

    da_cafe_may_detrended_stacked = da_cafe_may_detrended.stack(sample=('init_date', 'lead_time', 'ensemble'))

    n_population = len(da_fcst_may_detrended_stacked['sample'])
    sample_size = n_init_dates * n_ensembles

    #simple bootstrapping
    null_correlations = []
    n_repeats = 100
    for repeat in range(n_repeats):
        mean_corr = random_mean_ensemble_correlation(da_fcst_may_detrended_stacked,
                                                     n_init_dates,
                                                     n_ensembles)
        null_correlations.append(mean_corr)
    null_correlations = xr.concat(null_correlations, "k")
    null_correlations = null_correlations.chunk({'k': -1})

    may_lower_bound = float(null_correlations.sel({'region': 'all'}).quantile(0.025).values)
    may_upper_bound = float(null_correlations.sel({'region': 'all'}).quantile(0.975).values)

    mean_corr_may.sel({'region': 'all'}).plot(color='blue', marker='o', linestyle='None', label=f'May starts')
    mean_corr_nov.sel({'region': 'all'}).plot(color='orange', marker='o', linestyle='None', label=f'Nov starts')

    lead_bounds = [1, ds_cafe_may['lead_time'].max()]

    plt.plot(lead_bounds, [may_lower_bound, may_lower_bound], color='blue', linestyle='--')
    plt.plot(lead_bounds, [nov_upper_bound, nov_upper_bound], color='orange', linestyle='--')
 
    plt.ylabel('correlation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
#    parser.add_argument("test", type=str,
#                        choices=('kolmogorov_smirnov'),
#                        help="Similarity test")
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument("--dask_config", type=str,
                        help="YAML file specifying dask client configuration")

    parser.add_argument("--reference_time_period", type=str, nargs=2, default=None,
                        help="Start and end date (YYYY-MM-DD format)")
    parser.add_argument("--output_chunks", type=str, nargs='*', action=general_utils.store_dict,
                        default={}, help="Chunks for writing data to file (e.g. init_date=-1 lead_time=-1)")

    args = parser.parse_args()
    _main(args)
