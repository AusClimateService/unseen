"""Command line program for similarity testing."""

import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-1])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)
sys.path.insert(1, '/home/599/dbi599/xks/')

import pdb
import argparse

import xarray as xr

import fileio
import general_utils


def univariate_ks_test(fcst_stacked, obs_stacked):
    """Univariate KS test.

    If p < 0.05 you can reject the null hypothesis
    that the two samples are from different populations.
    """

    ks_distances = []
    pvals = []
    for lead_time in fcst_stacked['lead_time'].values:
        ks_distance, pval = xks.ks1d2s(obs_stacked, fcst_stacked.sel({'lead_time': lead_time}), 'sample')
        ks_distance = ks_distance.rename({'pr': 'ks'})
        pval = pval.rename({'pr': 'pval'})
        ks_distances.append(ks_distance['ks'])
        pvals.append(pval['pval'])

    ks_distances = xr.concat(ks_distances, 'lead_time')
    pvals = xr.concat(pvals, 'lead_time')
    ds = xr.merge([ks_distances, pvals])

    return ds


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
        
    fcst_stacked = ds_cafe.stack({'sample': ['ensemble', 'init_date']})
    fcst_stacked = fcst_stacked.chunk({'sample': -1, 'region': 1})

    obs_stacked = ds_obs.rename(time='sample')
    obs_stacked = obs_stacked.chunk({'sample': -1, 'region': 1})

    ds_similarity = univariate_ks_test(fcst_stacked, obs_stacked)

    infile_logs = {args.fcst_file: ds_fcst.attrs['history'],
                   args.obs_file: ds_obs.attrs['history']}
    ds_similarity.attrs['history'] = fileio.get_new_log(infile_logs=infile_logs,
                                                        repo_dir=repo_dir)

    if args.output_chunks:
        ds_similarity = ds_similarity.chunk(args.output_chunks)
    fileio.to_zarr(ds_scores, args.outfile)


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
