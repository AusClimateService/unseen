"""Command line program for fidelity testing."""

import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-1])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import pdb
import argparse

import fileio


def _main(args):
    """Run the command line program."""

    if args.dask_config:
        client = dask_setup.launch_client(args.dask_config)
        print(client)

    ds_fcst = fileio.open_file(args.fcst_file)
    da_fcst = ds_fcst[args.var]

    ds_obs = fileio.open_file(args.obs_file, variables=[args.var])
    da_obs = ds_obs[args.var]
    
    fcst_stacked = ds_cafe.stack({'sample': ['ensemble', 'init_date', 'lead_time']})
    fcst_stacked = fcst_stacked.chunk({'sample': -1, 'region': 1})

    obs_stacked = ds_obs.rename(time='sample')
    obs_stacked = obs_stacked.chunk({'sample': -1, 'region': 1})

    ks_obs = xclim.analog.spatial_analogs(target=obs_stacked,
                                          candidates=fcst_stacked,
                                          dist_dim='sample',
                                          method='kolmogorov_smirnov')

    n_population = len(fcst_stacked['sample'])
    n_samples = len(obs_stacked['sample'])

    ## TODO replace loop with dask
    rs_list = []
    for repeat in range(n_repeats):
        random_indexes = np.random.choice(n_population, size=n_samples, replace=True)
        fcst_random_sample = fcst_stacked.isel({'sample': random_indexes})
        ks_random = xclim.analog.spatial_analogs(target=fcst_random_sample,
                                                 candidates=fcst_stacked,
                                                 dist_dim='sample',
                                                 method='kolmogorov_smirnov')
        rs_list.append(ks_random)

    da_scores = xr.concat(rs_list, dim='k')
    ds_scores = da_scores.to_dataset()
    ## TODO: Add the obs score to the dataset
    
    infile_logs = {args.fcst_file: ds_fcst.attrs['history'],
                   args.obs_file: ds_obs.attrs['history']}
    ds_scores.attrs['history'] = fileio.get_new_log(infile_logs=infile_logs,
                                                     repo_dir=repo_dir)

    if args.output_chunks:
        ds_scores = ds_scores.chunk(args.output_chunks)
    fileio.to_zarr(ds_scores, args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("test", type=str,
                        choices=('kolmogorov_smirnov'),
                        help="Fidelity test")
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument("--dask_config", type=str,
                        help="YAML file specifying dask client configuration")

    parser.add_argument("--time_period", type=str, nargs=2,
                        help="Start and end date (YYYY-MM-DD format)")
    parser.add_argument("--output_chunks", type=str, nargs='*', action=general_utils.store_dict,
                        default={}, help="Chunks for writing data to file (e.g. init_date=-1 lead_time=-1)")

    args = parser.parse_args()
    _main(args)
