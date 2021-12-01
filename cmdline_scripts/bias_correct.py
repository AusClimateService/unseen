"""Command line program for bias correction."""

import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-1])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import pdb
import argparse

import xarray as xr

import fileio
import array_handling
import bias_correction
import general_utils
import time_utils


def _main(args):
    """Run the command line program."""

    ds_obs = fileio.open_file(args.obs_file, variables=[args.var])
    da_obs = ds_obs[args.var]

    ds_fcst = fileio.open_file(args.fcst_file, variables=[args.var])
    da_fcst = ds_fcst[args.var]
    init_dates = time_utils.cftime_to_str(da_fcst['init_date'])
    n_lead_steps = int(da_fcst['lead_time'].values.max()) + 1
    
    bias = bias_correction.get_bias(da_fcst, da_obs, args.method,
                                    time_period=args.base_period)
    da_fcst_bc = bias_correction.remove_bias(da_fcst, bias, args.method)
    
    ds_fcst_bc = da_fcst_bc.to_dataset()
    infile_logs = {args.fcst_file: ds_fcst.attrs['history'],
                   args.obs_file: ds_obs.attrs['history']}
    ds_fcst_bc.attrs['history'] = fileio.get_new_log(infile_logs=infile_logs,
                                                     repo_dir=repo_dir)

    if args.output_chunks:
        ds_fcst_bc = ds_fcst_bc.chunk(args.output_chunks)
    fileio.to_zarr(ds_fcst_bc, args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("method", type=str,
                        choices=('multiplicative', 'additive'),
                        help="Bias correction method")
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument("--base_period", type=str, nargs=2,
                        help="Start and end date for baseline (YYYY-MM-DD format)")
    parser.add_argument("--output_chunks", type=str, nargs='*', action=general_utils.store_dict,
                        default={}, help="Chunks for writing data to file (e.g. init_date=-1 lead_time=-1)")

    args = parser.parse_args()
    _main(args)
