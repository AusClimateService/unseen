"""Bias correction."""

import sys
repo_dir = sys.path[0]
import pdb
import argparse

import git
import xarray as xr
import cmdline_provenance as cmdprov

import myfuncs


# TODO: Clarify how the monthly climatology works at:
# https://github.com/dougiesquire/hydro_tasmania/blob/main/2020/myfuncs.py#L242


def main(args):
    """Run the command line program."""

    ds_model = xr.open_zarr(args.model_file)
    da_model = ds_model[args.var]
    model_clim = da_model.mean(['ensemble', 'init_date'])
    
    ds_obs = xr.open_zarr(args.obs_file)
    da_obs = ds_obs[args.var]
    obs_clim = da_obs.mean('init_date')

    if args.method == 'additive':
        bias = model_clim - obs_clim
        corrected_model = da_model - bias
    elif args.method == 'multiplicative':
        bias = model_clim / obs_clim
        corrected_model = da_model / bias
    
    ds = corrected_model.to_dataset()

    repo = git.Repo(repo_dir)
    repo_url = repo.remotes[0].url.split('.git')[0]
    new_log = cmdprov.new_log(code_url=repo_url)
    ds.attrs['history'] = new_log

    ds = ds.chunk({'init_date': -1, 'lead_time': -1})
    ds.to_zarr(args.outfile, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
                                     
    parser.add_argument("model_file", type=str, help="Model file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("method", type=str,
                        choices=('multiplicative', 'additive'),
                        help="Bias correction method")
    parser.add_argument("outfile", type=str, help="Output file")

    args = parser.parse_args()
    main(args)
