"""Plot SeaTac tXx histogram"""

import pdb
import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-2])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import genextreme as gev

import fileio
import general_utils
import indices


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    general_utils.set_plot_params(args.plotparams)
    
    ds_obs = fileio.open_file(args.obs_file)
    obs_shape, obs_loc, obs_scale = indices.fit_gev(ds_obs['tasmax'].values)
    logging.info(f'Observations GEV fit: shape={obs_shape}, location={obs_loc}, scale={obs_scale}')

    ds_raw = fileio.open_file(args.raw_model_file)
    ds_raw_stacked = ds_raw.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    raw_shape, raw_loc, raw_scale = indices.fit_gev(ds_raw_stacked['tasmax'].values, use_estimates=True)
    logging.info(f'Model (raw) GEV fit: shape={raw_shape}, location={raw_loc}, scale={raw_scale}')

    ds_bias = fileio.open_file(args.bias_corrected_model_file)
    ds_bias_stacked = ds_bias.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    bias_shape, bias_loc, bias_scale = indices.fit_gev(ds_bias_stacked['tasmax'].values, use_estimates=True)
    logging.info(f'Model (bias corrected) GEV fit: shape={bias_shape}, location={bias_loc}, scale={bias_scale}')

    fig, ax = plt.subplots(figsize=[10, 8])
    bins = np.arange(23, 49)
    gev_xvals = np.arange(22, 49, 0.1)
    
    ds_bias_stacked['tasmax'].plot.hist(bins=bins,
                                        density=True,
                                        rwidth=0.9,
                                        alpha=0.7,
                                        color='tab:blue',
                                        label='ACCESS-D')
    bias_pdf = gev.pdf(gev_xvals, bias_shape, bias_loc, bias_scale)
    plt.plot(gev_xvals, bias_pdf, color='tab:blue', linewidth=2.0)
    raw_pdf = gev.pdf(gev_xvals, raw_shape, raw_loc, raw_scale)
    plt.plot(gev_xvals, raw_pdf, color='tab:blue', linestyle='--', linewidth=2.0)

    ds_obs['tasmax'].plot.hist(bins=bins,
                               density=True,
                               rwidth=0.9,
                               alpha=0.7,
                               color='tab:orange',
                               label='Station Observations')
    obs_pdf = gev.pdf(gev_xvals, obs_shape, obs_loc, obs_scale)
    plt.plot(gev_xvals, obs_pdf, color='tab:orange', linewidth=2.0)

    plt.legend()
    plt.xlabel('TXx (C)')
    plt.ylabel('probability')
    plt.title('Histogram of TXx: SeaTac')

    infile_logs = {args.bias_corrected_model_file: ds_bias.attrs['history'],
                   args.obs_file: ds_obs.attrs['history']}
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("obs_file", type=str, help="Observations data file")
    parser.add_argument("raw_model_file", type=str, help="Model file (raw)")
    parser.add_argument("bias_corrected_model_file", type=str, help="Model file (bias corrected)")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension')
    
    args = parser.parse_args()
    _main(args)
