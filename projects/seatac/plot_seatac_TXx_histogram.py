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


def _main(args):
    """Run the command line program."""

    logging.basicConfig(level=logging.INFO, filename='seatac_tXx_histogram.log')
    general_utils.set_plot_params(args.plotparams)
    
    ds_obs = fileio.open_file(args.obs_file,
                              metadata_file=args.obs_config,
                              time_freq='A-DEC',
                              time_agg='max')
    obs_gev_shape, obs_gev_loc, obs_gev_scale = gev.fit(ds_obs['tasmax'].values)
    logging.info(f'Observations GEV fit: shape={obs_gev_shape}, location={obs_gev_loc}, scale={obs_gev_scale}')

    ds_ensemble = fileio.open_file(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()
    ensemble_gev_shape, ensemble_gev_loc, ensemble_gev_scale = gev.fit(ds_ensemble_stacked['tasmax'].values)
    logging.info(f'Ensemble GEV fit: shape={ensemble_gev_shape}, location={ensemble_gev_loc}, scale={ensemble_gev_scale}')

    fig, ax = plt.subplots(figsize=[10, 8])
    bins = np.arange(23, 49)
    gev_xvals = np.arange(22, 49, 0.1)
    
    ds_ensemble_stacked['tasmax'].plot.hist(bins=bins,
                                           density=True,
                                           rwidth=0.9,
                                           alpha=0.7,
                                           color='blue',
                                           label='ACCESS-D')

    ensemble_gev_pdf = gev.pdf(gev_xvals, ensemble_gev_shape, ensemble_gev_loc, ensemble_gev_scale)
    plt.plot(gev_xvals, ensemble_gev_pdf, color='blue')

    ds_obs['tasmax'].plot.hist(bins=bins,
                               density=True,
                               rwidth=0.9,
                               alpha=0.7,
                               color='orange',
                               label='Station Observations')
    obs_gev_pdf = gev.pdf(gev_xvals, obs_gev_shape, obs_gev_loc, obs_gev_scale)
    plt.plot(gev_xvals, obs_gev_pdf, color='orange')

    plt.legend()
    plt.xlabel('TXx (C)')
    plt.ylabel('probability')
    plt.title('Histogram of TXx: SeaTac')

    infile_logs = {args.ensemble_file : ds_ensemble.attrs['history']}
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("obs_file", type=str, help="Observations data file")
    parser.add_argument("obs_config", type=str, help="Observations configuration file")
    parser.add_argument("ensemble_file", type=str, help="Model ensemble file")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    
    args = parser.parse_args()
    _main(args)
