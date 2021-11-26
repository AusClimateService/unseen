"""Plot SeaTac TXx return periods"""

import pdb
import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-2])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import argparse
import warnings
warnings.filterwarnings('ignore')
import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import genextreme as gev

import fileio
import general_utils
import indices


def return_period(data, score):
    """Calculate the return period for a given score"""
    
    n_exceedance_events = (data > score).sum()
    exceedance_probability = n_exceedance_events / len(data)
    
    return 1. / exceedance_probability


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    general_utils.set_plot_params(args.plotparams)
    
    ds_ensemble = fileio.open_file(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()

    population_size = ds_ensemble_stacked['tasmax'].size
    threshold = 42.2
    n_repeats = 1000

    full_model_return_period = return_period(ds_ensemble_stacked['tasmax'].values, threshold)
    logging.info(f'TXx={threshold}C return period in full model ensemble: {full_model_return_period}')

    gev_shape, gev_loc, gev_scale = indices.fit_gev(ds_ensemble_stacked['tasmax'].values, use_estimates=True)
    gev_data = gev.rvs(gev_shape, loc=gev_loc, scale=gev_scale, size=args.gev_samples)
    full_gev_return_period = return_period(gev_data, threshold)
    logging.info(f'TXx={threshold}C return period from GEV fit to full model ensemble: {full_gev_return_period}')

    df_model_return_period = pd.DataFrame([full_model_return_period]*n_repeats, columns=[population_size])
    df_gev_return_period = pd.DataFrame([full_gev_return_period]*n_repeats, columns=[population_size])

    for sample_size in [10, 50, 100, 500, 1000, 5000, 10000]:
        print(sample_size)
        model_estimates = []
        gev_estimates = []
        for resample in range(n_repeats):
            random_indexes = np.random.choice(population_size, size=sample_size, replace=False)
            #random_indexes.sort()
            model_subsample = ds_ensemble_stacked['tasmax'].isel({'sample': random_indexes})
            model_return_period = return_period(model_subsample.values, threshold)
            model_estimates.append(model_return_period)
            gev_shape, gev_loc, gev_scale = indices.fit_gev(model_subsample.values, use_estimates=False)
            gev_data = gev.rvs(gev_shape, loc=gev_loc, scale=gev_scale, size=args.gev_samples)  
            gev_return_period = return_period(gev_data, threshold)
            gev_estimates.append(gev_return_period)
        df_model_return_period[sample_size] = model_estimates
        df_gev_return_period[sample_size] = gev_estimates

    df_model_return_period = df_model_return_period.reindex(sorted(df_model_return_period.columns), axis=1)
    df_gev_return_period = df_gev_return_period.reindex(sorted(df_gev_return_period.columns), axis=1)

    fig, (ax1, ax2) = plt.subplots(2, figsize=[10, 12])
    df_model_return_period.boxplot(ax=ax1)
    ax1.set_title('(a) Return periods from model samples')
    ax1.set_xlabel(' ')
    ax1.set_ylabel('return period for TXx=42.2C (years)')
    
    df_gev_return_period.mask(df_gev_return_period > 1000).boxplot(ax=ax2)
    ax2.set_title('(b) Return periods from GEV fits to model samples')
    ax2.set_xlabel('sample size')
    ax2.set_ylabel('return period for TXx=42.2C (years)')

    infile_logs = {args.ensemble_file : ds_ensemble.attrs['history']}
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("ensemble_file", type=str, help="Model ensemble file")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension)')
    parser.add_argument('--gev_samples', type=int, default=10000,
                        help='number of times to sample the GEVs')
    
    args = parser.parse_args()
    _main(args)
