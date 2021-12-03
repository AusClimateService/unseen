"""Plot SeaTac TXx likelihoods"""

import pdb
import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-2])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import fileio
import general_utils


def likelihood_curve(da, threshold):
    """Calculate the likelihood curve for a particular TXx threshold"""
    
    population_size = da.size
    n_repeats = 1000
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, population_size]
    likelihoods = []
    for sample_size in sample_sizes:
        event_count = []
        for resample in range(n_repeats):
            random_indexes = np.random.choice(population_size, size=sample_size, replace=False)
            random_indexes.sort()
            tasmax_max = float(da.isel({'sample': random_indexes}).max().values)
            event_count.append(tasmax_max > threshold)
        n_events = np.sum(event_count)
        likelihood = (n_events / n_repeats) * 100
        likelihoods.append(likelihood)
    
    return likelihoods, sample_sizes


def _main(args):
    """Run the command line program."""

    general_utils.set_plot_params(args.plotparams)
    
    ds_ensemble = fileio.open_file(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()

    fig, ax = plt.subplots(figsize=[10, 6])
    thresholds = [37, 39, 41, 43, 45]
    for threshold in thresholds:
        likelihoods, sample_sizes = likelihood_curve(ds_ensemble_stacked['tasmax'], threshold)
        xvals = np.arange(len(sample_sizes)) + 1
        ax.step(xvals, likelihoods, where='post', label=threshold, marker='o')
        #ax.plot(xvals, likelihoods, 'o')
    ax.set_xticklabels([0] + sample_sizes)
    ax.set_ylabel('likelihood (%)')
    ax.set_xlabel('sample size')
    ax.set_title('Likelihoods from model ensemble')
    ax.legend()

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
    
    args = parser.parse_args()
    _main(args)
