"""Plot SeaTac TXx sample size distribution"""

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

import fileio
import general_utils


def _main(args):
    """Run the command line program."""

    general_utils.set_plot_params(args.plotparams)
    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    
    ds_ensemble = fileio.open_file(args.ensemble_file)
    ds_ensemble_stacked = ds_ensemble.stack({'sample': ['ensemble', 'init_date', 'lead_time']}).compute()

    population_size = ds_ensemble_stacked['tasmax'].size
    n_repeats = 1000
    maximum = float(ds_ensemble_stacked['tasmax'].max().values)
    logging.info(f'Maximum TXx: {maximum}C')

    df_random = pd.DataFrame([maximum]*n_repeats, columns=[population_size])
      
    for sample_size in [10, 50, 100, 500, 1000, 5000, 10000]:
        estimates = []
        for resample in range(n_repeats):
            random_indexes = np.random.choice(population_size, size=sample_size, replace=False)
            #random_indexes.sort()
            tasmax_max = float(ds_ensemble_stacked['tasmax'].isel({'sample': random_indexes}).max().values)
            estimates.append(tasmax_max)
        df_random[sample_size] = estimates

    df_random = df_random.reindex(sorted(df_random.columns), axis=1)

    fig = plt.figure(figsize=[10, 6])
    df_random.boxplot()
    plt.title('Maximum TXx from model ensemble')
    plt.xlabel('sample size')
    plt.ylabel('TXx (C)')

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
                        help='name of logfile (default = same as outfile but with .log extension')
    
    args = parser.parse_args()
    _main(args)
