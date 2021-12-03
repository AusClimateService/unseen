"""Plot maximum TXx by year"""

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
import xarray as xr

import fileio
import array_handling
import general_utils


def log_sample_counts(years, counts):
    """Log sample count for each year"""

    logging.info('Sample counts for each year:') 
    for year, count in zip(years, counts):
        logging.info(f'{year}: {count}')


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')
    general_utils.set_plot_params(args.plotparams)

    ds_init = xr.open_zarr(args.ensemble_file)
    ds_time = array_handling.reindex_forecast(ds_init)

    max_by_year = ds_time['tasmax'].max(dim=('ensemble', 'init_date'), keep_attrs=True)
    max_by_year = max_by_year.resample(time='A-DEC').max('time', keep_attrs=True)

    count = ds_time['tasmax'].notnull(keep_attrs=True)
    count = count.sum(dim=('ensemble', 'init_date'), keep_attrs=True)
    count = count.resample(time='A-DEC').sum('time', keep_attrs=True)

    fig, ax1 = plt.subplots(figsize=[12, 6])

    color = 'tab:blue'
    xvals1 = max_by_year['time'].dt.year.values
    yvals1 = max_by_year.values
    ax1.set_xlabel('year')
    ax1.set_ylabel('maximum TXx (C)', color=color)
    ax1.plot(xvals1, yvals1, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.set_ylim(40.5, 47.5)
    ax1.axvspan(2004, 2021, alpha=0.5, color='0.8')

    ax2 = ax1.twinx() 

    color = '0.6'
    xvals2 = count['time'].dt.year.values
    yvals2 = count.values
    log_sample_counts(xvals2, yvals2)
    ax2.set_ylabel('number of samples', color=color)
    ax2.plot(xvals2, yvals2, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(110, 2850)

    plt.title('Maximum TXx from model ensemble')
    fig.tight_layout()

    infile_logs = {args.ensemble_file : ds_init.attrs['history']}
    new_log = fileio.get_new_log(infile_logs=infile_logs, repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plt.savefig(args.outfile, metadata={metadata_key: new_log}, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("ensemble_file", type=str, help="Model ensemble file")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument('--logfile', type=str, default=None,
                        help='name of logfile (default = same as outfile but with .log extension)')
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    
    args = parser.parse_args()
    _main(args)
