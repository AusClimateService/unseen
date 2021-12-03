"""Plot TXx distribution by year"""

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
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr

import fileio
import array_handling
import general_utils


def _main(args):
    """Run the command line program."""

    logfile = args.logfile if args.logfile else args.outfile.split('.')[0] + '.log'
    logging.basicConfig(level=logging.INFO, filename=logfile, filemode='w')

    general_utils.set_plot_params(args.plotparams)

    ds_init = xr.open_zarr(args.ensemble_file)
    ds_time = array_handling.reindex_forecast(ds_init)

    fig = plt.figure(figsize=[14, 8])

    years = np.arange(2004, 2022)
    color = iter(matplotlib.cm.hot_r(np.linspace(0.3, 1, len(years))))
    for year in years:
        c = next(color)
        year_da = ds_time['tasmax'].sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
        year_array = year_da.to_masked_array().flatten().compressed()
        nsamples = year_array.shape[0]
        logging.info(f'{nsamples} samples for the year {year}')
        year_df = pd.DataFrame(year_array)
        sns.kdeplot(year_df[0], color=c, label=str(year))

    plt.xlim(26, 46)
    plt.title('TXx distribution from model ensemble')
    plt.xlabel('TXx (C)')
    plt.legend(ncol=2)

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
