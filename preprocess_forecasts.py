"""Collect forecast data into an ensemble."""

import sys
repo_dir = sys.path[0]
import pdb
import argparse

import git
import numpy as np
import xarray as xr
import cmdline_provenance as cmdprov

import myfuncs


def to_init_lead(da):
    """Switch out time axis for init_date and lead_time."""

    lead_time = range(len(da['time']))
    init_date = np.datetime64(da['time'].values[0].strftime('%Y-%m-%d'))
    new_coords = {'lead_time': lead_time, 'init_date': init_date}
    da = da.rename({'time': 'lead_time'})
    da = da.assign_coords(new_coords)

    return da


def open_mfforecast(infiles, var, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    for infile in infiles:
        da = myfuncs.open_file(infile, var, **kwargs)
        da = to_init_lead(da)
        datasets.append(da)
    da = xr.concat(datasets, dim='init_date')

    time_values = [da.get_index('init_date').shift(int(lead), 'D') for lead in da['lead_time']]
    time_dimension = xr.DataArray(time_values,
                                  dims={'lead_time': da['lead_time'],
                                        'init_date': da['init_date']})
    da = da.assign_coords({'time': time_dimension})
    da['lead_time'].attrs['units'] = 'D'

    return da


def _main(args):
    """Run the command line program."""

    da = open_mfforecast(args.infiles, args.var,
                         region=args.region,
                         no_leap_days=args.no_leap_days,
                         dataset=args.dataset,
                         units=args.units)

    ds = da.to_dataset()
    repo = git.Repo(repo_dir)
    repo_url = repo.remotes[0].url.split('.git')[0]
    new_log = cmdprov.new_log(code_url=repo_url)
    ds.attrs['history'] = new_log

    ds = ds.chunk({'init_date': -1, 'lead_time': -1})
    ds.to_zarr(args.outfile, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)                                     
    parser.add_argument("infiles", type=str, nargs='*', help="Input files")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("--dataset", type=str, choices=('cafe'),
                        help="Dataset name for custom metadata handling")
    parser.add_argument("--units", type=str, default=None,
                        help="Convert to these units")
    parser.add_argument("--region", type=str, choices=myfuncs.regions.keys(),
                        help="Select region from data")
    parser.add_argument("--no_leap_days", action="store_true", default=False,
                        help="Remove leap days from time series [default=False]")
    args = parser.parse_args()
    _main(args)

