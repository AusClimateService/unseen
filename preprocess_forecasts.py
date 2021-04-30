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


def open_and_clip(infile, var, region=None, no_leap_days=False):
    """Open file and select region"""
    
    ds = xr.open_zarr(infile, consolidated=True, use_cftime=True)
    
    da = ds['precip']
    for drop_coord in ['average_DT', 'average_T1', 'average_T2', 'zsurf', 'area']:
        if drop_coord in da.coords:
            da = da.drop(drop_coord)
    
    if no_leap_days:
        da = da.sel(time=~((da['time'].dt.month == 2) & (da['time'].dt.day == 29)))

    if region:
        region = myfuncs.box_regions[region]
        da = myfuncs.get_region(da, region)
    
    if var == 'precip':
        da = myfuncs.convert_pr_units(da)

    lead_time = range(len(da['time']))
    init_date = np.datetime64(da['time'].values[0].strftime('%Y-%m-%d'))
    new_coords = {'lead_time': lead_time, 'init_date': init_date}
    da = da.rename({'time': 'lead_time'})
    da = da.assign_coords(new_coords)

    return da


def main(args):
    """Run the command line program."""

    datasets = []
    for infile in args.infiles:
        da = open_and_clip(infile, args.var,
                           region=args.region,
                           no_leap_days=args.no_leap_days)
        datasets.append(da)
    da = xr.concat(datasets, dim='init_date')

    time_values = [da.get_index('init_date').shift(int(lead), 'D') for lead in da['lead_time']]
    time_dimension = xr.DataArray(time_values,
                                  dims={'lead_time': da['lead_time'],
                                        'init_date': da['init_date']})
    da = da.assign_coords({'time': time_dimension})
    da['lead_time'].attrs['units'] = 'D'

    ds = da.to_dataset()

    repo = git.Repo(repo_dir)
    repo_url = repo.remotes[0].url.split('.git')[0]
    new_log = cmdprov.new_log(code_url=repo_url)
    ds.attrs['history'] = new_log

    ds = ds.chunk({'init_date': 1, 'lead_time': 50})
    ds.to_zarr(args.outfile, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
                                     
    parser.add_argument("infiles", type=str, nargs='*', help="Input files")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument("--region", type=str, choices=myfuncs.box_regions.keys(),
                        help="Select spatial region from data")
    parser.add_argument("--no_leap_days", action="store_true", default=False,
                        help="Remove leap days from time series [default=False]")

    args = parser.parse_args()
    main(args)

