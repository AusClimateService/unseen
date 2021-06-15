"""Collect forecast data into an ensemble."""

import pdb
import argparse

import numpy as np
import xarray as xr

import myfuncs


def to_init_lead(ds):
    """Switch out time axis for init_date and lead_time."""

    lead_time = range(len(ds['time']))
    init_date = np.datetime64(ds['time'].values[0].strftime('%Y-%m-%d'))
    new_coords = {'lead_time': lead_time, 'init_date': init_date}
    ds = ds.rename({'time': 'lead_time'})
    ds = ds.assign_coords(new_coords)

    return ds


def open_mfforecast(infiles, **kwargs):
    """Open multi-file forecast."""

    datasets = []
    for infile in infiles:
        ds = myfuncs.open_file(infile, **kwargs)
        ds = to_init_lead(ds)
        datasets.append(ds)
    ds = xr.concat(datasets, dim='init_date')

    time_values = [ds.get_index('init_date').shift(int(lead), 'D') for lead in ds['lead_time']]
    time_dimension = xr.DataArray(time_values,
                                  dims={'lead_time': ds['lead_time'],
                                        'init_date': ds['init_date']})
    ds = ds.assign_coords({'time': time_dimension})
    ds['lead_time'].attrs['units'] = 'D'

    return ds


def _main(args):
    """Run the command line program."""

    ds = open_mfforecast(args.infiles,
                         metadata_file=args.metadata_file,
                         no_leap_days=args.no_leap_days,
                         region=args.region,
                         units=args.units,
                         variables=args.variables)
    ds.attrs['history'] = myfuncs.get_new_log()
    ds = ds.chunk({'init_date': -1, 'lead_time': -1})
    ds.to_zarr(args.outfile, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("infiles", type=str, nargs='*', help="Input files")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument("--metadata_file", type=str,
                        help="YAML file specifying required file metadata changes")
    parser.add_argument("--no_leap_days", action="store_true", default=False,
                        help="Remove leap days from time series [default=False]")
    parser.add_argument("--region", type=str, choices=myfuncs.regions.keys(),
                        help="Select region from data")
    parser.add_argument("--units", type=str, nargs='*', action=myfuncs.store_dict,
                        help="Variable / new unit pairs (e.g. precip=mm/day)")
    parser.add_argument("--variables", type=str, nargs='*',
                        help="Variables to select")

    args = parser.parse_args()
    _main(args)

