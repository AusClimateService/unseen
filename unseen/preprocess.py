"""Preprocess data (e.g. select variables, regions, units, etc)."""

import pdb
import argparse

import myfuncs
import indices
#import dask_setup


def indices_setup(kwargs, variables):
    """Set variables and units for index calculation."""

    index = ''
    if 'ffdi' in variables:
        kwargs['variables'] = ['pr', 'hur', 'tasmax',
                               'uas', 'vas']
        kwargs['units'] = {'pr': 'mm/day',
                           'tasmax': 'C',
                           'uas': 'km/h',
                           'vas': 'km/h',
                           'hur': '%'}
        index = 'ffdi'

    return kwargs, index


def _main(args):
    """Run the command line program."""

    #dask_setup.local()

    kwargs = {'metadata_file': args.metadata_file,
              'no_leap_days': args.no_leap_days,
              'region': args.region,
              'units': args.units,
              'variables': args.variables,
              'isel': args.isel,
              'chunks': args.input_chunks,
             }

    kwargs, index = indices_setup(kwargs, args.variables)

    if args.data_type == 'obs':
        assert len(args.infiles) == 1
        ds = myfuncs.open_file(args.infiles[0], **kwargs)
        temporal_dim = 'time'
    elif args.data_type == 'forecast':
        ds = myfuncs.open_mfforecast(args.infiles, **kwargs)
        temporal_dim = 'lead_time'
    else:
        raise ValueError(f'Unrecognised data type: {args.data_type}')

    if index == 'ffdi':
        ds['ffdi'] = indices.calc_FFDI(ds, dim=temporal_dim)

    if args.output_chunks:
        ds = ds.chunk(args.output_chunks)
    ds = ds[args.variables]

    ds.attrs['history'] = myfuncs.get_new_log()
    myfuncs.to_zarr(ds, args.outfile, zip=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
                                     
    parser.add_argument("infiles", type=str, nargs='*', help="Input files")
    parser.add_argument("data_type", type=str, choices=('forecast', 'obs'), help='Data type')
    parser.add_argument("outfile", type=str, help="Output file")

    parser.add_argument("--metadata_file", type=str,
                        help="YAML file specifying required file metadata changes")
    parser.add_argument("--no_leap_days", action="store_true", default=False,
                        help="Remove leap days from time series [default=False]")
    parser.add_argument("--region", type=str, choices=myfuncs.regions.keys(),
                        help="Select region from data")
    parser.add_argument("--units", type=str, nargs='*', default={}, action=myfuncs.store_dict,
                        help="Variable / new unit pairs (e.g. precip=mm/day)")
    parser.add_argument("--variables", type=str, nargs='*',
                        help="Variables to select (or index to calculate)")
    parser.add_argument("--isel", type=str, nargs='*', action=myfuncs.store_dict,
                        help="Index selection along dimensions (e.g. ensemble=1:5)")
    parser.add_argument("--input_chunks", type=str, nargs='*', action=myfuncs.store_dict,
                        default='auto', help="Chunks for reading data (e.g. time=-1)")
    parser.add_argument("--output_chunks", type=str, nargs='*', action=myfuncs.store_dict,
                        default={}, help="Chunks for writing data to file (e.g. lead_time=50)")
    

    args = parser.parse_args()
    _main(args)
    
