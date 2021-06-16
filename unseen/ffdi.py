"""Calculate the Forest Fire Danger Index."""

import pdb
import argparse

import myfuncs
import indices


def _main(args):
    """Run the command line program."""

    kwargs = {'metadata_file': args.metadata_file,
              'no_leap_days': args.no_leap_days,
              'region': args.region,
              'units': {'pr': 'mm/day',
                        'tasmax': 'C',
                        'uas': 'km/h',
                        'vas': 'km/h',
                        'hur': '%'},
              'variables': ['pr', 'hur', 'tasmax', 'uas', 'vas'],
             }

    if args.data_type == 'obs':
        assert len(args.infiles) == 1
        ds = myfuncs.open_file(args.infiles[0], **kwargs)
        temporal_dim = 'time'
        ds = ds.chunk({'time': -1})
    elif args.data_type == 'forecast':
        ds = myfuncs.open_mfforecast(args.infiles, **kwargs)
        temporal_dim = 'lead_time'
        ds = ds.chunk({'init_date': -1, 'lead_time': -1})
    else:
        raise ValueError(f'Unrecognised data type: {args.data_type}')

    ffdi = indices.calc_FFDI(ds, dim=temporal_dim)

    ffdi.attrs['history'] = myfuncs.get_new_log()
    ffdi.to_zarr(args.outfile, mode='w')


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

    args = parser.parse_args()
    _main(args)
    
