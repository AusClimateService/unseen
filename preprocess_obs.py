"""Process observational data into forecast (i.e. initial date / lead time) format."""

import pdb
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import cftime

import myfuncs


def check_cftime(time_dim):
    """Check that time dimension is cftime.

    Args:
      time_dim (xarray DataArray) : Time dimension
    """

    t0 = time_dim.values[0]
    assert type(t0) in cftime._cftime.DATE_TYPES.values(), \
        'Time dimension must use cftime objects'


def str_to_cftime(datestring, calendar):
    """Convert a date string to cftime object"""
    
    dt = datetime.strptime(datestring, '%Y-%m-%d')
    cfdt = cftime.datetime(dt.year, dt.month, dt.day, calendar=calendar)
     
    return cfdt


def stack_by_init_date(ds, init_dates, n_lead_steps, freq='D'):
    """Stack timeseries array in inital date / lead time format.

    Args:
      da (xarray Dataset or DataArray)
      init_dates (list) : Initial dates in YYYY-MM-DD format
      n_lead_steps (int) : Maximum lead time
      freq (str) : Time-step frequency
    """

    myfuncs.check_date_format(init_dates)
    check_cftime(ds['time'])

    rounded_times = ds['time'].dt.floor(freq).values
    ref_time = init_dates[0]
    ref_calendar = rounded_times[0].calendar
    ref_var = list(ds.keys())[0]
    ref_array = ds[ref_var].sel(time=ref_time).values    

    time2d = np.empty((len(init_dates), n_lead_steps), 'object')
    init_date_indexes = []
    offset = n_lead_steps - 1
    for ndate, date in enumerate(init_dates):
        date_cf = str_to_cftime(date, ref_calendar)
        start_index = np.where(rounded_times == date_cf)[0][0]
        end_index = start_index + n_lead_steps
        time2d[ndate, :] = ds['time'][start_index:end_index].values
        init_date_indexes.append(start_index + offset)

    ds = ds.rolling(time=n_lead_steps, min_periods=1).construct("lead_time")
    ds = ds.assign_coords({'lead_time': ds['lead_time'].values})
    ds = ds.rename({'time': 'init_date'})
    ds = ds.isel(init_date=init_date_indexes)
    ds = ds.assign_coords({'init_date': time2d[:, 0]})
    ds = ds.assign_coords({'time': (['init_date', 'lead_time'], time2d)})
    ds['lead_time'].attrs['units'] = freq

    actual_array = ds[ref_var].sel({'init_date': ref_time, 'lead_time': 0}).values
    np.testing.assert_allclose(actual_array, ref_array)
    
    # TODO: Return nans if requested times lie outside of the available range
    
    return ds


def _main(args):
    """Run the command line program."""

    ds = myfuncs.open_file(args.infile,
                           dataset=args.dataset,
                           new_names=args.new_names,
                           no_leap_days=args.no_leap_days,
                           region=args.region,
                           units=args.units,
                           variables=args.variables)
    ds = stack_by_init_date(ds, args.init_dates, args.n_lead_steps)
    ds.attrs['history'] = myfuncs.get_new_log()
    ds = ds.chunk({'init_date': -1, 'lead_time': -1})
    ds.to_zarr(args.outfile, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
                                     
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    
    parser.add_argument("--init_dates", type=str, nargs='*', required=True,
                        help="Initial dates (YYYY-MM-DD format)")
    parser.add_argument("--n_lead_steps", type=int, required=True,
                        help="Number of lead time steps")

    parser.add_argument("--dataset", type=str, choices=('JRA-55', 'AWAP'),
                        help="Dataset name for custom metadata handling")
    parser.add_argument("--new_names", type=str, nargs='*', action=myfuncs.store_dict, 
                        help="Variable / new name pairs (e.g. precip=pr temp=tas)")
    parser.add_argument("--no_leap_days", action="store_true", default=False,
                        help="Remove leap days from time series [default=False]")
    parser.add_argument("--region", type=str, choices=myfuncs.regions.keys(),
                        help="Select spatial region from data")
    parser.add_argument("--units", type=str, nargs='*', action=myfuncs.store_dict,
                        help="Variable / new unit pairs (e.g. precip=mm/day)")
    parser.add_argument("--variables", type=str, nargs='*',
                        help="Variables to select")

    args = parser.parse_args()
    _main(args)
    