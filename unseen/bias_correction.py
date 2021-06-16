"""Bias correction."""

import pdb
import argparse

import xarray as xr

import myfuncs


def mask_time_period(da, period):
    """Mask a period of time.

    Args:
      da (xarray DataArray)
      period (list) : Start and stop dates (in YYYY-MM-DD format)

    Only works for cftime objects.
    """

    myfuncs.check_date_format(period)
    start, stop = period

    if 'time' in da.dims:
        masked = da.sel({'time': slice(start, stop)})
    elif 'time' in da.coords:
        calendar = da['time'].calendar_type.lower()
        time_bounds = xr.cftime_range(start=start, end=stop,
                                      periods=2, freq=None,
                                      calendar=calendar)
        time_values = da['time'].compute()
        mask = (time_values >= time_bounds[0]) & (time_values <= time_bounds[1])
        masked = da.where(mask)
    else:
        raise ValueError(f'No time axis for masking')
    masked.attrs = da.attrs

    return masked


def get_monthly_clim(da, dim, time_period=None):
    """Calculate monthly climatology

    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate climatology
        (e.g. init_date)
      time_period (list) : Time period
    """

    if time_period is not None:
        da = mask_time_period(da.copy(), period)
        da.attrs['climatological_period'] = str(period)
    
    clim = da.groupby(f'{dim}.month').mean(dim, keep_attrs=True)

    return clim


def get_bias(fcst, obs, method, time_period=None):
    """Calculate forecast bias.

    Args:
      fcst (xarray DataArray) : Forecast data
      obs (xarray DataArray) : Observational data
      method (str) : Bias removal method
      time_period (list) : Start and end dates (in YYYY-MM-DD format)
    """

    fcst_ensmean = fcst.mean('ensemble', keep_attrs=True)
    fcst_clim = get_monthly_clim(fcst_ensmean, 'init_date',
                                 time_period=time_period)
    obs_clim = get_monthly_clim(obs, 'init_date',
                                time_period=time_period)

    if method == 'additive':
        bias = fcst_clim - obs_clim
    elif method == 'multiplicative':
        bias = fcst_clim / obs_clim
    else:
        raise ValueError(f'Unrecognised bias removal method {method}')

    bias.attrs['bias_correction_method'] = method
    if time_period:
        bias.attrs['bias_correction_period'] = '-'.join(time_period)

    return bias


def remove_bias(fcst, bias, method):
    """Remove model bias.

    Args:
      fcst (xarray DataArray) : Forecast data
      bias (xarray DataArray) : Bias
      method (str) : Bias removal method
    """

    if method == 'additive':
        fcst_bc = (fcst.groupby('init_date.month') - bias).drop('month')
    elif method == 'multiplicative':
        fcst_bc = (fcst.groupby('init_date.month') / bias).drop('month')
    else:
        raise ValueError(f'Unrecognised bias removal method {method}')

    fcst_bc.attrs['bias_correction_method'] = bias.attrs['bias_correction_method']
    try:
        fcst_bc.attrs['bias_correction_period'] = bias.attrs['bias_correction_period']
    except KeyError:
        pass

    return fcst_bc


def _main(args):
    """Run the command line program."""

    ds_fcst = xr.open_zarr(args.fcst_file, use_cftime=True)
    da_fcst = ds_fcst[args.var]
    init_dates = myfuncs.cftime_to_str(da_fcst['init_date'])
    n_lead_steps = da_fcst['lead_time'].values.max() + 1

    ds_obs = xr.open_zarr(args.obs_file, use_cftime=True)
    ds_obs = myfuncs.stack_by_init_date(ds_obs, init_dates, n_lead_steps)
    da_obs = ds_obs[args.var]
    
    bias = get_bias(da_fcst, da_obs, args.method,
                    time_period=args.base_period) 
    fcst_bc = remove_bias(da_fcst, bias, args.method)
    
    ds_fcst_bc = fcst_bc.to_dataset()
    infile_logs = {args.fcst_file: ds_fcst.attrs['history'],
                   args.obs_file: ds_obs.attrs['history']}
    ds_fcst_bc.attrs['history'] = myfuncs.get_new_log(infile_logs=infile_logs)

    ds_fcst_bc = ds_fcst_bc.chunk({'init_date': -1, 'lead_time': -1})
    ds_fcst_bc.to_zarr(args.outfile, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fcst_file", type=str, help="Forecast file")
    parser.add_argument("obs_file", type=str, help="Observations file")
    parser.add_argument("var", type=str, help="Variable name")
    parser.add_argument("method", type=str,
                        choices=('multiplicative', 'additive'),
                        help="Bias correction method")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("--base_period", type=str, nargs=2,
                        help="Start and end date for baseline (YYYY-MM-DD format)")
    args = parser.parse_args()
    _main(args)
