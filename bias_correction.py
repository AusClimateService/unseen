"""Bias correction."""

import sys
repo_dir = sys.path[0]
import pdb
import argparse

import git
import xarray as xr
import cmdline_provenance as cmdprov

import myfuncs


def mask_time_period(ds, period, dim='time'):
    """Mask a period of time.

    Args:
      da (xarray DataArray)
      period (list) - Start and stop dates (in YYYY-MM-DD format)
      dim (str)

    Only works for cftime objects.
    """

    myfuncs.check_date_format(period)
    start, stop = period

    if dim in ds.dims:
        masked = ds.sel({dim: slice(start, stop)})
    elif dim in ds.coords:
        calendar = ds.time.calendar_type.lower()
        time_bounds = xr.cftime_range(start=start, end=stop,
                                      periods=2, freq=None,
                                      calendar=calendar)
        mask = (ds[dim].compute() >= time_bounds[0]) & (ds[dim].compute() <= time_bounds[1])
        masked = ds.where(mask)
    masked.attrs = ds.attrs

    return masked


def get_monthly_clim(da, dim, time_period=None):
    """Calculate monthly climatology

    Args:
      da (xarray DataArray)
      dim (str) : Dimension over which to calculate climatology
        (e.g. init_date)
      time_period () : Time period
    """

    if period is not None:
        ds = mask_time_period(ds.copy(), period)
        ds.attrs['climatological_period'] = str(period)
    
    clim = ds.groupby(f'{dim}.month').mean(dim, keep_attrs=True)

    return clim


def get_bias(da_fcst, da_obs, method, time_period=None):
    """Calculate model bias.

    Args:
      da_fcst (xarray DataArray) : Forecast data
      da_obs (xarray DataArray) : Observational data
      method (str) : Bias removal method
    """

    fcst_ensmean = da_fcst.mean('ensemble', keep_attrs=True)
    fcst_clim = get_monthly_clim(fcst_ensmean, 'init_date',
                                 time_period=time_period)



    obs_clim = da_obs.mean('init_date')
    if method == 'additive':
        bias = fcst_clim - obs_clim
    elif method == 'multiplicative':
        bias = fcst_clim / obs_clim
    else:
        raise ValueError('Invalid bias removal method')
    bias.attrs['bias_correction_method'] = method

    return bias



    fcst_clim = get_monthly_clim(
        mask_time_period(fcst.mean('ensemble', keep_attrs=True), period), 
        dim='init_date')
    obsv_clim = get_monthly_clim(
        mask_time_period(obsv, period),
        dim='init_date')
    if method == 'additive':
        bias = fcst_clim - obsv_clim
    elif method == 'multiplicative':
        bias = fcst_clim / obsv_clim
    else: 
        raise ValueError(f'Unrecognised mode {mode}')
    bias.attrs['bias_correction_method'] = method
    bias.attrs['bias_correction_period'] = str(period)




def remove_bias(da, bias, method):
    """Remove model bias.

    Args:
      da (xarray DataArray) : Data
      bias (xarray DataArray) : Bias
      method (str) : Bias removal method
    """

    if method == 'additive':
        da_corrected = da - bias
    elif method == 'multiplicative':
        da_corrected = da / bias
    else:
        raise ValueError('Invalid bias removal method')
    da_corrected.attrs['bias_correction_method'] = bias.attrs['bias_correction_method']

    return da_corrected


def _main(args):
    """Run the command line program."""

    ds_fcst = xr.open_zarr(args.fcst_file)
    da_fcst = ds_fcst[args.var]
    ds_obs = xr.open_zarr(args.obs_file)
    da_obs = ds_obs[args.var]

    bias = get_bias(da_fcst, da_obs, args.method) 
    da_corrected = remove_bias(da_model, bias, args.method)
    
    ds = da_corrected.to_dataset()
    repo = git.Repo(repo_dir)
    repo_url = repo.remotes[0].url.split('.git')[0]
    new_log = cmdprov.new_log(code_url=repo_url)
    ds.attrs['history'] = new_log

    ds = ds.chunk({'init_date': -1, 'lead_time': -1})
    ds.to_zarr(args.outfile, mode='w')


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
    parser.add_argument("--", type=str, nargs='*', required=True,
                        help="Initial dates (YYYY-MM-DD format)")
    args = parser.parse_args()
    _main(args)
