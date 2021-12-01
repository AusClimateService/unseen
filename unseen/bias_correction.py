"""Functions for bias correction."""

import pdb
import operator

import xarray as xr

import time_utils


def get_bias(fcst, obs, method, time_period=None):
    """Calculate forecast bias.

    Args:
      fcst (xarray DataArray) : Forecast data
      obs (xarray DataArray) : Observational data
      method (str) : Bias removal method
      time_period (list) : Start and end dates (in YYYY-MM-DD format)
    """

    fcst_clim = time_utils.get_clim(fcst, ['ensemble', 'init_date'],
                                    time_period=time_period,
                                    groupby_init_month=True)

    obs_stacked = array_handling.stack_by_init_date(obs,
                                                    init_dates=fcst['init_date'],
                                                    n_lead_steps=fcst.sizes['lead_time'])
    obs_clim = time_utils.get_clim(obs_stacked, ['ensemble', 'init_date'],
                                   time_period=time_period,
                                   groupby_init_month=True)

    with xr.set_options(keep_attrs=True):
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
        op = operator.sub
    elif method == 'multiplicative':
        op = operator.div
    else:
        raise ValueError(f'Unrecognised bias removal method {method}')

    with xr.set_options(keep_attrs=True):
        fcst_bc = op(fcst.groupby('init_date.month'), bias).drop('month')

    fcst_bc.attrs['bias_correction_method'] = bias.attrs['bias_correction_method']
    try:
        fcst_bc.attrs['bias_correction_period'] = bias.attrs['bias_correction_period']
    except KeyError:
        pass

    return fcst_bc

