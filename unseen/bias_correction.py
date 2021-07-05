"""Functions for bias correction."""

import pdb

import time_utils


def get_bias(fcst, obs, method, time_period=None):
    """Calculate forecast bias.

    Args:
      fcst (xarray DataArray) : Forecast data
      obs (xarray DataArray) : Observational data
      method (str) : Bias removal method
      time_period (list) : Start and end dates (in YYYY-MM-DD format)
    """

    fcst_ensmean = fcst.mean('ensemble', keep_attrs=True)
    fcst_clim = time_utils.get_monthly_clim(fcst_ensmean, 'init_date',
                                            time_period=time_period)
    obs_clim = time_utils.get_monthly_clim(obs, 'init_date',
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

