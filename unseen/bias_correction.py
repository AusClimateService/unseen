"""Functions for bias correction."""

import pdb
import operator

import xarray as xr

import time_utils


def get_bias(fcst, obs, method, time_period=None, monthly=False):
    """Calculate forecast bias.

    Args:
      fcst (xarray DataArray) : Forecast data
      obs (xarray DataArray) : Observational data
      method (str) : Bias removal method
      time_period (list) : Start and end dates (in YYYY-MM-DD format)
      monthly (bool) : Use monthly climatology
    """

    fcst_ensmean = fcst.mean("ensemble", keep_attrs=True)
    fcst_clim = time_utils.get_clim(
        fcst_ensmean, "init_date", time_period=time_period, monthly=monthly
    )
    obs_clim = time_utils.get_clim(
        obs, "time", time_period=time_period, monthly=monthly
    )

    with xr.set_options(keep_attrs=True):
        if method == "additive":
            bias = fcst_clim - obs_clim
        elif method == "multiplicative":
            bias = fcst_clim / obs_clim
        else:
            raise ValueError(f"Unrecognised bias removal method {method}")

    bias.attrs["bias_correction_method"] = method
    if time_period:
        bias.attrs["bias_correction_period"] = "-".join(time_period)

    return bias


def remove_bias(fcst, bias, method, monthly=False):
    """Remove model bias.

    Args:
      fcst (xarray DataArray) : Forecast data
      bias (xarray DataArray) : Bias
      method (str) : Bias removal method
      monthly (bool) : Monthly bias removal
    """

    if method == "additive":
        op = operator.sub
    elif method == "multiplicative":
        op = operator.div
    else:
        raise ValueError(f"Unrecognised bias removal method {method}")

    with xr.set_options(keep_attrs=True):
        if monthly:
            fcst_bc = op(fcst.groupby("init_date.month"), bias).drop("month")
        else:
            fcst_bc = op(fcst, bias)

    fcst_bc.attrs["bias_correction_method"] = bias.attrs["bias_correction_method"]
    try:
        fcst_bc.attrs["bias_correction_period"] = bias.attrs["bias_correction_period"]
    except KeyError:
        pass

    return fcst_bc
