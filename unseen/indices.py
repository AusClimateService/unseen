"""Climate indices."""

import numpy as np
import xarray as xr
from scipy.stats import genextreme as gev


def calc_drought_factor(pr, time_dim="time", scale_dims=["time"]):
    """Calculate the Drought Factor index.

    20-day accumulated rainfall scaled to lie between 0 and 10,
      with larger values indicating less precipitation

    Args:
      pr (xarray DataArray) : Daily precipitation data
      time_dim (str) : Dimension along which to accumulate precipitation
      scale_dims (list) : Dimension(s) over which to compute the max and min
        precipitation to scale the accumulated precipitation

    Returns:
      df (xarray DataArray) : Drought Factor

    Calculation:
      df = -10 * (pr20 - min(pr20)) / (max(pr20) - min(pr20)) + 10

        pr20 is the 20 day running sum
        min/max are along a temporal axis
    """

    pr20 = pr.rolling({time_dim: 20}).sum()
    pr20_min = pr20.min(scale_dims)
    pr20_max = pr20.max(scale_dims)
    df = -10 * (pr20 - pr20_min) / (pr20_max - pr20_min) + 10
    df = df.rename("drought_factor")

    return df


def calc_FFDI(ds, time_dim="time", scale_dims=["time"]):
    """Calculate the McArthur Forest Fire Danger Index.

    Args:
      ds (xarray Dataset): Containing the following variables:
        pr (daily precipitation in mm)
        tasmax (daily maximum surface temperature in degrees C)
        hur (daily surface relative humidity)
        uas (daily eastward wind speed in km/h)
        vas (daily northward wind speed in km/h)
      dim (str): Temporal dimension over which to calculate FFDI

    Calculation:
      FFDI = DF**0.987 * e^(0.0338*tmax - 0.0345*rhmax + 0.0234*wmax + 0.243147)

      DF is the Drought Factor index
      - 20-day accumulated rainfall scaled to lie between 0 and 10, with larger values indicating less precipitation
      tmax is the daily maximum 2m temperature [mm]
      rhmax is the daily maximum 2m relative humidity [%] (or similar, depending on data availability)
      - Richardson et al (2021) uses mid-afternoon relative humidity at 2m
      - Squire et al (2021) uses daily mean relative humidity at 1000 hPa
      wmax the daily maximum 10m wind speed [km/h] (or similar, depending on data availability)
      - Squire et al. (2021) uses daily mean wind speed

    Reference:
      Dowdy, A. J. (2018). Climatological Variability of Fire Weather in Australia.
        Journal of Applied Meteorology and Climatology 57.2, pp. 221-234.
        doi: 10.1175/JAMC-D-17-0167.1.
    """

    xr.set_options(keep_attrs=False)
    ds["df"] = calc_drought_factor(ds["pr"], time_dim=time_dim, scale_dims=scale_dims)
    ds["wsp"] = calc_wind_speed(ds)

    ffdi = (ds["df"] ** 0.987) * np.exp(
        (0.0338 * ds["tasmax"]) - (0.0345 * ds["hur"]) + (0.0234 * ds["wsp"]) + 0.243147
    )

    return ffdi


def calc_wind_speed(ds):
    """Calculate wind speed.

    Args:
      ds (xarray Dataset): Containing the following variables:
        uas (daily eastward wind speed)
        vas (daily northward wind speed)

    Return:
      wsp (xarray DataArray): Wind speed
    """

    wsp = xr.ufuncs.sqrt(ds["uas"] ** 2 + ds["vas"] ** 2)

    return wsp


def fit_gev(data, use_estimates=False):
    """Fit a GEV by providing fit and scale estimates.

    Args:
      data (numpy ndarray)
      use_estimates (bool) : Fit GEV to data subset first to estimate parameters.
                             Useful for large datasets.
    """

    if use_estimates:
        shape_estimate, loc_estimate, scale_estimate = gev.fit(data[::2])
        shape, loc, scale = gev.fit(data, loc=loc_estimate, scale=scale_estimate)
    else:
        shape, loc, scale = gev.fit(data)

    return shape, loc, scale
