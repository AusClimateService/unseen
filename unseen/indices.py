"""Climate indices."""

import pdb

import numpy as np
import xarray as xr


def calc_drought_factor(da, dim='time'):
    """Calculate the Drought Factor index.

    Args:
      pr20 (xarray DataArray) : Daily precipitation data
      dims (list) : Dimensions over which to perform calculation 

    Calculation:
      D = -10 * (pr20 - min(pr20)) / (max(pr20) - min(pr20)) + 10
        pr20 is the 20 day running sum
        min/max are along a temporal axis
    """

    pr20 = da.rolling({dim: 20}).sum()
    pr20_min = pr20.min(dim)
    pr20_max = pr20.max(dim)
    df = -10 * (pr20 - pr20_min) / (pr20_max - pr20_min) + 10
    df = df.rename('drought_factor')

    return df 
    

def calc_FFDI(ds, dim='time'):
    """Calculate the Forest Fire Danger Index.

    Args:
      ds (xarray Dataset): Containing the following variables:
        pr (daily precipitation)
        tasmax (daily maximum surface temperature) 
        hur (daily surface relative humidity)
        uas (daily eastward wind speed)
        vas (daily northward wind speed)
      dim (str): Temporal dimension over which to calculate FFDI

    Calculation:
      FFDI = DF**0.987 * e^(0.0338*tmax - 0.0345*rhmax + 0.0234*wmax + 0.243147)
        where df is the Drought Factor index,
              tmax the daily maximum 2m temperature 
              rhmax the daily maximum 2m relative humidity
              wmax the daily maximum 10m wind speed

    Reference:
      Richardson 2021
    """
    
    xr.set_options(keep_attrs=False)
    df = calc_drought_factor(ds['pr'], dim=dim)
    wsp = calc_wind_speed(ds['uas'], ds['vas'])

    ffdi_da = ( df ** 0.987 ) * np.exp( (0.0338 * ds['tasmax']) - (0.0345 * ds['hur']) + (0.0234 * wsp) + 0.243147 )
    ffdi_ds = ffdi_da.to_dataset(name='ffdi')

    return ffdi_ds


def calc_wind_speed(u, v):
    """Calculate wind speed."""

    wsp = xr.ufuncs.sqrt(u ** 2 + v ** 2)

    return wsp



