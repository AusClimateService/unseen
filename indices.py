"""Climate indices."""

import xarray as xr


def calc_drought_factor(pr20, dims):
    """Calculate the Drought Factor index.

    Args:
      pr20 (xarray DataArray) : Precipitation data (20 day running sum)
      dims (list) : dimensions over which to perform calculation 

    Calculation:
      D = -10 * (pr20 - min(pr20)) / (max(pr20) - min(pr20)) + 10
        where the min and max are over time
    """

    pr20_min = pr20.min(dims)
    pr20_max = pr20.max(dims)
    df = -10 * (pr20 - pr20_min) / (pr20_max - pr20_min) + 10
    df = df.rename('drought_factor')

    return df 
    

def calc_FFDI(df, tmax, rhmax, wmax):
    """Calculate the Forest Fire Danger Index.

    Args:
      df (xarray DataArray):  Drought Factor index
      tmax (xarray DataArray) : Daily maximum 2m temperature 
      rhmax (xarray DataArray) : Daily maximum 2m relative humidity
      wmax (xarray DataArray) : Daily maximum 10m wind speed

    Reference:
      Richardson 2021
    """
    
    ffdi = ( ( df ** 0.987 ) * np.exp( (0.0338 * tmax) - (0.0345 * rhmax) + (0.0234 * wmax) + 0.243147 ) )

    return ffdi


def calc_wind_speed(u, v):
    """Calculate wind speed."""

    wsp = xr.ufuncs.sqrt(u ** 2 + v ** 2).to_dataset(name='V_ref')

    return wsp



