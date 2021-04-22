import xarray as xr
import numpy as np
import zarr
import pandas as pd

import myfuncs


jra_file = '/g/data/xv83/ds0092/data/csiro-dcfp-jra55/surface_daily_cafe-grid.zarr/'

ds_jra = xr.open_zarr(jra_file, consolidated=True, use_cftime=True)
ds_jra = ds_jra.rename({'initial_time0_hours': 'time'})

da_jra = ds_jra['TPRAT_GDS0_SFC']
da_jra = da_jra.rename('precip')
da_jra = myfuncs.get_region(da_jra, myfuncs.AUS_BOX)

stack_dates = np.array([np.datetime64('1984-11-01T00:00:00.000000000'),
                        np.datetime64('1985-11-01T00:00:00.000000000'),
                        np.datetime64('1986-11-01T00:00:00.000000000'),
                        np.datetime64('1987-11-01T00:00:00.000000000')])
N_lead_steps = 3653

da_jra = myfuncs.stack_by_init_date(da_jra, stack_dates, N_lead_steps)
