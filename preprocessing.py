import pathlib
import xarray as xr


def open_file(filename, variable, selection=None,
              to_datetime64=True, region=None, 
              time_var='time', lat_name='lat', lon_name='lon'):
    """Open a file and select data.
    
    Args:
      filename (str): File name
      variable (str): Variable to select from file
      selection (dict):
      to_datetime (bool): 
    
    """
    
    path = pathlib.Path(filename)    
    assert '.zarr' in path.suffixes, "Zarr files only"
    
    ds = xr.open_zarr(filename, consolidated=True, use_cftime=True)
    ds = ds[variable]
    
    #if to_datetime64: 
    #    ds = as_datetime64(ds, time_var=time_var)
        
    #for drop_coord in ['average_DT', 'average_T1', 'average_T2', 'zsurf', 'area']:
    #    if drop_coord in ds.coords:
    #        ds = ds.drop(drop_coord)
    
    if selection:
        ds = ds.sel(sel)
            
#    truncate_latitudes(get_region(ds, REGION, lat_name=lat_name, lon_name=lon_name), 
#                       lat_name).chunk({lat_name:-1,lon_name:-1})

    return ds 