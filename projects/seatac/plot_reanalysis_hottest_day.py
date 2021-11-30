"""Plot hottest SeaTac day in observations"""

import pdb
import sys
script_dir = sys.path[0]
repo_dir = '/'.join(script_dir.split('/')[:-2])
module_dir = repo_dir + '/unseen'
sys.path.insert(1, module_dir)

import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr

import fileio
import general_utils


def plot_usa(da_tasmax, da_h500, outfile, metadata_key, command_log,
             data_source, point=None):
    """Plot map of USA

    Args:
      da_tasmax (xarray DataArray) : maximum temperature data
      da_h500 (xarray DataArray) : 500hPa geopotential height data
      outfile (str) : output file name
      metadata_key (str) : key for image metadata entry
      command_log (str) : commnad log for metadata entry
      data_source (str) : data source for title
      point (list) : coordinates of point to plot (lon, lat)
    """
    
    assert data_source in ['Observations', 'Model']

    fig = plt.figure(figsize=[12,5])
    map_proj = ccrs.LambertConformal(central_longitude=262.5, central_latitude=38.5, standard_parallels=[38.5, 38.5])
    h500_levels = np.arange(5000, 6300, 50)
    
    ax = fig.add_subplot(111, projection=map_proj)
    
    da_tasmax.plot(ax=ax,
                   transform=ccrs.PlateCarree(),
                   cmap=plt.cm.hot_r,
                   vmin=10, vmax=52,
                   cbar_kwargs={'label': 'maximum temperature (C)'}) #alpha=0.7
    
    lines = da_h500.plot.contour(ax=ax,
                                 transform=ccrs.PlateCarree(),
                                 levels=h500_levels,
                                 colors=['0.1'])
    ax.clabel(lines, colors=['0.1'], manual=False, inline=True)
    if point:
        lon, lat = point
        ax.plot(lon, lat, 'bo', transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([-140, -60, 20, 70])
    ax.gridlines(linestyle='--', draw_labels=True)
    ax.set_title(f'Hottest day: {data_source}')
    plt.savefig(outfile, metadata={metadata_key: command_log}, bbox_inches='tight', facecolor='white')


def _main(args):
    """Run the command line program."""

    general_utils.set_plot_params(args.plotparams)
    
    ds_hgt = xr.open_dataset(args.hgt_file, engine='cfgrib')
    da_h500 = ds_hgt['z'].mean('time')
    da_h500 = da_h500 / 9.80665

    ds_tas = xr.open_dataset(args.tas_file, engine='cfgrib')
    da_tasmax = ds_tas['t2m'].max('time')
    da_tasmax = da_tasmax - 273.15

    new_log = fileio.get_new_log(repo_dir=repo_dir)
    metadata_key = fileio.image_metadata_keys[args.outfile.split('.')[-1]]
    plot_usa(da_tasmax, da_h500, args.outfile, metadata_key, new_log,
             'Observations', point=args.point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("hgt_file", type=str, help="geopotential height file")
    parser.add_argument("tas_file", type=str, help="temperature file")
    parser.add_argument("outfile", type=str, help="output file")
    
    parser.add_argument('--point', type=float, nargs=2, metavar=('lon', 'lat'),
                        default=None, help='plot marker at this point')
    parser.add_argument('--plotparams', type=str, default=None,
                        help='matplotlib parameters (YAML file)')
    
    args = parser.parse_args()
    _main(args)
