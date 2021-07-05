"""General utility functions."""

import pdb
import argparse

import xclim


class store_dict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, val = value.split('=')
            if ':' in val:
                start, end = val.split(':')
                try:
                    start = int(start)
                    end = int(end)
                except ValueError:
                    pass
                val = slice(start, end)
            else:
                try:
                    val = int(val)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = val


def convert_units(da, target_units):
    """Convert units.
    
    Args:
      da (xarray DataArray)
      target_units (str)
    """

    xclim_unit_check = {'deg_k': 'degK'}

    if da.attrs['units'] in xclim_unit_check:
        da.attrs['units'] = xclim_unit_check[da.units]

    da = xclim.units.convert_units_to(da, target_units)

    return da
