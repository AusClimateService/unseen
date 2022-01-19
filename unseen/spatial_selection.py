"""Functions for spatial selection."""

import numpy as np
import xarray as xr
import geopandas as gp
import regionmask


def select_region(
    ds,
    coords=None,
    shapefile=None,
    header=None,
    combine_shapes=False,
    agg=None,
    lat_dim="lat",
    lon_dim="lon",
):
    """Select point, box or shapefile region.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    coords : list, optional
        Coordinates for point or box selection.
        List of length 2 [lat, lon] or 4 [south bound, north bound, east bound, west bound]
    shapefile : str, optional
        Shapefile for spatial subseting
    header : str, optional
        Name of the shapefile column containing the region names
    combine_shapes : bool, default False
        Add region that combines all shapes in shapefile
    agg : {'mean', 'sum'}, optional
        Spatial aggregation method
    lat_dim: str, default 'lat'
        Name of the latitude dimension in ds
    lon_dim: str, default 'lon'
        Name of the longitude dimension in ds

    Returns
    -------
    ds : xarray DataArray or Dataset
    """

    if coords is None:
        pass
    elif len(coords) == 4:
        ds = select_box_region(ds, coords)
    elif len(coords) == 2:
        ds = select_point_region(ds, coords)
    else:
        msg = "coordinate selection must be None, a box (list of 4 floats) or a point (list of 2 floats)"
        raise ValueError(msg)

    if shapefile:
        ds = select_shapefile_regions(
            ds, shapefile, agg=agg, header=header, combine_shapes=combine_shapes
        )

    if (agg == "sum") and not shapefile:
        ds = ds.sum(dim=("lat", "lon"))
    elif (agg == "mean") and not shapefile:
        ds = ds.mean(dim=("lat", "lon"))
    elif (agg is None) or shapefile:
        pass
    else:
        raise ValueError("""agg must be None, 'sum' or 'mean'""")

    return ds


def add_combined_shape(mask):
    """Add new region to mask that combines all other regions."""

    new_region_number = int(mask["region"].max()) + 1
    mask_combined = mask.max(dim="region")
    mask_combined = mask_combined.assign_coords(region=new_region_number).expand_dims(
        "region"
    )
    mask = xr.concat([mask, mask_combined], "region")

    return mask


def select_shapefile_regions(
    ds,
    shapefile,
    agg=None,
    header=None,
    combine_shapes=False,
    lat_dim="lat",
    lon_dim="lon",
):
    """Select region using a shapefile.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    shapefile : str
        File path for shapefile
    agg : {'mean', 'sum'}, optional
        Spatial aggregation method
    header : str, optional
        Name of the shapefile column containing the region names
    lat_dim: str, default 'lat'
        Name of the latitude dimension in ds
    lon_dim: str, default 'lon'
        Name of the longitude dimension in ds

    Returns
    -------
    ds : xarray DataArray or Dataset

    Notes
    -----
    regionmask requires the names of the horizontal spatial dimensions
    to be 'lat' and 'lon'

    """

    new_dim_names = {}
    if not lat_dim == "lat":
        new_dim_names[lat_dim] = "lat"
    if not lon_dim == "lon":
        new_dim_names[lon_dim] = "lon"
    if new_dim_names:
        ds = ds.rename_dims(new_dim_names)
    assert "lat" in ds.coords, "Latitude coordinate must be called lat"
    assert "lon" in ds.coords, "Longitude coordinate must be called lon"

    lons = ds["lon"].values
    lats = ds["lat"].values

    shapes = gp.read_file(shapefile)
    if agg is None:
        mask = regionmask.mask_geopandas(shapes, lons, lats)
        mask = xr.where(mask.notnull(), True, False)
        ds = ds.where(mask)
        if new_dim_names:
            old_dim_names = {y: x for x, y in new_dim_names.items()}
            ds = ds.rename_dims(old_dim_names)
    elif agg == "sum":
        mask = regionmask.mask_geopandas(shapes, lons, lats)
        ds = ds.groupby(mask).sum(keep_attrs=True)
    elif agg == "mean":
        mask = regionmask.mask_3D_geopandas(shapes, lons, lats)
        if combine_shapes:
            mask = add_combined_shape(mask)
        weights = np.cos(np.deg2rad(ds["lat"]))
        ds = ds.weighted(mask * weights).mean(dim=("lat", "lon"), keep_attrs=True)

    if header:
        shape_names = shapes[header].to_list()
        if combine_shapes:
            shape_names.append("all")
        ds = ds.assign_coords(region=shape_names)

    return ds


def select_box_region(ds, box, lat_dim="lat", lon_dim="lon"):
    """Select grid points that fall within a lat/lon box.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    box : list
        Box coordinates: [south bound, north bound, east bound, west bound]
    lat_dim: str, default 'lat'
        Name of the latitude dimension in ds
    lon_dim: str, default 'lon'
        Name of the longitude dimension in ds

    Returns
    -------
    ds : xarray DataArray or Dataset
    """

    lat_south_bound, lat_north_bound, lon_east_bound, lon_west_bound = box
    assert -90 <= lat_south_bound <= 90, "Valid latitude range is [-90, 90]"
    assert -90 <= lat_north_bound <= 90, "Valid latitude range is [-90, 90]"
    assert lat_south_bound < lat_north_bound, "South bound greater than north bound"

    lon_east_bound = (lon_east_bound + 360) % 360
    lon_west_bound = (lon_west_bound + 360) % 360
    assert 0 <= lon_east_bound <= 360, "Valid longitude range is [0, 360]"
    assert 0 <= lon_west_bound <= 360, "Valid longitude range is [0, 360]"

    ds = ds.assign_coords({lon_dim: (ds[lon_dim] + 360) % 360})
    ds = ds.sortby(ds[lon_dim])

    selection_lat = (ds[lat_dim] >= lat_south_bound) & (ds[lat_dim] <= lat_north_bound)
    if lon_east_bound < lon_west_bound:
        selection_lon = (ds[lon_dim] >= lon_east_bound) & (
            ds[lon_dim] <= lon_west_bound
        )
    else:
        selection_lon = (ds[lon_dim] >= lon_east_bound) | (
            ds[lon_dim] <= lon_west_bound
        )

    ds = ds.where(selection_lat & selection_lon, drop=True)

    return ds


def select_point_region(ds, point, lat_dim="lat", lon_dim="lon"):
    """Select a single grid point.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    point : list
        Point coordinates: [lat, lon]
    lat_dim: str, default 'lat'
        Name of the latitude dimension in ds
    lon_dim: str, default 'lon'
        Name of the longitude dimension in ds

    Returns
    -------
    ds : xarray DataArray or Dataset
    """

    ds = ds.assign_coords({lon_dim: (ds[lon_dim] + 360) % 360})
    ds = ds.sortby(ds[lon_dim])

    lat, lon = point
    lon = (lon + 360) % 360
    ds = ds.sel({lat_dim: lat, lon_dim: lon}, method="nearest", drop=True)

    return ds
