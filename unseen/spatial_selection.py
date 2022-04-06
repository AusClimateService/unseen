"""Functions for spatial selection."""

import math

import numpy as np
import xarray as xr
import geopandas as gp
import regionmask


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


def select_box_region(ds, box, agg="none", lat_dim="lat", lon_dim="lon"):
    """Select grid points that fall within a lat/lon box.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    box : list
        Box coordinates: [south bound, north bound, east bound, west bound]
    agg : {'mean', 'sum', 'weighted_mean', 'none'}, default 'none'
        Spatial aggregation method
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

    if agg == "sum":
        ds = ds.sum(dim=(lat_dim, lon_dim))
    elif agg == "mean":
        ds = ds.mean(dim=(lat_dim, lon_dim))
    elif agg == "weighted_mean":
        weights = np.cos(np.deg2rad(ds[lat_dim]))
        ds = ds.weighted(weights).mean(dim=(lat_dim, lon_dim), keep_attrs=True)
    elif agg == "none":
        pass
    else:
        raise ValueError("""Invalid spatial aggregation method""")

    return ds


def select_shapefile_regions(
    ds,
    shapefile,
    agg="none",
    method="centre",
    min_overlap=0.5,
    header=None,
    combine_shapes=False,
    lat_dim="lat",
    lon_dim="lon",
):
    """Select region/s using a shapefile.

    Parameters
    ----------
    ds : xarray DataArray or Dataset
    shapefile : str
        File path for shapefile
    agg : {'mean', 'sum', 'weighted_mean', 'none'}, default 'none'
        Spatial aggregation method
    method : {'centre', 'touch', 'fraction'}, default 'centre'
        Grid cell selection method
    min_overlap : float, default 0.5
        Minimum fractional overlap used for 'fraction' method
    header : str, optional
        Name of the shapefile column containing the region names
    combine_shape : bool, default False
        Create an extra region which combines them all
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

    if combine_shapes:
        assert agg == "weighted_mean", "Combining shapes only works for weighted mean"

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

    if method == "centre":
        outdim = "3D" if ("weighted" in agg) else "2D"
        mask = centre_mask(shapes, lons, lats, output=outdim)
    elif method == "fraction":
        mask = fraction_overlap_mask(shapes, lons, lats, min_overlap)
    elif method == "touch":
        if "weighted" in agg:
            mask = fraction_overlap_mask(shapes, lons, lats, 0.001)
        else:
            mask = touch_mask(shapes, lons, lats)
    else:
        return ValueError("Invalid selection method")

    if (mask.ndim == 3) and not ("weighted" in agg):
        mask = mask.max("region")

    if agg == "none" :
        if mask.ndim == 2:
            mask = _mask_to_np_bool(mask)
        ds = ds.where(mask)
        ds = ds.dropna(lat_dim, how="all")
        ds = ds.dropna(lon_dim, how="all")
    elif agg == "sum":
        ds = ds.groupby(mask).sum(keep_attrs=True)
        ds = _squeeze_and_drop_region(ds)
    elif agg == "mean":
        ds = ds.groupby(mask).mean(keep_attrs=True)
        ds = _squeeze_and_drop_region(ds)
    elif agg == "weighted_mean":
        if combine_shapes:
            mask = _add_combined_shape(mask)
        else:
            mask = _squeeze_and_drop_region(mask)
        weights = np.cos(np.deg2rad(ds["lat"]))
        ds = ds.weighted(mask * weights).mean(dim=("lat", "lon"), keep_attrs=True)

    if header:
        shape_names = shapes[header].to_list()
        if combine_shapes:
            shape_names.append("all")
        ds = ds.assign_coords(region=shape_names)

    return ds


def centre_mask(shapes_gp, lons, lats, output="2D"):
    """Create an array indicating grid cells whose centre falls within each shape.

    Parameters
    ----------
    shapes_gp : geopandas GeoDataFrame
        Shapes/regions
    lons : numpy ndarray
        Grid longitude values
    lats : numpy ndarray
        Grid latitude values
    output : {'2D', '3D'}
        Dimensions for output array

    Returns
    -------
    mask : xarray DataArray
        For 2D (i.e. lat/lon) output values are a region number or NaN
        For 3D (i.e. region/lat/lon) output values are bool
    """

    if output == "2D":
        mask = regionmask.mask_geopandas(shapes_gp, lons, lats)
    elif output == "3D":
        mask = regionmask.mask_3D_geopandas(shapes_gp, lons, lats)
    else:
        raise ValueError("""Output argument must be '2D' and '3D'""")

    return mask


def touch_mask(shapes_gp, lons, lats):
    """Create a 2D region value array indicating grid cells touched by each shape.

    Parameters
    ----------
    shapes_gp : geopandas GeoDataFrame
        Shapes/regions
    lons : numpy ndarray
        Grid longitude values
    lats : numpy ndarray
        Grid latitude values

    Returns
    -------
    mask_2D : numpy ndarray
        Two dimensional (i.e. lat/lon) region value array.
        Values are a region number or NaN.

    Notes
    -----
    From https://github.com/regionmask/regionmask/issues/225#issuecomment-915033614
    2D region value arrays don't allow for overlapping regions.
    Must be a regularly spaced grid.
    """

    _check_regular_grid(lons)
    _check_regular_grid(lats)

    shapes_rm = regionmask.from_geopandas(shapes_gp)
    mask_2D = regionmask.core.mask._mask_rasterize(
        lons,
        lats,
        shapes_rm.polygons,
        shapes_rm.numbers,
        all_touched=True
    )

    return mask_2D


def fraction_overlap_mask(shapes_gp, lons, lats, min_overlap):
    """Create a 3D boolean array for grid cells over the shape overlap threshold.

    Parameters
    ----------
    shapes_gp : geopandas GeoDataFrame
        Shapes/regions
    lons : numpy ndarray
        Grid longitude values
    lats : numpy ndarray
        Grid latitude values
    threshold : float
        Minimum fractional overlap

    Returns
    -------
    mask_3D : xarray DataArray
        Three dimensional (i.e. region/lat/lon) boolean array
    """

    assert min_overlap > 0.0, "Minimum overlap must be fractional value > 0"
    assert min_overlap <= 1.0, "Minimum overlap must be fractional value <= 1.0"
    _check_regular_grid(lons)
    _check_regular_grid(lats)

    shapes_rm = regionmask.from_geopandas(shapes_gp)
    fraction = overlap_fraction(shapes_rm, lons, lats)
    fraction = _squeeze_and_drop_region(fraction)
    mask_3D = fraction > min_overlap

    return mask_3D


def overlap_fraction(shapes_rm, lons, lats):
    """Calculate the fraction of overlap of regions with lat/lon grid cells.

    Parameters
    ----------
    shapes_rm : regionmask.Regions
        Shapes/regions
    lons : numpy ndarray
        Grid longitude values
    lats : numpy ndarray
        Grid latitude values

    Returns
    -------
    mask_sampled : xarray DataArray
        Three dimensional (i.e. region/lat/lon) array of overlap fractions

    Notes
    -----
    From https://github.com/regionmask/regionmask/issues/38
    Assumes an equally spaced lat/lon grid
    """

    # sample with 10 times higher resolution
    lons_sampled = _sample_coord(lons)
    lats_sampled = _sample_coord(lats)

    mask = shapes_rm.mask(lons_sampled, lats_sampled)
    isnan = np.isnan(mask.values)
    numbers = np.unique(mask.values[~isnan])
    numbers = numbers.astype(np.int)

    mask_sampled = list()
    for num in numbers:
        # coarsen the mask again
        mask_coarse = (mask == num).coarsen(lat=10, lon=10).mean()
        mask_sampled.append(mask_coarse)

    mask_sampled = xr.concat(
        mask_sampled, dim="region", compat="override", coords="minimal"
    )
    mask_sampled = mask_sampled.assign_coords(region=("region", numbers))

    return mask_sampled


def _sample_coord(coord):
    """Sample coordinates for the fractional overlap calculation."""

    d_coord = coord[1] - coord[0]
    n_cells = len(coord)
    left = coord[0] - d_coord / 2 + d_coord / 20
    right = coord[-1] + d_coord / 2 - d_coord / 20

    return np.linspace(left, right, n_cells * 10)


def _check_regular_grid(dim_values):
    """Check that a grid (e.g. lat or lon) has uniform spacing."""

    spaces = np.diff(dim_values)
    min_spacing = np.max(spaces)
    max_spacing = np.min(spaces)
    assert math.isclose(min_spacing, max_spacing), "Grid spacing must be uniform"


def _add_combined_shape(mask):
    """Add new region to mask that combines all other regions."""

    new_region_number = int(mask["region"].max()) + 1
    mask_combined = mask.max(dim="region")
    mask_combined = mask_combined.assign_coords(region=new_region_number).expand_dims(
        "region"
    )
    mask = xr.concat([mask, mask_combined], "region")

    return mask


def _mask_to_np_bool(mask):
    """Convert a mask to a numpy array of dtype bool."""

    if type(mask) == np.ndarray:
        if mask.dtype != "bool":
            # Case 1: Numpy Array of NaNs and region numbers
            mask = ~np.isnan(mask)
    elif mask.values.dtype != "bool":
        # Case 2: xarray DataArray of NaNs and region numbers
        mask = xr.where(mask.notnull(), True, False)
        mask = mask.values
    else:
        # Case 3: xarray DataArray of bools
        mask = mask.values

    assert type(mask) == np.ndarray
    assert mask.dtype == "bool"

    return mask


def _squeeze_and_drop_region(ds):
    """Squeeze and drop region dimension if necessary."""

    ds = ds.squeeze()
    try:
        if ds['region'].size <= 1:
            ds = ds.drop('region')
    except KeyError:
        pass

    return ds
