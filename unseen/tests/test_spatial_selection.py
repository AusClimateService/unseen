import pytest

import geopandas as gp
from shapely.geometry import Polygon
import numpy as np
import xarray as xr

from unseen.spatial_selection import (
    select_point,
    subset_lon,
    select_shapefile_regions,
)


# Box selection tests


@pytest.mark.parametrize("example_da_lon", [360], indirect=True)
@pytest.mark.parametrize(
    "lon_bnds",
    [
        [120, 60],
        [60, 120],
    ],
)
@pytest.mark.parametrize("data_object", ["DataArray", "Dataset"])
def test_selected_lon_bnds(example_da_lon, lon_bnds, data_object):
    """Test longitudes returned by subset_lon."""
    west_bound, east_bound = lon_bnds
    data = example_da_lon
    if data_object == "Datatset":
        data = data.to_dataset(name="var")

    selection = subset_lon(data, lon_bnds)

    if east_bound > west_bound:
        assert selection.data.max() == east_bound
        assert selection.data.min() == west_bound
    else:
        assert selection.data.max() == 359
        assert selection.data.min() == 0
        assert np.diff(selection["lon"].values).max() == west_bound - east_bound


# Point selection tests


@pytest.mark.parametrize("example_da_lon", [180, 360], indirect=True)
@pytest.mark.parametrize("point", [[-55, 50], [60, -80]])
@pytest.mark.parametrize("data_object", ["DataArray", "Dataset"])
def test_selected_point_lon(example_da_lon, point, data_object):
    """Test longitude of point returned by select_point_region."""
    point_lat, point_lon = point
    point_lon360 = (point_lon + 360) % 360

    data = example_da_lon
    if data_object == "Datatset":
        data = data.to_dataset(name="var")

    selection = select_point(data, point)
    if data_object == "Datatset":
        data_lon = float(selection["var"].values)
    else:
        data_lon = float(selection.values)

    assert point_lon360 == data_lon


# Shape selection tests


@pytest.fixture
def shapes():
    nw = Polygon([(120, -20.25), (122, -20.25), (122, -21.75), (120, -21.75)])
    ne = Polygon([(122, -20.25), (125, -20.25), (125, -21.75), (122, -21.75)])
    sw = Polygon([(120, -21.75), (123, -21.75), (123, -23.25), (120, -23.25)])
    se = Polygon([(123, -21.75), (125, -21.75), (125, -23.25), (123, -23.25)])
    gdf = gp.GeoDataFrame(
        {
            "region": ["north-west", "north-east", "south-west", "south-east"],
            "geometry": gp.GeoSeries([nw, ne, sw, se]),
        }
    )
    return gdf


@pytest.fixture
def example_da():
    data = np.reshape(np.arange(0, 7 * 4), [4, 7])
    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={
            "lat": [-23.5, -22.5, -21.5, -20.5],
            "lon": [119.5, 120.5, 121.5, 122.5, 123.5, 124.5, 125.5],
        },
    )
    da = da.expand_dims({"ensemble": [0, 1, 2]})
    return da


@pytest.mark.parametrize("agg", ["sum", "weighted_mean"])
@pytest.mark.parametrize("overlap", [None, 0.5, 0.1])
@pytest.mark.parametrize("data_object", ["DataArray", "Dataset"])
def test_shape_agg(example_da, shapes, agg, overlap, data_object):
    """Test shape aggregates.

    See shapefile_test_sandbox.ipynb for details.
    """
    ds = example_da
    if data_object == "Dataset":
        ds = ds.to_dataset(name="var")

    ds_agg = select_shapefile_regions(
        ds,
        shapes,
        agg=agg,
        overlap_fraction=overlap,
    )

    expected_results = {
        (0.1, "sum"): np.array([76, 129, 81, 69]),
        (0.5, "sum"): np.array([76, 129, 27, 23]),
        (None, "sum"): np.array([76, 129, 27, 23]),
        (0.1, "weighted_mean"): np.array(
            [19.01172476, 21.51172476, 9.03373887, 11.53373887]
        ),
        (0.5, "weighted_mean"): np.array([19.01172476, 21.51172476, 9, 11.5]),
        (None, "weighted_mean"): np.array([19.01172476, 21.51172476, 9, 11.5]),
    }
    expected_result = expected_results[(overlap, agg)]
    if data_object == "Dataset":
        actual_result = ds_agg["var"].isel({"ensemble": 0}).values
    else:
        actual_result = ds_agg.isel({"ensemble": 0}).values

    assert np.allclose(expected_result, actual_result)
