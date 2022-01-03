import pytest

from unseen.spatial_selection import (
    select_box_region,
    select_point_region,
)


@pytest.mark.parametrize("example_da_lon", [180, 360], indirect=True)
@pytest.mark.parametrize(
    "box",
    [
        [-45, 20, 60, 120],
        [20, 30, 120, 60],
        [-35, -15, -170, -120],
        [0, 20, -120, -170],
    ],
)
@pytest.mark.parametrize("data_object", ["DataArray", "Dataset"])
def test_selected_box_lon(example_da_lon, box, data_object):
    """Test longitudes of box returned by select_box_region."""
    south_bound, north_bound, east_bound, west_bound = box
    east_bound360 = (east_bound + 360) % 360
    west_bound360 = (west_bound + 360) % 360

    data = example_da_lon
    if data_object == "Datatset":
        data = data.to_dataset(name="var")

    selection = select_box_region(data, box)

    if east_bound360 < west_bound360:
        assert selection.data.min() == east_bound360
        assert selection.data.max() == west_bound360
    else:
        segment1_max = float(selection.where(selection < east_bound360).max().data)
        segment2_min = float(selection.where(selection > west_bound360).min().data)
        assert segment1_max == west_bound360
        assert segment2_min == east_bound360


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

    selection = select_point_region(data, point)
    if data_object == "Datatset":
        data_lon = float(selection["var"].values)
    else:
        data_lon = float(selection.values)

    assert point_lon360 == data_lon
