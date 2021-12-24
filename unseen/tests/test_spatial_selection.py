import pytest

from unseen.spatial_selection import (
    select_point_region,
)


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
