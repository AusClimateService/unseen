import pytest

import numpy as np
from xarray.coding.times import cftime_to_nptime

from unseen.time_utils import (
    select_time_period,
)


@pytest.mark.parametrize("example_da_forecast", ["numpy"], indirect=True)
@pytest.mark.parametrize("add_nans", [False, True])
@pytest.mark.parametrize("data_object", ["DataArray", "Dataset"])
def test_select_time_period(example_da_forecast, add_nans, data_object):
    """Test values returned by select_time_period"""
    PERIOD = ["2000-06-01", "2001-06-01"]

    data = example_da_forecast
    if data_object == "Datatset":
        data = data.to_dataset(name="var")

    if add_nans:
        time_nans = data[pytest.TIME_DIM].where(data[pytest.LEAD_DIM] > 3)
        data = data.assign_coords({pytest.TIME_DIM: time_nans})

    masked = select_time_period(data, PERIOD)

    min_time = cftime_to_nptime(data[pytest.TIME_DIM].where(masked.notnull()).min())
    max_time = cftime_to_nptime(data[pytest.TIME_DIM].where(masked.notnull()).max())

    assert min_time >= np.datetime64(PERIOD[0])
    assert max_time <= np.datetime64(PERIOD[1])
