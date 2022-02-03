Worked examples
===============

Wheatbelt rainfall
------------------

The year 2019 was the driest on record for the Australian wheatbelt.
So dry, in fact, that wheat was imported into the country for the first time since 2006.

In this worked example,
we'll put this record dry year in context by applying the UNSEEN approach to
an observational dataset (`AGCD <http://www.bom.gov.au/metadata/catalogue/19115/ANZCW0503900567>`__)
and a large forecast ensemble (`CAFE <https://www.publish.csiro.au/ES/justaccepted/ES21024>`__).

Observational data
^^^^^^^^^^^^^^^^^^

We can use the AGCD rainfall data regridded to the CAFE model grid for our analysis.
(See `this repo <https://github.com/AusClimateService/agcd>`__ for details.)

.. code:: python

   agcd_file = '/g/data/ia39/agcd/post-processed/data/agcd_v2_precip_total_cafe-grid_monthly_1900-2020.zarr.zip'


The `fileio.open_dataset` function can be used to open a data file/s as an xarray Dataset:

.. code:: python

    agcd_ds = fileio.open_dataset(
        agcd_file,
        variables=['pr'],
        shapefile='wheatbelt.zip',
        spatial_agg='mean',
        shape_label_header='region',
        time_freq='A-DEC',
        time_agg='sum',
        input_freq='M',
        metadata_file='../../config/dataset_agcd_monthly.yml',
        complete_time_agg_periods=True
    )


In addition to opening the AGCD file,
we've asked the function to:

-  Edit the metadata of the data file / xarray Dataset according to the details in a :doc:`configuration file <configuration_files>`
-  Select the precipitation variable from the Dataset
-  Calculate the spatial mean across the wheatbelt (as defined in a shapefile)
-  Convert the monthly timescale data to an annual sum and only retain years where data for all months are available 

The order of operations here
(e.g. spatial before temporal selection and aggregation)
is set within the `open_dataset` function,
so if you require a different order you can use the relevant functions
from the `spatial_selection` and `time_utils` modules on their own
to acheive the order you need.

We can then simple sqeeze the redundant `region` dimension
(there's only one region in the shapefile)
and drop the years that are NaN because they didn't have data for all months:

.. code:: python

   agcd_ds = agcd_ds.squeeze(drop=True)
   agcd_ds = agcd_ds.dropna('time')
   agcd_ds


..code:: text

   <xarray.Dataset>
   Dimensions:  (time: 120)
   Coordinates:
     * time     (time) object 1900-12-31 00:00:00 ... 2019-12-31 00:00:00
   Data variables:
       pr       (time) float64 dask.array<chunksize=(1,), meta=np.ndarray>
   Attributes: (12/29)
       Conventions:               CF-1.6, ACDD-1.3
       acknowledgment:            The Australian Government, Bureau of Meteorolo...
       agcd_version:              AGCD v2.0.0 Snapshot (1900-01-01 to 2020-05-31)
       analysis_components:       total: the gridded accumulation of rainfall.
       attribution:               Data should be cited as : Australian Bureau of...
       cdm_data_type:             Grid
       ...                        ...
       summary:                   The monthly rainfall data represents the amoun...
       time_coverage_end:         1900-12-31T00:00:00
       time_coverage_start:       1900-01-01T00:00:00
       title:                     Interpolated Rain Gauge Precipitation
       url:                       http://www.bom.gov.au/climate/
       uuid:                      43596dc1-c56e-42a2-ba87-4e3b726a6e60


Model data
^^^^^^^^^^

The CAFE model data comes in multiple files:

..code:: python

   import glob

   cafe_files1990s = glob.glob('/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]*/atmos_isobaric_daily.zarr.zip')
   cafe_files2000s = glob.glob('/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2*/atmos_isobaric_daily.zarr.zip')
   cafe_files = cafe_files1990s + cafe_files2000s
   cafe_files.sort()
   cafe_files

Blah

