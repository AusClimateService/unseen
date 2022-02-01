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

The first step is the read the data file,
subsetting and aggregating in space and time as required:

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

TODO: Complete worked example. 

