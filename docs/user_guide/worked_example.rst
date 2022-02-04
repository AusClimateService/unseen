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


The ``fileio.open_dataset`` function can be used to open a data file/s as an xarray Dataset:

.. code:: python

    from unseen import fileio

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
is set within the ``open_dataset`` function,
so if you require a different order you can use the relevant functions
from the ``spatial_selection`` and ``time_utils`` modules on their own
to acheive the order you need.

We can then squeeze the redundant ``region`` dimension
(there's only one region in the shapefile)
and drop the years that are NaN because they didn't have data for all months:

.. code:: python

   agcd_ds = agcd_ds.squeeze(drop=True)
   agcd_ds = agcd_ds.dropna('time')
   print(agcd_ds)


::

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


If working in a notebook it can be a good idea to compute the Dataset before going to far,
otherwise the dask task graph can get out of control.

.. code:: python

   agcd_ds = agcd_ds.compute()


Model data
^^^^^^^^^^

The CAFE dataset consists of multiple forecast files - one for each initialisation date:

.. code:: python

   import glob

   cafe_files1990s = glob.glob('/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-199[5,6,7,8,9]*/atmos_isobaric_daily.zarr.zip')
   cafe_files2000s = glob.glob('/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-2*/atmos_isobaric_daily.zarr.zip')
   cafe_files = cafe_files1990s + cafe_files2000s
   cafe_files.sort()
   cafe_files


::

   ['/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-19950501/atmos_isobaric_daily.zarr.zip',
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-19951101/atmos_isobaric_daily.zarr.zip',
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-19960501/atmos_isobaric_daily.zarr.zip',
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-19961101/atmos_isobaric_daily.zarr.zip',
    ...
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-20190501/atmos_isobaric_daily.zarr.zip',
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-20191101/atmos_isobaric_daily.zarr.zip',
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-20200501/atmos_isobaric_daily.zarr.zip',
    '/g/data/xv83/dcfp/CAFE-f6/c5-d60-pX-f6-20201101/atmos_isobaric_daily.zarr.zip']


In order to open and combine a multi-file forecast data,
we can use the ``fileio.open_mfforecast`` function:

.. code:: python

   cafe_ds = fileio.open_mfforecast(cafe_files,
       variables=['pr'],
       spatial_coords=[-44, -11, 113, 154],
       shapefile='wheatbelt.zip',
       spatial_agg='mean',
       time_freq='A-DEC',
       time_agg='sum',
       input_freq='D',
       reset_times=True,
       metadata_file='../../config/dataset_cafe_monthly.yml',
       complete_time_agg_periods=True,
       units={'pr': 'mm day-1'},
       units_timing='middle'
   )


We've used similar keyword arguments as for the AGCD data
(``open_mfforecast`` uses ``open_dataset`` to open each individual file)
with a couple of additions:

-  Selecting a box region (using the ``spatial_coords`` argument) around your shapefile region can help reduce the memory required to work with the shapefile
-  The ``reset_times`` option ensures that after resampling (e.g. here we calculate the annual mean from daily data) the month assigned to each time axis value matches the initialisation month 
-  The ``units`` option allows you to convert the units of particular variables. You can choose (using the ``units_timing`` option) for the conversion to happen at the start (before spatial and temporal operations), middle (after the spatial but before the temporal operations) or end.

The only other thing we need to do is once again remove the redundant dimension:

.. code:: python

   cafe_ds = cafe_ds.squeeze(drop=True)
   cafe_ds = cafe_ds.compute()
   cafe_ds
   

::

   <xarray.Dataset>
   Dimensions:    (ensemble: 96, init_date: 52, lead_time: 11)
   Coordinates:
     * lead_time  (lead_time) int64 0 1 2 3 4 5 6 7 8 9 10
     * ensemble   (ensemble) int64 1 2 3 4 5 6 7 8 9 ... 88 89 90 91 92 93 94 95 96
     * init_date  (init_date) object 1995-05-01 00:00:00 ... 2020-11-01 00:00:00
       time       (lead_time, init_date) object 1995-05-01 12:00:00 ... 2030-11-...
   Data variables:
       pr         (init_date, lead_time, ensemble) float64 dask.array<chunksize=(1, 1, 96), meta=np.ndarray>
   Attributes:
       comment:    pressure level interpolator, version 3.0, precision=double
       filename:   atmos_isobaric_daily.zarr
       grid_tile:  N/A
       grid_type:  regular
       title:      AccessOcean-AM2


Bias correction
^^^^^^^^^^^^^^^

In order to bias correct the model data,
we can use the ``bias_correction`` module:

.. code:: python

   from unseen import bias_correction

   bias = bias_correction.get_bias(
       cafe_ds['pr'],
       agcd_ds['pr'],
       'additive',
       time_rounding='A',
       time_period=['2004-01-01', '2019-12-31']
   )
   bias


::

   TODO: Add dump of what the bias variable looks like to show it's a different bias for each lead time.


In this case we're using the additive (as opposed to multiplicative) bias correction method.
The bias represents the difference between model (CAFE) and observed (AGCD) climatology over the period 2004-2019. 
The first initialisation date is 1995 and each forecast is run for 10 years,
so each year over the 2004-2019 period is sampled the same number of times.
A separate bias is calculated for each lead time/year.

.. code:: python

   cafe_da_bc = bias_correction.remove_bias(cafe_ds['pr'], bias, 'additive')
   cafe_da_bc = cafe_da_bc.compute()
   cafe_da_bc


::

   TODO: Add dump of what this variable looks like
   
   
Independence testing
^^^^^^^^^^^^^^^^^^^^

We want to ensure that each sample in our model data is independent.
To do this, we can use the ``independence`` module:

.. code:: python

   mean_correlations, null_correlation_bounds = independence.run_tests(cafe_da_bc)


For each initialisation time/month,
``run_tests`` calculates the mean correlation between all the ensemble members (for each lead time)
as well as the bounds on zero correlation based on random sampling.

.. code:: python
    
   print(mean_correlations)   


::

   {5: <xarray.DataArray (lead_time: 11)>
 dask.array<mean_agg-aggregate, shape=(11,), dtype=float64, chunksize=(11,), chunktype=numpy.ndarray>
 Coordinates:
   * lead_time  (lead_time) int64 0 1 2 3 4 5 6 7 8 9 10,
 11: <xarray.DataArray (lead_time: 11)>
 dask.array<mean_agg-aggregate, shape=(11,), dtype=float64, chunksize=(11,), chunktype=numpy.ndarray>
 Coordinates:
   * lead_time  (lead_time) int64 0 1 2 3 4 5 6 7 8 9 10} 


The mean correlations and null correlation bounds can then be plotted:

.. code:: python

   independence.create_plot(
       mean_correlations,
       null_correlation_bounds,
       'wheatbelt_independence.png'
   )


.. image:: wheatbelt_independence.png
   :width: 500


(Lead time 0 and 10 aren't present because they didn't contain data for the full year.)


