# UNSEEN

[![tests](https://github.com/AusClimateService/unseen/actions/workflows/tests.yml/badge.svg)](https://github.com/AusClimateService/unseen/actions/workflows/tests.yml)
[![pre-commit](https://github.com/AusClimateService/unseen/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/AusClimateService/unseen/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/AusClimateService/unseen/branch/master/graph/badge.svg?token=OCNJ29LV5C)](https://codecov.io/gh/AusClimateService/unseen)

Python package for implementing the UNSEEN (UNprecedented Simulated Extremes using ENsembles;
[Thompson et al 2017](https://doi.org/10.1038/s41467-017-00275-3))
approach to assessing the likelihood of extreme events.

## Dependencies

The UNSEEN package depends on the following libraries: 
```
cftime cmdline_provenance dask dask-jobqueue geopandas gitpython matplotlib netCDF4 numpy pytest pyyaml regionmask scipy xarray xclim xskillscore zarr
```
... plus `xks` if you're doing similarity testing.

All these libraries (except `xks`) can be installed via conda.
To create a new environment with all the libraries installed,
use either of the following commands: 

```bash
$ conda create -n unseen cftime cmdline_provenance dask dask-jobqueue ...
$ conda env create -f environment.yml
```

The `environment.yml` file includes other useful analysis libraries
such as cartopy and jupyter.

The `xks` package is only needed for similarity testing
(i.e. when using `unseen/similarity.py`).
It isn't available on PyPI,
so in order to install it in your environment you'll need to clone
the `xks` repository and pip install as follows:

```bash
$ git clone https://github.com/dougiesquire/xks
$ cd xks
$ pip install .
```

## Installation

The UNSEEN package isn't currently available on PyPI,
so in order to install it in your conda environemnt along with all the dependencies
you'll need to close this repository and pip install as follows:

```bash
$ git clone https://github.com/AusClimateService/unseen
$ cd unseen
$ pip install .
```

If you're thinking of modifying and possibly contributing changes to the package,
follow the installation instructions in CONTRIBUTING.md instead.


## Module index

Key functions for UNSEEN analysis are contained within the modules listed below.
A number can also be run as command line programs.
For example, once you've pip installed the unseen package,
the `bias_correction.py` module can be run as a command line program
by simply running `bias_correction` at the command line
(use the `-h` option for details).

#### `array_handling.py`
Functions for array handling and manipulation:
- `stack_by_init_date`: Stack time series array in initial date / lead time format  
- `reindex_forecast`: Switch out lead_time axis for time axis (or vice versa) in a forecast dataset  
- `time_to_lead`: Convert from time to (a newly created) lead_time dimension

#### `bias_correction.py` 
Functions (and command line program) for bias correction:
- `get_bias`: Calculate forecast bias
- `remove_bias`: Remove model bias

#### `boostrap.py`
Utilities for repeated random sampling:
- `n_random_resamples`: Repeatedly randomly resample from provided xarray objects and return the results of the subsampled dataset passed through a provided function

#### `dask_setup.py`
Setup dask scheduling client:
- `launch_client`: Launch a dask client

#### `fileio.py`
Functions and command line program for file I/O:
- `open_file`: create an xarray dataset from a single netCDF or zarr file, with many processing options (spatial/temporal selection and aggregation, unit conversion, etc)
- `open_mfzarr`: open multiple zarr files
- `open_mfforecast`: open multi-file forecast
- `get_new_log`: generate command log for output file
- `to_zarr`: write to zarr file

#### `general_utils.py`
General utility functions:
- `convert_units`: Convert units

#### `independence.py`
Functions and command line program for independence testing

#### `indices.py`
Climate indices:
- `calc_drought_factor`: Calculate the Drought Factor index
- `calc_FFDI`: Calculate the McArthur Forest Fire Danger Index
- `calc_wind_speed`: Calculate wind speed
- `fit_gev`: Fit a GEV

#### `similarity.py` 
Funcitons and command line program for similarity testing:
- `univariate_ks_test`: Univariate KS test

#### `spatial_selection.py`
Functions for spatial selection (point, box and shapefile)

#### `time_utils.py`
Utilities for working with time axes and values


## Configuration files

### Data file metadata

When reading an input file using `fileio.open_file`
you can use the `metadata_file` keyword argument to pass
a YAML file specifying required file metadata changes.

The valid keys for the YAML file are
`rename`, `drop_coords`, `round_coords` and `units`.
The `rename` key is typically used to enforce standard variable names
(`lat`, `lon` and `time` are required by many functions).
For example,
the following YAML file renames a bunch of variables
(including `lat`, `lon` and `time`),
adds a missing units attribute,
deletes some unneeded coordinates,
and rounds coordinate values 
(see `fileio.fix_metadata` for details).

```yaml
rename:
  initial_time0_hours: time
  latitude: lat
  longitude: lon
  precip: pr
  u_ref: uas
  v_ref: vas

units:
  pr: 'mm d-1'

drop_coords:
  - average_DT
  - average_T1
  - average_T2
  - zsurf
  - area

round_coords:
  - lat
```

The `config/` directory contains a series of YAML files
that describe changes that need to be made to the metadata
associated with a number of different datasets.

### Dask

When launching a dask client using `dask_setup.launch_client` a configuration YAML file is required.
These files might look something like
the following for a local cluster, PBS Cluster (e.g. on NCI) and SLURM Cluster respectively:

```yaml
LocalCluster: {}
temporary_directory: /g/data/xv83/dbi599/
```

```yaml
PBSCluster:
  processes: 1
  walltime: '01:00:00'
  cores: 24
  memory: 48GB
  job_extra:
    - '-l ncpus=24'
    - '-l mem=48GB'
    - '-P ux06'
    - '-l jobfs=100GB'
    - '-l storage=gdata/xv83+gdata/v14+scratch/v14'
  local_directory: $PBS_JOBFS
  header_skip:
    - select
```

```yaml    
SLURMCluster:
  cores: 12
  memory: 72GB
  walltime: '02:00:00'
```

In other words, the YAML files contain the keyword arguments for
`dask.distributed.LocalCluster`, `dask_jobqueue.PBSCluster`
or `dask_jobqueue.SLURMCluster`.

