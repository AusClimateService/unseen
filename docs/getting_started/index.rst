Getting Started
===============

Dependencies
------------

The UNSEEN package depends on the following libraries: ::

   cftime cmdline_provenance dask dask-jobqueue geopandas gitpython matplotlib netCDF4 numpy pytest pyyaml regionmask scipy xarray xclim xskillscore zarr

... plus ``xks`` if you're doing similarity testing.

All these libraries (except ``xks``) can be installed via conda.
To create a new environment with all the libraries installed,
use either of the following commands: 

.. code:: bash

   $ conda create -n unseen cftime cmdline_provenance dask dask-jobqueue ...
   $ conda env create -f environment.yml


The ``environment.yml`` file includes other useful analysis libraries
such as cartopy and jupyter.

The ``xks`` package is only needed for similarity testing
(i.e. when using `unseen/similarity.py`).
It isn't available on PyPI,
so in order to install it in your environment you'll need to clone
the ``xks`` repository and pip install as follows:

.. code:: bash

   $ git clone https://github.com/dougiesquire/xks
   $ cd xks
   $ pip install .


Installation
------------

The UNSEEN package isn't currently available on PyPI,
so in order to install it in your conda environemnt (along with all the dependencies)
you'll need to clone this repository and pip install as follows:

.. code:: bash

   $ git clone https://github.com/AusClimateService/unseen
   $ cd unseen
   $ pip install .


If you're thinking of modifying and possibly contributing changes to the package,
follow the installation instructions in
`CONTRIBUTING.md <https://github.com/AusClimateService/unseen/blob/master/CONTRIBUTING.md>`__
instead.
