Getting Started
===============

Dependencies
------------

The UNSEEN package depends on the following libraries: ::

   cftime cmdline_provenance dask dask-jobqueue geopandas gitpython matplotlib netCDF4 numpy pytest pyyaml regionmask scipy xarray xclim xskillscore xstatstests zarr

All these libraries can be installed via conda except ``xstatstests``,
which can be installed using pip.
To create a new environment with all the libraries installed,
use either of the following commands: 

.. code:: bash

   $ conda create -n unseen cftime cmdline_provenance dask dask-jobqueue ...
   $ conda activate unseen
   $ pip install xstatstests
   
or

.. code:: bash

   $ conda env create -f environment.yml


The ``environment.yml`` file includes other useful analysis libraries
such as cartopy and jupyter.


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
