Configuration files
===================

Data file metadata
------------------

When reading an input file using ``fileio.open_dataset``
you can use the ``metadata_file`` keyword argument to pass
a YAML file specifying required file metadata changes.

The valid keys for the YAML file are
``rename``, ``drop_coords``, ``round_coords`` and ``units``.
The ``rename`` key is typically used to enforce standard variable names
(``lat``, ``lon`` and ``time``).
For example,
the following YAML file renames a bunch of variables
(including ``lat``, ``lon`` and ``time``),
adds a missing units attribute,
deletes some unneeded coordinates,
and rounds coordinate values 
(see ``fileio._fix_metadata`` for details).

.. code:: yaml

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


The ``config/`` directory contains a series of YAML files
that describe changes that need to be made to the metadata
associated with a number of different datasets.


Dask
----

When launching a dask client using ``dask_setup.launch_client`` a configuration YAML file is required.
These files might look something like
the following for a local cluster, PBS Cluster (e.g. on NCI) and SLURM Cluster respectively:

.. code:: yaml

   LocalCluster: {}
   temporary_directory: /g/data/xv83/dbi599/


.. code:: yaml

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


.. code:: yaml

   SLURMCluster:
     cores: 12
     memory: 72GB
     walltime: '02:00:00'


In other words, the YAML files contain the keyword arguments for
``dask.distributed.LocalCluster``, ``dask_jobqueue.PBSCluster``
or ``dask_jobqueue.SLURMCluster``.
