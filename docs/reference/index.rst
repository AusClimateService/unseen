API Reference
=============

Key functions for large forecast ensemble analysis are contained within the modules listed below.
A number can also be run as command line programs.
For example, once you've installed the unseen package,
the ``bias_correction.py`` module can be run as a command line program
by running ``bias_correction`` at the command line
(use the ``-h`` option for details).

.. autosummary::
   :toctree: ../_autosummary
   :template: custom-module-template.rst
   :recursive:

   unseen.array_handling
   unseen.bias_correction
   unseen.bootstrap
   unseen.dask_setup
   unseen.eva
   unseen.fileio
   unseen.general_utils
   unseen.independence
   unseen.indices
   unseen.similarity
   unseen.spatial_selection
   unseen.tests
   unseen.time_utils
