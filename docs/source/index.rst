.. pyplsc documentation master file, created by
   sphinx-quickstart on Wed Apr  1 14:37:32 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyplsc
====================

A python implementation of partial least squares correlation (PLSC). For details on this method, see `Krishnan et al. (2011) <https://doi.org/10.1016/j.neuroimage.2010.07.034>`_.

Installation
------------

``pyplsc`` can be installed from PyPI via:

.. code-block::

   pip install pyplsc

Quickstart
----------

``pyplsc`` uses ``scikit-learn``-style syntax for model fitting:

.. code-block::

   from pyplsc import PLSC

   plsc = PLSC().fit(data=X, covariates=Y, labels=table)
   plsc.permute(1000) # Permutation testing to assess significance
   plsc.bootstrap(1000) # Bootstrap resampling to assess reliability

The examples and API explain these methods in greater depth.

.. toctree::
   :maxdepth: 1
   :hidden:

   notebooks/background
   examples
   api