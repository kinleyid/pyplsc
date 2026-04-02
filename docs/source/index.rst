.. pyplsc documentation master file, created by
   sphinx-quickstart on Wed Apr  1 14:37:32 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyplsc
====================

A python implementation of partial least squares correlation (PLSC).

Installation
------------

pyplsc can be installed from pypi via:

.. code-block::

   pip install pyplsc

Quickstart
----------

The following is a brief example illustrating how 

.. code-block::

   from pyplsc import PLSC

   plsc = BDA().fit(data=X, covariates=Y design=table, between='group')
   plsc.permute(1000) # Permutation testing to assess significance
   plsc.bootstrap(1000) # Bootstrap resampling to assess reliability

   # Display results
   f, ax = plt.subplots(1, 2)

   # Plot bootstrap ratios
   ax[0].plot(mod.bootstrap_ratios_[:, 0])

   # Plot brain scores
   ax[1].bar(
      x=mod.get_labels()['between'],
      height=mod.boot_stat_[:, 0],
      yerr=mod.get_boot_stat_yerr(0))

Main objects
------------



.. toctree::
   :maxdepth: 2

   BDA
   PLSC
   notebooks/little-example
   notebooks/BDA