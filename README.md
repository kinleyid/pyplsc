# Python implementation of partial least squares correlation (PLSC)

![tests](https://github.com/kinleyid/pyplsc/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/kinleyid/pyplsc/graph/badge.svg?token=9B0F14E8XW)](https://codecov.io/gh/kinleyid/pyplsc)

## Background

PLSC is a multivariate statistical technique used in neuroscience ([McIntosh et al., 1994](https://doi.org/10.1006/nimg.1996.0016); [McIntosh & Lobaugh, 2004](https://doi.org/10.1016/j.neuroimage.2004.07.020); [Krishnan et al., 2011](https://doi.org/10.1016/j.neuroimage.2010.07.034)), among other fields. It uses compact singular value decomposition (SVD) to analyze relationships between a multivariate data array and a design matrix. When the object of study is brain-behaviour correlations or functional connectivity, this method is referred to as "behaviour PLSC" or "seed PLSC". In `pyplsc`, these are implemented by the `PLSC()` model class.

Multivariate categorical differences across experimental conditions can also be analyzed by applying SVD to matrices of condition-wise averages. This approach is called "mean-centred PLSC" or "barycentric discriminant analysis" (BDA; [Abdi et al., 2018](https://doi.org/10.1007/978-1-4614-7163-9_110192-2)) and is implemented in `pyplsc` by the `BDA` model class.

## Installation

`pyplsc` can be installed from PyPI with:

```
pip install pyplsc
```

`pyplsc` is tested with Python 3.10 and above but may also work with earlier versions.

## Usage

`pyplsc` replicates the statistical functionality of the [PLS Matlab package](https://www.rotman-baycrest.on.ca/index.php?section=84), much like the [`pyls`](https://github.com/netneurolab/pypyls) library. A major difference is that `pyplsc` uses a scikit-learn-style model-fitting syntax and accepts tabular (`pandas.DataFrame`) input:

```python
from pyplsc import PLSC, BDA

mod = PLSC(random_state=123)
mod.fit(data=data_array, covariates=cov_table)
```

Permutation testing and bootstrap resampling are then run as separate steps (possibly in parallel using the `n_jobs` parameter):

```python
perm_dist = mod.permute(n_perm=1000, n_jobs=3)
boot_dist = mod.bootstrap(n_boot=1000)
```

In contrast to other PLS implementations, `pyplsc` does not require data to be pre-sorted by (between-participant) group and (within-participant) condition:

```python
mod = BDA()
mod.fit(data=data_array,
		design=design_matrix_dataframe,
		between='group',
		within='cond',
		participant='subj')
```

See the [documentation](https://pyplsc.readthedocs.io/) for more details and examples.