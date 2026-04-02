
Introduction
============

Partial least squares correlation (PLSC) is a statistical technique that uses compact singular value decomposition to identify associations between a set of multivarite observations and a design matrix. Background information on this technique in the context of neuroimaging can be found in `Krishnan et al. (2011) <https://doi.org/10.1016/j.neuroimage.2010.07.034>`__

Unfortunately, a major challenge when learning PLSC is the vocabulary: different terms are sometimes used for the same underlying mathematical objects. The terminology 

A matrix :math:`X` with as many rows as we have observations and as many columns as we have variables. There is also a design matrix :math:`Y` that encodes information about experimental conditions and covariates collected. These are combined to create a matrix :math:`R` which is decomposed by compact singular value decomposition:

.. math::
	R = USV

Where :math:`U` is a matrix of "left singular vectors" and :math:`V` is a matrix of "right singular vectors". Singular vectors are also called "saliences" and are analogous to loadings in PCA. For clarity, in the context of neuroimaging, the terms "brain saliences" for :math:`V` and "design saliences" for :math:`U` are sometimes used. This lets us avoid having to remember an arbitrary left--right mapping. ``pyplsc`` uses the non-neuroimaging-specific terms "data saliences" for :math:`V` and "design saliences" for :math:`U`. The main point is that the latent variables for the multivariate data array are computed as :math:`XV` and the latent variables for the design matrix are computed using :math:`Y` and :math:`U`.

