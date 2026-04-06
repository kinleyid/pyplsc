
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from pdb import set_trace

def get_stratifier(design, output='ints'):
    # Get unique combinations of between and within factors
    multi_idx = pd.MultiIndex.from_frame(design[['between', 'within']])
    if output == 'ints':
        stratifier, _ = multi_idx.factorize()
    elif output == 'tuples':
        stratifier = multi_idx.to_list()
    return stratifier

def get_groupwise_means(data, group_idx):
    groups = np.unique(group_idx)
    # Initialize
    groupwise_means = np.zeros((len(groups), data.shape[1]), dtype=data.dtype)
    for i, group in enumerate(groups):
        groupwise_means[i] = data[group_idx == group].mean(axis=0)
    return groupwise_means

def pre_centre(data, design, pre_subtract):
    # Pre-subtract between- or within-wise means if applicable
    group_idx = design[pre_subtract].cat.codes
    rowwise_group_means = get_groupwise_means(data, group_idx)[group_idx]
    return data - rowwise_group_means

def get_mean_centred(data, design, stratifier=None, pre_subtract=None):
    if pre_subtract is not None:
        data = pre_centre(data, design, pre_subtract)
    # Compute group-wise means
    if stratifier is None: # Might not be pre-computed
        stratifier = get_stratifier(design)
    groupwise_means = get_groupwise_means(data, stratifier)
    # Mean centre
    mean_centred = groupwise_means - groupwise_means.mean(axis=0)
    return mean_centred

def corr(data, Y):
    # Compute a rectangular correlation matrix between data and Y
    datac = data - data.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    
    denom = data.shape[0] - 1
    stddata = np.sqrt((datac ** 2).sum(axis=0) / denom)
    stdY = np.sqrt((Yc ** 2).sum(axis=0) / denom)
    
    datan = datac / stddata
    Yn = Yc / stdY
    return datan.T @ Yn / denom

def get_stacked_cormats(data, covariates, stratifier):
    submatrices = []
    n_levels = stratifier.max() + 1
    for level in range(n_levels):
        idx = stratifier == level
        submatrix = corr(covariates[idx], data[idx])
        submatrices.append(submatrix)
    R = np.concat(submatrices)
    return R

def align(v, s, target_v, alignment_method):
    # Align with original decomposition
    if alignment_method == 'rotate':
        # Via rotation
        R, _ = orthogonal_procrustes(v, target_v, check_finite=False)
        aligned = v*s @ R
    elif alignment_method == 'flip':
        # Via correcting apparent sign flips
        flips = np.sign(np.diag(v.T @ target_v))
        aligned = v*s * flips
    return aligned
