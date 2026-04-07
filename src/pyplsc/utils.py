
import numpy as np
import pandas as pd

from pdb import set_trace

def get_design_for_sorted(group_sizes, n_cond=1):
    between = []
    within = []
    participant = []
    min_ptpt_id = 0
    for group_id, group_size in enumerate(group_sizes):
        between += [group_id]*group_size*n_cond
        for cond_id in range(n_cond):
            within += [cond_id]*group_size
            participant += list(range(min_ptpt_id, min_ptpt_id + group_size))
        min_ptpt_id += group_size
    
    design = pd.DataFrame({
        'between': between,
        'within': within,
        'participant': participant})
    return design

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
