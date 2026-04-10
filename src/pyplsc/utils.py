
import numpy as np
import pandas as pd
from scipy.stats import zscore

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
        'between': pd.Categorical(between),
        'within': pd.Categorical(within),
        'participant': pd.Categorical(participant)})
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
    # Initialize
    n_groups = group_idx.max() + 1
    groupwise_means = np.zeros((n_groups, data.shape[1]), dtype=data.dtype)
    for i in range(n_groups):
        groupwise_means[i] = data[group_idx == i].mean(axis=0)
    return groupwise_means

def corr(cov, data):
    # Compute a rectangular correlation matrix between data and Y
    # z-score data and covariate
    data_z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    cov_z = (cov - cov.mean(axis=0)) / cov.std(axis=0, ddof=1)
    return (cov_z.T @ data_z) / (len(data_z) - 1)

def get_stacked_cormats(data, covariates, stratifier):
    submatrices = []
    n_levels = stratifier.max() + 1
    for level in range(n_levels):
        idx = stratifier == level
        submatrix = corr(covariates[idx], data[idx])
        submatrices.append(submatrix)
    R = np.concat(submatrices)
    return R
