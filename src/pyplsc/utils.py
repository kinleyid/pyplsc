
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

def get_covariates_array(design, covariates):
    # Take flexible input, return covariates as array and list of names
    if isinstance(covariates, np.ndarray):
        # covariates is an array---add custom names
        covariate_array = covariates
        if covariate_array.ndim == 1:
            # Reshape to column array
            covariate_array = covariate_array.reshape((len(covariates), 1))
        covariate_names = ['cov%s' % i for i in range(covariate_array.shape[1])]
    else:
        if isinstance(covariates, pd.DataFrame):
            # covariates is a dataframe
            covariate_array = covariates.to_numpy()
            covariate_names = covariates.columns.to_list()
        if isinstance(covariates, pd.Series):
            covariate_array = covariates.to_numpy().reshape(len(covariates), 1)
            covariate_names = [covariates.name]
        else:
            try:
                # covariates is a column indexer
                if isinstance(covariates, str):
                    covariates = [covariates]
                covariate_array = design[covariates].to_numpy()
                covariate_names = list(covariates)
            except:
                raise ValueError('Covariates must be a DataFrame or ndarray, or the names of the columns in the design matrix that contain the covariates')
    if covariate_array.ndim == 1:
        # Reshape to column array
        # Not redundant with earlier
        covariate_array = covariate_array.reshape((len(covariates), 1))
    return covariate_array, covariate_names

def get_groupwise_means(data, group_idx):
    n_groups = group_idx.max() + 1
    # Initialize
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

def stratified_average(data, labels, modeled):
    while any(~modeled):
        if len(modeled) == 1:
            # No more hierarchical structure
            # Average within this final level
            data = data.mean(axis=0, keepdims=True)
            break
        else:
            # Find lowest unmodeled level to average over
            avg_level = np.where(~modeled)[0][-1]
            # Stratify by all but the level at which averages are taken
            stratify = np.array([True]*len(modeled))
            stratify[avg_level] = False
            label_mi = pd.MultiIndex.from_arrays(labels[:, stratify].T)
            label_sets = label_mi.unique()
            Ms = []
            for label_set in label_sets:
                idx = label_mi == label_set
                M = data[idx].mean(axis=0)
                Ms.append(M)
            data = np.stack(Ms)
            # Create new, smaller labels matrix and modeled indicator
            labels = np.stack(label_sets)
            modeled = modeled[stratify]
    return data

def stratified_corrs(data, covariates, labels, modeled):
    # Compute correlations within clusters, and possible average within higher-level clusters
    assert any(~modeled)
    n_levels = labels.shape[1]
    assert len(modeled) == n_levels
    # Compute correlations across lowest unmodeled level
    corr_level = np.where(~modeled)[0][-1]
    # Stratify by all but the level at which correlations are computed
    stratify = np.array([True]*n_levels)
    stratify[corr_level] = False
    if n_levels == 1:
        R_mat = corr(covariates, data)
    else:
        label_mi = pd.MultiIndex.from_arrays(labels[:, stratify].T)
        label_sets = label_mi.unique()
        Rs = []
        for label_set in label_sets:
            idx = label_mi == label_set
            R = corr(covariates[idx], data[idx])
            Rs.append(R)
        R_mat = np.stack(Rs)
        # Update labels matrix to reflect the fact that the covariate level is no longer present
        modeled = modeled[stratify]
        if any(~modeled):
            labels = np.stack(label_sets)
            # First z-transform for averaging correlations
            z_mat = np.arctanh(R_mat)
            # Average within higher unmodeled levels
            z_mat = stratified_average(z_mat, labels, modeled)
            # Back-transform to R
            R_mat = np.tanh(z_mat)
        # Stack such that covariate is represented along first axis
        R_mat = np.concat(R_mat)
    return R_mat
        
def cluster_permute(labels, permute, rng, return_cov_perm=False):
    permuted_labels = labels.copy()
    n_obs, n_levels = labels.shape
    # Permute 0th level? Special case because 0th level can never be clustered
    if permute[0]:
        if n_levels == 1:
            # Labels at 0th level apply to individual observations
            depth = 1
        else:
            # Labels at 0th level apply to the clusters at the next level down
            depth = 2
        clusters = pd.MultiIndex.from_arrays(labels[:, :depth].T)
        unique_clusters = clusters.unique()
        # Shuffle the labels
        perm = rng.permutation(len(unique_clusters))
        permuted_level_labels = unique_clusters.get_level_values(0)[perm]
        # Expand labels
        mapping = dict(zip(unique_clusters, permuted_level_labels))
        permuted_level_labels = np.array([mapping[c] for c in clusters])
        permuted_labels[:, 0] = permuted_level_labels
    if n_levels > 1:
        for parent_level in range(n_levels - 1):
            level = parent_level + 1
            if permute[level]:
                parent_clusters = pd.MultiIndex.from_arrays(labels[:, :level].T)
                unique_parent_clusters = parent_clusters.unique()
                child_level = level + 1
                # Stratify permutations to be within the parent clusters
                for parent_cluster in unique_parent_clusters:
                    in_parent = parent_clusters == parent_cluster
                    if child_level == n_levels:
                        # No child---we're permuting the lowest-level observation labels
                        perm = rng.permutation(in_parent.sum())
                        permuted_labels[in_parent, level] = permuted_labels[in_parent, level][perm]
                    else:
                        # Permute labels at child level within clusters at current level
                        child_clusters = pd.MultiIndex.from_arrays(labels[in_parent, level:(child_level+1)].T)
                        unique_child_clusters = child_clusters.unique()
                        unique_level_labels = unique_child_clusters.get_level_values(0)
                        perm = rng.permutation(len(unique_level_labels))
                        permuted_level_labels = unique_level_labels[perm]
                        mapping = dict(zip(unique_child_clusters, permuted_level_labels))
                        permuted_level_labels = np.array([mapping[c] for c in child_clusters])
                        permuted_labels[in_parent, level] = permuted_level_labels
    out = (permuted_labels,)
    if return_cov_perm:
        cov_perm = np.arange(n_obs)
        if n_levels == 1:
            # Don't cluster
            perm = rng.permutation(n_obs)
            cov_perm = cov_perm[perm]
        else:
            # Cluster by second-lowest level
            level = n_levels - 1
            parent_clusters = pd.MultiIndex.from_arrays(labels[:, :level].T)
            unique_parent_clusters = parent_clusters.unique()
            # Stratify permutations within the parent clusters
            for parent_cluster in unique_parent_clusters:
                in_parent = parent_clusters == parent_cluster
                perm = rng.permutation(in_parent.sum())
                cov_perm[in_parent] = cov_perm[in_parent][perm]
        out += (cov_perm,)
    return out

def cluster_resample(labels, resample, rng):
    labels = labels.copy()
    n_obs, n_levels = labels.shape
    obs_id = np.arange(n_obs)
    # Resample 0th level? Special case because 0th level can never be clustered
    if resample[0]:
        # Get unique vals at 0th level
        level_0 = labels[:, 0]
        unique_vals = np.unique(level_0)
        # Resample them
        resampled_vals = rng.choice(unique_vals, len(unique_vals))
        # Map from values to rows
        mapping = {val: np.where(level_0 == val)[0] for val in unique_vals}
        resampled_rows = np.concat([mapping[val] for val in resampled_vals])
        labels = labels[resampled_rows]
        obs_id = obs_id[resampled_rows]
    for level in range(n_levels - 1):
        if resample[level + 1]:
            # curr_labels = labels[:, level]
            curr_clusters = pd.MultiIndex.from_arrays(labels[:, :(level + 1)].T)
            # Stratify children by current level
            # unique_labels = np.unique(curr_labels)
            unique_clusters = curr_clusters.unique()
            child_level = level + 1
            resampled_rows = []
            for unique_cluster in unique_clusters:
                # Get unique "children" to resample
                children = labels[curr_clusters == unique_cluster, child_level]
                unique_children = np.unique(children)
                # Resample them
                resampled_children = rng.choice(unique_children, len(unique_children))
                # Map from child to rows
                mapping = {child: np.where(labels[:, child_level] == child)[0] for child in unique_children}
                resampled_rows += [mapping[child] for child in resampled_children]
            resampled_rows = np.concat(resampled_rows)
            labels = labels[resampled_rows]
            obs_id = obs_id[resampled_rows]
    return resampled_rows

def df_to_tuples(df):
    return list(df.itertuples(index=False))