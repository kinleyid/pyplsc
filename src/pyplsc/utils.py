
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

def corr(cov, data):
    # Compute a rectangular correlation matrix between data and Y
    # z-score data and covariate
    data_z = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    cov_z = (cov - cov.mean(axis=0)) / cov.std(axis=0, ddof=1)
    return (cov_z.T @ data_z) / (len(data_z) - 1)

def mean_center(matrix):
    out = matrix - matrix.mean(axis=0)
    return out

def stratified_average(data, labels, modeled, baseline=None):
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
            
            unique_labels, label_ids = np.unique(labels[:, stratify], axis=0, return_inverse=True)
            Ms = []
            for label_id in range(len(unique_labels)):
                mask = label_ids == label_id
                M = data[mask].mean(axis=0)
                Ms.append(M)
            
            data = np.stack(Ms)
            # Create new, smaller labels matrix and modeled indicator
            labels = np.stack(unique_labels)
            modeled = modeled[stratify]
    '''
    if baseline is not None:
        if baseline == 'add':
            baseline_val = 0
        elif baseline == 'div':
            baseline_val = 1
        baseline_row = baseline_val*np.ones_like(data[[0]])
        # baseline_row = baseline_val*np.ones_like(data)
        data = np.concat((data, baseline_row))
    '''
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
        unique_labels, label_ids = np.unique(labels[:, stratify], axis=0, return_inverse=True)        
        Rs = []
        for label_id in range(len(unique_labels)):
            mask = label_ids == label_id
            R = corr(covariates[mask], data[mask])
            Rs.append(R)
            
        R_mat = np.stack(Rs)
        # Update labels matrix to reflect the fact that the covariate level is no longer present
        modeled = modeled[stratify]
        if any(~modeled):
            labels = np.stack(unique_labels)
            # First z-transform for averaging correlations
            z_mat = np.arctanh(R_mat)
            # Average within higher unmodeled levels
            z_mat = stratified_average(z_mat, labels, modeled)
            # Back-transform to R
            R_mat = np.tanh(z_mat)
            # R_mat = z_mat
        # Stack such that covariate is represented along first axis
        R_mat = np.concat(R_mat)
    # R_mat = np.abs(R_mat).mean(axis=0, keepdims=True)
    # R_mat = R_mat - R_mat.mean()
    return R_mat

def cluster_permute(labels, permute, rng, return_cov_perm=False, return_flips=False):
    permuted_labels = labels.copy()
    n_obs, n_levels = labels.shape

    # Vectorized row comparisons
    def rows_equal(arr, row):
        return np.all(arr == row, axis=1)

    # Permute values within groups
    def permute_level_within_groups(parent_cols, level_col, child_col=None):
        """
        For every unique combination in parent_cols, shuffle the label in
        level_col (and, when child_col is provided, track its grouping too).
        Returns updated level_col.
        """
        result = level_col.copy()
        unique_parents, parent_inv = np.unique(parent_cols, axis=0, return_inverse=True)
        for idx in range(len(unique_parents)):
            mask = parent_inv == idx
            if child_col is None:
                # Lowest level of labels: shuffle individual observations
                result[mask] = rng.permutation(level_col[mask])
            else:
                # Shuffle child-cluster labels
                sub = np.stack([level_col[mask], child_col[mask]], axis=1)
                unique_sub, sub_inv = np.unique(sub, axis=0, return_inverse=True)
                unique_vals = unique_sub[:, 0]
                perm = rng.permutation(len(unique_vals))
                result[mask] = unique_vals[perm][sub_inv]
        return result

    # Level 0 (special case, cannot have a parent level)
    if permute[0]:
        if n_levels == 1:
            depth = 1
        else:
            depth = 2 # Labels at this level apply to a child level
        clusters = labels[:, :depth]
        unique_clusters, inv = np.unique(clusters, axis=0, return_inverse=True)
        perm = rng.permutation(len(unique_clusters))
        permuted_labels[:, 0] = unique_clusters[perm, 0][inv]   # vectorized remap

    # Remaining levels, if any
    if n_levels > 1:
        for level in range(1, n_levels):
            if permute[level]:
                parent_cols = labels[:, :level] # Observations are not just stratified by the immediate parent level but by all "ancestor" levels (in case of, e.g., repeated condition labels within some higher level of labels)
                # Is there a child level?
                if (level + 1) < n_levels:
                    child_col = labels[:, level + 1]
                else:
                    child_col = None
                permuted_labels[:, level] = permute_level_within_groups(
                    parent_cols, labels[:, level], child_col
                )

    # Permute covariates?
    out = (permuted_labels,)
    if return_cov_perm:
        cov_perm = np.arange(n_obs)
        if n_levels == 1:
            # No stratification, just shuffle
            cov_perm = rng.permutation(n_obs)
        else:
            # Shuffle within clusters
            level = n_levels - 1
            parent_cols = labels[:, :level]
            _, parent_inv = np.unique(parent_cols, axis=0, return_inverse=True)
            for idx in range(parent_inv.max() + 1):
                mask = parent_inv == idx
                cov_perm[mask] = cov_perm[mask][rng.permutation(mask.sum())]
        out += (cov_perm,)
    
    # Flip to model baseline?
    if return_flips:
        flips = rng.random(len(permuted_labels)) < 0.5
        out += (flips,)

    return out

def cluster_resample(labels, resample, rng):
    labels = labels.copy()
    n_obs, n_levels = labels.shape
    obs_id = np.arange(n_obs) # Identifiers for individual observations

    # Level 0 (special case: not clustered within any parent level)
    if resample[0]:
        unique_vals, inv = np.unique(labels[:, 0], return_inverse=True)
        resampled_val_ids = rng.choice(len(unique_vals), len(unique_vals))   # indices
        # For each resampled value, grab all rows that carry it
        resampled_rows = np.concatenate([np.where(inv == val_id)[0] for val_id in resampled_val_ids])
        labels = labels[resampled_rows]
        obs_id = obs_id[resampled_rows]

    # Remaining levels
    for level in range(1, n_levels):
        if resample[level]:
            parent_cols = labels[:, :level]
            child_col = labels[:, level]
    
            # Integer parent-cluster id per row
            unique_parents, parent_inv = np.unique(parent_cols, axis=0, return_inverse=True)
    
            resampled_rows = []
            for p_idx in range(len(unique_parents)):
                parent_mask  = parent_inv == p_idx # rows in this parent
                child_vals = child_col[parent_mask]
                unique_children, child_inv = np.unique(child_vals, return_inverse=True)
    
                # Resample child cluster indices (with replacement)
                resampled_child_idx = rng.choice(len(unique_children), len(unique_children))
    
                # Collect rows for each resampled child, scoped to this parent
                parent_positions = np.where(parent_mask)[0] # absolute row indices
                for c_idx in resampled_child_idx:
                    resampled_rows.append(parent_positions[child_inv == c_idx])
    
            labels = labels[np.concatenate(resampled_rows)]
            obs_id = obs_id[np.concatenate(resampled_rows)]

    return obs_id