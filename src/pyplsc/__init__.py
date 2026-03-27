
import numpy as np
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import svd
from joblib import Parallel, delayed

from pdb import set_trace

class BaseClass():
    def _setup_data(self, X, between, within, participant):
        if within is not None and participant is None:
            raise ValueError('Participants must be differentiated if there is a within-participants factor')
        self.design_, sort_idx = _get_design_matrix(len(X), between, within, participant)
        self.stratifier_ = _get_stratifier(self.design_)
        self.X_ = X[sort_idx]
        return sort_idx
    def _initial_decomposition(self, to_factorize):
        u, s, v = svd(to_factorize, full_matrices=False, compute_uv=True)
        self.singular_vals_ = s
        self.n_lv_ = len(s)
        self.variance_explained_ = s / sum(s)
        self.design_sals_ = u
        self.brain_sals_ = v.T
    def permute(self, n_perm=5000, n_jobs=None):
        if n_perm < 1:
            raise ValueError('n_perm must be a positive integer')
        perm_singvals = Parallel(n_jobs=n_jobs)(
            delayed(self._single_permutation)()
            for _ in tqdm(range(n_perm), desc="Permutations")
        )
        null_dist = np.stack(perm_singvals)
        pvals = (np.sum(null_dist >= self.singular_vals_, axis=0) + 1) / (n_perm + 1)
        self.pvals_ = pvals
        return null_dist
    def bootstrap(self, n_boot=5000, confint_level=0.025, n_jobs=None):
        if n_boot < 1:
            raise ValueError('n_boot must be a positive integer')
        self.n_boot_ = n_boot
        self.confint_level_ = confint_level
        # Get variables needed for bootstrapping
        resample_vars = _get_vars_for_resampling(self.design_)
        brain_resampled = []
        design_resampled = []
        boot_results = []
        boot_results = Parallel(n_jobs=n_jobs)(
            delayed(self._single_bootstrap_resample)(*resample_vars)
            for _ in tqdm(range(n_boot), desc="Resamples")
        )
        design_resampled, brain_resampled = zip(*boot_results)
        # Compute standard deviations for brain saliences to get bootstrap ratios
        stds = np.stack(brain_resampled).std(axis=0)
        self.bootstrap_ratios_ = (self.brain_sals_ @ np.diag(self.singular_vals_)) / stds
        # Compute confidence intervals for design saliences
        self.bootstrap_ci_ = np.quantile(np.stack(design_resampled), [confint_level, 1 - confint_level], axis=0)
        
class BDA(BaseClass):
    def __init__(self, subtract=None):
        self.subtract = subtract
    def fit(self, X, between=None, within=None, participant=None):
        if between is None and within is None:
            raise ValueError('Observations must be differentiated by some categorical variable (specified via "between" or "within") for BDA')
        self._setup_data(X, between, within, participant)
        # TODO: enfore categoricity? I.e., check indicator arrays for float values
        # TODO: check whether there are multiple levels of within and between factors
        # TODO: check whether subtract option is possible given availability of factors
        # TODO: make sure lengths of inputs are all the same
        # TODO: enforce one between condition per participant
        # SVD decomposition
        mean_centred = _get_mean_centred(
            X=self.X_,
            design=self.design_,
            subtract=self.subtract)
        self._initial_decomposition(mean_centred)
        self.design_stat_ = mean_centred @ self.brain_sals_ # Score per barycentre
        return self
    def transform_brain(self, X=None):
        # Brain scores
        if X is None:
            X = self.X_
        brain_scores = X @ self.brain_sals_
        return brain_scores
    def transform_design(self, Y=None):
        # Design scores
        if Y is None:
            Y = self.stratifier_
        design_scores = self.design_sals_[Y]
        return design_scores
    def _single_permutation(self):
        perm_idx = _get_permutation(self.design_)
        mean_centred = _get_mean_centred(
            X=self.X_,
            design=self.design_[perm_idx],
            stratifier=self.stratifier_[perm_idx],
            subtract=self.subtract)
        s = svd(mean_centred, full_matrices=False, compute_uv=False)
        return s
    def _single_bootstrap_resample(self, *resample_vars):
        # Get indices of resample
        resample_idx = _get_resample_idx(*resample_vars)
        # Run decomposition
        mean_centred = _get_mean_centred(
            X=self.X_[resample_idx],
            design=self.design_[resample_idx],
            stratifier=self.stratifier_[resample_idx],
            subtract=self.subtract)
        u, s, v = _svd_and_align(to_factorize=mean_centred,
                                 target_v=self.brain_sals_)
        brain_estimate = v @ np.diag(s)
        # Brain scores
        design_estimate = mean_centred @ self.brain_sals_
        return design_estimate, brain_estimate
  
class PLSC(BaseClass):
    def __init__(self):
        # No initialization variables
        pass
    def fit(self, X, covariates, between=None, within=None, participant=None):
        sort_idx = self._setup_data(X, between, within, participant)
        self.covariates_ = covariates[sort_idx]
        R = _get_stacked_cormats(
            self.X_,
            self.covariates_,
            self.stratifier_)
        self._initial_decomposition(R)
        # Correlation between brain scores and covariates
        brain_scores = self.X_ @ self.brain_sals_
        self.design_stat_ = _get_stacked_cormats(brain_scores,
                                                 self.covariates_,
                                                 self.stratifier_)
    def _single_permutation(self):
        perm_idx = _get_permutation(self.design_)
        R = _get_stacked_cormats(
            self.X_,
            self.covariates_[perm_idx],
            self.stratifier_[perm_idx])
        s = svd(R, full_matrices=False, compute_uv=False)
        return s
    def _single_bootstrap_resample(self, *resample_vars):
        all_same = True
        while all_same:    
            # Get indices of resample
            resample_idx = _get_resample_idx(*resample_vars)
            # Check for no unique observations within any level
            all_same = _validate_resample(resample_idx, self.stratifier_)
        # Run decomposition
        resampled_X = self.X_[resample_idx]
        resampled_cov = self.covariates_[resample_idx]
        stacked_cormats = _get_stacked_cormats(
            resampled_X,
            resampled_cov,
            self.stratifier_) # Because we're resampling within levels of the stratifier, we don't need to explicitly apply the resample_idx to stratifier. stratifier[resample_idx] == stratifier, always
        u, s, v = _svd_and_align(to_factorize=stacked_cormats,
                                 target_v=self.brain_sals_)
        brain_estimate = v @ np.diag(s)
        # Correlation between covariates and brain scores
        design_estimate = _get_stacked_cormats(resampled_X @ self.brain_sals_, # Brain scores
                                               resampled_cov,
                                               self.stratifier_)
        return design_estimate, brain_estimate

def _get_permutation(design):
    # n_obs, between=None, participant=None)
    if design[-1, 1] == 0: # If no within-participants factor:
        # No between-participant conditions---just shuffle all rows
        perm_idx = np.random.permutation(len(design))
    else:
        participant = design[:, 2]
        if design[-1, 0] > 0: # If a between-participants factor:
            # Shuffle participants
            n_participants = participant[-1] + 1 # Max participant idx + 1
            participant_permutation = np.random.permutation(n_participants)
            # This next line works because "participant" is both an array of
            # integer labels and an integer index that could be used to index
            # an array of unique participant IDs
            participant = participant_permutation[participant]
        # Shuffle within participants
        perm_idx = np.lexsort((np.random.rand(len(participant)), participant))
    return perm_idx

def _get_stratifier(design):
    # Get unique combinations of between and within factors
    _, stratifier = np.unique(design[:, :2], axis=0, return_inverse=True)
    return stratifier

def _pre_centre(X, design, subtract):
    # Pre-subtract between- or within-wise means if applicable
    if subtract == 'between':
        group_idx = design[:, 0]
    elif subtract == 'within':
        group_idx = design[:, 1]
    rowwise_group_means = _get_groupwise_means(X, group_idx)[group_idx]
    return X - rowwise_group_means

def _get_mean_centred(X, design, stratifier=None, subtract=None):
    if subtract is not None:
        X = _pre_centre(X, design, subtract)
    # Compute group-wise means
    if stratifier is None: # Might not be pre-computed
        stratifier = _get_stratifier(design)
    groupwise_means = _get_groupwise_means(X, stratifier)
    # Mean centre
    mean_centred = groupwise_means - groupwise_means.mean(axis=0)
    return mean_centred

def _get_groupwise_means(X, group_idx):
    n_groups = group_idx.max() + 1
    # Pre-allocate memory
    groupwise_means = np.zeros((n_groups, X.shape[1]), dtype=X.dtype)
    for group in range(n_groups):
        groupwise_means[group] = X[group_idx == group].mean(axis=0)
    return groupwise_means

def _get_vars_for_resampling(design):
    # Set up variables used for resampling
    row_idx = np.arange(len(design))
    # Set up dummy indicators if needed
    between, within, participant = design[:, :3].T
    row_idx_by_participant = np.split(row_idx, np.cumsum(np.bincount(participant)[:-1]))
    between_by_participant = between[np.cumsum(np.bincount(participant)) - 1]
    participants_by_between = np.split(
        np.arange(len(row_idx_by_participant)),
        np.cumsum(np.bincount(between_by_participant)[:-1])
    )
    participant_offsets = np.cumsum([0] + [len(r) for r in row_idx_by_participant])
    return row_idx, participants_by_between, participant_offsets

def _get_resample_idx(row_idx, participants_by_between, participant_offsets):
    sampled_rows = []
    for ps in participants_by_between:
        samp = ps[np.random.randint(len(ps), size=len(ps))]
        # sampled_rows.extend(row_idx_by_participant[p] for p in samp)
        sampled_rows.extend(row_idx[participant_offsets[p]:participant_offsets[p+1]] for p in samp)
    resample_idx = np.concatenate(sampled_rows)
    return resample_idx

def _get_design_matrix(n_obs, between=None, within=None, participant=None):
    # Assign null column of zeros if absent, otherwise assign integer labels
    null_col = np.zeros((n_obs,), dtype=np.int64)
    if between is None:
        between = null_col
    else:
        _, between = np.unique(between, return_inverse=True)
    if within is None:
        within = null_col
        participant = np.arange(n_obs)
    else:
        _, within = np.unique(within, return_inverse=True)
        _, participant = np.unique(participant, return_inverse=True)
    
    # Sort by between, then participant, then within, if applicable
    sort_idx = np.lexsort((within, participant, between))
    design_matrix = np.column_stack((between, within, participant))
    design_matrix = design_matrix[sort_idx]
    return design_matrix, sort_idx

def _get_stacked_cormats(X, covariates, stratifier):
    submatrices = []
    n_levels = stratifier.max() + 1
    for level in range(n_levels):
        idx = stratifier == level
        submatrix = _corr(covariates[idx], X[idx])
        submatrices.append(submatrix)
    R = np.concat(submatrices)
    return R

def _corr(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    
    n = X.shape[0] - 1
    stdX = np.sqrt((Xc ** 2).sum(axis=0) / n)
    stdY = np.sqrt((Yc ** 2).sum(axis=0) / n)
    
    Xn = Xc / stdX
    Yn = Yc / stdY
    return Xn.T @ Yn / n

def _validate_resample(resample_idx, stratifier):
    # Ensure that each stratfier level contains at least 2 unique observations
    # To do this quickly, compute min and max observation idx within category
    # and check that min != max
    resampled_levels = stratifier[resample_idx]
    order = np.argsort(resampled_levels)
    stratifier = stratifier[order]
    obs = resample_idx[order]
    # Stratifier level boundaries
    boundaries = np.flatnonzero(np.diff(stratifier)) + 1
    starts = np.r_[0, boundaries]
    # Min/max observation per category
    mins = np.minimum.reduceat(obs, starts)
    maxs = np.maximum.reduceat(obs, starts)
    # Invalid if all observations are identical within any level
    invalid = (mins == maxs).any()
    return invalid

def _svd_and_align(to_factorize, target_v):
    u, s, v = svd(to_factorize, full_matrices=False)
    v = v.T
    # Rotate to align with original decomposition
    R, _ = orthogonal_procrustes(v, target_v, check_finite=False)
    v = v @ R
    return u, s, v