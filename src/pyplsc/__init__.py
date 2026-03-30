
import numpy as np
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import svd as lapack_svd
from joblib import Parallel, delayed
import pandas as pd

from pdb import set_trace

class BaseClass():
    # Parent class for PLSC and BDA
    def __init__(self, svd_method='lapack', random_state=None):
        # Private properties for tracking whether permutation testing and bootstrap resampling have been done
        self.__perm_done = False
        self.__boot_done = False
        self.svd_method = svd_method
        self.random_state = random_state
    def _setup_data(self, X):
        valid_X = True
        if not isinstance(X, np.ndarray):
            valid_X = False
        else:
            if not X.ndim == 2:
                valid_X = False
        if not valid_X:
            raise ValueError('data must be a 2-dimensional numpy array')
        self.X_ = X
    def _setup_design_matrix(self, design=None, between=None, within=None, participant=None):
        if participant is None:
            if within is not None:
                raise ValueError('Participants must be differentiated if there is a within-participants factor')
            else:
                participant = np.arange(len(self.X_))
        # Assign null column of zeros if absent, otherwise assign categorical labels
        null_col = pd.Categorical([0]*len(self.X_))
        cols = {'between': between,
                'within': within,
                'participant': participant}
        if design is None:
            # Scenario 1: no dataframe, columns provided individually
            design = {}
            for colname, col in cols.items():
                if col is None:
                    design[colname] = null_col.copy()
                else:
                    design[colname] = pd.Categorical(col)
            design = pd.DataFrame(design)
        else:
            # Scenario 2: dataframe but also possibly columns provided individually
            for colname, col in cols.items():
                if col is None:
                    design[colname] = null_col.copy()
                elif isinstance(col, str):
                    design[colname] = pd.Categorical(design[col])
                else:
                    design[colname] = pd.Categorical(col)
            design = design[['between', 'within', 'participant']]
        self.design_ = design
        self.stratifier_ = _get_stratifier(design)
    def get_labels(self):
        """
        Get the labels corresponding to each row of the design saliences. For BDA, this is the between- and within-participant condition labels. For PLSC, covariate labels are also included.

        Returns
        -------
        labels : pandas.DataFrame
            A dataframe with one column corresponding to each label and one row corresponding to each row of the design saliences.

        """
        condition_labels = self.design_[['between','within']].drop_duplicates()
        if 'covariates_' in dir(self):
            # Create a MultiIndex from product of conditions and covariates
            index = pd.MultiIndex.from_product(
                [condition_labels.index, self.covariates_.columns],
                names=['condition_combo', 'covariate']
            )
            
            # Build the expanded dataframe
            labels = (
                condition_labels
                .loc[index.get_level_values('condition_combo')]  # repeat each combo row
                .reset_index(drop=True)
                .assign(covariate=index.get_level_values('covariate'))
            )
        else:
            labels = condition_labels
        return labels
    def _svd(self, M, compute_uv=True):
        if self.svd_method == 'lapack':
            if compute_uv:
                u, s, v = lapack_svd(M, full_matrices=False, compute_uv=True)
            else:
                s = lapack_svd(M, full_matrices=False, compute_uv=False)
        elif self.svd_method == 'randomized':
            u, s, v = randomized_svd(M, n_components=len(M))
        if compute_uv:
            out = u, s, v.T
        else:
            out = s
        return out
    def _initial_decomposition(self, to_factorize):
        u, s, v = self._svd(to_factorize)
        self.singular_vals_ = s
        self.n_lv_ = len(s)
        self.variance_explained_ = s / sum(s)
        self.design_sals_ = u
        self.brain_sals_ = v
    def flip_signs(self, lv_idx):
        """
        Flips the signs of one or more latent variables, to aid with interpretation.

        Parameters
        ----------
        lv_idx : integer or list
            The index or indices of latent variables whose signs should be flipped.

        Returns
        -------
        None.

        """
        self.design_sals_[:, lv_idx] *= -1
        self.brain_sals_[:, lv_idx] *= -1
        self.design_stat_[:, lv_idx] *= -1
        if self.__boot_done:
            self.bootstrap_ratios_[:, lv_idx] *= -1
            self.bootstrap_ci_[..., lv_idx] *= -1
            self.bootstrap_ci_ = self.bootstrap_ci_[(1, 0), ...]
    def transform(self, X=None, lv_idx=None):
        """
        Compute brain scores---i.e., coordinates of brain data in the new basis defined by the latent variables.

        Parameters
        ----------
        X : numpy.ndarray, optional
            Brain data to transform. The default is None, which yields brain scores for the data on which the model was fit.
        lv_idx : index, optional
            Index of latent variable(s) for which to compute brain scores. Default is None, which computes brain scores for all latent variables.

        Returns
        -------
        brain_scores : numpy.ndarray
            A 2D array of brain scores where rows correspond to different observations and columns correspond to different latent variables.

        """
        if X is None:
            X = self.X_
        sals = self.brain_sals_
        if lv_idx is not None:
            sals = sals[:, lv_idx]
        brain_scores = X @ sals
        return brain_scores
    def permute(self, n_perm=5000, return_null_dist=False, n_jobs=1):
        """
        Perform permutation testing to assess the significance of the latent variables. p values become available after running this method through the pvals_ property.

        Parameters
        ----------
        n_perm : int, optional
            Number of permutations t operform. The default is 5000.
        return_null_dist : bool, optional
            If true, permutation samples will be returned.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.

        Raises
        ------
        ValueError
            # TODO: document
            DESCRIPTION.

        Returns
        -------
        null_dist : numpy.ndarray
            2D array containing null distribution of singular values, where each row is a different permutation and each columns is a different singular value.

        """
        if n_perm < 1:
            raise ValueError('n_perm must be a positive integer')
        # Pre-generate perm_idx
        rng = np.random.default_rng(self.random_state)
        perms = [_get_permutation(rng, self.design_)
                 for _ in tqdm(range(n_perm), desc='Getting permutations')]
        perm_singvals = Parallel(n_jobs=n_jobs)(
            delayed(self._single_permutation)(perm)
            for perm in tqdm(perms, desc='Permuting')
        )
        null_dist = np.stack(perm_singvals)
        pvals = (np.sum(null_dist >= self.singular_vals_, axis=0) + 1) / (n_perm + 1)
        self.pvals_ = pvals
        self.__perm_done = True
        if return_null_dist:
            return null_dist
    def bootstrap(self, n_boot=5000, confint_level=0.95, alignment_method='rotate', svd_method='lapack', return_boot_dist=False, n_jobs=1):
        """
        Perform bootstrap resampling to assess the reliability of saliences.

        Parameters
        ----------
        n_boot : int, optional
            Number of bootstrap resamples to compute. The default is 5000.
        confint_level : float, optional
            The confidence level of the quantile-based confidence intervals to compute. The default is 0.95.
        alignment_method : string, optional
            Method to be used for aligning recomputed brain saliences with original brain saliences. 'rotate' uses the solution to the orthogonal Proctrustes problem. 'flip' flips the signs of the resampled saliences so that their inner products with original saliences are positive. The default is 'rotate'.
        svd_method : string, optional
            # TODO: remove Method to use for singular value decomposition. Options are 'lapack' (numpy.linalg.svd, default) or 'randomized' (sklearn.utils.extmath.randomized_svd).
        return_boot_dist : bool, optional
            # If true, bootstrap distribution from resampling is returned. This is thre distribution used to compute confidence intervals.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        design_resampled : numpy.ndarray
            # TODO: describe

        """
        if n_boot < 1:
            raise ValueError('n_boot must be a positive integer')
        self.n_boot_ = n_boot
        self.confint_level_ = confint_level
        # Get variables needed for bootstrapping
        # Pre-generate bootstrap samples
        rng = np.random.default_rng(self.random_state)
        boot_idxs = [self._get_resample(rng)
                     for _ in tqdm(range(n_boot), desc='Getting resamples')]
        boot_results = Parallel(n_jobs=n_jobs)(
            delayed(self._single_bootstrap_resample)(boot_idx, alignment_method)
            for boot_idx in tqdm(boot_idxs, desc="Resampling")
        )
        design_resampled, brain_resampled = zip(*boot_results)
        # Compute standard deviations for brain saliences to get bootstrap ratios
        stds = np.stack(brain_resampled).std(axis=0)
        self.bootstrap_ratios_ = (self.brain_sals_ @ np.diag(self.singular_vals_)) / stds
        # Compute confidence intervals for design saliences
        design_resampled = np.stack(design_resampled)
        alpha = 1 - confint_level
        self.bootstrap_ci_ = np.quantile(design_resampled, [alpha/2, 1 - alpha/2], axis=0)
        self.__boot_done = True
        return design_resampled
    def get_design_yerr(self, lv_idx):
        """
        Get yerr for matplotlib barplots.

        Parameters
        ----------
        lv_idx : int
            Integer indexing the latent variable of interest.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        yerr : numpy.ndarray
            yerr value that can be passed to matplotib's pyplot.bar().

        """
        if not self.__boot_done:
            raise ValueError('Bootstrap resampling must be done before confidence intervals can be extracted')
        if not isinstance(lv_idx, int):
            raise ValueError('lv_idx must be an integer index of a single latent variable')
        est = self.design_stat_[:, lv_idx]
        ci = self.bootstrap_ci_[..., lv_idx]
        yerr = np.array([ci[1] - est,
                         est - ci[0]])
        return yerr
        
class BDA(BaseClass):
    def __init__(self, pre_subtract=None, svd_method='lapack', random_state=None):
        super().__init__(svd_method=svd_method, random_state=random_state)
        self.pre_subtract = pre_subtract
    def fit(self, X, design=None, between=None, within=None, participant=None):
        # TODO: document
        if between is None and within is None:
            raise ValueError('Observations must be differentiated by some categorical variable (specified via "between" or "within") for BDA')
        if self.pre_subtract is not None:
            if self.pre_subtract == 'between':
                if between is None:
                    raise ValueError('Pre-subtracting between-participant condition means is not possible when no between-participant condition is defined')
                if within is None:
                    raise Warning('No effect of between-participant condition will be detectable if between-participant condition means are pre-subtracted and no within-participant condition is defined.')
            if self.pre_subtract == 'within':
                if within is None:
                    raise ValueError('Pre-subtracting within-participant condition means is not possible when no within-participant condition is definted')
                if between is None:
                    raise Warning('No effect of within-participant condition will be detectable if within-participant condition means are pre-subtracted and no between-participant condition is defined.')
        self._setup_data(X)
        self._setup_design_matrix(design, between, within, participant)
        if len(np.unique(self.stratifier_)) == 1:
            raise ValueError('The conjunction of between- and within-participant factors has only one unique level. I.e., the data cannot be stratified for BDA.')
        # TODO: enforce one between condition per participant
        mean_centred = _get_mean_centred(
            X=self.X_,
            design=self.design_,
            pre_subtract=self.pre_subtract)
        self._initial_decomposition(mean_centred)
        self.design_stat_ = mean_centred @ self.brain_sals_ # Score per barycentre
        return self
    def transform_design(self, Y=None, lv_idx=None):
        """
        Compute design scores

        Parameters
        ----------
        Y : numpy.ndarray, optional
            # TODO: explain. The default is None.
        lv_idx : index, optional

        Returns
        -------
        design_scores : numpy.ndarray
            A 2D array of design scores where rows correspond to different observations and columns correspond to different latent variables.

        """
        if Y is None:
            Y = self.stratifier_
        sals = self.design_sals_
        if lv_idx is not None:
            sals = sals[:, lv_idx]
        design_scores = sals[Y]
        return design_scores
    def _single_permutation(self, perm_idx):
        # Compute SVD for this permutation
        mean_centred = _get_mean_centred(
            X=self.X_,
            design=self.design_.iloc[perm_idx],
            stratifier=self.stratifier_[perm_idx],
            pre_subtract=self.pre_subtract)
        s = self._svd(mean_centred, compute_uv=False)
        return s
    def _get_resample(self, rng):
        resample = _get_resample(rng, self.design_)
        return resample
    def _single_bootstrap_resample(self, resample_idx, alignment_method):
        # Run decomposition
        mean_centred = _get_mean_centred(
            X=self.X_[resample_idx],
            design=self.design_.iloc[resample_idx],
            # stratifier=self.stratifier_[resample_idx],
            pre_subtract=self.pre_subtract)
        _, s, v = self._svd(mean_centred)
        v = _align(v, self.brain_sals_, alignment_method)
        brain_estimate = v @ np.diag(s)
        # Brain scores
        design_estimate = mean_centred @ self.brain_sals_
        return design_estimate, brain_estimate
  
class PLSC(BaseClass):
    def __init__(self, svd_method='lapack', random_state=None):
        super().__init__(svd_method=svd_method, random_state=random_state)
    def _setup_covariates(self, design, covariates):
        if isinstance(covariates, np.ndarray):
            covariates = pd.DataFrame(covariates)
            covariates.columns = ['cov%s' % col for col in covariates.columns]
        else:
            if not isinstance(covariates, pd.DataFrame):
                try:
                    covariates = design[covariates]
                except:
                    raise ValueError('Covariates must be a DataFrame or ndarray, or the names of the columns in the design matrix that contain the covariates')
        if len(covariates) != len(self.X_):
            raise ValueError('Must be as many covariate rows as data rows')
        self.covariates_ = covariates
    def fit(self, X, covariates, design=None, between=None, within=None, participant=None):
        self._setup_data(X)
        self._setup_design_matrix(design, between, within, participant)
        self._setup_covariates(design, covariates)        
        R = _get_stacked_cormats(
            self.X_,
            self.covariates_,
            self.stratifier_)
        self._initial_decomposition(R)
        # Correlation between brain scores and covariates
        brain_scores = self.transform()
        self.design_stat_ = _get_stacked_cormats(brain_scores,
                                                 self.covariates_,
                                                 self.stratifier_)
    def transform_design(self, Y=None, lv_idx=None):
        if Y is None:
            Y = self.covariates_
        sals = self.design_sals_
        if lv_idx is not None:
            sals = sals[:, lv_idx]
        return Y.T @ sals
    def _single_permutation(self, perm_idx):
        R = _get_stacked_cormats(
            self.X_,
            self.covariates_.iloc[perm_idx],
            self.stratifier_[perm_idx])
        s = self._svd(R, compute_uv=False)
        return s
    def _get_resample(self, rng):
        valid_resample = False
        while not valid_resample:
            # Get indices of resample
            resample = _get_resample(rng, self.design_)
            # Check only one unique observation within a level
            obs_per_level = self.design_.iloc[resample].groupby(['between', 'within'])['participant'].nunique()
            valid_resample = all(obs_per_level > 1)
        return resample
    def _single_bootstrap_resample(self, resample_idx, alignment_method):
        # Run decomposition
        resampled_X = self.X_[resample_idx]
        resampled_cov = self.covariates_.iloc[resample_idx]
        resampled_strat = self.stratifier_[resample_idx]
        R = _get_stacked_cormats(resampled_X,
                                 resampled_cov,
                                 resampled_strat)
        _, s, v = self._svd(R)
        v = _align(v, self.brain_sals_, alignment_method)
        brain_estimate = v @ np.diag(s)
        # Correlation between covariates and brain scores
        design_estimate = _get_stacked_cormats(resampled_X @ self.brain_sals_, # Brain scores
                                               resampled_cov,
                                               resampled_strat)
        return design_estimate, brain_estimate

def _get_permutation(rng, design):
    if design['within'].nunique() == 1:
        # If no within-participants factor, just shuffle all rows
        perm = rng.permutation(len(design))
    else:
        # There is a within-participants factor
        rows_by_ptpt = design.groupby('participant').indices
        # Shuffle within-participant condition
        for rows in rows_by_ptpt.values():
            rng.shuffle(rows)
        if design['between'].nunique() > 1:
            # If there is a between-participants cond, shuffle which set of rows (and hence which between-participants condition) is assigned to each participant
            # (This is just shuffling mapping between dict keys and values)
            ptpts, row_sets = zip(*rows_by_ptpt.items())
            row_sets = rng.permutation(row_sets)
            rows_by_ptpt = dict(zip(ptpts, row_sets))
        # Assign new rows to participants
        perm = np.zeros((len(design),), dtype=np.int64)
        for ptpt, rows in rows_by_ptpt.items():
            perm[design['participant'] == ptpt] = rows
    return perm

def _get_stratifier(design):
    # Get unique combinations of between and within factors
    stratifier, _ = pd.MultiIndex.from_frame(design[['between', 'within']]).factorize()
    return stratifier

def _pre_centre(X, design, pre_subtract):
    # Pre-subtract between- or within-wise means if applicable
    group_idx = design[pre_subtract]
    rowwise_group_means = _get_groupwise_means(X, group_idx)[group_idx]
    return X - rowwise_group_means

def _get_mean_centred(X, design, stratifier=None, pre_subtract=None):
    if pre_subtract is not None:
        X = _pre_centre(X, design, pre_subtract)
    # Compute group-wise means
    if stratifier is None: # Might not be pre-computed
        stratifier = _get_stratifier(design)
    groupwise_means = _get_groupwise_means(X, stratifier)
    # Mean centre
    mean_centred = groupwise_means - groupwise_means.mean(axis=0)
    return mean_centred

def _get_groupwise_means(X, group_idx):
    groups = np.unique(group_idx)
    # Pre-allocate memory
    groupwise_means = np.zeros((len(groups), X.shape[1]), dtype=X.dtype)
    for group in groups:
        groupwise_means[group] = X[group_idx == group].mean(axis=0)
    return groupwise_means

def _get_resample(rng, design):
    if design['between'].nunique() > 1:
        # Sample participants stratified by between factor
        ptpts_by_between = design.groupby('between')['participant'].unique()
        resampled_by_between = [rng.choice(ptpts, len(ptpts)) for ptpts in ptpts_by_between]
        resampled_ptpts = np.concatenate(resampled_by_between)
    else:
        # Sample participants at random
        unique_ptpts = design['participant'].unique()
        resampled_ptpts = rng.choice(unique_ptpts, len(unique_ptpts))
    rows_by_ptpt = design.groupby('participant').indices
    rows = np.concatenate([rows_by_ptpt[ptpt] for ptpt in resampled_ptpts])
    return rows

def _validate_resample(resample, stratifier):
    # Ensure that each stratfier level contains at least 2 unique observations
    # To do this quickly, compute min and max observation idx within category
    # and check that min != max
    resampled_levels = stratifier[resample]
    order = np.argsort(resampled_levels)
    stratifier = stratifier[order]
    obs = resample[order]
    # Stratifier level boundaries
    boundaries = np.flatnonzero(np.diff(stratifier)) + 1
    starts = np.r_[0, boundaries]
    # Min/max observation per category
    mins = np.minimum.reduceat(obs, starts)
    maxs = np.maximum.reduceat(obs, starts)
    # Invalid if all observations are identical within any level
    invalid = (mins == maxs).any()
    return invalid

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
    # Compute a rectangular correlation matrix between X and Y
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    
    denom = X.shape[0] - 1
    stdX = np.sqrt((Xc ** 2).sum(axis=0) / denom)
    stdY = np.sqrt((Yc ** 2).sum(axis=0) / denom)
    
    Xn = Xc / stdX
    Yn = Yc / stdY
    return Xn.T @ Yn / denom

def _align(v, target_v, alignment_method):
    # Align with original decomposition
    if alignment_method == 'rotate':
        # Via rotation
        R, _ = orthogonal_procrustes(v, target_v, check_finite=False)
        aligned = v @ R
    elif alignment_method == 'flip':
        # Via correcting apparent sign flips
        flips = np.sign(np.diag(v.T @ target_v))
        aligned = v * flips
    return aligned