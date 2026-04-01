
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
    def __init__(self, svd_method='lapack', boot_stat=None, validate_resamples=False, random_state=None):
        # Private properties for tracking whether permutation testing and bootstrap resampling have been done
        self.__perm_done = False
        self.__boot_done = False
        self.__validate_resamples = validate_resamples
        self.svd_method = svd_method
        self.boot_stat = boot_stat
        self.random_state = random_state
    def _setup_data(self, data):
        # Add data_ as a property
        valid_data = True
        if not isinstance(data, np.ndarray):
            valid_data = False
        else:
            if not data.ndim == 2:
                valid_data = False
        if not valid_data:
            raise ValueError('data must be a 2-dimensional numpy array')
        self.data_ = data
    def _setup_design_matrix(self, design=None, between=None, within=None, participant=None):
        # Add design_matrix_ and stratifier_ as properties
        if participant is None:
            if within is not None:
                raise ValueError('Participants must be differentiated if there is a within-participants factor')
            else:
                participant = np.arange(len(self.data_))
        # Assign null column of zeros if absent, otherwise assign categorical labels
        null_col = pd.Categorical([0]*len(self.data_))
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
        self.stratifier_ = _get_stratifier(design) # This is handy to pre-compute because it's used many times later on
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
        # Single function to perform svd using the specified method
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
        # Initial fit and add various properties
        u, s, v = self._svd(to_factorize)
        self.singular_vals_ = s
        self.n_lv_ = len(s)
        self.variance_explained_ = s / sum(s)
        self.design_sals_ = u
        self.data_sals_ = v
    def flip_signs(self, lv_idx=None):
        """
        Flips the signs of one or more latent variables, to aid with interpretation.

        Parameters
        ----------
        lv_idx : integer or list
            The index or indices of latent variables whose signs should be flipped. If None (default), signs are flipped for all latent variables.

        Returns
        -------
        None.

        """
        if lv_idx is None:
            lv_idx = range(self.n_lv_)
        self.design_sals_[:, lv_idx] *= -1
        self.data_sals_[:, lv_idx] *= -1
        self.boot_stat_[:, lv_idx] *= -1
        if self.__boot_done:
            self.bootstrap_ratios_[:, lv_idx] *= -1
            self.boot_stat_ci_[..., lv_idx] *= -1
            self.boot_stat_ci_ = self.boot_stat_ci_[(1, 0), ...]
    def transform(self, data=None, lv_idx=None):
        """
        Compute scores, i.e., coordinates of data in the new basis defined by the latent variables.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to transform. The default is None, which yields scores for the data on which the model was fit.
        lv_idx : index, optional
            Index of latent variable(s) for which to compute scores. Default is None, which computes scores for all latent variables.

        Returns
        -------
        data_scores : numpy.ndarray
            A 2D array of scores where rows correspond to different observations and columns correspond to different latent variables.

        """
        if data is None:
            data = self.data_
        sals = self.data_sals_
        if lv_idx is not None:
            sals = sals[:, lv_idx]
        data_scores = data @ sals
        return data_scores
    def _get_permutations(self, n_perm):
        # Get indices that can be used to permute
        rng = np.random.default_rng(self.random_state)
        if self.design_['within'].nunique() == 1:
            # If no within-participants factor, just shuffle all rows
            case = 'only-between'
        else:
            unshuffled_rows_by_ptpt = self.design_.groupby('participant').indices
            if self.design_['between'].nunique() > 1:
                case = 'between-and-within'
            else:
                case = 'only-within'
        n_obs = len(self.design_)
        perms = []
        for perm_n in tqdm(range(n_perm), desc='Getting permutations'):
            if case == 'only-between':
                # Just shuffle all rows
                perm = rng.permutation(n_obs)
            else:
                # There is a within-participants factor
                # Create a copy to shuffle
                rows_by_ptpt = unshuffled_rows_by_ptpt.copy()
                # Shuffle within-participant condition
                for rows in rows_by_ptpt.values():
                    rng.shuffle(rows)
                if case == 'between-and-within':
                    # If there is a between-participants cond, shuffle which set of rows (and hence which between-participants condition) is assigned to each participant
                    # (This is just shuffling mapping between dict keys and values)
                    ptpts, row_sets = zip(*rows_by_ptpt.items())
                    row_sets = rng.permutation(row_sets)
                    rows_by_ptpt = dict(zip(ptpts, row_sets))
                # Assign new rows to participants
                perm = np.zeros((n_obs,), dtype=np.int64) # Pre-allocate to later index
                for ptpt, rows in rows_by_ptpt.items():
                    perm[unshuffled_rows_by_ptpt[ptpt]] = rows # Assign participant's new rows to their old rows
            perms.append(perm)
        return perms
    def permute(self, n_perm=5000, return_null_dist=False, n_jobs=1):
        """
        Perform permutation testing to assess the significance of the latent variables. p values become available after running this method through the pvals_ property.

        Parameters
        ----------
        n_perm : int, optional
            Number of permutations t operform. The default is 5000.
        return_null_dist : bool, optional
            If true, permutation samples will be returned as a 2D (n. perms, n. latent vars) array.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.

        Returns
        -------
        null_dist : numpy.ndarray
            2D array containing null distribution of singular values, where each row is a different permutation and each columns is a different singular value.

        """
        if n_perm < 1:
            raise ValueError('n_perm must be a positive integer')
        # Pre-generate perm_idx
        perms = self._get_permutations(n_perm)
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
    def _get_resamples(self, n_boot, validate=False):
        rng = np.random.default_rng(self.random_state)
        rows_by_ptpt = self.design_.groupby('participant').indices
        if self.design_['between'].nunique() > 1:
            case = 'between'
            ptpts_by_between = self.design_.groupby('between')['participant'].unique()
        else:
            case = 'no-between'
            unique_ptpts = self.design_['participant'].unique()
        if validate:
            strat_levels = np.unique(self.stratifier_)
            ptpts = self.design_['participant'].to_numpy()
        resamples = []
        for boot_n in tqdm(range(n_boot)):
            validated = False
            while not validated:
                if case == 'between':
                    # Sample participants stratified by between factor
                    resampled_by_between = [rng.choice(ptpts, len(ptpts)) for ptpts in ptpts_by_between]
                    resampled_ptpts = np.concatenate(resampled_by_between)
                else:
                    # Sample participants at random
                    resampled_ptpts = rng.choice(unique_ptpts, len(unique_ptpts))
                resample = np.concatenate([rows_by_ptpt[ptpt] for ptpt in resampled_ptpts])
                if validate:
                    # Check for levels with only one participant
                    resampled_strat = self.stratifier_[resample]
                    resampled_ptpts = ptpts[resample]
                    validated = all([len(np.unique(resampled_ptpts[resampled_strat == lvl])) > 1
                                     for lvl in strat_levels])
                else:
                    validated = True
            resamples.append(resample)
        return resamples
    def bootstrap(self, n_boot=5000, confint_level=0.95, alignment_method='rotate', return_boot_dist=False, n_jobs=1):
        """
        Perform bootstrap resampling to assess the reliability of saliences.

        Parameters
        ----------
        n_boot : int, optional
            Number of bootstrap resamples to compute. The default is 5000.
        confint_level : float, optional
            The confidence level of the quantile-based confidence intervals to compute. The default is 0.95.
        alignment_method : string, optional
            Method to be used for aligning recomputed data saliences with original data saliences. 'rotate' uses the solution to the orthogonal Proctrustes problem. 'flip' flips the signs of the resampled saliences so that their inner products with original saliences are positive. The default is 'rotate'.
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
            If return_boot_dist is true, returns the bootstrap distribution of boot_stat_

        """
        if n_boot < 1:
            raise ValueError('n_boot must be a positive integer')
        self.n_boot_ = n_boot
        self.confint_level_ = confint_level
        # Pre-generate bootstrap samples
        boot_idxs = self._get_resamples(n_boot, validate=self.__validate_resamples)
        boot_results = Parallel(n_jobs=n_jobs)(
            delayed(self._single_bootstrap_resample)(boot_idx, alignment_method)
            for boot_idx in tqdm(boot_idxs, desc="Resampling"))
        design_resampled, data_resampled = zip(*boot_results)
        # Compute standard deviations for data saliences to get bootstrap ratios
        stds = np.stack(data_resampled).std(axis=0)
        self.bootstrap_ratios_ = (self.data_sals_ @ np.diag(self.singular_vals_)) / stds
        # Compute confidence intervals for design saliences
        design_resampled = np.stack(design_resampled)
        alpha = 1 - confint_level
        self.boot_stat_ci_ = np.quantile(design_resampled, [alpha/2, 1 - alpha/2], axis=0)
        self.__boot_done = True
        if return_boot_dist:
            return design_resampled
    def get_boot_stat_yerr(self, lv_idx):
        """
        Get yerr for boot_stat_ that can be passed to a matplotlib bar plot.

        Parameters
        ----------
        lv_idx : int
            Integer indexing the latent variable of interest.

        Returns
        -------
        yerr : numpy.ndarray
            2D array with shape (2, n. design saliences) that can be passed to matplotib's pyplot.bar() as the yerr= argument.

        """
        if not self.__boot_done:
            raise ValueError('Bootstrap resampling must be done to obtain confidence intervals')
        if not isinstance(lv_idx, int):
            raise ValueError('lv_idx must be an integer index of a single latent variable')
        est = self.boot_stat_[:, lv_idx]
        ci = self.boot_stat_ci_[..., lv_idx]
        yerr = np.array([ci[1] - est,
                         est - ci[0]])
        return yerr
        
class BDA(BaseClass):
    def __init__(self, pre_subtract=None, boot_stat='condwise-scores-centred', svd_method='lapack', random_state=None):
        super().__init__(svd_method=svd_method,
                         boot_stat=boot_stat,
                         random_state=random_state,
                         validate_resamples=False)
        self.pre_subtract = pre_subtract
    def fit(self, data, design=None, between=None, within=None, participant=None):
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
        self._setup_data(data)
        self._setup_design_matrix(design, between, within, participant)
        if len(np.unique(self.stratifier_)) == 1:
            raise ValueError('The conjunction of between- and within-participant factors has only one unique level. I.e., the data cannot be stratified for BDA.')
        # TODO: enforce one between condition per participant
        mean_centred = _get_mean_centred(
            data=self.data_,
            design=self.design_,
            stratifier=self.stratifier_,
            pre_subtract=self.pre_subtract)
        self._initial_decomposition(mean_centred)
        # Compute design scores
        self.design_scores_ = self.design_sals_[self.stratifier_]
        if self.boot_stat == 'condwise-scores-centred':
            self.boot_stat_ = mean_centred @ self.data_sals_
        elif self.boot_stat == 'condwise-scores':
            data_scores = self.transform()
            self.boot_stat_ = _get_groupwise_means(data_scores, self.stratifier_)
        return self
    def _single_permutation(self, perm_idx):
        # Compute SVD for this permutation
        mean_centred = _get_mean_centred(
            data=self.data_,
            design=self.design_.iloc[perm_idx],
            stratifier=self.stratifier_[perm_idx],
            pre_subtract=self.pre_subtract)
        s = self._svd(mean_centred, compute_uv=False)
        return s
    def _single_bootstrap_resample(self, resample_idx, alignment_method):
        # Run decomposition
        resampled_data = self.data_[resample_idx]
        resampled_design = self.design_.iloc[resample_idx]
        resampled_strat = self.stratifier_[resample_idx]
        mean_centred = _get_mean_centred(
            data=resampled_data,
            design=resampled_design,
            stratifier=resampled_strat,
            pre_subtract=self.pre_subtract)
        _, s, v = self._svd(mean_centred)
        v = _align(v, self.data_sals_, alignment_method)
        resampled_data_sals = v @ np.diag(s)
        # Brain scores
        if self.boot_stat == 'condwise-scores-centred':
            boot_stat = mean_centred @ self.data_sals_
        elif self.boot_stat == 'condwise-scores':
            scores = resampled_data @ self.data_sals_
            boot_stat = _get_groupwise_means(scores, resampled_strat)
        return boot_stat, resampled_data_sals
  
class PLSC(BaseClass):
    def __init__(self, svd_method='lapack', random_state=None):
        super().__init__(svd_method=svd_method,
                         random_state=random_state,
                         validate_resamples=True)
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
        if len(covariates) != len(self.data_):
            raise ValueError('Must be as many covariate rows as data rows')
        self.covariates_ = covariates
    def _get_design_scores(self):
        # Initialize
        design_scores = np.zeros((len(self.covariates_), self.n_lv_), dtype=self.design_sals_.dtype)
        # Align the observationsn with the design saliences, level-wises
        sal_labels = self.get_labels()
        sal_levels = _get_stratifier(sal_labels, output='tuples')
        obs_levels = _get_stratifier(self.design_, output='tuples')
        for curr_lvl in set(obs_levels):
            obs_mask = [i for i, obs_lvl in enumerate(obs_levels) if obs_lvl == curr_lvl]
            sal_mask = [i for i, sal_lvl in enumerate(sal_levels) if sal_lvl == curr_lvl]
            obs_submat = self.covariates_.to_numpy()[obs_mask]
            sal_submat = self.design_sals_[sal_mask]
            # Ensure each covariate is being multiplied by the appropriate salience
            assert all(sal_labels['covariate'].iloc[sal_mask] == self.covariates_.columns)
            design_scores[obs_mask] = obs_submat @ sal_submat
    def fit(self, data, covariates, design=None, between=None, within=None, participant=None):
        # TODO: document
        self._setup_data(data)
        self._setup_design_matrix(design, between, within, participant)
        self._setup_covariates(design, covariates)
        R = _get_stacked_cormats(
            self.data_,
            self.covariates_,
            self.stratifier_)
        self._initial_decomposition(R)
        self.design_scores_ = self._get_design_scores()
        # Correlation between data scores and covariates
        data_scores = self.transform()
        self.boot_stat_ = _get_stacked_cormats(data_scores,
                                                 self.covariates_,
                                                 self.stratifier_)
    def _single_permutation(self, perm_idx):
        R = _get_stacked_cormats(
            self.data_,
            self.covariates_.iloc[perm_idx],
            self.stratifier_[perm_idx])
        s = self._svd(R, compute_uv=False)
        return s
    def _single_bootstrap_resample(self, resample_idx, alignment_method):
        # Run decomposition
        resampled_data = self.data_[resample_idx]
        resampled_cov = self.covariates_.iloc[resample_idx]
        resampled_strat = self.stratifier_[resample_idx]
        R = _get_stacked_cormats(resampled_data,
                                 resampled_cov,
                                 resampled_strat)
        _, s, v = self._svd(R)
        v = _align(v, self.data_sals_, alignment_method)
        resampled_data_sals = v @ np.diag(s)
        if self.boot_stat == 'score-covariate-corr':
            # Correlation between covariates and data scores
            boot_stat = _get_stacked_cormats(resampled_data @ self.data_sals_, # Brain scores
                                             resampled_cov,
                                             resampled_strat)
        elif self.boot_stat == 'condwise-scores':
            scores = resampled_data @ self.data_sals_
            boot_stat = _get_groupwise_means(scores, resampled_strat)
        return boot_stat, resampled_data_sals

def _get_stratifier(design, output='ints'):
    # Get unique combinations of between and within factors
    multi_idx = pd.MultiIndex.from_frame(design[['between', 'within']])
    if output == 'ints':
        stratifier, _ = multi_idx.factorize()
    elif output == 'tuples':
        stratifier = multi_idx.to_list()
    return stratifier

def _pre_centre(data, design, pre_subtract):
    # Pre-subtract between- or within-wise means if applicable
    group_idx = design[pre_subtract]
    rowwise_group_means = _get_groupwise_means(data, group_idx)[group_idx]
    return data - rowwise_group_means

def _get_mean_centred(data, design, stratifier=None, pre_subtract=None):
    if pre_subtract is not None:
        data = _pre_centre(data, design, pre_subtract)
    # Compute group-wise means
    if stratifier is None: # Might not be pre-computed
        stratifier = _get_stratifier(design)
    groupwise_means = _get_groupwise_means(data, stratifier)
    # Mean centre
    mean_centred = groupwise_means - groupwise_means.mean(axis=0)
    return mean_centred

def _get_groupwise_means(data, group_idx):
    groups = np.unique(group_idx)
    # Pre-allocate memory
    groupwise_means = np.zeros((len(groups), data.shape[1]), dtype=data.dtype)
    for group in groups:
        groupwise_means[group] = data[group_idx == group].mean(axis=0)
    return groupwise_means

def _get_stacked_cormats(data, covariates, stratifier):
    submatrices = []
    n_levels = stratifier.max() + 1
    for level in range(n_levels):
        idx = stratifier == level
        submatrix = _corr(covariates[idx], data[idx])
        submatrices.append(submatrix)
    R = np.concat(submatrices)
    return R

def _corr(data, Y):
    # Compute a rectangular correlation matrix between data and Y
    datac = data - data.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    
    denom = data.shape[0] - 1
    stddata = np.sqrt((datac ** 2).sum(axis=0) / denom)
    stdY = np.sqrt((Yc ** 2).sum(axis=0) / denom)
    
    datan = datac / stddata
    Yn = Yc / stdY
    return datan.T @ Yn / denom

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
