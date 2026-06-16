
import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import svd as lapack_svd
from joblib import Parallel, delayed, effective_n_jobs
import pandas as pd
import copy

from . import utils

from pdb import set_trace
from scipy.linalg import orthogonal_procrustes

class NotFittedError(Exception):
    def __init__(self):
        self.message = '.fit() has not yet been called to fit model'
        super().__init__(self.message)

class BadStrArgError(Exception):
    def __init__(self, argname, provided, allowed):
        self.message = '%s is not a valid value for "%s". Must be one of %s' % (provided, argname, allowed)
        super().__init__(self.message)

def _check_str_arg(argname, provided, allowed):
    if provided not in allowed:
        raise BadStrArgError(argname, provided, allowed)

class BaseClass():
    def __init__(self, svd_method='lapack', boot_stat=None, random_state=None):
        _check_str_arg('svd_method', svd_method, ('lapack', 'randomized'))
        # Private attributes for tracking whether permutation testing and bootstrap resampling have been done
        self._fitted = False
        self._reset()
        # Document inheritable attributes in one place; requires initializing many to None but worth it imo
        # Attributes set by __init__()
        self.svd_method = svd_method #: ``str``: SVD method used
        self.boot_stat = boot_stat #: ``str``: Name of statistic whose distribution is derived during bootstrap resampling.
        self.random_state = random_state #: ``int``: Random state for reproducible permutation and bootstrap resampling.
        # Attributes set by fit()
        self.boot_stat_val_ = None #: ``numpy.ndarray`` Point estimate from initial decomposition of statistic whose distribution is derived during :meth:`bootstrap` resampling. Set by :meth:`fit`.
        self.data_ = None #: ``numpy.ndarray``: Data used to fit model. Set by :meth:`fit`.
        self.data_sals_ = None #: ``numpy.ndarray``: Right saliences/singular vectors used to compute data scores. Shape (n. observed vars, n. latent vars). Set by :meth:`fit`.
        self.design_sals_ = None #: ``numpy.ndarray``: Left saliences/singular vectors used to compute design scores. Shape (n. design saliences, n.latent variables). Set by :meth:`fit`.
        self.n_sv_ = None #: ``int``: Number of singular values, i.e., the number of latent variable pairs in the model. Set by :meth:`fit`.
        self.singular_vals_ = None #: ``numpy.ndarray``:  Singular values from the decomposition of the mean-centred data. Set by :meth:`fit`.
        self.variance_explained_ = None #: ``np.ndarray``: Proportion of variance explained by each latent variable pair. Set by :meth:`fit`.
        # Attributes set by permute()
        self.pvals_ = None #: ``numpy.ndarray``: Permutation p values for the latent variable pairs. Set by :meth:`permute`.
        # Attributes set by bootstrap()
        self.boot_stat_ci_ = None #: ``numpy.ndarray``: Confidence interval on stat named by :attr:`boot_stat` derived from bootstrap resampling. CI level is determined by :attr:`confint_level_`. Set by :meth:`bootstrap`.
        self.confint_level_ = None #: ``float``: Level of confidence interval on stat named by :attr:`boot_stat` to derive during bootstrap resampling (e.g., 0.95). Set by :meth:`bootstrap`.
        self.data_sals_std_ = None #: ``numpy.ndarray``: Standard deviations of data saliences (:attr:`data_sals_`) as estimated during bootstrap resampling. Set by :meth:`bootstrap`.
        self.data_sals_z_ = None #: ``numpy.ndarray``: Data saliences (:attr:`data_sals_`) divided by their standard deviations (:attr:`data_sals_std_`) as estimated during bootstrap resampling. Set by :meth:`bootstrap`.
        self.n_boot_ = None #: ``int``: Number of bootstrap resamples used. Set by :meth:`bootstrap`.
        if boot_stat is None:
            # Fill in with defaults
            if self._has_covariates:
                self.boot_stat = 'score-covariate-corr'
            else:
                self.boot_stat = 'condwise-scores-centred'
    def _reset(self):
        # Reset variables that track whether perm_done and boot_done
        self._perm_done = False
        self._boot_done = False
    def _setup_data(self, data):
        # Add data_ as a property
        valid_data = True
        if not isinstance(data, np.ndarray):
            valid_data = False
        else:
            data = data.copy()
            if data.ndim == 1:
                # Reshape to column array
                data = data.reshape((len(data), 1))
            elif data.ndim != 2:
                valid_data = False
        if not valid_data:
            raise ValueError('data must be a 1- or 2-dimensional numpy array')
        self.data_ = data
    def _setup_labels(self, labels):
        # Convert to dataframe
        if isinstance(labels, np.ndarray):
            labels = pd.DataFrame(labels)
            labels.columns = ['label_%s' % i for i in range(len(labels.columns))]
        elif not isinstance(labels, pd.DataFrame):
            raise ValueError('Data labels must be pandas DataFrame or numpy array')
        # Convert columns to categorical
        for col in labels.columns:
            labels[col] = pd.Categorical(labels[col])
        # Store frame
        self.label_frame_ = labels
        # Create numpy array of integer codes
        mat_cols = [labels[col].cat.codes for col in labels.columns]
        self.label_mat_ = np.stack(mat_cols).T
        # Check that each observation can be uniquely identified
        clusters = pd.MultiIndex.from_arrays(self.label_mat_.T)
        if len(clusters.unique()) < len(labels):
            raise ValueError('Individual observations cannot be uniquely identified with the current data labels. Consider adding a final "obs" column populated by np.arange(num_rows).')
    def _setup_stratification(self, modeled):
        # Set up attributes that determine how data will be stratified
        self.modeled_ = np.array(modeled)
        self.resample_ = ~self.modeled_ # TODO: set as needed
        self.permute_ = self.modeled_
    def _svd(self, M, compute_uv=True):
        # Single function to perform svd using the specified method
        if self.svd_method == 'lapack':
            if compute_uv:
                u, s, v = lapack_svd(M, full_matrices=False, compute_uv=True)
            else:
                s = lapack_svd(M, full_matrices=False, compute_uv=False)
        elif self.svd_method == 'randomized':
            u, s, v = randomized_svd(M, n_components=len(M))
        s[self.rank_:] = 0
        if compute_uv:
            out = u, s, v.T
        else:
            out = s
        return out
    def _initial_decomposition(self, to_factorize):
        # Initial fit and add various attributes
        u, s, v = self._svd(to_factorize)
        self.singular_vals_ = s
        self.n_sv_ = len(s)
        self.variance_explained_ = s**2 / sum(s**2)
        self.design_sals_ = u
        self.data_sals_ = v
        self._fitted = True
    def summary(self):
        """
        Summarize the model.

        Returns
        -------
        :class:`pandas.DataFrame`
            Data frame with one row per latent variable pair.
        
        Examples
        --------
        >>> mod.summary()
        """
        df = pd.DataFrame({
            'LV index': range(self.n_sv_),
            'singular value': self.singular_vals_,
            'variance explained': self.variance_explained_})
        if self._perm_done:
            pvals = self.pvals_
        else:
            pvals = float('nan')
        df['p value'] = pvals
        return df
    def flip_signs(self, lv_idx=None):
        """
        Flips the signs of one or more latent variables, to aid with interpretation.

        :param lv_idx: The index or indices of latent variables whose signs should be flipped. If None (default), signs are flipped for all latent variables.

        Parameters
        ----------
        lv_idx : indexer
            The index or indices of latent variables whose signs should be flipped. If None (default), signs are flipped for all latent variables.

        Examples
        --------
        >>> mod.flip_signs() # Flip all signs
        >>> mod.flip_signs(0) # Flip signs for the first latent variable
        >>> mod.flip_signs([0, 1]) # Flip signs for the first two   latent variables

        """
        if not self._fitted:
            raise NotFittedError()
        if lv_idx is None:
            lv_idx = range(self.n_sv_)
        self.design_sals_[:, lv_idx] *= -1
        self.design_scores_[:, lv_idx] *= -1 # These are just computed once so need to be reversed
        self.data_sals_[:, lv_idx] *= -1
        self.boot_stat_val_[:, lv_idx] *= -1
        if self._boot_done:
            self.data_sals_z_[:, lv_idx] *= -1
            self.boot_stat_ci_[..., lv_idx] *= -1
            self.boot_stat_ci_ = self.boot_stat_ci_[(1, 0), ...]
    def transform(self, data=None, lv_idx=None):
        """
        Compute data scores, i.e., coordinates of array data in the new basis defined by the latent variables, by multiplying a data array by the data saliences (the :attr:`data_sals_` property)

        Parameters
        ----------
        data : numpy.ndarray, optional
            Data to transform. The default is None, which yields scores for the data on which the model was fit (the :attr:`data_` property).
        lv_idx : indexer, optional
            Index of latent variable(s) for which to compute scores. Default is None, which computes scores for all latent variables.

        Returns
        -------
        data_scores : numpy.ndarray
            A 2D array of scores where rows correspond to different observations and columns correspond to different latent variables.

        Examples
        --------
        >>> scores = mod.transform() # Get scores for data used to fit model
        >>> scores = mod.transform(new_data) # Get scores for new data
        """
        if not self._fitted:
            raise NotFittedError()
        if data is None:
            data = self.data_
        sals = self.data_sals_
        if lv_idx is not None:
            sals = sals[:, lv_idx]
        data_scores = data @ sals
        return data_scores
    def _get_design_sal_labels(self):
        if any(self.modeled_):
            # Stratify the same way as when computing the matrix to factorize
            modeled_labels = self.label_frame_.iloc[:, self.modeled_]
            stratifier = pd.MultiIndex.from_arrays(self.label_mat_[:, self.modeled_].T)
            label_sets = stratifier.unique()
            rows = []
            for label_set in label_sets:
                # Create row
                row = []
                for col_idx, cat_idx in enumerate(label_set):
                    cat = modeled_labels.iloc[:, col_idx].cat.categories[cat_idx]
                    row.append(cat)
                if self._has_covariates:
                    # Add extra column for covariate
                    row.append(None) # Placeholder
                    for cov_name in self.covariate_names_:
                        row = row.copy()
                        row[-1] = cov_name
                        rows.append(row)
                else:
                    rows.append(row)
            df = pd.DataFrame(rows)
            colnames = self.label_frame_.columns[self.modeled_]
            if self._has_covariates:
                colnames = colnames.to_list()
                colnames.append('covariate')
            df.columns = colnames
        else:
            df = pd.DataFrame({'covariate': self.covariate_names_})
        return df
    def get_design_matrix(self):
        """
        Get design matrix, including any covariates, as a dataframe.

        Returns
        -------
        pd.DataFrame
            Design matrix as a dataframe
        """
        df = self.label_frame_.copy()
        if self._has_covariates:
            for i, cov in enumerate(self.covariate_names_):
                df[cov] = self.covariates_[:, i]
        return df
    def get_scores_frame(self, lv_idx=None):
        """
        Get dataframe containing design and data scores for each observation in :attr:`data_`, alongside condition information from the data labels.

        Parameters
        ----------
        lv_idx : indexer, optional
            Index of latent variable(s) for which to include design and data scores. The default is None, which includes scores for all latent variables.

        Returns
        -------
        df : pandas.dataframe
            Dataframe containing design and data scores for each observation.
            
        Notes
        -----
        Data is in long format, with a column specifying the latent variable corresponding to each score.
            
        Examples
        --------
        >>> mod.get_scores_frame().to_csv('scores.csv')
        """
        if not self._fitted:
            raise NotFittedError()
        if lv_idx is None:
            lv_idx = list(range(self.n_sv_))
        else:
            try:
                len(lv_idx)
            except:
                lv_idx = [lv_idx]
        lv_idxs = lv_idx
        lv_subdfs = []
        for lv_idx in lv_idxs:
            sub_df = self.get_design_matrix()
            sub_df['lv_idx'] = lv_idx
            sub_df['design_score'] = self.design_scores_[:, lv_idx]
            sub_df['data_score'] = self.transform(lv_idx=lv_idx)
            lv_subdfs.append(sub_df)
        df = pd.concat(lv_subdfs)
        df = df.reset_index(drop=True)
        return df
    def permute(self, n_perm=5000, return_null_dist=True, n_jobs=1, print_prog=True):
        """
        Perform permutation testing to assess the significance of the latent variables. p values become available after running this method through the :attr:`pvals_` property.

        Parameters
        ----------
        n_perm : int, optional
            Number of permutations t operform. The default is 5000.
        return_null_dist : bool, optional
            If ``True``, permutation samples will be returned as a 2D (n. perms, n. latent vars) array. Default is ``True``.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.
        print_prog : bool, optional
            Specifies whether to display a progress bar. Default is ``True``.

        Returns
        -------
        null_dist : numpy.ndarray
            2D array containing null distribution of singular values, where each row is a different permutation and each columns is a different singular value.

        Examples
        --------
        >>> mod.permute(n_perm=1000, n_jobs=-1)
        >>> print(mod.pvals_)
        """
        if not self._fitted:
            raise NotFittedError()
        if n_perm < 1:
            raise ValueError('n_perm must be a positive integer')
        silent = not print_prog
        # Pre-generate perm_idx
        perms = self._get_permutations(n_perm, n_jobs, silent)
        perm_singvals = Parallel(n_jobs=n_jobs)(
            delayed(self._single_permutation)(*perm)
            for perm in tqdm(perms, desc='Permuting', disable=silent)
        )
        null_dist = np.stack(perm_singvals)
        pvals = (np.sum(null_dist >= self.singular_vals_, axis=0) + 1) / (n_perm + 1)
        # Nullify p vals based on the rank of the matrix being decomposed
        pvals[self.rank_:] = np.nan
        self.pvals_ = pvals
        self._perm_done = True
        if return_null_dist:
            return null_dist
    def _get_permutations(self, n_perm, n_jobs, silent):
        ss = np.random.SeedSequence(self.random_state)
        child_sequences = ss.spawn(n_perm)
        perms = Parallel(n_jobs=n_jobs)(
            delayed(utils.cluster_permute)(self.label_mat_,
                                           self.permute_,
                                           np.random.default_rng(child_seq),
                                           return_cov_perm=self._has_covariates)
            for child_seq in tqdm(child_sequences, desc='Getting permutations', disable=silent)
        )
        '''
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=n_perm)
        perms = Parallel(n_jobs=n_jobs)(
            delayed(utils.cluster_permute)(self.label_mat_,
                                           self.modeled_,
                                           rng,
                                           return_cov_perm=self._has_covariates)
            for _ in tqdm(perms, desc='Getting permutations', disable=silent)
        )
        perms = [
            utils.cluster_permute(self.label_mat_,
                                  self.modeled_,
                                  rng,
                                  return_cov_perm=self._has_covariates)
            for perm_n in tqdm(range(n_perm), desc='Getting permutations', disable=silent)
        ]
        '''
        return perms
    def bootstrap(self, n_boot=5000, confint_level=0.95, alignment_method='rotate-design-sals', return_boot_stat_dist=True, n_jobs=1, print_prog=True):
        """
        Perform (stratified) bootstrap resampling to assess the reliability of the data saliences.

        Parameters
        ----------
        n_boot : int, optional
            Number of bootstrap resamples to compute. The default is 5000.
        confint_level : float, optional
            The confidence level of the quantile-based confidence intervals to compute. The default is 0.95.
        alignment_method : string, optional
            Method to be used for aligning recomputed data saliences with original data saliences. Must be one of:
            
            - ``'rotate-design-sals'`` (default): Find the rotation that solves the orthogonal procrustes problem to align the recomputed and original design saliences, then apply this to the recomputed data saliences. This is the what is computed in the original Matlab version of PLS.
            - ``'rotate-data-sals'``: Find the rotation that solves the orthogonal procrustes problem to align the recomputed and original data saliences, then apply this to the recomputed data saliences.
            - ``'flip-design-sals'``: Find the set of sign flips that ensures the inner product of the recomputed and original design saliences are positive, then apply these sign flips to the recomputed data saliences.
            - ``'flip-data-sals'``: Find the set of sign flips that ensures the inner product of the recomputed and original data saliences are positive, then apply these sign flips to the recomputed data saliences.
            - ``'none'``: Perform no alignment.
        return_boot_stat_dist : bool, optional
            If ``True``, distribution of ``boot_stat`` from resampling is returned. This is the distribution used to compute quantile-based confidence intervals. Default is ``True``.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.
        print_prog : bool, optional
            Specifies whether to display a progress bar. Default is ``True``.

        Returns
        -------
        :class:`numpy.ndarray`
            If `return_boot_dist` is true, returns the bootstrap distribution of the statistic named by :attr:`boot_stat`

        Examples
        --------
        >>> mod.bootstrap(1000, n_jobs=-1)
        >>> print(mod.data_sals_z_)
        >>> print(mod.boot_stat_ci[..., 0]) # Print CI of boot_stat for first LV
        """
        if not self._fitted:
            raise NotFittedError()
        if n_boot < 1:
            raise ValueError('n_boot must be a positive integer')
        _check_str_arg('alignment_method',
                        alignment_method,
                        ('rotate-design-sals', 'rotate-data-sals', 'flip-design-sals', 'flip-data-sals', 'none'))
        n_jobs = effective_n_jobs(n_jobs)
        silent = not print_prog
        self.n_boot_ = n_boot
        self.confint_level_ = confint_level
        # Pre-generate bootstrap samples
        boot_idxs = self._get_resamples(n_boot, min_unique=self._min_unique, silent=silent)
        # Run bootstraps in parallel
        # Set up container for boot_stat
        boot_stat_dist = []
        # Set up variables for Welford's algorithm
        old_mean = np.zeros_like(self.data_sals_)
        mean = np.zeros_like(self.data_sals_)
        M2 = np.zeros_like(self.data_sals_)
        results = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(self._single_resample)(boot_idx, alignment_method)
            for boot_idx in boot_idxs
        )
        count = 0
        for boot_stat, resampled_data_sals in tqdm(results, total=n_boot, desc='Resampling', disable=silent):
            boot_stat_dist.append(boot_stat)
            # Welford's algorithm
            count += 1
            old_mean[:] = mean
            mean += (resampled_data_sals - old_mean) / count
            M2 += (resampled_data_sals - old_mean) * (resampled_data_sals - mean)
        # Compute standard deviations for data saliences to get bootstrap ratios
        std_data_sals = np.sqrt(M2 / (n_boot - 1))
        self.data_sals_z_ = (self.data_sals_ @ np.diag(self.singular_vals_)) / std_data_sals
        self.data_sals_std_ = std_data_sals
        # Compute confidence intervals for design saliences
        boot_stats = np.stack(boot_stat_dist)
        alpha = 1 - confint_level
        self.boot_stat_ci_ = np.quantile(boot_stats, [alpha/2, 1 - alpha/2], axis=0)
        self._boot_done = True
        if return_boot_stat_dist:
            return boot_stats
    def _get_resamples(self, n_boot, min_unique, silent):
        rng = np.random.default_rng(self.random_state)
        # Set up to verify min unique
        corr_level = np.where(~self.modeled_)[0][-1]
        stratify = np.array([True]*len(self.modeled_))
        stratify[corr_level] = False
        resamples = []
        for resample_n in tqdm(range(n_boot), desc='Getting resamples', disable=silent):
            validated = False
            while not validated:
                resample = utils.cluster_resample(self.label_mat_,
                                                  self.resample_,
                                                  rng)
                # Check for min unique 
                resampled_label_mat_ = self.label_mat_[resample]
                stratifier = pd.MultiIndex.from_arrays(resampled_label_mat_[:, stratify].T)
                labels = stratifier.unique()
                # Assume valid, break on first invalid resample
                validated = True
                for label in labels:
                    obs_id = resample[stratifier == label]
                    if len(np.unique(obs_id)) < min_unique:
                        validated = False
                        break
            resamples.append(resample)
        return resamples        
    def _align(self, u, s, v, method):
        if method == 'rotate-data-sals':
            A, _ = orthogonal_procrustes(v, self.data_sals_, check_finite=False)
        elif method == 'rotate-design-sals':
            A, _ = orthogonal_procrustes(u, self.design_sals_, check_finite=False)
        elif method == 'flip-data-sals':
            A = np.sign(np.diag(v.T @ self.data_sals_))
        elif method == 'flip-design-sals':
            A = np.sign(np.diag(u.T @ self.design_sals_))
        elif method == 'none':
            A = np.diag([1]*self.n_sv_)
        aligned = v*s @ A
        return aligned
    def get_boot_stat_frame(self, lv_idx=None):
        """
        Get :attr:`boot_stat` as a dataframe, including upper and lower confidence limits if bootstrap resampling has been done.

        Parameters
        ----------
        lv_idx : indexer, optional
            Index of latent variable the dataframe should cover. The default is None, which yields a dataframe covering all latent variables.

        Returns
        -------
        df : pandas.dataframe
            :attr:`boot_stat` as a dataframe.

        """
        if not self._fitted:
            raise NotFittedError()
        if lv_idx is None:
            lv_idx = list(range(self.n_sv_))
        else:
            try:
                len(lv_idx)
            except:
                lv_idx = [lv_idx]
        lv_idxs = lv_idx
        lv_subdfs = []
        for lv_idx in lv_idxs:
            sub_df = self.design_sal_labels_.copy()
            sub_df['lv_idx'] = lv_idx
            sub_df['stat'] = self.boot_stat_val_[:, lv_idx]
            if self._boot_done:
                sub_df['L_CI'] = self.boot_stat_ci_[0, :, lv_idx]
                sub_df['U_CI'] = self.boot_stat_ci_[1, :, lv_idx]
            lv_subdfs.append(sub_df)
        df = pd.concat(lv_subdfs)
        df = df.reset_index(drop=True)
        return df
    def get_boot_stat_yerr(self, lv_idx):
        """
        Get yerr for statistic named by :attr:`boot_stat` that can be passed to a matplotlib bar plot.

        Parameters
        ----------
        lv_idx : int
            Integer indexing the latent variable of interest.

        Returns
        -------
        yerr : numpy.ndarray
            2D array with shape (2, n. design saliences) that can be passed to matplotib's pyplot.bar() as the yerr= argument.

        Examples
        --------
        >>> # Make bar plot of boot_stat
        >>> x = mod.design_sal_labels_['between']
        >>> lv_idx = 0 # First latent variable
        >>> height = mod.boot_stat_val_[:, lv_idx]
        >>> yerr = mod.get_boot_stat_yerr(lv_idx)
        >>> matplotlib.pyplot.bar(x=x, height=height, yerr=yerr)
        """
        if not self._fitted:
            raise NotFittedError()
        if not self._boot_done:
            raise ValueError('Bootstrap resampling must be done to obtain confidence intervals')
        est = self.boot_stat_val_[:, lv_idx]
        if len(est.shape) == 2:
            raise ValueError('lv_idx must index a single latent variable')
        ci = self.boot_stat_ci_[..., lv_idx]
        yerr = np.array([ci[1] - est,
                         est - ci[0]])
        return yerr

class PLSC(BaseClass):
    """
    Class for PLSC models, based on singular value decomposition of cross-correlation matrices stacked by condition.
    
    Parameters
    ----------
    boot_stat : str, optional
        Name of statistic to recompute on each bootstrap resample to get a confidence interval. Must be one of:

        - ``'score-covariate-corr'`` (default): Correlations between covariates and data scores (i.e., output of :meth:`transform`), computed within participants and averaged. Covariates and data may be original or resampled but scores are always computed by multiplying data by :attr:`data_sals_` (i.e., the saliences from the initial decomposition). This is the what is computed in the original Matlab version of PLS.
        - ``'condwise-scores'``: Condition-wise average data (original or resampled) multiplied by :attr:`data_sals_`, computed within participants and averaged. 
    svd_method : str, optional
        Method to use for singular value decomposition. Must be one of:
            
        - ``'lapack'`` (default): use ``numpy.linalg.svd``.
        - ``'randomized'``: use ``sklearn.utils.extmath.randomized_svd``.
    random_state : int, optional
        Random state of model for reproducible premutation and bootstrap resampling. Passed to ``numpy.random.default_rng`` internally. Default is ``None``.
    """
    _min_unique = 2 # For resampling
    _has_covariates = True
    def _setup_covariates(self, covariates):
        if isinstance(covariates, pd.DataFrame):
            self.covariate_names_ = covariates.columns
            self.covariates_ = covariates.to_numpy()
        elif isinstance(covariates, np.ndarray):
            if covariates.ndim != 2:
                # Reshape to 2d column vector
                covariates = covariates.reshape((-1, 1))
            n_cov = covariates.shape[1]
            self.covariates_ = covariates
            self.covariate_names_ = ['cov_%s' % i for i in range(n_cov)]
        elif isinstance(covariates, pd.Series):
            self.covariate_names_ = [covariates.name]
            self.covariates_ = covariates.to_numpy().reshape((-1, 1))
        else:
            raise ValueError('Covariates must be a pandas DataFrame or a numpy array')
        # Convert covariate names to array
        self.covariate_names_ = np.array(self.covariate_names_)
    def _get_design_scores(self):
        # Initialize
        design_scores = np.zeros((len(self.covariates_), self.n_sv_), dtype=self.design_sals_.dtype)
        # Align the observations with the design saliences, level-wise
        design_sal_labels = list(self.design_sal_labels_.itertuples(index=False, name=None))
        modeled_labels = self.label_frame_.iloc[:, self.modeled_]
        obs_labels = list(modeled_labels.itertuples(index=False, name=None))
        # Loop over levels of stratifying variables
        for curr_label in set(obs_labels):
            # Find where the observations are at this level, and which design saliences correspond to it
            obs_mask = [i for i, obs_label in enumerate(obs_labels) if obs_label == curr_label]
            sal_mask = [i for i, sal_label in enumerate(design_sal_labels) if sal_label[:-1] == curr_label]
            # Get the sub-matrices of the observed covariates and design saliences
            obs_submat = self.covariates_[obs_mask]
            sal_submat = self.design_sals_[sal_mask]
            # Ensure each covariate is being multiplied by the appropriate salience
            assert all(self.design_sal_labels_['covariate'].iloc[sal_mask] == self.covariate_names_)
            # Multiply them to get the current design scores
            design_scores[obs_mask] = obs_submat @ sal_submat
        return design_scores
    def fit(self, data, covariates, labels, modeled):
        """
        Fit a PLSC model.

        Parameters
        ----------
        data : numpy.ndarray
            Data array of shape (n. observations, n. features).
        covariates : numpy.ndarray | pd.DataFrame
            Covariate array or dataframe of shape (n. observations, n. covariates).
        labels : numpy.ndarray | pd.DataFrame
            Data label array or dataframe of shape (n. observations, n. levels) where n. levels refers to the number of levels at which the data are labeled. The hierarchy of labels moves from left to right---i.e., the broadest classifications should be in the leftmost column and the most granular classifications in the rightmost column.
        modeled : numpy.ndarray | list
            Iterable of booleans of length n. levels, each specifying whether the corresponding column in ``labels`` is used to stratify the data (``True``) or not (``False``).

        Returns
        -------
        self : :class:`PLSC`
            PLSC model fit to the data provided.
        
        Examples
        --------
        >>> # Simulate null data
        >>> n_var = 10
        >>> ptptwise_n_trials = [10, 10, 9, 8, 12]
        >>> ptptwise_conds = ['a', 'a', 'b', 'b', 'b']
        >>> data = np.concat([np.random.normal(size=(n_trials, n_var)) for n_trials in ptptwise_n_trials])
        >>> covs = np.concat([np.random.normal(size=(n_trials, 1)) for n_trials in ptptwise_n_trials])
        >>> # Generate labels for between-participants condition, participant IDs, and arbitrary trial indices
        >>> cond_labels = np.concat([cond]*n_trials for cond, n_trials in zip(ptptwise_conds, ptptwise_n_trials)])
        >>> ptpt_ids = np.concat([ptpt_id]*n_trials for ptpt_id, n_trials in enumerate(ptptwise_n_trials)])
        >>> trial_labels = np.arange(np.sum(ptptwise_n_trials))
        >>> labels = pd.DataFrame({'cond': cond_labels, 'ptpt': ptpt_ids, 'trial': trial_labels})
        >>> # Stratify only by condition, not by participant or trial
        >>> modeled = [True, False, False]
        >>> # Fit model
        >>> mod = pyplsc.PLSC()
        >>> mod.fit(data=data, covariates=covs, modeled=modeled)
        """
        # Compute within-participant stacked correlation matrices
        self._setup_data(data)
        self._setup_labels(labels)
        self._setup_covariates(covariates)
        self._setup_stratification(modeled)
        R = utils.stratified_corrs(self.data_,
                                   self.covariates_,
                                   self.label_mat_,
                                   self.modeled_)
        self.rank_ = np.linalg.matrix_rank(R)
        self._initial_decomposition(R)
        self.design_sal_labels_ = self._get_design_sal_labels() # TODO: implement
        self.design_scores_ = self._get_design_scores() # TODO: implement
        # Compute boot stat
        scores = self.transform()
        if self.boot_stat == 'score-covariate-corr':
            self.boot_stat_val_ = utils.stratified_corrs(scores,
                                                         self.covariates_,
                                                         self.label_mat_,
                                                         self.modeled_)
        elif self.boot_stat == 'condwise-scores':
            self.boot_stat_val_ = utils.stratified_average(scores,
                                                           self.label_mat_,
                                                           self.modeled_)
        return self
    def _single_permutation(self, permuted_labels, cov_perm):
        R = utils.stratified_corrs(self.data_,
                                   self.covariates_[cov_perm],
                                   permuted_labels,
                                   self.modeled_)
        s = self._svd(R, compute_uv=False)
        return s
    def _single_resample(self, resample, alignment_method):
        # Compute stacked cormats within ptpts and then average
        resampled_data = self.data_[resample]
        resampled_covs = self.covariates_[resample]
        resampled_label_mat_ = self.label_mat_[resample]
        R = utils.stratified_corrs(resampled_data,
                                   resampled_covs,
                                   resampled_label_mat_,
                                   self.modeled_)
        u, s, v = self._svd(R)
        resampled_data_sals = self._align(u, s, v, method=alignment_method)
        scores = self.transform(resampled_data)
        if self.boot_stat == 'score-covariate-corr':
            boot_stat = utils.stratified_corrs(scores,
                                               resampled_covs,
                                               resampled_label_mat_,
                                               self.modeled_)
        elif self.boot_stat == 'condwise-scores':
            boot_stat = utils.stratified_average(scores,
                                                 resampled_label_mat_,
                                                 self.modeled_)
        return boot_stat, resampled_data_sals

class BDA(BaseClass):
    """
    Barycentric discriminant analysis model, also known as mean-centred PLS. Used for analyzing condition-wise differences.
    
    Parameters
    ----------
    boot_stat : str, optional
        Name of statistic to recompute on each bootstrap resample to get a confidence interval. Must be one of:

        - ``'condwise-scores-centred'`` (default): Mean-centred condition-wise average data (original or resampled) multiplied by :attr:`data_sals_`. This is the what is computed in the original Matlab version of PLS.
        - ``'condwise-scores'``: Condition-wise average data (original or resampled) multiplied by :attr:`data_sals_`. 
    svd_method : str, optional
        Method to use for singular value decomposition. Must be one of:
            
        - ``'lapack'`` (default): use ``numpy.linalg.svd``.
        - ``'randomized'``: use ``sklearn.utils.extmath.randomized_svd``.
    random_state : int, optional
        Random state of model for reproducible premutation and bootstrap resampling. Passed to ``numpy.random.default_rng`` internally. Default is ``None``.
    """
    _min_unique = 1 # For resampling
    _has_covariates = False
    def _mean_center(self, matrix):
        out = matrix - matrix.mean(axis=0)
        return out
    def _get_design_scores(self):
        # Align individual observations with design saliences
        design_sal_labels = list(self.design_sal_labels_.itertuples(index=False, name=None))
        design_scores = []
        for obs_label in self.label_frame_.iloc[:, self.modeled_].itertuples(index=False, name=None):
            idx = design_sal_labels.index(obs_label)
            design_scores.append(self.design_sals_[idx])
        return np.stack(design_scores)
    def fit(self, data, labels, modeled):
        """
        Fit a BDA model.

        Parameters
        ----------
        data : numpy.ndarray
            Data array of shape (n. observations, n. features).
        covariates : numpy.ndarray | pd.DataFrame
            Covariate array or dataframe of shape (n. observations, n. covariates).
        labels : numpy.ndarray | pd.DataFrame
            Data label array or dataframe of shape (n. observations, n. levels) where n. levels refers to the number of levels at which the data are labeled. The hierarchy of labels moves from left to right---i.e., the broadest classifications should be in the leftmost column and the most granular classifications in the rightmost column.
        modeled : numpy.ndarray | list
            Iterable of booleans of length n. levels, each specifying whether the corresponding column in ``labels`` is used to stratify the data (``True``) or not (``False``).

        Returns
        -------
        self : :class:`PLSC`
            PLSC model fit to the data provided.
        
        Examples
        --------
        >>> # Simulate null data
        >>> n_var = 10
        >>> ptptwise_n_trials = [10, 10, 9, 8, 12]
        >>> data = [np.random.normal(size=(n_trials, n_var)) for n_trials in ptptwise_n_trials]
        >>> covs = [np.random.normal(size=(n_trials, 1)) for n_trials in ptptwise_n_trials] 
        >>> # Fit model
        >>> mod = pyplsc.WPLSC()
        >>> mod.fit(data=data, covariates=covs, weighted=True)
        """
        # Compute within-participant stacked correlation matrices
        self._setup_data(data)
        self._setup_labels(labels)
        self._setup_stratification(modeled)
        M = utils.stratified_average(self.data_,
                                     self.label_mat_,
                                     self.modeled_)
        M = self._mean_center(M)
        self.rank_ = np.linalg.matrix_rank(M)
        self._initial_decomposition(M)
        self.design_sal_labels_ = self._get_design_sal_labels() # TODO: implement
        # self.design_scores_ = None # TODO: implement
        self.design_scores_ = self._get_design_scores()
        # Compute boot stat
        scores = self.transform()
        SM = utils.stratified_average(scores,
                                      self.label_mat_,
                                      self.modeled_)
        if self.boot_stat == 'condwise-scores-centred':
            self.boot_stat_val_ = self._mean_center(SM)
        elif self.boot_stat == 'condwise-scores':
            self.boot_stat_val_ = SM
        return self
    def _single_permutation(self, permuted_labels):
        M = utils.stratified_average(self.data_,
                                     permuted_labels,
                                     self.modeled_)
        M = self._mean_center(M)
        s = self._svd(M, compute_uv=False)
        return s
    def _single_resample(self, resample, alignment_method):
        # Compute stacked cormats within ptpts and then average
        resampled_data = self.data_[resample]
        resampled_label_mat_ = self.label_mat_[resample]
        M = utils.stratified_average(resampled_data,
                                     resampled_label_mat_,
                                     self.modeled_)
        M = self._mean_center(M)
        u, s, v = self._svd(M)
        resampled_data_sals = self._align(u, s, v, method=alignment_method)
        scores = self.transform(resampled_data)
        SM = utils.stratified_average(scores,
                                      resampled_label_mat_,
                                      self.modeled_)
        if self.boot_stat == 'condwise-scores-centred':
            boot_stat = self._mean_center(SM)
        elif self.boot_stat == 'condwise-scores':
            boot_stat = SM
        return boot_stat, resampled_data_sals