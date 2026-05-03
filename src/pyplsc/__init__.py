
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
    # Parent class for PLSC, BDA, and WPLSC.
    def __init__(self, svd_method='lapack', boot_stat=None, min_unique=False, random_state=None):
        _check_str_arg('svd_method', svd_method, ('lapack', 'randomized'))
        # Private attributes for tracking whether permutation testing and bootstrap resampling have been done
        self._fitted = False
        self._reset()
        self._min_unique = min_unique
        self._has_covarites = False
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
    def _setup_design_matrix(self, design=None, between=None, within=None, participant=None):
        # Add design_matrix_ and stratifier_ as attributes
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
            design = design.copy()
            for colname, col in cols.items():
                if col is None:
                    design[colname] = null_col.copy()
                elif isinstance(col, str):
                    design[colname] = pd.Categorical(design[col])
                else:
                    design[colname] = pd.Categorical(col)
            design = design[['between', 'within', 'participant']]
        # Validation
        between_per_participant = design.groupby('participant')['between'].nunique()
        if not all(between_per_participant == 1):
            raise Warning('Some participants belong to more than one between-participants condition')
        within_per_participant = design.groupby('participant')['within'].nunique()
        if len(within_per_participant.unique()) > 1:
            raise Warning('Participants do not all have the same number of within-participants conditions')
        self.design_ = design
        self.stratifier_ = utils.get_stratifier(design) # This is handy to pre-compute because it's used many times later on
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
    def _get_design_sal_labels(self):
        """
        # TODO: move this documentation to the attribute doc
        Get the labels corresponding to each row of the design saliences. For BDA, each row corresponds to a combination of the between- and within-participant conditions. For PLSC, eqch row additionally corresponds to a covariate.
        
        Parameters
        ----------
        which : indexer
            Specifies which labels should be returned. Can be "between", "within", and "covariate" (for PLSC) or an iterable containing a combination of these.
        
        Returns
        -------
        labels : pandas.DataFrame
            A dataframe with one column corresponding to each label and one row corresponding to each row of the design saliences.

        Examples
        --------
        >>> labels = mod.get_labels()
        >>> btwn_labels = mod.get_labels('between')
        >>> cond_labels = mod.get_labels(['between', 'within'])
        """
        condition_labels = self.design_[['between', 'within']].drop_duplicates()
        if self._has_covarites:
            # Covariates included; create product of conditions and covariates
            index = pd.MultiIndex.from_product(
                [condition_labels.index, self.covariate_names_],
                names=['condition_combo', 'covariate']
            )
            # Build the full dataframe
            labels = (
                condition_labels
                .loc[index.get_level_values('condition_combo')]  # repeat each combo row
                .reset_index(drop=True)
                .assign(covariate=index.get_level_values('covariate'))
            )
        else:
            labels = condition_labels
        labels = labels.reset_index(drop=True)
        return labels
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
    def _get_data(self):
        # This exists to be overwritten by WPLSC
        return self.data_
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
            data = self._get_data()
        sals = self.data_sals_
        if lv_idx is not None:
            sals = sals[:, lv_idx]
        data_scores = data @ sals
        return data_scores
    def get_design_matrix(self):
        """
        Get design matrix, including any covariates, as a dataframe.

        Returns
        -------
        pd.DataFrame
            Design matrix as a dataframe
        """
        df = self.design_.copy()
        if self._has_covarites:
            for i, cov in enumerate(self.covariate_names_):
                df[cov] = self.covariates_[:, i]
        return df
    def get_scores_frame(self, lv_idx=None):
        """
        Get dataframe containing design and data scores for each observation in :attr:`data_`, alongside condition information from the design matrix (:attr:`design_`).

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
    def _get_permutations(self, n_perm, silent):
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
        for perm_n in tqdm(range(n_perm), desc='Getting permutations', disable=silent):
            if case == 'only-between':
                # Just shuffle all rows
                perm = rng.permutation(n_obs)
            else:
                # There is a within-participants factor
                # Create a copy to shuffle
                rows_by_ptpt = copy.deepcopy(unshuffled_rows_by_ptpt)
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
        perms = self._get_permutations(n_perm, silent)
        perm_singvals = Parallel(n_jobs=n_jobs)(
            delayed(self._single_permutation)(perm)
            for perm in tqdm(perms, desc='Permuting', disable=silent)
        )
        null_dist = np.stack(perm_singvals)
        pvals = (np.sum(null_dist >= self.singular_vals_, axis=0) + 1) / (n_perm + 1)
        if isinstance(self, BDA):
            # Nullify p vals based on the rank of the matrix being decomposed
            pvals[self.rank_:] = np.nan
        self.pvals_ = pvals
        self._perm_done = True
        if return_null_dist:
            return null_dist
    def _get_resamples(self, n_boot, min_unique=1, silent=False):
        rng = np.random.default_rng(self.random_state)
        rows_by_ptpt = self.design_.groupby('participant').indices
        # Is there a between-participants condition?
        # If so, we'll do stratified resampling
        if self.design_['between'].nunique() > 1:
            case = 'between'
            ptpts_by_between = self.design_.groupby('between')['participant'].unique()
        else:
            case = 'no-between'
            unique_ptpts = self.design_['participant'].unique()
        if min_unique > 1:
            strat_levels = np.unique(self.stratifier_)
            ptpts = self.design_['participant'].to_numpy()
        resamples = []
        for boot_n in tqdm(range(n_boot), desc='Getting resamples', disable=silent):
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
                if min_unique > 1:
                    # Check for levels with only one participant
                    resampled_strat = self.stratifier_[resample]
                    resampled_ptpts = ptpts[resample]
                    validated = all([len(np.unique(resampled_ptpts[resampled_strat == lvl])) >= min_unique
                                     for lvl in strat_levels])
                else:
                    validated = True
            resamples.append(resample)
        return resamples
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
  
class BtwnClass(BaseClass):
    # Class for models that are not trial-level
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Document inheritable attributes unique to these model types
        self.design_ = None #: ``pandas.DataFrame``: Design matrix with columns "between", "within", and "participant". Set by :meth:`fit`.
        self.design_sal_labels_ = None #: ``pandas.DataFrame``: Dataframe with rows corresponding to rows of the design saliences and columns specifying between-participants conditions, within-participants conditions, and covarites. Set by :meth:`fit`.
        self.stratifier_ = None #: ``numpy.ndarray``: Integer array that indexes each unique combination of between- and within-participants condition. Used to stratify the data for mean-centering. Set by :meth:`fit`.
        self.design_scores_ = None #: ``numpy.ndarray`` Design scores for the data used to fit the model. Set by :meth:`fit`.

class PLSC(BtwnClass):
    """
    Partial least squares correlation model, also known as behavioural PLS. Used for analyzing between-participants relationships between data and covariates across multiple conditions. For analyzing within-participant correlations, see :class:`WPLSC`.
    
    Parameters
    ----------
    boot_stat : str, optional
        Name of statistic to recompute on each bootstrap resample to get a confidence interval. Must be one of:

        - ``'score-covariate-corr'`` (default): Correlations between covariates and data scores (i.e., output of :meth:`transform`). Covariates and data may be original or resampled but scores are always computed by multiplying data by :attr:`data_sals_` (i.e., the saliences from the initial decomposition). This is the what is computed in the original Matlab version of PLS.
        - ``'condwise-scores'``: Condition-wise average data (original or resampled) multiplied by :attr:`data_sals_`. 
    svd_method : str, optional
        Method to use for singular value decomposition. Must be one of:
            
        - ``'lapack'`` (default): use ``numpy.linalg.svd``.
        - ``'randomized'``: use ``sklearn.utils.extmath.randomized_svd``.
    random_state : int, optional
        Random state of model for reproducible premutation and bootstrap resampling. Passed to ``numpy.random.default_rng`` internally. Default is ``None``.
    """
    def __init__(self, boot_stat='score-covariate-corr', svd_method='lapack', random_state=None):
        _check_str_arg('boot_stat',
                        boot_stat,
                        ('score-covariate-corr', 'condwise-scores'))
        super().__init__(boot_stat=boot_stat,
                         svd_method=svd_method,
                         random_state=random_state,
                         min_unique=2)
        self.covariates_ = None #: ``pd.dataframe``: Data frame containing covariate data. One column per covariate, one row per observation. Set by :meth:`fit`.
    def _setup_covariates(self, design, covariates):
        covariates_array, covariate_names = utils.get_covariates_array(design, covariates)
        if len(covariates_array) != len(self.data_):
            raise ValueError('Must be as many covariate rows as data rows')
        self.covariates_ = covariates_array
        self.covariate_names_ = covariate_names
        self._has_covarites = True
    def _get_design_scores(self):
        # Initialize
        design_scores = np.zeros((len(self.covariates_), self.n_sv_), dtype=self.design_sals_.dtype)
        # Align the observations with the design saliences, level-wise
        sal_levels = utils.get_stratifier(self.design_sal_labels_, output='tuples')
        obs_levels = utils.get_stratifier(self.design_, output='tuples')
        for curr_lvl in set(obs_levels):
            obs_mask = [i for i, obs_lvl in enumerate(obs_levels) if obs_lvl == curr_lvl]
            sal_mask = [i for i, sal_lvl in enumerate(sal_levels) if sal_lvl == curr_lvl]
            obs_submat = self.covariates_[obs_mask]
            sal_submat = self.design_sals_[sal_mask]
            # Ensure each covariate is being multiplied by the appropriate salience
            assert all(self.design_sal_labels_['covariate'].iloc[sal_mask] == self.covariate_names_)
            design_scores[obs_mask] = obs_submat @ sal_submat
        return design_scores
    def fit(self, data, covariates, design=None, between=None, within=None, participant=None):
        """
        Fit a partial least squares correlation model.

        Parameters
        ----------
        data : numpy.ndarray
            Data array of shape (n. observations, n. features). Each row should contain the average data for a participant, possibly the average for some within-participants condition for a participant.
        covariates : numpy.ndarray or pandas.DataFrame or list
            2D data of size (n. observations, n. features) to be used as covariates, or names of columns in ``design`` containing covariates.
        design : pandas.DataFrame, optional
            DataFrame with columns to indicate between-participant group membership, within-participant condition, and/or participant identity, as applicable. The default is None.
        between : str or iterable, optional
            Between-participants condition. This can be specified as a string referring to the appropriate column in ``design`` or as an iterable containing an indicator of group membership (e.g., a list of strings or integers). The default is None, indicating an absence of between-participant conditions.
        within : str or iterable, optional
            Within-participants condition. This can be specified as a string referring to the appropriate column in ``design`` or as an iterable containing an indicator of condition (e.g., a list of strings or integers). The default is None, indicating an absence of within-participant conditions.
        participant : str or iterable, optional
            Participant identifier. This can be specified as a string referring to the appropriate column in ``design`` or as an iterable containing an indicator of participant identity (e.g., a list of strings or integers). The default is None, which is only permitted when there are no within-participant conditions.

        Examples
        --------
        >>> mod = pyplsc.PLSC()
        >>> data = numpy.random.normal(size=(4, 3)) # 4 observations of 3 variables
        >>> covariates = numpy.random.normal(size=(4, 2)) # 4 observations of 2 covariates
        >>> design = pandas.DataFrame({'group': [0, 0, 1, 1]})
        >>> # Pattern 1: provide design matrix, specify column names of condition indicators
        >>> mod.fit(data, design, between='group')
        >>> # Pattern 2: provide condition indicators directly as iterables
        >>> mod.fit(data, between)
        """
        self._reset()
        self._setup_data(data)
        self._setup_design_matrix(design, between, within, participant)
        self._setup_covariates(design, covariates)
        self.design_sal_labels_ = self._get_design_sal_labels()
        R = utils.get_stacked_cormats(
            self.data_,
            self.covariates_,
            self.stratifier_)
        self.rank_ = np.linalg.matrix_rank(R)
        self._initial_decomposition(R)
        self.design_scores_ = self._get_design_scores()
        # Compute boot stat
        scores = self.transform()
        if self.boot_stat == 'score-covariate-corr':
            # Correlation between covariates and data scores
            val = utils.get_stacked_cormats(scores,
                                            self.covariates_,
                                            self.stratifier_)
        elif self.boot_stat == 'condwise-scores':
            # Condition-wise brain scores
            val = utils.get_groupwise_means(scores, self.stratifier_)
        self.boot_stat_val_ = val
    def _single_permutation(self, perm):
        R = utils.get_stacked_cormats(
            self.data_,
            self.covariates_[perm],
            self.stratifier_[perm])
        s = self._svd(R, compute_uv=False)
        return s
    def _single_resample(self, resample_idx, alignment_method):
        # Run decomposition
        resampled_data = self.data_[resample_idx]
        resampled_cov = self.covariates_[resample_idx]
        resampled_strat = self.stratifier_[resample_idx]
        R = utils.get_stacked_cormats(resampled_data,
                                      resampled_cov,
                                      resampled_strat)
        u, s, v = self._svd(R)
        resampled_data_sals = self._align(u, s, v, method=alignment_method)
        if self.boot_stat == 'score-covariate-corr':
            # Correlation between covariates and data scores
            boot_stat = utils.get_stacked_cormats(resampled_data @ self.data_sals_, # Brain scores
                                                  resampled_cov,
                                                  resampled_strat)
        elif self.boot_stat == 'condwise-scores':
            scores = resampled_data @ self.data_sals_
            boot_stat = utils.get_groupwise_means(scores, resampled_strat)
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
    def __init__(self, boot_stat='condwise-scores-centred', svd_method='lapack', random_state=None):
        _check_str_arg('boot_stat',
                       boot_stat,
                       ('condwise-scores-centred', 'condwise-scores'))
        super().__init__(svd_method=svd_method,
                         boot_stat=boot_stat,
                         random_state=random_state,
                         min_unique=1)
        self.effects = None #: ``tuple``: Effects present in the model. Set by :meth:`fit`.
    def _setup_effects(self, effects, between, within):
        if between is None and within is None:
            raise ValueError('Observations must be differentiated by some categorical variable (specified via "between" or "within") for BDA')
        possible_effects = set()
        if between is not None:
            possible_effects |= {'between'}
        if within is not None:
            possible_effects |= {'within'}
        if possible_effects == {'between', 'within'}:
            possible_effects |= {'interaction'}
        self.__possible_effects = possible_effects
        if effects == 'all':
            # Test all possible effects
            self.effects = self.__possible_effects
        elif isinstance(effects, str):
            # Single effect
            self.effects = {effects}
        else:
            self.effects = set(effects)
            if len(self.effects) == 0:
                raise ValueError('At least one effect must be specified.')
        invalid_effects = self.effects - self.__possible_effects
        if len(invalid_effects) > 0:
            raise ValueError('Effect(s) %s cannot be measured given the conditions defined in the design matrix' % invalid_effects)
    def _get_mean_centred(self, data=None, design=None, stratifier=None):
        if data is None:
            data = self.data_
        if design is None:
            design = self.design_
        if stratifier is None:
            stratifier = self.stratifier_
        rm_effects = self.__possible_effects - self.effects
        labels = self.design_sal_labels_
        # Begin with matrix including all effects, then remove as specified
        groupwise_means = utils.get_groupwise_means(data, stratifier)
        if len(rm_effects) > 0:
            # Compute effects to remove
            effect_mats = {}
            if len({'between', 'interaction'} & rm_effects) > 0:
                levelwise_means = utils.get_groupwise_means(groupwise_means, labels['between'].cat.codes)
                effect_mats['between'] = levelwise_means[labels['between'].cat.codes]
            if len({'within', 'interaction'} & rm_effects) > 0:
                levelwise_means = utils.get_groupwise_means(data, design['within'].cat.codes)
                effect_mats['within'] = levelwise_means[labels['within'].cat.codes]
            if 'interaction' in rm_effects:
                effect_mats['interaction'] = groupwise_means - effect_mats['between'] - effect_mats['within']
            # Remove specified effects
            for rm_effect in rm_effects:
                groupwise_means -= effect_mats[rm_effect]
        # Mean centre
        groupwise_means -= groupwise_means.mean(axis=0)
        return groupwise_means
    def fit(self, data, design=None, between=None, within=None, participant=None, effects='all'):
        """
        Fit a barycentric discriminant analysis model.

        Parameters
        ----------
        data : numpy.ndarray
            Data array of shape (n. observations, n.features). Each row should contain the average data for a participant, possibly the average for some within-participants condition for a participant.
        design : pandas.DataFrame, optional
            DataFrame with columns to indicate between-participant group membership, within-participant condition, and/or participant identity, as applicable. The default is None.
        between : str or iterable, optional
            Between-participants factor. This can be specified as a string referring to the appropriate column in ``design`` or as an iterable containing an indicator of group membership (e.g., a list of strings or integers). The default is None, indicating an absence of between-participant conditions.
        within : str or iterable, optional
            Within-participants factor. This can be specified as a string referring to the appropriate column in ``design`` or as an iterable containing an indicator of condition (e.g., a list of strings or integers). The default is None, indicating an absence of within-participant conditions.
        participant : str or iterable, optional
            Participant identifier. This can be specified as a string referring to the appropriate column in ``design`` or as an iterable containing an indicator of participant identity (e.g., a list of strings or integers). The default is None, which is only permitted when there are no within-participant conditions.
        effects : str or iterable, optional
            Effects to be included in the model. If only a between-participants factor is specified, then only a main effect of between-participants condition can be measured (same goes for within-participants, mutatis mutandis). However, if both a between- and a within-participants factor are specified, then any of the following effects can be specified:
            
            - ``'between'``: main effect of between-participants condition
            - ``'within'``: main effect of within-participants condition
            - ``'interaction'``: interaction of between- and within-participants factors
            
            In the original Matlab PLS, the default behaviour is to remove the between-participants factor, which is equivalent to ``effects=('within', 'interaction')``.
            
        Examples
        --------
        >>> mod = pyplsc.BDA()
        >>> data = numpy.random.normal(size=(4, 3))
        >>> design = pandas.DataFrame({'group': [0, 0, 1, 1]})
        >>> # Pattern 1: provide design matrix, specify column names of condition indicators
        >>> mod.fit(data, design, between='group')
        >>> # Pattern 2: provide condition indicators directly as iterables
        >>> mod.fit(data, between=design['group'])
        >>> # Specifying effects
        >>> mod.fit(data, between=between, within=within, participant=participant,
        ...         effects=('within', 'interaction'))

        """
        self._reset()
        self._setup_effects(effects, between, within)
        self._setup_data(data)
        self._setup_design_matrix(design, between, within, participant)
        self.design_sal_labels_ = self._get_design_sal_labels()
        if len(np.unique(self.stratifier_)) == 1:
            raise ValueError('The conjunction of between- and within-participant factors has only one unique level. I.e., the data cannot be stratified for BDA.')
        mean_centred = self._get_mean_centred()
        self.rank_ = np.linalg.matrix_rank(mean_centred)
        self._initial_decomposition(mean_centred)
        # Compute design scores
        self.design_scores_ = self.design_sals_[self.stratifier_]
        if self.boot_stat == 'condwise-scores-centred':
            val = mean_centred @ self.data_sals_
        elif self.boot_stat == 'condwise-scores':
            data_scores = self.transform()
            val = utils.get_groupwise_means(data_scores, self.stratifier_)
        self.boot_stat_val_ = val
        return self
    def _single_permutation(self, perm):
        # Compute SVD for this permutation
        mean_centred = self._get_mean_centred(
            design=self.design_.iloc[perm],
            stratifier=self.stratifier_[perm])
        s = self._svd(mean_centred, compute_uv=False)
        return s
    def _single_resample(self, resample_idx, alignment_method):
        # Run decomposition
        resampled_data = self.data_[resample_idx]
        resampled_design = self.design_.iloc[resample_idx]
        resampled_strat = self.stratifier_[resample_idx]
        mean_centred = self._get_mean_centred(
            data=resampled_data,
            design=resampled_design,
            stratifier=resampled_strat)
        u, s, v = self._svd(mean_centred)
        resampled_data_sals = self._align(u, s, v, method=alignment_method)
        # Brain scores
        if self.boot_stat == 'condwise-scores-centred':
            boot_stat = mean_centred @ self.data_sals_
        elif self.boot_stat == 'condwise-scores':
            scores = resampled_data @ self.data_sals_
            boot_stat = utils.get_groupwise_means(scores, resampled_strat)
        return boot_stat, resampled_data_sals
    
class WPLSC(BaseClass):
    """
    Within-participants PLSC (`Roberts et al., 2016 <https://doi.org/10.1016/j.neuroimage.2016.04.028>`_). Used for analyzing within-partcipants correlations. Cross-correlation matrices are computed within participants, averaged, and submitted to singular value decomposition.
    
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
    def __init__(self, boot_stat='score-covariate-corr', svd_method='lapack', random_state=None):
        _check_str_arg('boot_stat',
                        boot_stat,
                        ('score-covariate-corr', 'condwise-scores'))
        super().__init__(boot_stat=boot_stat,
                         svd_method=svd_method,
                         random_state=random_state)
        self.models_ = None #: ``list``: Participant-specific :class:`PLSC` models. Set by :meth:`fit`.
        self.participant_labels_ = None #: ``pandas.Categorical``: Participants labels. Set by :meth:`fit`.
        self.weights_ = None #: ``numpy.ndarray``: Weights, based on number of trials, applied to participant-level cross-correlation matrices when averaged together. Set by ``weighted`` argument to :meth:`fit`.
    def fit(self, data, covariates, design=None, within=None, participant=None, weighted=False):
        """
        Fit a within-participants PLSC model.

        Parameters
        ----------
        data : list
            List of participant-specific data arrays. Each should be a ``numpy.ndarray`` of shape (n. trials, n. observed vars).
        covariates : list or str
            List of participant-specific covariates (in which case each list element must be a valid ``covariates`` argument to :class:`PLSC.fit`), or the names of the columns in ``design`` that contain the covariates.
        design : list, optional
            List of participant-specific design matrices. Each list element must be a valid ``design`` argument to :class:`PLSC.fit`. The default is ``None``.
        within : list or str, optional
            List of participant-specific indicators of within-participant condition (in which case each list element must be a valid ``between`` argument to :class:`PLSC.fit`), or the names of the columns in ``design`` that contain the within-participant condition indicators.
        participant : list, optional
            A list of participant identifiers (integers or strings).
        weighted : bool, optional
            Specifies whether participant-level cross-covariance matrices should weighted by number of trials when averaged together. Default is False.

        Returns
        -------
        None
        
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
        if len(set(arr.shape[1] for arr in data)) > 1:
            raise ValueError('All data arrays must contain the same number of variables')
        if design is None:
            design = [None]*len(data)
        if within is None:
            within = [None]*len(data)
        if participant is None:
            participant = ['participant-%s' % n for n in range(len(data))]
        # Handle case where covariates are column names
        # Turn single strings into one-element lists
        if isinstance(covariates, str):
            covariates = [covariates]
        # If str, just replicate for each participant
        if isinstance(covariates[0], str):
            covariates = [covariates]*len(data)
        if isinstance(within[0], str):
            within = [within]*len(data)
        if len(covariates) != len(data):
            raise ValueError('Must be as many participant-specific sets of covariates as there are participant-specific data arrays.')
        if len(within) != len(data):
            raise ValueError('Must be as many participant-specific sets of condition indicators as there are participant-specific data arrays.')
        if len(np.unique(participant)) != len(data):
            raise ValueError('Must be as many unique participant identifiers as there are participant-specific data arrays.')
        self.participant_labels_ = pd.Categorical(participant)
        
        # Set up participant-specific PLSC models        
        self.models_ = []
        # Get seeds for participant-specific models
        ss = np.random.SeedSequence(self.random_state)
        seeds = ss.spawn(len(data))
        for ptpt_data, ptpt_cov, ptpt_design, ptpt_within, seed in zip(data, covariates, design, within, seeds):
            model = PLSC(random_state=seed)
            # Set up data using internal API but don't fit participant-level models
            model._setup_data(ptpt_data)
            model._setup_design_matrix(design=ptpt_design,
                                       between=ptpt_within) # Note that a within-participants condition is between in the sense of "between-trials"
            model._setup_covariates(design=ptpt_design,
                                    covariates=ptpt_cov)
            model.design_sal_labels_ = model._get_design_sal_labels()
            self.models_.append(model)
        # Set weights
        if weighted:
            weights = []
            for model in self.models_:
                weights.append(len(model.data_) - 3) # Weight by expected variance of z
            weights = np.array(weights).astype(np.float64)
        else:
            weights = np.array([1]*len(self.models_)).astype(np.float64)
        self.weights_ = weights
        # Compute within-participant stacked correlation matrices
        ptpt_Rs = []
        for model, weight in zip(self.models_, self.weights_):
            R = utils.get_stacked_cormats(
                model.data_,
                model.covariates_,
                model.stratifier_)
            R = np.arctanh(R) # Fisher z-transform
            ptpt_Rs.append(R)
        mean_R = np.average(ptpt_Rs, axis=0, weights=self.weights_)
        mean_R = np.tanh(mean_R) # Inverse z-transform
        self.rank_ = np.linalg.matrix_rank(mean_R)
        self._initial_decomposition(mean_R)
        # Decomposition is on the average rather than the participant-wise Rs
        # Thus copy decomposition results to individual models for computing scores
        # Note that this isn't wasteful because each array is one object in memory
        for model in self.models_:
            for attr in ['design_sals_',
                         'singular_vals_',
                         'data_sals_',
                         'n_sv_']:
                setattr(model, attr, getattr(self, attr))
            model._fitted = True
            model.data_scores_ = model.transform()
            model.design_scores_ = model._get_design_scores()
        self.design_sal_labels_ = self.models_[0]._get_design_sal_labels()
        self.design_scores_ = np.concat([model.design_scores_ for model in self.models_],
                                        axis=0)
        # Compute boot_stat within ptpts and then average
        ptpt_boot_stats = []
        for model in self.models_:
            scores = model.transform()
            if self.boot_stat == 'score-covariate-corr':
                stat = utils.get_stacked_cormats(
                    scores,
                    model.covariates_,
                    model.stratifier_)
                stat = np.atanh(stat) # z-transform since this is a correlation
            elif self.boot_stat == 'condwise-scores':
                stat = utils.get_groupwise_means(
                    scores,
                    model.stratifier_)
            ptpt_boot_stats.append(stat)
        boot_stat = np.average(ptpt_boot_stats,
                               axis=0,
                               weights=self.weights_)
        if self.boot_stat == 'score-covariate-corr':
            boot_stat = np.tanh(boot_stat) # back-transform from fisher z
        self.boot_stat_val_ = boot_stat
    def get_scores_frame(self, lv_idx=None):
        """
        Get dataframe containing design and data scores for each trial.

        Parameters
        ----------
        lv_idx : indexer, optional
            Index of latent variable(s) for which to include design and data scores. The default is None, which includes scores for all latent variables.

        Returns
        -------
        df : pandas.dataframe
            Dataframe containing design and data scores for each trial.
            
        Notes
        -----
        Data is in long format, with a column specifying the latent variable corresponding to each score.
            
        Examples
        --------
        >>> mod.get_scores_frame().to_csv('scores.csv')
        """
        ptpt_dfs = []
        for model, label in zip(self.models_, self.participant_labels_):
            df = model.get_scores_frame(lv_idx)
            df['participant'] = label
            ptpt_dfs.append(df)
        return pd.concat(ptpt_dfs)
    def _get_data(self):
        # Get big array of all participant-wise data
        return np.concat([model._get_data() for model in self.models_],
                         axis=0)
    def _get_permutations(self, n_perm, silent):
        ptptwise_perms = []
        for model in tqdm(self.models_, desc='Getting permutations', disable=silent):
            ptpt_perms = model._get_permutations(n_perm=n_perm, silent=True)
            ptptwise_perms.append(ptpt_perms)
        return list(zip(*ptptwise_perms))
    def _single_permutation(self, perms):
        ptpt_Rs = []
        for model, perm in zip(self.models_, perms):
            R = utils.get_stacked_cormats(
                model.data_,
                model.covariates_[perm],
                model.stratifier_[perm])
            R = np.arctanh(R)
            ptpt_Rs.append(R)
        mean_R = np.average(ptpt_Rs, axis=0, weights=self.weights_)
        mean_R = np.tanh(mean_R)
        s = self._svd(mean_R, compute_uv=False)
        return s
    def _get_resamples(self, n_boot, min_unique, silent):
        rng = np.random.default_rng(self.random_state)
        # Do this in 2 steps. First, for each resample, generate an array indexing
        # the participants to be included in the resample. I.e. get a list of
        # resample-wise participant indices
        resamplewise_ptpts = []
        for boot_n in tqdm(range(n_boot), desc='Getting participant resamples', disable=silent):
            # TODO: between condition (actually quite simple; see _get_resamples above)
            ptpt_resample = rng.choice(self.participant_labels_.codes,
                                       len(self.participant_labels_))
            resamplewise_ptpts.append(ptpt_resample)
        # Next, for each participant, pre-generate resampled trials. Do this for
        # as many times as the participant will be resampled. The reason we're
        # doing this in 2 steps is because _get_resamples resets the RNG so we can
        # only call it once per participant
        ptpt_ids, idwise_counts = np.unique(np.concat(resamplewise_ptpts), return_counts=True)
        ptpt_counts = np.array([0]*len(np.unique(self.participant_labels_)))
        for ptpt_id, idwise_count in zip(ptpt_ids, idwise_counts):
            ptpt_counts[ptpt_id] = idwise_count # This may seem roundabout but its to ensure that ptpt_counts[ptpt_id] yields something for each possible ptpt_id
        ptptwise_trial_resamples = []
        for ptpt_idx in tqdm(range(len(ptpt_counts)), desc='Getting trial resamples', disable=silent):
            n_ptpt_boot = ptpt_counts[ptpt_idx]
            if n_ptpt_boot > 0:
                model = self.models_[ptpt_idx]
                ptpt_boots = model._get_resamples(n_boot=n_ptpt_boot,
                                                  min_unique=3, # 2 would yield a correlation of +/- 1, which gives infinite atanh
                                                  silent=True)
                ptpt_boots = iter(ptpt_boots) # When we're looping over resamples, we'll call next() to get a unique trial resample
                ptptwise_trial_resamples.append(ptpt_boots)
            else:
                ptptwise_trial_resamples.append(iter([])) # This only occurs if a participant NEVER appears in any resample, which should only happen when way too few resamples are being done (e.g. in dev work)
        # Next, combine them
        final_resample_idx = []
        for boot_n in range(n_boot): # No need for tqdm because this should be ultra quick---it's just re-organizing data
            # Which participants were resampled on this iteration?
            ptpt_idxs = resamplewise_ptpts[boot_n]
            trial_idxs = []
            # For each resampled participant, get a trial indexer specific to this iteration
            for ptpt_idx in ptpt_idxs:
                trial_resamples = ptptwise_trial_resamples[ptpt_idx]
                trial_idxs.append(next(trial_resamples))
            # Final result that specifies the current resample is a set of participant
            # identifiers (used for accessing a participant's data) and a set of
            # corresponding trial indices (used for accessing that paticipant's)
            # resampled trials
            final_resample_idx.append((ptpt_idxs, trial_idxs))
        return final_resample_idx
    def _single_resample(self, resample_idx, alignment_method):
        # Compute stacked cormats within ptpts and then average
        ptpt_Rs = []
        ptpt_idxs, ptptwise_trial_idx = resample_idx
        for ptpt_idx, trial_idx in zip(ptpt_idxs, ptptwise_trial_idx):
            model = self.models_[ptpt_idx]
            R = utils.get_stacked_cormats(
                model.data_[trial_idx],
                model.covariates_[trial_idx],
                model.stratifier_[trial_idx])
            R = np.arctanh(R)
            ptpt_Rs.append(R)
        # Decompose avg cormat
        mean_R = np.average(ptpt_Rs,
                            axis=0,
                            weights=self.weights_[ptpt_idxs]) # !!! Note that resample needs to be applied to weights
        mean_R = np.tanh(mean_R)
        u, s, v = self._svd(mean_R)
        resampled_data_sals = self._align(u, s, v, method=alignment_method)
        # Compute boot_stat within ptpts and then average
        ptpt_boot_stats = []
        for ptpt_idx, trial_idx in zip(ptpt_idxs, ptptwise_trial_idx):
            model = self.models_[ptpt_idx]
            scores = model.transform(model.data_[trial_idx])
            if self.boot_stat == 'score-covariate-corr':
                stat = utils.get_stacked_cormats(
                    scores,
                    model.covariates_[trial_idx],
                    model.stratifier_[trial_idx])
                stat = np.atanh(stat) # z-transform since this is a correlation
            elif self.boot_stat == 'condwise-scores':
                stat = utils.get_groupwise_means(
                    scores,
                    model.stratifier_[trial_idx])
            ptpt_boot_stats.append(stat)
        boot_stat = np.average(ptpt_boot_stats,
                               axis=0,
                               weights=self.weights_[ptpt_idxs])
        if self.boot_stat == 'score-covariate-corr':
            boot_stat = np.tanh(boot_stat) # back-transform from fisher z
        return boot_stat, resampled_data_sals
