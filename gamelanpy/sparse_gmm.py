import numpy as np
import scipy
import scipy.linalg
import scipy.cluster
from sklearn.mixture import GMM
from coreset import get_coreset
import nonparanormal

from sklearn.covariance import graph_lasso
from weighted_kmeans import WeightedKMeans


def _learn_sparse_gaussian(data, l1_penalty_range, l1_search_depth, l1_search_repeat, atol):
    '''
    Learn single sparse Gaussian Graphical Model using graphical lasso
    L1-parameter search is done by heuristic method using column shuffling

    :param data: array of shape(n_samples, n_dimensions)
    :param l1_penalty_range: min and max values for l1_penalty search
    :param l1_search_depth: number of depth in binary search for l1_penalty
    :param l1_search_repeat: number of trials for l1_penalty search
    :param atol: absolute tolerance for recovering zero or non-zero entries in precision matrix
    :return: covarinace matrix, precision matrix, final l1_penalty
    '''
    n_frames, n_vars = data.shape

    # copy data for column shuffling
    data_s = data.copy()
    l1_penalty_sum = 0.
    for i in range(l1_search_repeat):
        l1_min, l1_max = l1_penalty_range
        l1_mid = (l1_min + l1_max) / 2
        for j in range(l1_search_depth):
            # shuffle columns
            for p in range(n_vars):
                data_s[:, p] = data_s[np.random.permutation(n_frames), p]
            cov, pre = graph_lasso(np.cov(data_s.T), l1_mid)

            # conservative binary search on l1_param
            if ((np.isclose(pre, 0, atol=atol) == False) == np.eye(n_vars)).all():
                l1_max = (l1_max + l1_mid) / 2
            else:
                l1_min = (l1_min + l1_mid) / 2
            l1_mid = (l1_min + l1_max) / 2
        l1_penalty_sum += l1_mid

    l1_penalty = l1_penalty_sum / l1_search_repeat

    # final graphical lasso on the original data
    cov, pre = graph_lasso(np.cov(data.T), l1_penalty)

    pre[np.abs(pre) < atol] = 0.
    cov = np.linalg.inv(pre)
    return cov, pre, l1_penalty


class SparseGMM(GMM):
    def __init__(self, n_components=1, nonparanormal=False, random_state=None, thresh=None, min_covar=0.001,
                 n_iter=100, n_init = 1, params='wmc', init_params='wmc', tol=1e-4):
        GMM.__init__(self, n_components=n_components, covariance_type='full', random_state=random_state,
                     thresh=thresh, min_covar=min_covar, n_iter=n_iter, n_init=n_init, params=params,
                     init_params=init_params)
        self.nonparanormal = nonparanormal
        self.tol=tol
        self.l1_penalty_ = None

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the greater the better)
        """
        n_vars = self.means_.shape[1]
        score = self.score(X).sum()
        df = 0
        for k in range(self.n_components):
            df += (np.abs(scipy.triu(self.precs_[k])) > self.tol).sum()
            df += n_vars

        return score - df * np.log(X.shape[0]) / 2

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float (the greater the better)
        """
        n_vars = self.means_.shape[1]
        score = self.score(X).sum()
        df = 0
        for k in range(self.n_components):
            df += (np.abs(scipy.triu(self.precs_[k])) > self.tol).sum()
            df += n_vars

        return score - df

    def fit(self, X, subsample_method, subsample_size,
            l1_penalty_range=[0.0001, 10.0], l1_search_depth=20, l1_search_repeat=10, npn_sample_ratio=None):
        n_frames, n_vars = X.shape

        samples = None
        sample_idx = None

        if subsample_method == 'coreset':
            coreset, weights = get_coreset(X, self.n_components, subsample_size)
            kmeans = WeightedKMeans(n_clusters=self.n_components)
            kmeans.fit(coreset, weights=weights)
            X_idx = kmeans.predict(X)

            samples = X
            sample_idx = X_idx
        elif subsample_method == 'coreset2':
            coreset, weights = get_coreset(X, self.n_components, subsample_size)
            kmeans = WeightedKMeans(n_clusters=self.n_components)
            kmeans.fit(coreset, weights=weights)
            coreset_idx = kmeans.predict(coreset)

            samples = coreset
            sample_idx = coreset_idx

        elif subsample_method == 'uniform':
            subsamples = X[np.random.permutation(n_frames)[:subsample_size], :]
            kmeans = WeightedKMeans(n_clusters=self.n_components)
            kmeans.fit(subsamples)

            samples = X
            sample_idx = kmeans.predict(X)
        elif subsample_method == 'None':
            kmeans = WeightedKMeans(n_clusters=self.n_components)
            kmeans.fit(X)
            samples = X
            sample_idx = kmeans.predict(X)
        else:
            raise ValueError('Method %s is not supported' % subsample_method)

        self.covars_ = np.empty((self.n_components, n_vars, n_vars), dtype=float)
        self.precs_ = np.empty((self.n_components, n_vars, n_vars), dtype=float)
        self.means_ = np.empty((self.n_components, n_vars), dtype=float)
        self.weights_ = np.empty(self.n_components, dtype=float)
        self.l1_penalty_ = np.empty(self.n_components, dtype=float)
        if self.nonparanormal:
            self.npns_ = [None] * self.n_components

        for k in range(self.n_components):
            data_k = samples[sample_idx == k, :]

            if self.nonparanormal:
                self.npns_[k] = nonparanormal.NPNTransformer()
                self.npns_[k].set_data(data_k, ratio=npn_sample_ratio)
                data_k = self.npns_[k].transform(data_k)

            self.means_[k] = data_k.mean(axis=0)

            cov, pre, l1_penalty = \
                _learn_sparse_gaussian(X, l1_penalty_range, l1_search_depth, l1_search_repeat, self.tol)
            self.covars_[k] = cov
            self.precs_[k] = pre
            self.l1_penalty_[k] = l1_penalty
            self.weights_[k] = data_k.shape[0]

        self.weights_ /= self.weights_.sum()







