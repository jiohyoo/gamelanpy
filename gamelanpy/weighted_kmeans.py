# Original Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
#
# Edited to cover weighted K-means algorithm by Ji Oh Yoo <jioh.yoo@gmail.com>
# License: BSD 3 clause

import numpy as np
from numpy import random
from sklearn.metrics.pairwise import euclidean_distances

import _weighted_kmeans


def weighted_kmeans(X, n_clusters, weights=None, n_init=10, max_iter=300, tol=1e-4):
    """Weighted K-means clustering algorithm.
    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    weights : array-like, shape(n_samples,), optional, default=None
        Weights for the given data

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    tol : float, optional
        The relative increment in the results before declaring convergence.


    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).
    """
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    if n_init <= 0:
        raise ValueError("Number of iteration n_init=%d must be bigger than zero." % n_init)
    if n_clusters < 0:
        raise ValueError("Number of clusters n_clusters=%d must be at least 1." % n_clusters)
    if n_clusters > X.shape[0]:
        raise ValueError("Number of clusters n_clusters=%d is larger than number of samples %d"
                         % (n_clusters, X.shape[0]))

    X = np.array(X, dtype=float)
    tol = _tolerance(X, tol)

    # subtract mean of X for more accurate distance computations
    X_mean = X.mean(axis=0)
    X -= X_mean

    # precompute squared norms of data points
    x_squared_norms = np.einsum('ij,ij->i', X, X)

    best_labels, best_inertia, best_centers = None, None, None
    # For a single thread, less memory is needed if we just store one set
    # of the best results (as opposed to one set per run per thread).
    for it in range(n_init):
        # run a k-means once
        labels, inertia, centers, n_iter_ = _weighted_kmeans_single(
            X, n_clusters, weights=weights, max_iter=max_iter, tol=tol,
            x_squared_norms=x_squared_norms)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

    # add mean of X to original distribution
    best_centers += X_mean
    return best_centers, best_labels, best_inertia


def _weighted_kmeans_single(X, n_clusters, x_squared_norms, weights=None, max_iter=300, tol=1e-4):
    """A single run of k-means, assumes preparation completed prior.

    Parameters
    ----------
    X: array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    n_clusters: int
        The number of clusters to form as well as the number of
        centroids to generate.

    weights : array-like, shape(n_samples,), optional, default=None
        Weights for the given data

    max_iter: int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    tol: float, optional
        The relative increment in the results before declaring convergence.

    x_squared_norms: array
        Precomputed x_squared_norms.

    Returns
    -------
    centroid: float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label: integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia: float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _k_init(X, n_clusters, x_squared_norms=x_squared_norms)

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=np.float64)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, x_squared_norms, centers, weights=weights,
                            distances=distances)

        # computation of the means is also called the M-step of EM
        centers = _weighted_kmeans._centers_dense(X, labels, n_clusters, distances, weights)

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        # break if diff is less than tol at this iteration
        diff = centers_old - centers
        if (diff * diff).sum() <= tol:
            break
    return best_labels, best_inertia, best_centers, i + 1


def _labels_inertia_precompute_dense(X, x_squared_norms, centers, distances, weights=None):
    """Compute labels and inertia using a full distance matrix.

    This will overwrite the 'distances' array in-place.

    Parameters
    ----------
    X : numpy array, shape (n_sample, n_features)
        Input data.

    x_squared_norms : numpy array, shape (n_samples,)
        Precomputed squared norms of X.

    centers : numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.

    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.

    weights : (optional) numpy array, shape (n_samples,)
        Weights for each point in X. Default is None

    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.

    inertia : float
        Sum of distances of samples to their closest cluster center.

    """
    n_samples = X.shape[0]
    k = centers.shape[0]
    all_distances = euclidean_distances(centers, X, x_squared_norms,
                                        squared=True)
    labels = np.empty(n_samples, dtype=np.int32)
    labels.fill(-1)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for center_id in range(k):
        dist = all_distances[center_id]
        labels[dist < mindist] = center_id
        mindist = np.minimum(dist, mindist)
    if n_samples == distances.shape[0]:
        # distances will be changed in-place
        distances[:] = mindist
    if weights is None:
        inertia = mindist.sum()
    else:
        inertia = (mindist * weights).sum()
    return labels, inertia


def _labels_inertia(X, x_squared_norms, centers, weights=None, distances=None):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.

    Parameters
    ----------
    X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.

    x_squared_norms: array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers: float64 array, shape (k, n_features)
        The cluster centers.

    weights : array-like, shape(n_samples,), optional, default=None
        Weights for the given data

    distances: float64 array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.

    Returns
    -------
    labels: int array of shape(n)
        The resulting assignment

    inertia : float
        Sum of distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    # distances will be changed in-place
    return _labels_inertia_precompute_dense(X, x_squared_norms, centers, distances, weights=weights)


def _k_init(X, n_clusters, x_squared_norms, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    -----------
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters: integer
        The number of seeds to choose

    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state: numpy.RandomState
        The generator used to initialize the centers.

    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features))

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0], X, Y_norm_squared=x_squared_norms, squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    variances = np.var(X, axis=0)
    return np.mean(variances) * tol


class WeightedKMeans():
    """Weighted K-Means clustering

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.


    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Notes
    ------
    The k-means problem is solved using Lloyd's algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    """

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init

    def _check_fitted(self):
        if not hasattr(self, "cluster_centers_"):
            raise AttributeError("Model has not been trained yet.")

    def fit(self, X, weights=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.cluster_centers_, self.labels_, self.inertia_ = \
            weighted_kmeans(
                X, n_clusters=self.n_clusters, weights=weights,
                n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
        return self

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        X = self._check_fit_data(X)
        return self.fit(X)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        self._check_fitted()
        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        self._check_fitted()
        x_squared_norms = np.einsum('ij,ij->i', X, X)
        return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]

    def score(self, X):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        self._check_fitted()
        x_squared_norms = np.einsum('ij,ij->i', X, X)
        return -_labels_inertia(X, x_squared_norms, self.cluster_centers_)[1]
