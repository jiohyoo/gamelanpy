# Original Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <larsmans@gmail.com>
#
# Edited to cover weighted K-means algorithm by Ji Oh Yoo <jioh.yoo@gmail.com>
# License: BSD 3 clause


cimport numpy as np
import numpy as np
cimport cython

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _centers_dense(np.ndarray[DOUBLE, ndim=2] X,
        np.ndarray[INT, ndim=1] labels, int n_clusters,
        np.ndarray[DOUBLE, ndim=1] distances,
        np.ndarray[DOUBLE, ndim=1] gamma=None):
    """M step of the K-means EM algorithm

    Computation of cluster centers / means.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)

    labels: array of integers, shape (n_samples)
        Current label assignment

    n_clusters: int
        Number of desired clusters

    distances: array-like, shape (n_samples)
        Distance to closest cluster for each sample.

    gamma: array-like, shape (n_samples), optional
        Weights on each sample

    Returns
    -------
    centers: array, shape (n_clusters, n_features)
        The resulting centers
    """
    ## TODO: add support for CSR input
    cdef int n_samples, n_features
    n_samples = X.shape[0]
    n_features = X.shape[1]
    cdef int i, j, c
    cdef np.ndarray[DOUBLE, ndim=2] centers = np.zeros((n_clusters, n_features))

    if gamma is None:
        n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)
    else:
        n_samples_in_cluster = np.zeros(n_clusters)
        for i in range(n_samples):
            n_samples_in_cluster[labels[i]] += gamma[i]

    empty_clusters = np.where(n_samples_in_cluster == 0)[0]
    # maybe also relocate small clusters?

    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

    for i, cluster_id in enumerate(empty_clusters):
        # XXX two relocated clusters could be close to each other
        new_center = X[far_from_centers[i]]
        centers[cluster_id] = new_center
        if gamma is None:
            n_samples_in_cluster[cluster_id] = 1
        else:
            n_samples_in_cluster[cluster_id] = gamma[far_from_centers[i]]

    if gamma is None:
        for i in range(n_samples):
            centers[labels[i], :] += X[i, :]
        centers /= n_samples_in_cluster[:, np.newaxis]
    else:
        for i in range(n_samples):
            centers[labels[i], :] += X[i, :] * gamma[i]
        centers /= n_samples_in_cluster[:, np.newaxis]

    return centers
