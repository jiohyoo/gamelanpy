__author__ = 'jiohyoo'

import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal
from sklearn.mixture import GMM
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from sparse_gmm import SparseGMM


def predict_missing_values(model, data):
    if not isinstance(model, GMM):
        raise ValueError('The given model is not an instance of GMM or SparseGMM')

    predicted_data = data.copy()

    if len(predicted_data.shape) == 1:
        predicted_data = predicted_data[:, np.newaxis]

    n_frames, n_vars = predicted_data.shape
    for idx_frame in range(n_frames):
        row = predicted_data[idx_frame:idx_frame+1]
        idx_missing = np.isnan(row[0])
        c, mu, sigma_none = get_conditional_dist(model, row, skip_covar=True)
        predicted_data[idx_frame][idx_missing] = mu
        if model.nonparanormal:
            predicted_data[idx_frame] = \
                model.npns_[c].inverse_transform(predicted_data[idx_frame:idx_frame+1])

    return predicted_data

def sample_missing_values(model, data, n_samples=10):
    if not isinstance(model, GMM):
        raise ValueError('The given model is not an instance of GMM or SparseGMM')

    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    n_frames, n_vars = data.shape
    predicted_data = np.empty((n_frames * n_samples, n_vars))

    for idx_frame in range(n_frames):
        row = data[idx_frame:idx_frame+1]
        c, mu, sigma = get_conditional_dist(model, row, skip_covar=False)

        idx_missing = np.isnan(row[0])

        for i in range(n_samples):
            if len(mu.shape) == 0:
                mu = mu.reshape((1,))
            sampled_vals = multivariate_normal(mu, sigma)
            predicted_data[idx_frame * n_samples + i] = row
            predicted_data[idx_frame * n_samples + i][idx_missing] = sampled_vals
            if model.nonparanormal:
                predicted_data[idx_frame * n_samples + i] = \
                    model.npns_[c].inverse_transform(
                        predicted_data[idx_frame * n_samples + i : idx_frame * n_samples + i + 1])

    return predicted_data


def get_conditional_dist(model, row, skip_covar=True):
    if not isinstance(model, GMM):
        raise ValueError('The given model is not an instance of GMM or SparseGMM')

    row = np.asarray(row, dtype='float')
    if len(row.shape) == 1:
        row = row[np.newaxis, :]

    n_vars = row.shape[1]

    idx_given = []
    idx_missing = []
    for i in range(n_vars):
        if np.isnan(row[0, i]):
            idx_missing.append(i)
        else:
            idx_given.append(i)

    if len(idx_given) == 0:
        raise ValueError('Row must have at least one given value!')
    if len(idx_missing) == 0:
        raise ValueError('The given row has all values filled already!')

    # prepare parts of the params
    mu_g = model.means_[:, idx_given]
    mu_m = model.means_[:, idx_missing]

    sigma_gg = model.covars_[np.ix_(range(model.n_components), idx_given, idx_given)]
    sigma_mg = model.covars_[np.ix_(range(model.n_components), idx_missing, idx_given)]
    sigma_mm = model.covars_[np.ix_(range(model.n_components), idx_missing, idx_missing)]

    # marginal probability for each component
    if isinstance(model, SparseGMM) and model.nonparanormal:
        marginal_probs = np.empty(model.n_components)
        for c in range(model.n_components):
            row_t = model.npns_[c].transform(row)
            vals_given = row_t[0, np.ix_(idx_given)]
            marginal_probs[c] = np.exp(_log_multivariate_normal_density_full(vals_given, mu_g[c:c+1], sigma_gg[c:c+1]))
    else:
        vals_given = row[0, np.ix_(idx_given)]
        marginal_probs = np.exp(_log_multivariate_normal_density_full(vals_given, mu_g, sigma_gg))

    # weighted marginal probability for each component
    probs = marginal_probs * model.weights_

    # find the most probable component
    k = np.argmax(probs)

    # conditional distribution
    mu_c = mu_m[k] + sigma_mg[k].dot(inv(sigma_gg[k])).dot(vals_given[0] - mu_g[k])[0]
    if skip_covar:
        return k, mu_c, None
    else:
        sigma_c = sigma_mm[k] + sigma_mg[k].dot(inv(sigma_gg[k])).dot(sigma_mg[k].T)
        return k, mu_c, sigma_c

