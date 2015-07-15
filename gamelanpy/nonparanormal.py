import numpy as np
from scipy.stats import norm

class NPNTransformer():
    """
    Transformation function for multi-variate non-paranormal data

    self.vals[j][idx] = j-th dimension, val of the sample at index idx when sorted
    self.cnts[j][idx] = j-th dimension, cdf at index idx
    self.mu[j] = sample mean of j-th dimension data
    self.sigma[j] = sample std of j-th dimension data

    self.delta = if not specified, follow Liu (2009) The Nonparanormal paper
    """
    def __init__(self):
        self.is_data_set = False
        self.n_dims = 0
        self.n_samples = 0
        self.cdf = None
        self.vals = None
        self.delta = 0
        self.mu = None
        self.sigma = None
        self.ratio = None

    def set_data(self, data, delta=None, ratio=1.0):
        if self.is_data_set is True:
            raise RuntimeError('Data is already set!')

        if len(data.shape) == 1:
            data = data.reshape((data.size, 1))

        if ratio <= 0 or ratio > 1.0:
            raise ValueError('The sample ratio for NPNTransformer %f must be beween 0 and 1!' % ratio)
        self.ratio = ratio

        n_samples = int(round(data.shape[0] * ratio))
        if n_samples != data.shape[0]:
            data = data[np.random.permutation(data.shape[0])[:n_samples]]
        else:
            data = data.copy()

        self.n_samples, self.n_dims = data.shape

        if delta is None:
            self.delta = 1.0 / (4 * (self.n_samples ** 0.25) * np.sqrt(np.pi * np.log(self.n_samples)))
        else:
            self.delta = delta

        self.cdf = [None] * self.n_dims
        self.vals = [None] * self.n_dims
        for dim in range(self.n_dims):
            sorted_col = np.sort(data[:, dim])
            n_unique = np.unique(data[:, dim]).size
            self.cdf[dim] = np.zeros(n_unique, dtype=float)
            self.vals[dim] = np.zeros(n_unique, dtype=float)

            prev_val = None
            prev_idx = -1
            for i in range(sorted_col.size):
                if sorted_col[i] == prev_val:
                    self.cdf[dim][prev_idx] += 1
                else:
                    self.cdf[dim][prev_idx + 1] = self.cdf[dim][prev_idx] + 1
                    self.vals[dim][prev_idx + 1] = sorted_col[i]
                    prev_val = sorted_col[i]
                    prev_idx += 1
            self.cdf[dim] /= self.n_samples

        self.mu = data.mean(axis=0)
        self.sigma = data.std(axis=0)
        self.is_data_set = True

    def transform(self, X):
        if self.is_data_set is False:
            raise RuntimeError('Data is not set yet!')

        if len(X.shape) == 1:
            Xt = np.zeros((X.size, 1), dtype=float)
            X = X.reshape((X.size, 1))
        else:
            Xt = np.zeros(X.shape, dtype=float)

        n_samples, n_dims = Xt.shape
        if n_dims != self.n_dims:
            raise ValueError('dimension of the given X %d is not equal to the transformation\'s dimension %d' %
                             (n_dims, self.n_dims))

        # get empirical distribution
        for sample_idx in range(n_samples):
            for dim in range(n_dims):
                v = X[sample_idx, dim]
                if np.isnan(v):
                    Xt[sample_idx, dim] = np.nan
                else:
                    if v < self.vals[dim][0]:
                        Xt[sample_idx, dim] = 0.
                    elif v >= self.vals[dim][-1]:
                        Xt[sample_idx, dim] = 1.
                    else:
                        max = self.vals[dim].size - 1
                        min = 0
                        mid = -1
                        while min <= max:
                            mid = (max + min) / 2
                            if self.vals[dim][mid] > v:
                                max = mid - 1
                            elif self.vals[dim][mid] < v:
                                min = mid + 1
                            else:
                                break
                        if self.vals[dim][mid] > v:
                            Xt[sample_idx, dim] = self.cdf[dim][mid - 1]
                        else:
                            Xt[sample_idx, dim] = self.cdf[dim][mid]

        # winsorization
        Xt[np.logical_and(np.isnan(Xt) == False, Xt < self.delta)] = self.delta
        Xt[np.logical_and(np.isnan(Xt) == False, Xt > 1 - self.delta)] = 1 - self.delta

        # apply inverse CDF
        Xt = norm.ppf(Xt)

        # apply mean and std
        Xt = Xt * self.sigma + self.mu
        return Xt

    def inverse_transform(self, Xt):

        if self.is_data_set is False:
            raise RuntimeError('Data is not set yet!')

        X = (Xt - self.mu) / self.sigma
        X = norm.cdf(X)

        n_samples, n_dims = Xt.shape
        if n_dims != self.n_dims:
            raise ValueError('dimension of the given X %d is not equal to the transformation\'s dimension %d' %
                             (n_dims, self.n_dims))

        # get the reverse of the empirical distribution
        for sample_idx in range(n_samples):
            for dim in range(n_dims):
                v = X[sample_idx, dim]
                if not np.isnan(v):
                    # if v < self.vals[dim][0]:
                    #     Xt[sample_idx, dim] = 0.
                    # elif v >= self.vals[dim][-1]:
                    #     Xt[sample_idx, dim] = 1.
                    # else:
                    max = self.cdf[dim].size - 1
                    min = 0
                    mid = -1
                    while min <= max:
                        mid = (max + min) / 2
                        if self.cdf[dim][mid] > v:
                            max = mid - 1
                        elif self.cdf[dim][mid] < v:
                            min = mid + 1
                        else:
                            break
                    if self.cdf[dim][mid] > v:
                        X[sample_idx, dim] = self.vals[dim][mid - 1]
                    else:
                        X[sample_idx, dim] = self.vals[dim][mid]

        return X

