__author__ = 'jiohyoo'


import numpy as np
import gamelan

data = np.random.rand(10000, 10)

s_gmm = gamelan.SparseGMM(n_components=3)

# learn using coreset subsampling
s_gmm.fit(data, 'coreset', 1000)

# learn using coreset subsampling
s_gmm.fit(data, 'coreset2', 1000)

# learn using naive uniform subsampling
s_gmm.fit(data, 'uniform', 1000)


test = np.random.rand(100, 10)
print 'Test negative log-likelihood:'
print -s_gmm.score(test).sum()
