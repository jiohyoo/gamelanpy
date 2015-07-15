__author__ = 'jiohyoo'


import numpy as np
import gamelan

data = np.random.rand(10000, 10)
test = np.random.rand(100, 10)

n_components_candidates = range(3, 6)
bic_scores = []
best_score = -np.inf
best_model = None

for n_comp in n_components_candidates:
    s_gmm = gamelan.SparseGMM(n_components=n_comp)

    # learn using coreset subsampling
    s_gmm.fit(data, 'coreset', 1000)
    bic_score = s_gmm.bic(test)
    bic_scores.append(bic_score)

    if bic_score > best_score:
        best_score = bic_score
        best_model = s_gmm

print 'Best model has number of components: %d' % best_model.n_components
print 'BIC scores'
print bic_scores

print 'Test negative log-likelihood:'
print -s_gmm.score(test).sum()
