from sklearn import mixture
import numpy as np


def train_gmm(Y_features):
    bic = []
    lowest_bic = np.infty
    n_init = 100
    max_iter = 100
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          n_init=n_init,
                                          covariance_type=cv_type,
                                          random_state=2)
            gmm.fit(Y_features)
            bic.append(gmm.bic(Y_features))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm


"""
    # Fit Gaussian Mixture Models with features
    covariance_type = 'full'  # whiten = True in PCA --> ?
    n_init = 100
    max_iter = 100
    # define the GMM
    gmm = mixture.GaussianMixture(n_components=n_components,
                                  n_init=n_init,
                                  covariance_type=covariance_type,
                                  max_iter=max_iter,
                                  random_state=2)
    # train the GMM using the features
    gmm.fit(Y_features)
    return gmm
"""
