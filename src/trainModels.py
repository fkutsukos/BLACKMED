from sklearn import mixture


def train_gmm(Y_features,n_components):
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
