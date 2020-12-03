from sklearn import mixture
import numpy as np
import datetime
import tensorflow as tf


def create_model():
    model = tf.keras.models.Sequential()
    SEED = 1234
    model.add(tf.keras.Input(shape=(12,)))
    model.add(tf.keras.layers.Dense(units=10,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=10,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=3,
                                    activation='softmax',
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)))
    return model


def train_gmm(Y_features, logger):
    bic = []
    lowest_bic = np.infty
    n_init = 100
    max_iter = 100
    n_components_range = range(1, 12)
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
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' GMM covariance type "{}", with {} dimensions'.format(best_gmm.covariance_type,
                                                                                                                            best_gmm.weights_.shape))

    return best_gmm



