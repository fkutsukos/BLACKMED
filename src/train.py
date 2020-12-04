from sklearn import mixture
import os
import numpy as np
import datetime
import tensorflow as tf


def create_model(input_shape):
    model = tf.keras.models.Sequential()
    SEED = 1234
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(units=128,
                                    activation='relu',
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(tf.keras.layers.Dense(units=3,
                                    activation='softmax',
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    metrics = ['accuracy']
    loss = tf.keras.losses.CategoricalCrossentropy()
    # Setting the initial Learning Rate:
    lr = 0.001
    # Setting the Optimizer to be used:
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_mlp(mlp_model, train_dataset, valid_dataset, steps_per_epoch,validation_steps, logger):
    callbacks = []

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('checkpoints', 'mlp_model'),
        save_best_only=True,
        save_weights_only=False)  # False to save the model directly
    callbacks.append(ckpt_callback)

    early_stop = True
    if early_stop:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=10,
                                                       restore_best_weights=True)
        callbacks.append(es_callback)
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Training MLP...')
    history = mlp_model.fit(x=train_dataset,
                            y=None,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=valid_dataset,
                            validation_steps=validation_steps,
                            epochs=1000,
                            callbacks=callbacks)
    return history


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
        str(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ' GMM covariance type "{}", with {} dimensions'.format(best_gmm.covariance_type,
                                                                                           best_gmm.weights_.shape))

    return best_gmm
