from sklearn import mixture
import os
import numpy as np
import datetime
import tensorflow as tf


def create_model(input_shape, regression=False):
    """
    This function creates a sequential model for the MLP model. It can be either:
     - a regression model with linear output neurons and MSE error function or
     - a classification model with softmax output neurons and CategoricalCrossEntropy error function
    depending on the user choice.
    :param input_shape: the input tensor shape
    :param regression: this variable defines if the model will be
    :return: model: the actual MLP model to be compiled
    """
    model = tf.keras.models.Sequential()
    SEED = 1234
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(units=128,
                                    activation='relu',
                                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

    # Regression settings
    if regression:
        model.add(tf.keras.layers.Dense(units=3,
                                        activation='linear',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001)))

        metrics = [tf.keras.metrics.MeanSquaredError()]
        loss = tf.keras.losses.MeanSquaredError()

    # Classification settings
    else:
        model.add(tf.keras.layers.Dense(units=3,
                                        activation='softmax',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
        metrics = [tf.keras.metrics.CategoricalAccuracy()]
        loss = tf.keras.losses.CategoricalCrossentropy()

    # Global Settings
    lr = 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_mlp(mlp_model, train_dataset, valid_dataset, steps_per_epoch, validation_steps, logger, regression=False):
    """

    :param mlp_model: the MLP sequential model
    :param train_dataset: the dataset that will be used for training
    :param valid_dataset: the dataset that will be used for validation
    :param steps_per_epoch: is the number of batches in the training data
    :param validation_steps: is the number of valid_x, valid_y shape[0]
    :param logger: the logger entity
    :param regression: if true then the model will be stored in the mlp_model_regression checkpoint
    :return:
    """
    callbacks = []
    if regression:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('checkpoints', 'mlp_model_regression'),
            save_best_only=True,
            save_weights_only=False)  # False to save the model directly
    else:
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('checkpoints', 'mlp_model_class'),
            save_best_only=True,
            save_weights_only=False)  # False to save the model directly
    callbacks.append(ckpt_callback)

    early_stop = True
    if early_stop:
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=20,
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