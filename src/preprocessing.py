from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from numpy import loadtxt
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

plt.style.use('seaborn')


def to_cast_regression(x_, y_):
    """
    This functions casts the X and y data to tensorflow data types
    :param x_: X data
    :param y_: y data
    :return: the casted data
    """
    return tf.cast(x_, tf.float32), tf.cast(y_, tf.float32)


def to_cast_classification(x_, y_):
    """
    This functions casts the X and y data to tensorflow data types
    :param x_: the X data
    :param y_: the y data
    :return: the casted data
    """
    return tf.cast(x_, tf.float32), tf.cast(y_, tf.uint8)


# 1-hot encoding <- for categorical cross entropy
def to_categorical(x_, y_):
    """
    Converts y data to binary class matrix.
    :param x_: the X data are not modified
    :param y_: the y data that are converted to one hot encoding
    :return: the converted x,y
    """
    return x_, tf.one_hot(y_, depth=3)


def build_x_y_regression(datasets, logger):
    """
    This function builds the X, y numpy arrays in the format expected for training the model.
    The y is loaded from user annotated .csv file
    :param datasets: the X dataset for training
    :param logger: the logger entity
    :return: returns the X, y numpy arrays
    """
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Create X,y for Regression...')

    # Building the X
    X = np.array([[]])
    for index, d in enumerate(datasets):
        X_dataset = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')
        if index == 0:
            X = X_dataset
        else:
            X = np.concatenate((X, X_dataset), axis=0)

    # Building the y
    df = pd.read_excel('highLevelFeatures/y.xlsx', index_col=None, usecols="B:D", engine='openpyxl')
    y = df.dropna().to_numpy()
    return X, y


def build_x_y_classification(datasets, logger):
    """
    This function builds the X, y numpy arrays in the format expected for training the model.
    :param datasets: the X dataset for training
    :param logger: the logger entity
    :return: returns the X, y numpy arrays
    """
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Create X,y for Classification... ')
    X = np.array([[]])
    y = np.array([[]])
    for index, d in enumerate(datasets):
        X_dataset = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')

        # Create the X and y ground truth vectors
        if index == 0:
            X = X_dataset
            y = np.zeros((X_dataset.shape[0],))
        else:
            X = np.concatenate((X, X_dataset), axis=0)
            y = np.concatenate((y, (np.ones((X_dataset.shape[0],)) * index)), axis=0)

    return X, y


def wrapped_svm_method(X_train, X_test, y_train, y_test, logger):
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Wrapped method... ')
    pipeline = Pipeline([
        ('select', SelectKBest(score_func=f_classif)),
        ('clf', SVC())]
    )

    param_grid = {'select__k': range(1, 12),
                  'clf__C': [0.001, 0.1, 1, 2, 5, 10, 100],
                  'clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1.0],
                  'clf__kernel': ['rbf']}

    grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=10, n_jobs=1)
    grid_search.fit(X_train, y_train)
    print(grid_search.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(grid_search.best_params_)

    y_true, y_pred = y_test, grid_search.predict(X_test)
    print(classification_report(y_true, y_pred))