from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
plt.style.use('seaborn')


def to_cast(x_, y_):
    return tf.cast(x_, tf.float32), tf.cast(y_, tf.uint8)


# 1-hot encoding <- for categorical cross entropy
def to_categorical(x_, y_):
    return x_, tf.one_hot(y_, depth=3)


def build_x_y(datasets, logger):
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Create X,y for Testing Selection ... ')
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


def reduced_variance_selection(X_features, logger):
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Reduced Variance Feature Selection ... ')
    feature_selection_variance_model = VarianceThreshold(threshold=(.9 * (1 - .9)))
    X_selected_features_variance = feature_selection_variance_model.fit_transform(X_features)

    # mask = feature_selection_variance_model.get_support()  # list of booleans
    print("Reduced data set shape = ", X_selected_features_variance.shape)
    # print("     Selected features = ", X_selected_features_variance[mask])
    return X_selected_features_variance


def univariate_selection(X, y, logger):
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Univariate Feature Selection ... ')

    # apply the procedure to take the best k variables based on mutual_info_regression
    feature_selection_univariate_model = SelectKBest(mutual_info_regression, k=3)

    # fit the feature selection model and select the four variables
    X_selected_features_univariate = feature_selection_univariate_model.fit_transform(X, y)

    mask = feature_selection_univariate_model.get_support()  # list of booleans
    print("Reduced data set shape = ", X_selected_features_univariate.shape)
    print("     Selected features = ", mask)
    return X_selected_features_univariate


# def pca(X_features, n_components, logger):
def pca_analysis(X_features):
    # logger.info(
    #    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Applying PCA for {} components '.format(
    #        n_components))

    full_pca_model = PCA()
    X_std = StandardScaler().fit_transform(X_features)
    full_fitted_model = full_pca_model.fit(X_std)

    print("Explained variance ratio: ", full_fitted_model.explained_variance_ratio_)
    print(sum(full_fitted_model.explained_variance_ratio_[:6]))
    plt.plot(full_fitted_model.explained_variance_ratio_, '--o')
    plt.xticks(np.arange(0, 13, 1), labels=np.arange(1, 14, 1))
    plt.xlabel("Feature")
    plt.ylabel("Percentage of explained variance")
    plt.xticks(np.arange(0, 13, 1), labels=np.arange(1, 14, 1))
    plt.yticks(np.arange(0.0, 0.51, .1), labels=["%.0f%%" % (x * 100) for x in np.arange(0.0, 0.51, .1)])
    plt.ylim([0.0, 0.51])
    plt.show()


def pca_components(X_features, dataset, logger):
    # Preprocessing the low level features with PCA - Selecting the best 10 low level features
    n_components = 1
    max_n_components = X_features.shape[1]
    model = PCA(n_components=max_n_components, whiten=True)
    full_fitted_model = model.fit(X_features)
    for n in range(1, max_n_components):
        if (sum(full_fitted_model.explained_variance_ratio_[:n])) > 0.9:
            n_components = n
            break

    model = PCA(n_components=n_components, whiten=True)
    model.fit(X_features)

    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
        ' Applying PCA for {} with {} components describing {} % of variance'.format(dataset, n_components, sum(
            model.explained_variance_ratio_[:n_components]).round(3) * 100))

    Y_features = model.transform(X_features)

    return Y_features, n_components
