from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import numpy as np


def pca(X_features, n_components):
    # Preprocessing the low level features with PCA - Selecting the best 10 low level features
    n_components = n_components
    model = decomposition.PCA(n_components=n_components, whiten=True)
    model.fit(X_features)
    Y_features = model.transform(X_features)

    return Y_features
