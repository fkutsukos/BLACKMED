import logging
import datetime
import matplotlib.pyplot as plt
import sklearn
from numpy import savetxt
from colorlog import ColoredFormatter
import warnings
from numpy import loadtxt
import audioFeatures
import trainModels
import joblib
import numpy as np
import scipy as sp
import pandas as pd

plt.style.use('seaborn')

# filter out warnings regarding librosa.load for mp3s
warnings.filterwarnings('ignore', '.*PySoundFile failed. Trying audioread instead*', )

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('colorlogger')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)
logging.basicConfig(filename='blckmd.log')

# TODO: Create function to retrieve and analyse the next track online.

datasets = ['Darkness', 'Dynamicity', 'Jazz']

# Retrain the Classicity, Darkness, Dynamicity datasets

featureExtraction = False
gmmTraining = False

# Predict values for new track
prediction = True

# PCA and GMM Configuration
configuration = True
pca_components = 3
# gmm_components = 3

if __name__ == '__main__':
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting BLCKMD')

    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
                ' FeatureExtraction: {}, GMM: {} , Prediction: {}'.format(featureExtraction, gmmTraining, prediction))

    # Task 1 - Low level feature extraction from the updated datasets to be used for training the GMMs
    # a) We read the tracks based on their tagging,
    # b) We compute the low level audio features and store them in csv files
    if featureExtraction:
        # Building Labeled Features matrix for each category and then save features to csv file
        for index, d in enumerate(datasets):
            lowLevelFeatures = audioFeatures.compute_features_dataset(d, logger)
            savetxt('lowLevelFeatures/X_{}.csv'.format(d), lowLevelFeatures, delimiter=',')
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Ended low level feature extraction for'
                                                                         ' training...')
    # Task 2 - GMM Training using the updated data
    # a) Read the low level features and normalize them using the maximum values of the descriptors.
    # b) Apply PCA to reduce the number of features
    # b) Fit the GGM using the reduced data
    # c) Store the GMM models to csv file, to be used in prediction
    # for prediction
    if gmmTraining:
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started GMM Training...')
        for index, d in enumerate(datasets):
            # Applying normalization
            X_features_training = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')
            X_features_training_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X_features_training)
            # In the configuration step we decide which features to use to train the GMM
            if configuration:
                # Apply PCA to reduce the dimensionality
                pca = sklearn.decomposition.PCA(n_components=pca_components, whiten=True)
                Y_features_training = pca.fit_transform(X_features_training_scaled)
            else:
                Y_features_training = X_features_training_scaled
            gmm_model = trainModels.train_gmm(Y_features_training)
            joblib.dump(gmm_model, 'models/gmm_{}.sav'.format(d))
            logger.info(
                str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Saved ggm_{} model'.format(d))

    if prediction:
        # Task 3 - Predict the Darkness, Dynamicity,Classicity of the new tracks
        # Generating a high-level feature means to properly train the related Gaussian Mixture Model, exploiting audio
        # signals strictly related to the meaning of the descriptor. For each feature, the generation process is composed
        # of two steps, namely configuration and training:

        # CONFIGURATION step (configuration == True). Here it can be decided which are the low-level
        # features to use to train the GMM. For example, we may consider all the 12 descriptors related to the summary
        # vector or only a subset

        # TRAINING step. Once the subset of low-level features has been selected, the training phase allows to build the GMM exploiting
        # a set of audio signals that show characteristics belonging to the semantic meaning of the current high-level
        # feature.

        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Prediction ...')

        # Extract the audio features for the Prediction tracks
        X_features_predict = audioFeatures.compute_features_dataset('Predict', logger)

        # savetxt('lowLevelFeatures/X_Prediction.csv', X_features_predict, delimiter=',')
        # X_features_predict = loadtxt('lowLevelFeatures/X_Prediction.csv', delimiter=',')

        # Create the final high level features matrix
        n_files = X_features_predict.shape[0]
        high_level_features = np.zeros((n_files, 3))

        # The Prediction data are standardized and reduced using PCA based on training data:
        for index, d in enumerate(datasets):
            # Apply normalization of the Prediction based on the Training data
            X_features_training = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(X_features_training)
            X_features_predict_scaled = scaler.transform(X_features_predict)

            if configuration:
                # apply PCA on the Prediction based on Training data
                X_features_training_scaled = scaler.transform(X_features_training)
                pca = sklearn.decomposition.PCA(n_components=pca_components, whiten=True)
                pca.fit(X_features_training_scaled)
                Y_features_predict = pca.transform(X_features_predict_scaled)
            else:
                Y_features_predict = X_features_predict_scaled

            # Load the GMM trained on the specific Dataset Class
            filename = 'models/gmm_{}.sav'.format(d)
            gmm_model = joblib.load(filename)

            # predict using multivariate normal random variables
            # compute the pdf on the Predict data
            pdf = []

            n_components = len(gmm_model.weights_)
            for n in np.arange(n_components):
                gauss = sp.stats.multivariate_normal(gmm_model.means_[n],
                                                     gmm_model.covariances_[n],
                                                     allow_singular=False)
                pdf.append(gmm_model.weights_[n] * gauss.pdf(Y_features_predict))
                # print('gauss: ', gauss.pdf(Y_features_predict))

            pdf = np.sum(pdf, axis=0)
            # print('the sum of gauss', pdf)
            feature = np.log(1 + pdf)

            if n_files > 1:
                # Reshaping the result as column before inserting into the high level features
                feature = feature[:, None]
                # print('{} prediction: '.format(d), feature)
                high_level_features[:, index] = feature[:, 0]
            else:
                high_level_features[:, index] = feature

        high_level_features = sklearn.preprocessing.normalize(high_level_features, axis=1)

        high_level_features = pd.DataFrame(data=high_level_features, columns=datasets)
        print(high_level_features.round(1))
