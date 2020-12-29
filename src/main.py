import logging
import datetime
import matplotlib.pyplot as plt
import sklearn
from numpy import savetxt
from colorlog import ColoredFormatter
import warnings
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import audioFeatures
import train
import preprocessing
import joblib
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import os

plt.style.use('seaborn')

# fix random seed for reproducibility
np.random.seed(7)

# filter out warnings regarding librosa.load for mp3s
warnings.filterwarnings('ignore', '.*PySoundFile failed. Trying audioread instead*', )
warnings.filterwarnings('ignore', '.*tensorflow*', )

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
logging.basicConfig(filename='logs/blckmd.log')

# TODO: Create function to retrieve and analyse the next track online.

datasets = ['Darkness', 'Dynamicity', 'Jazz']


features = ['mSRO', 'mLOUD', 'mBW', 'mSFL', 'vSRO', 'vLOUD', 'vBW', 'vSFL', 'pSRO', 'pLOUD', 'pBW', 'pSFL', 'MFCC1',
            'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
            'MFCC13']

featureExtraction = False

configuration = True
pca_components = 3

# Configure Training & Prediction (None, GMM , MLP)
# ------------------------------
training = None
prediction = 'MLP'

if training == 'GMM':
    mfccs = False
    features = features[:12]
    gmm_prediction = True

if training == 'MLP':
    mfccs = True

if training is not None:
    featureExtraction = True

if prediction == 'GMM':
    mfccs = False
    features = features[:12]

if prediction == 'MLP':
    mfccs = True


if __name__ == '__main__':
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting BLCKMD')

    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) +
                ' FeatureExtraction: {}, Training: {} , Prediction: {}'.format(featureExtraction, training, prediction))

    # TASK 1 - Low level feature extraction
    # 1.1 Iterate over the files by DYNAMICITY, DARKNESS, JAZZICITY
    # 1.2 Extract audio features and save as .csv files
    if featureExtraction:
        # Building Labeled Features matrix for each category and then save features to csv file
        for index, d in enumerate(datasets):
            lowLevelFeatures, _ = audioFeatures.compute_dataset_features(d, mfccs, features, logger)
            savetxt('lowLevelFeatures/X_{}.csv'.format(d), lowLevelFeatures, delimiter=',')
            # lowLevelFeatures.to_csv('lowLevelFeatures/X_{}.csv'.format(d))
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Ended low level feature extraction for'
                                                                         ' training...')
    if training == 'GMM':
        # TASK 2 - GMM Training
        # 2.1 READ .csv low level features
        # 2.2 STANDARDIZE and PCA to reduce dimensionality
        # 2.3 FIT GGM
        # 2.4 SAVE the trained GMM model

        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started GMM Training...')

        for index, d in enumerate(datasets):

            # 2.1 READ .csv
            # ------------------------------
            X_features_training = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')
            # X_features_training = pd.read_csv('lowLevelFeatures/X_{}.csv'.format(d), header=0, index_col=0)

            ####################################################
            # preprocessing.pca_analysis(X_features_training)
            # X_features_training_reduced_variance = preprocessing.reduced_variance_selection(X_features_training, logger)
            ####################################################

            # CONFIGURATION step. Here it can be decided which are the low-level
            # features to use to train the GMM. For example, we may consider all the 12 descriptors related to the summary
            # vector or only a subset

            # 2.2 STANDARDIZE & PCA
            # ------------------------------
            if configuration:

                X_features_training_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X_features_training)

                if pca_components == 0:
                    # Apply PCA by finding the number of components that describe 90% of variance in the data
                    Y_features_training, _ = preprocessing.pca_components(X_features_training_scaled, d, logger)
                else:
                    # Apply PCA with fixed number of components
                    logger.info(str(datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")) + ' Applying PCA with fixed number of pca_components {} for dataset {}'.format(
                        pca_components, d))
                    pca = sklearn.decomposition.PCA(n_components=pca_components, whiten=True)
                    Y_features_training = pca.fit_transform(X_features_training_scaled)

            logger.info(
                str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Fitting GMM for {} dataset'.format(d))

            # 2.3 FIT GMM
            # ------------------------------
            gmm_model = train.train_gmm(Y_features_training, logger)

            # 2.4 SAVE MODEL
            # ------------------------------
            joblib.dump(gmm_model, 'models/gmm_{}.sav'.format(d))
            logger.info(
                str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Saved ggm_{} model'.format(d))

    if training == 'MLP':
        # TASK 2 - MLP Training
        # 2.1 GET TRAINING DATA AS X,y
        # 2.2 SPLIT DATA to TRAINING and VALIDATION
        # 2.3 NORMALIZE DATA
        # 2.4 BUILD DATASETS FOR TRAINING AND VALIDATION
        # 2.5 CREATE MODEL
        # 2.6 FIT MLP

        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started MLP Training...')

        # 2.1 GET TRAINING DATA AS X,y
        # ------------------------------
        X, y = preprocessing.build_x_y(datasets, logger)
        # preprocessing.wrapped_svm_method(X_train, X_test, y_train, y_test)
        # X_features_training_univariance = preprocessing.univariate_selection(X, y, logger)

        # 2.2 SPLIT DATA
        # ------------------------------
        # Sample 3 training sets while holding out 10%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

        # 2.3 NORMALIZE DATA
        # ------------------------------
        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)

        # 2.4 BUILD DATASETS FOR TRAINING AND TEST
        # ------------------------------
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # Shuffle
        train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0])
        # Cast values
        train_dataset = train_dataset.map(preprocessing.to_cast)
        # One-hot-encoding
        train_dataset = train_dataset.map(preprocessing.to_categorical)
        # Divide in batches
        bs = 32
        train_dataset = train_dataset.batch(bs)
        # Repeat
        train_dataset = train_dataset.repeat()

        # iterator = iter(train_dataset)
        # sample, target = next(iterator)
        # print(target)

        # REPEAT FOR VALID
        # ------------------------------
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        valid_dataset = valid_dataset.map(preprocessing.to_cast)
        valid_dataset = valid_dataset.map(preprocessing.to_categorical)
        valid_dataset = valid_dataset.batch(1)
        valid_dataset = valid_dataset.repeat()

        # 2.5 CREATE MLP MODEL
        # -------------------
        mlp_model = train.create_model(X_train.shape[1])

        acc_per_fold = []
        loss_per_fold = []

        # 2.6 FIT MLP
        # -------------------
        steps_per_epoch = int(np.ceil(X_train.shape[0] / bs))
        validation_steps = int(X_test.shape[0])
        train.train_mlp(mlp_model, train_dataset, valid_dataset, steps_per_epoch, validation_steps, logger)

    if prediction is not None:
        # Task 3 - Predict the Darkness, Dynamicity, Classicity high level features for the new tracks
        # Generating a high-level feature means to properly train the related Gaussian Mixture Model, exploiting audio
        # signals strictly related to the meaning of the descriptor. For each feature, the generation process is composed
        # of two steps, namely configuration and training:
        # TRAINING step. Once the subset of low-level features has been selected, the training phase allows to build the GMM exploiting
        # a set of audio signals that show characteristics belonging to the semantic meaning of the current high-level
        # feature.
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Prediction with {} ...'.format(prediction))
        # Extract the audio features for the Prediction tracks
        X_features_predict, track_names = audioFeatures.compute_dataset_features('Predict', mfccs, features, logger)
        # Create the final high level features matrix
        n_files = X_features_predict.shape[0]
        high_level_features = np.zeros((n_files, 3))

        if prediction == 'GMM':
            # The Prediction data are standardized and reduced using PCA based on training data:
            for index, d in enumerate(datasets):

                # Apply normalization of the Prediction data based on the Training data
                X_features_training = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')
                # X_features_training = pd.read_csv('lowLevelFeatures/X_{}.csv'.format(d), header=0, index_col=0)
                scaler = sklearn.preprocessing.StandardScaler()
                scaler.fit(X_features_training)
                X_features_training_scaled = scaler.transform(X_features_training)
                X_features_predict_scaled = scaler.transform(X_features_predict)

                if configuration:
                    if pca_components == 0:
                        _, pca_dataset_n_components = preprocessing.pca_components(X_features_training_scaled, d,
                                                                                   logger)
                        pca = sklearn.decomposition.PCA(n_components=pca_dataset_n_components, whiten=True)
                    else:
                        # apply PCA on the Prediction based on Training data
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

                gmm_n_components = len(gmm_model.weights_)
                print('gmm_model.n_components for {}: '.format(d), gmm_n_components)
                print('gmm_model.covariance_type for {}: '.format(d), gmm_model.covariance_type)
                print('gmm_model.covariances_.ndim for {}: '.format(d), gmm_model.covariances_.ndim)
                for n in np.arange(gmm_n_components):
                    if gmm_model.covariances_.ndim < 2:
                        gauss = sp.stats.multivariate_normal(gmm_model.means_[n],
                                                             gmm_model.covariances_[n],
                                                             allow_singular=True)

                    else:
                        gauss = sp.stats.multivariate_normal(gmm_model.means_[n, :],
                                                             gmm_model.covariances_[n, :],
                                                             allow_singular=True)
                    pdf.append(gmm_model.weights_[n] * gauss.pdf(Y_features_predict))
                    # print('gauss: ', gauss.pdf(Y_features_predict))

                pdf = np.array(pdf).transpose()

                # pdf = np.array(gmm_model.predict_proba(Y_features_predict))
                pdf_sum = np.sum(pdf, axis=-1)

                # Applying formula to convert probability to High level feature
                high_level_feature = np.log(1 + pdf_sum)
                # feature_predict_proba = np.log(1 + pdf_predict_proba_sum)

                # Reshaping the result as column before inserting into the high level features
                high_level_feature.reshape(-1, 1)

                # print('{} prediction: '.format(d), feature)
                high_level_features[:, index] = high_level_feature

        if prediction == 'MLP':

            X_features_predict = MinMaxScaler().fit_transform(X_features_predict)
            mlp_model = tf.keras.models.load_model('checkpoints/mlp_model')
            preds = mlp_model.predict(X_features_predict)
            high_level_features = np.log(1 + preds)

        high_level_features = sklearn.preprocessing.normalize(high_level_features, axis=1)

        '''
        for r in high_level_features.shape[0]:
            for c in high_level_features.shape[1]-1:
                high_level_features[r,c] = high_level_features[r, c] - high_level_features[r, c+1]
        '''
        high_level_features = pd.DataFrame(data=high_level_features, columns=datasets)
        high_level_features["Track"] = track_names

        print(high_level_features.round(1))
