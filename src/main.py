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

# Feature Extraction Config
featureExtraction = False
features = ['mSRO', 'mLOUD', 'mBW', 'mSFL', 'vSRO', 'vLOUD', 'vBW', 'vSFL', 'pSRO', 'pLOUD', 'pBW', 'pSFL']
mfccs = False
if mfccs:
    features.append(
        ['MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
         'MFCC13'])

# Training Type
gmmTraining = False
mlpTraining = True

# PCA and GMM Configuration
# Set pca_components=0 to search the number of components
configuration = True
pca_components = 3

prediction = True
gmm_prediction = False

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
            lowLevelFeatures, _ = audioFeatures.compute_dataset_features(d, mfccs, features, logger)
            savetxt('lowLevelFeatures/X_{}.csv'.format(d), lowLevelFeatures, delimiter=',')
            # lowLevelFeatures.to_csv('lowLevelFeatures/X_{}.csv'.format(d))
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Ended low level feature extraction for'
                                                                         ' training...')
    # Task 2 - GMM Training using the updated data
    # a) Read the low level features and standardize the values.
    # b) Apply PCA to reduce the number of features
    # b) Fit the GGM using the reduced data
    # c) Store the GMM models to csv file, to be used in prediction
    # for prediction

    if gmmTraining:
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started GMM Training...')

        for index, d in enumerate(datasets):

            X_features_training = loadtxt('lowLevelFeatures/X_{}.csv'.format(d), delimiter=',')
            # X_features_training = pd.read_csv('lowLevelFeatures/X_{}.csv'.format(d), header=0, index_col=0)

            ####################################################
            # preprocessing.pca_analysis(X_features_training)
            # X_features_training_reduced_variance = preprocessing.reduced_variance_selection(X_features_training, logger)
            ####################################################

            # CONFIGURATION step. Here it can be decided which are the low-level
            # features to use to train the GMM. For example, we may consider all the 12 descriptors related to the summary
            # vector or only a subset

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
            gmm_model = train.train_gmm(Y_features_training, logger)
            joblib.dump(gmm_model, 'models/gmm_{}.sav'.format(d))
            logger.info(
                str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Saved ggm_{} model'.format(d))

    if mlpTraining:
        # Build X,y for train-test
        X, y = preprocessing.build_x_y(datasets, logger)
        # preprocessing.wrapped_svm_method(X_train, X_test, y_train, y_test)
        # X_features_training_univariance = preprocessing.univariate_selection(X, y, logger)
        # Sample 3 training sets while holding out 10%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

        print('X_train shape', X_train.shape)
        print('y_train.shape', y_train.shape)

        # Normalize Audio

        X_train = MinMaxScaler().fit_transform(X_train)
        X_test = MinMaxScaler().fit_transform(X_test)

        # Create Training Dataset object
        # ------------------------------
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        # Shuffle
        train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0])

        train_dataset = train_dataset.map(preprocessing.to_cast)

        train_dataset = train_dataset.map(preprocessing.to_categorical)

        # iterator = iter(train_dataset)
        # sample, target = next(iterator)
        # print(target)

        # Divide in batches
        bs = 32
        train_dataset = train_dataset.batch(bs)

        # Repeat
        # Without calling the repeat function the dataset
        # will be empty after consuming all the images
        train_dataset = train_dataset.repeat()

        # Create Test Dataset
        # -------------------
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        test_dataset = test_dataset.map(preprocessing.to_cast)

        test_dataset = test_dataset.map(preprocessing.to_categorical)

        test_dataset = test_dataset.batch(1)

        test_dataset = test_dataset.repeat()

        mlp_model = train.create_model()

        metrics = ['accuracy']
        loss = tf.keras.losses.CategoricalCrossentropy()
        # Setting the initial Learning Rate:
        lr = 0.1
        # Setting the Optimizer to be used:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        mlp_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        callbacks = []

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('checkpoints', 'mlp_model'),
            save_best_only=True,
            save_weights_only=False)  # False to save the model directly
        callbacks.append(ckpt_callback)

        early_stop = True
        if early_stop:
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=50,
                                                           restore_best_weights=True)
            callbacks.append(es_callback)

        history = mlp_model.fit(x=train_dataset, y=None,
                                steps_per_epoch=int(np.ceil(X_train.shape[0] / bs)),
                                validation_data=test_dataset,
                                validation_steps=19,
                                epochs=1000,
                                callbacks=callbacks)


    if prediction:
        # Task 3 - Predict the Darkness, Dynamicity, Classicity high level features for the new tracks
        # Generating a high-level feature means to properly train the related Gaussian Mixture Model, exploiting audio
        # signals strictly related to the meaning of the descriptor. For each feature, the generation process is composed
        # of two steps, namely configuration and training:
        # TRAINING step. Once the subset of low-level features has been selected, the training phase allows to build the GMM exploiting
        # a set of audio signals that show characteristics belonging to the semantic meaning of the current high-level
        # feature.
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Prediction ...')
        # Extract the audio features for the Prediction tracks
        X_features_predict, track_names = audioFeatures.compute_dataset_features('Predict', mfccs, features, logger)
        # Create the final high level features matrix
        n_files = X_features_predict.shape[0]
        high_level_features = np.zeros((n_files, 3))

        if gmm_prediction:
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
                        _, pca_dataset_n_components = preprocessing.pca_components(X_features_training_scaled, d, logger)
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
        else:

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
