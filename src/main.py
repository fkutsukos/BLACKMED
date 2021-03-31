import logging
import datetime
import os
from numpy import savetxt
from colorlog import ColoredFormatter
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import audioFeatures
import getTracks
import train
import preprocessing
import prediction
import updateTracks
import numpy as np
import tensorflow as tf
import joblib

PREDICT_DATASET = 'Predict'

# fix random seed for reproducibility
np.random.seed(7)

# filter out warnings regarding librosa.load for mp3s
warnings.filterwarnings('ignore', '.*PySoundFile failed. Trying audioread instead*', )
warnings.filterwarnings('ignore', '.*TensorFlow*', )
warnings.filterwarnings('ignore', '.*Creating new thread pool*', )

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

datasets = ['Darkness', 'Dynamicity', 'Jazzicity']
features = ['mSRO', 'mLOUD', 'mBW', 'mSFL', 'vSRO', 'vLOUD', 'vBW', 'vSFL', 'pSRO', 'pLOUD', 'pBW', 'pSFL', 'MFCC1',
            'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
            'MFCC13']

# Enable/Disable Modules
featureExtraction = False
training = False
predict = False
regression = True

if __name__ == '__main__':
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting BLCKMD')
    while True:
        logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' \'train\' OR \'predict\' ?')
        try:
            cmd = input('> : ')
        except KeyboardInterrupt:
            logger.info('Shutting down!')
            break
        if cmd == 'predict':
            predict = True
            try:
                # Get tracks from Sanity
                getTracks.get_tracks(logger)

                # Analyze tracks and predict high level features
                predict_root = os.path.join('data', PREDICT_DATASET)
                predict_files = [f for f in os.listdir(predict_root) if f.endswith(('.wav', '.mp3', '.aiff', '.m4a'))]
                if len(predict_files) > 0:
                    high_level_features = prediction.predict_tracks(logger, features, datasets, PREDICT_DATASET,
                                                                    regression)

                    # Update tracks on Sanity
                    response = updateTracks.update_tracks(logger, high_level_features)

                    # Delete downloaded tracks from the local directory
                    if response.status_code == 200:
                        logger.info(
                            str(datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S")) + ' Deleting the downloaded tracks  from local directory...')
                        for file in predict_files:
                            os.remove(os.path.join(predict_root, file))
                    else:
                        logger.error(
                            str(datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S")) + ' Updating Sanity failed , {}'.format(response.text))

                else:
                    logger.info(
                        str(datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S")) + ' No tracks were found on Sanity that match the query specified...')
            except Exception as e:
                logger.error(e)

        elif cmd == 'train':
            try:
                # TASK 1 - Extract LLF
                # 1.1 Iterate over the files of DYNAMICITY, DARKNESS, JAZZICITY datasets
                # 1.2 Extract audio low level features and save as .csv files
                if featureExtraction:
                    # Building Labeled Features matrix for each category and then save features to csv file
                    for index, dataset in enumerate(datasets):
                        lowLevelFeatures, _, _, _, _, _, _ = audioFeatures.compute_dataset_features(logger, dataset,
                                                                                                    features)
                        savetxt('lowLevelFeatures/X_{}.csv'.format(dataset), lowLevelFeatures, delimiter=',')
                    logger.info(
                        str(datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S")) + ' Ended low level feature extraction for'
                                                    ' training...')
                # TASK 2 - MLP Training
                logger.info(
                    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Training...')
                logger.info(
                    str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Regression is set to {}'.format(regression))

                # 2.1 BUILD X,y
                # ------------------------------
                if regression:
                    X, y = preprocessing.build_x_y_regression(datasets, logger)
                else:
                    X, y = preprocessing.build_x_y_classification(datasets, logger)

                # 2.2 NORMALIZE DATA
                # ------------------------------
                X_scaler = MinMaxScaler().fit(X)
                X = X_scaler.transform(X)
                joblib.dump(X_scaler, 'scalers/X_scaler.gz')
                if regression:
                    y_scaler = MinMaxScaler().fit(y)
                    y = y_scaler.transform(y)
                    joblib.dump(y_scaler, 'scalers/y_scaler.gz')

                # 2.3 SPLIT DATA
                # ------------------------------
                # Sample training sets while holding out 20%
                if regression:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

                # 2.4 BUILD DATASETS FOR TRAINING AND TEST
                # ------------------------------
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                # Shuffle
                train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0])
                if regression:
                    # Cast values
                    train_dataset = train_dataset.map(preprocessing.to_cast_regression)
                else:
                    # Cast values
                    train_dataset = train_dataset.map(preprocessing.to_cast_classification)
                    # One-hot-encoding
                    train_dataset = train_dataset.map(preprocessing.to_categorical)
                # Divide in batches
                bs = 32
                train_dataset = train_dataset.batch(bs)
                # Repeat
                train_dataset = train_dataset.repeat()

                # REPEAT FOR VALID
                # ------------------------------
                valid_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

                if regression:
                    valid_dataset = valid_dataset.map(preprocessing.to_cast_regression)
                else:
                    valid_dataset = valid_dataset.map(preprocessing.to_cast_classification)
                    valid_dataset = valid_dataset.map(preprocessing.to_categorical)
                valid_dataset = valid_dataset.batch(1)
                valid_dataset = valid_dataset.repeat()

                # 2.5 CREATE MLP MODEL
                # -------------------
                mlp_model = train.create_model(X_train.shape[1], regression)
                acc_per_fold = []
                loss_per_fold = []

                # 2.6 FIT MLP
                # -------------------
                steps_per_epoch = int(np.ceil(X_train.shape[0] / bs))
                validation_steps = int(X_test.shape[0])
                train.train_mlp(mlp_model, train_dataset, valid_dataset, steps_per_epoch, validation_steps, logger,
                                regression)

            except Exception as e:
                logger.error(e)
