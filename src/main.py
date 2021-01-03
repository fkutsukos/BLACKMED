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

# TODO: Create function to retrieve and analyse the next track online.

datasets = ['Darkness', 'Dynamicity', 'Jazz']
features = ['mSRO', 'mLOUD', 'mBW', 'mSFL', 'vSRO', 'vLOUD', 'vBW', 'vSFL', 'pSRO', 'pLOUD', 'pBW', 'pSFL', 'MFCC1',
            'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
            'MFCC13']

# Enable/Disable Modules
featureExtraction = False
training = False
predict = False

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
                # Request tracks from Sanity
                getTracks.get_tracks(logger)
                # Analyze tracks and predict high level features
                predict_root = 'data/Predict'
                predict_files = [f for f in os.listdir(predict_root) if f.endswith(('.wav', '.mp3', '.aiff', '.m4a'))]
                if len(predict_files) > 0:
                    high_level_features = prediction.predict_tracks(logger, features, datasets)
                    response = updateTracks.update_tracks(logger, high_level_features)
                    # Delete files from the local directory
                    if response.status_code == 200:
                        logger.info(
                            str(datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S")) + ' Deleting the tracks files from local directory...')
                        for file in predict_files:
                            os.remove(os.path.join(predict_root, file))
                    else:
                        logger.error(
                            str(datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S")) + ' Updating Sanity failed , {}'.format(response.text))
                break
            except Exception as e:
                logger.error(e)
        elif cmd == 'train':
            try:
                # TASK 1 - Low level feature extraction
                # 1.1 Iterate over the files by DYNAMICITY, DARKNESS, JAZZICITY
                # 1.2 Extract audio features and save as .csv files
                if featureExtraction:
                    # Building Labeled Features matrix for each category and then save features to csv file
                    for index, d in enumerate(datasets):
                        lowLevelFeatures, _, _, _ = audioFeatures.compute_dataset_features(d, features, logger)
                        savetxt('lowLevelFeatures/X_{}.csv'.format(d), lowLevelFeatures, delimiter=',')
                        # lowLevelFeatures.to_csv('lowLevelFeatures/X_{}.csv'.format(d))
                    logger.info(
                        str(datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S")) + ' Ended low level feature extraction for'
                                                    ' training...')
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
                break
            except Exception as e:
                logger.error(e)
