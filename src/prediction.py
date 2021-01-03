import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import audioFeatures
import datetime
import pandas as pd


def predict_tracks(logger, features, datasets):

    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Prediction...')
    # Extract the audio features for the Prediction tracks
    X_features_predict, track_names, has_beat, tempo = audioFeatures.compute_dataset_features('Predict',
                                                                                              features, logger)

    X_features_predict = MinMaxScaler().fit_transform(X_features_predict)
    mlp_model = tf.keras.models.load_model('checkpoints/mlp_model')
    preds = mlp_model.predict(X_features_predict)
    # high_level_features = np.log(1 + preds)

    high_level_features = sklearn.preprocessing.normalize(preds, axis=1)

    high_level_features = pd.DataFrame(data=high_level_features, columns=datasets)
    high_level_features['Id'] = track_names
    high_level_features['Entropy'] = has_beat
    high_level_features['Tempo'] = tempo
    high_level_features['HasBeat'] = high_level_features.apply(lambda row: audioFeatures.check_beat(row), axis=1)

    return high_level_features
