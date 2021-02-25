from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import tensorflow as tf
import audioFeatures
import datetime
import pandas as pd


def predict_tracks(logger, features, high_level_features_names, predict_dataset):
    """
    This function receives as input a dataset of music tracks, extracts the low level features by calling the
    audioFeatures.compute_dataset_features() method and feeds them to a trained MLP model to predict the high level
    features.As output it returns a pandas dataframe which stores the track Id, HLF, Entropy, Tempo, HasBeat (boolean),
    LUFS for every track in the input dataset.
    :param logger: the logger instance used for writing logs to an external log file
    :param features: the list of low level features to be extracted
    :param high_level_features_names: the list of names of high level features
    :param predict_dataset: the dataset of tracks to be analysed
    :return: returns a pandas
    """
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Prediction...')
    # Extract the audio features for the Prediction tracks
    X_features_predict, track_names, has_beat, tempo, lufs = audioFeatures.compute_dataset_features(logger,
                                                                                                    predict_dataset,
                                                                                                    features)

    X_features_predict = MinMaxScaler().fit_transform(X_features_predict)
    mlp_model = tf.keras.models.load_model('checkpoints/mlp_model')
    preds = mlp_model.predict(X_features_predict)

    # high_level_features = np.log(1 + preds)
    # high_level_features = normalize(preds, axis=1)
    high_level_features = preds

    high_level_features = pd.DataFrame(data=high_level_features, columns=high_level_features_names)
    high_level_features['Id'] = track_names
    high_level_features['Entropy'] = has_beat
    high_level_features['Tempo'] = tempo
    high_level_features['HasBeat'] = high_level_features.apply(lambda row: audioFeatures.check_beat(row), axis=1)
    high_level_features['LUFS'] = lufs

    return high_level_features
