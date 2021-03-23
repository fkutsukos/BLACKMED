import tensorflow as tf
import audioFeatures
import datetime
import pandas as pd
import joblib


def predict_tracks(logger, features, high_level_features_names, dataset, regression=False):
    """
    This function receives as input a dataset of music tracks, extracts the low level features by calling the
    audioFeatures.compute_dataset_features() method and feeds them to a trained MLP model to predict the high level
    features.As output it returns a pandas dataframe which stores the track Id, HLF, Entropy, Tempo, HasBeat (boolean),
    LUFS for every track in the input dataset.
    :param regression: specifies if the model to be used for prediction is a regression model
    :param logger: the logger instance used for writing logs to an external log file
    :param features: the list of low level features to be extracted
    :param high_level_features_names: the list of names of high level features
    :param dataset: the dataset of tracks to be analysed
    :return: returns a pandas
    """
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Started Prediction...')
    # Extract the audio features for the Prediction tracks
    X_features_predict, track_ids, track_names, entropy_energy_mean, entropy_energy_std, entropy_energy_diff_mean, entropy_energy_diff_std, tempo, lufs = audioFeatures.compute_dataset_features(
        logger,
        dataset,
        features, predict=True)

    # Normalize the data according to training data
    X_scaler = joblib.load('scalers/X_scaler.gz')
    X_features_predict = X_scaler.transform(X_features_predict)
    if regression:
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Regression Prediction...')
        mlp_model = tf.keras.models.load_model('checkpoints/mlp_model_regression')
    else:
        logger.info(
            str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Classification Prediction...')
        mlp_model = tf.keras.models.load_model('checkpoints/mlp_model_class')
    preds = mlp_model.predict(X_features_predict)

    # high_level_features = np.log(1 + preds)
    # high_level_features = normalize(preds, axis=1)
    high_level_features = preds

    high_level_features = pd.DataFrame(data=high_level_features, columns=high_level_features_names)
    high_level_features['Id'] = track_ids
    high_level_features['Name'] = track_names
    high_level_features['Entropy'] = entropy_energy_mean
    high_level_features['Entropy_std'] = entropy_energy_std
    high_level_features['Entropy_diff'] = entropy_energy_diff_mean
    high_level_features['Entropy_diff_std'] = entropy_energy_diff_std
    high_level_features['Tempo'] = tempo
    high_level_features['HasBeat'] = high_level_features.apply(lambda row: audioFeatures.check_beat(row), axis=1)
    high_level_features['LUFS'] = lufs

    return high_level_features
