import requests
import datetime
import json
import numpy as np
from functools import singledispatch


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return round(np.float64(val), 2)


def update_tracks(logger, high_level_features):
    """
    This function is updating the algorithm fields for tracks on Sanity archive.
    It is building a dictionary with the high_level_features to be used in JSON post message
    :param logger: the logger instance used for writing logs to an external log file
    :param high_level_features: the pandas dataframe with the data to be updated on sanity
    :return: the response to the JSON post message from sanity
    """
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Updating Tracks on Sanity...')
    payload = {}
    patches = []

    for track in range(high_level_features.shape[0]):
        patch = {"id": high_level_features["Id"][track].split(".")[0],
                 "set": {"algorithm.darkness": round(high_level_features["Darkness"][track], 2),
                         "algorithm.dynamicity": round(high_level_features["Dynamicity"][track], 2),
                         "algorithm.jazzicity": round(high_level_features["Jazzicity"][track], 2),
                         "algorithm.hasBeat": bool(high_level_features["HasBeat"][track]),
                         "algorithm.lufs": round(high_level_features["LUFS"][track], 2)}
                 }

        patches.append({"patch": patch})
    payload["mutations"] = patches
    payload = json.dumps(payload, default=to_serializable)

    with open("config/sanity.txt", "r") as file:
        key = file.readlines()

    headers = {
        'Authorization': key[0]
    }

    url = "https://t0ciza9b.api.sanity.io/v1/data/mutate/production"
    response = requests.post(url=url, headers=headers, data=payload)
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Sanity Response: {}'.format(response.text))

    return response
