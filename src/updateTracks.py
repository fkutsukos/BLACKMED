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
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Updating Tracks on Sanity...')
    payload = {}
    patches = []

    for track in range(high_level_features.shape[0]):
        patch = {"id": high_level_features["Id"][track].split(".")[0],
                 "set": {"algorithm.darkness": round(high_level_features["Darkness"][track], 2),
                         "algorithm.dynamicity": (round(high_level_features["Dynamicity"][track], 2)),
                         "algorithm.jazzicity": (round(high_level_features["Jazz"][track], 2)),
                         "algorithm.hasBeat": bool(high_level_features["HasBeat"][track] == "True")}}

        patches.append({"patch": patch})
    payload["mutations"] = patches
    payload = json.dumps(payload, default=to_serializable)

    with open("data/sanity.txt", "r") as file:
        key = file.readlines()

    headers = {
        'Authorization': key[0]
    }

    url = "https://t0ciza9b.api.sanity.io/v1/data/mutate/production"
    response = requests.post(url=url, headers=headers, data=payload)
    logger.info(
        str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Sanity response: {}'.format(response.text))

    return response
