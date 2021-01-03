import json
import requests
import datetime
import urllib.parse


def get_tracks(logger):
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Querying Sanity for new tracks...')
    query = '*[_type == "track" && (!defined(algorithm.darkness))][0..1]{ ..., "file": file.asset->{url, mimeType} }'
    query = urllib.parse.quote(query)
    url = "https://t0ciza9b.api.sanity.io/v1/data/query/production?query=" + query

    with open("data/sanity.txt", "r") as file:
        key = file.readlines()

    payload = {}
    headers = {
        'Authorization': key[0]
    }
    response = requests.request("GET", url, headers=headers, data=payload)

    json_data = json.loads(response.text)

    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Number of tracks to update {}...'.format(
        len(json_data['result'])))
    for track in json_data['result']:
        urllib.request.urlretrieve(track['file']['url'],
                                   'data/Predict/{}.{}'.format(track['_id'], track['filename'].split('.')[-1]))
