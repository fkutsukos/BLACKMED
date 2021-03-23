import json
import requests
import datetime
import urllib.parse
import urllib.request
import zulu


def get_tracks(logger):
    """
    This function is used to download tracks from the Sanity archive based on a query.
    The tracks are stored to a local directory.
    :param logger: the logger instance used for writing logs to an external log file
    """
    dt = zulu.now()


    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Querying Sanity for new tracks...')
    query = '*[_type == "track" && !(_id in path("drafts.**")) && _updatedAt < \'' + dt.shift(hours=-1).isoformat() + '\'  ]{ ..., "file": file.asset->{url, mimeType} }[0...50]'
    # query = '*[_type == "track" && !(_id in path("drafts.**")) && algorithm.hasBeat == false]{ ..., "file": file.asset->{url, mimeType} }'
    # query = '*[_type == "track" && !(_id in path("drafts.**")) && algorithm.lufs]{ ..., "file": file.asset->{url, mimeType} }'
    # query = '*[_type == "track" && !(_id in path("drafts.**")) && !defined(algorithm.lufs)]{ ..., "file": file.asset->{url, mimeType} }'
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' with query: ' + query)
    query = urllib.parse.quote(query)
    url = "https://t0ciza9b.api.sanity.io/v1/data/query/production?query=" + query

    with open("config/sanity.txt", "r") as file:
        key = file.readlines()

    payload = {}
    headers = {
        'Authorization': key[0]
    }
    response = requests.request("GET", url, headers=headers, data=payload)

    json_data = json.loads(response.text)

    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Number of tracks to update {}...'.format(
        len(json_data['result'])))
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Downloading...')
    n_tracks = len(json_data['result'])
    for index, track in enumerate(json_data['result']):
        urllib.request.urlretrieve(track['file']['url'],
                                   'data/Predict/{}.{}'.format(track['_id'], track['file']['url'].split('.')[-1]))
        if (index+1) % 10 == 0 and index != 0:
            logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Downloaded ' + f'{index+1} out of ' + f'{n_tracks} tracks')

    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Downloading tracks complete...')
