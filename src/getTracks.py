import json
import requests
import datetime
import urllib.parse

query = '*[_type == "track" && (!defined(algorithm.darkness))] { ..., "file": file.asset->{url, mimeType} }'
query = urllib.parse.quote(query)

url = "https://t0ciza9b.api.sanity.io/v1/data/query/production?query=" + query

with open("../data/sanity.txt", "r") as file:
    key = file.readlines()

payload = {}
headers = {
    'Authorization': key[0]
}

response = requests.request("GET", url, headers=headers, data=payload)

json_data = json.loads(response.text)

dt = datetime.datetime.today()

json_path = "../data/sanity/sanity_response_{}{}{}.json".format(dt.year, dt.month, dt.day)
with open(json_path, 'w') as fp:
    json.dump(json_data, fp, indent=4)
