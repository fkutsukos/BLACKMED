import json
import datetime
import requests

with open("../data/sanity.txt", "r") as file:
    key = file.readlines()

headers = {
    'Authorization': key[0]
}

url = "https://t0ciza9b.api.sanity.io/v1/data/mutate/production"

payload = '{"mutations":[ { "patch": { "id": "01b3691b-a216-44da-baa6-73293b11b0d2", "set": { "algorithm.darkness": "0.5" } } } ] }'

response = requests.post(url=url, headers=headers, data=payload)

print(response.text)