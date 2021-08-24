import os
import requests
import json

CWD = os.getcwd()

ENDPOINT = 'https://mlops-census.herokuapp.com/'
CSV_SAMPLE_PATH = os.path.join(CWD, "tests", 'sample.csv')
JSON_SAMPLE_PATH = os.path.join(CWD, "tests", 'sample.json')


def request_get():
    r = requests.get(ENDPOINT)
    print(r.status_code)


def request_json():
    headers = {'Content-Type': 'application/json'}
    r = requests.post(ENDPOINT + 'inference',
                      json=json.load(open(JSON_SAMPLE_PATH, 'r')),
                      headers=headers)
    print(r.status_code)
    print(r.json())


def request_csv():
    csv_file = {'csv_file': open(CSV_SAMPLE_PATH, 'rb')}
    r = requests.post(ENDPOINT + 'batch_inference', files=csv_file)
    print(r.status_code)
    print(r.json())


if __name__ == '__main__':
    request_get()
    request_csv()
    request_json()