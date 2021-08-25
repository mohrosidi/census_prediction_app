import os
import requests
import json

CWD = os.getcwd()

ENDPOINT = 'https://mlops-census.herokuapp.com/'
CSV_SAMPLE_PATH = os.path.join(CWD, "data", 'sample.csv')
JSON_SAMPLE_PATH = os.path.join(CWD, "data", 'sample.json')


def root():
    r = requests.get(ENDPOINT)
    print(r.status_code)


def predict_json():
    headers = {'Content-Type': 'application/json'}
    r = requests.post(ENDPOINT + 'stream_inference',
                      json=json.load(open(JSON_SAMPLE_PATH, 'r')),
                      headers=headers)
    print(r.status_code)
    print(r.json())


def predict_csv():
    csv_file = {'csv_file': open(CSV_SAMPLE_PATH, 'rb')}
    r = requests.post(ENDPOINT + 'batch_inference', files=csv_file)
    print(r.status_code)
    print(r.json())


if __name__ == '__main__':
    root()
    predict_json()
    predict_csv()