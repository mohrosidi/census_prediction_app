import os
import json
from fastapi.testclient import TestClient
from web.app import app

client = TestClient(app)

CWD = os.getcwd()

CSV_SAMPLE_PATH = os.path.join(CWD,"tests", 'sample.csv')
JSON_SAMPLE_PATH = os.path.join(CWD,"tests", 'sample.json')


def test_get_root():
    r = client.get('http://localhost:5000/')
    get_content = r.content.decode('utf-8').strip('"')

    assert r.status_code == 200
    assert get_content == 'Hello from Census Predictor!'


def test_inference_csv():
    csv_file = {'csv_file': open(CSV_SAMPLE_PATH, 'rb')}
    r = client.post('http://localhost:5000/batch_inference', files=csv_file)

    assert r.status_code == 200
    assert r.json()['success']
    assert r.json()['error'] is None

def test_inference_json():
    headers = {'Content-Type': 'application/json'}
    r = client.post('http://localhost:5000/inference',
                    json=json.load(open(JSON_SAMPLE_PATH, 'r')),
                    headers=headers)

    assert r.status_code == 200
    assert r.json()['success']
    assert r.json()['error'] is None
