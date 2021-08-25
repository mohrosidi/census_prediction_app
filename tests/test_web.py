"""
Test Module for Web App

Author : Moh. Rosidi
Date   : August, 2021
"""

import os
import json
from fastapi.testclient import TestClient
from web.app import app

client = TestClient(app)

CWD = os.getcwd()

CSV_SAMPLE_PATH = os.path.join(CWD, "data", 'sample.csv')
JSON_SAMPLE_PATH = os.path.join(CWD, "data", 'sample.json')


def test_root():
    """
    Test root directory
    """
    result = client.get('http://localhost:5000/')
    get_content = result.content.decode('utf-8').strip('"')

    assert result.status_code == 200
    assert get_content == 'Welcome to census predictor app!'


def test_inference_csv():
    """
    Test batch inference path
    """
    csv_file = {'csv_file': open(CSV_SAMPLE_PATH, 'rb')}
    result = client.post('http://localhost:5000/batch_inference', files=csv_file)

    assert result.json()['error'] is None
    assert result.status_code == 200
    assert result.json()['success']

def test_inference_json():
    """
    Test inference path
    """
    headers = {'Content-Type': 'application/json'}
    result = client.post('http://localhost:5000/stream_inference',
                    json=json.load(open(JSON_SAMPLE_PATH, 'r')),
                    headers=headers)

    assert result.json()['error'] is None
    assert result.status_code == 200
    assert result.json()['success']
