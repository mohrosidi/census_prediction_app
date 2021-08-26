"""
Model artifact test module
"""
import os
from joblib import load

import yaml
import pytest
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import load_artifact, compute_slice_metrics, \
    inference, RFClassifier

CWD = os.getcwd()

# Loads config
config_path = os.path.join(CWD, 'starter', 'params.yaml')
with open(config_path, 'r') as fp:
    CONFIG = yaml.safe_load(fp)

MODEL_PATH = os.path.join(
    CWD,
    'model',
    CONFIG['model_output'])
BINARIZER_PATH = os.path.join(
    CWD,
    'model',
    CONFIG['label_binarizer_output'])
ENCODER_PATH = os.path.join(
    CWD,
    'model',
    CONFIG['encoder_output'])
DATA_PATH = os.path.join(
    CWD,
    'data',
    CONFIG['data'])

@pytest.fixture
def random_forest():
    return load(MODEL_PATH)


@pytest.fixture
def binarizer():
    return load(BINARIZER_PATH)


@pytest.fixture
def encoder():
    return load(ENCODER_PATH)


@pytest.fixture(scope='function')
def df():
    import pandas as pd

    dataframe = pd.read_csv(DATA_PATH)
    return dataframe


def test_rf_model_attributes(random_forest):
    model = RFClassifier(model=random_forest)
    assert hasattr(
        model,
        'model') and isinstance(
        model.model,
        RandomForestClassifier)
    assert hasattr(
        model,
        'binarizer') and isinstance(
        model.binarizer,
        LabelBinarizer)
    assert hasattr(
        model,
        'encoder') and isinstance(
        model.encoder,
        OneHotEncoder)


def test_load_model():
    assert isinstance(load_artifact(MODEL_PATH), RandomForestClassifier)

def test_load_binarizer():
    assert isinstance(load_artifact(BINARIZER_PATH), LabelBinarizer)

def test_load_encoder():
    assert isinstance(load_artifact(ENCODER_PATH), OneHotEncoder)

def test_compute_slice_metrics(df):
    SLICES = CONFIG['categorical_features']
    for elem in SLICES:
        predictions = compute_slice_metrics(df, elem)
        for feature, metrics in predictions.items():
            assert isinstance(feature, str)
            assert isinstance(metrics['precision'], float)
            assert isinstance(metrics['recall'], float)
            assert isinstance(metrics['fbeta'], float)


def test_process_data(encoder, binarizer, df):
    X, y, _, _ = process_data(df,
                              categorical_features=CONFIG['categorical_features'],
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)
    assert isinstance(X, np.ndarray)
    assert len(X) > 0
    assert isinstance(y, np.ndarray)

def test_encoder_artifact(encoder, binarizer, df):
    _, _, encoder, lb = process_data(df,
                              categorical_features=CONFIG['categorical_features'],
                              label='salary', training=True)
    assert isinstance(lb, LabelBinarizer)
    assert isinstance(encoder, OneHotEncoder)

def test_inference(random_forest, encoder, binarizer, df):
    X, y, _, _ = process_data(df,
                              categorical_features=CONFIG['categorical_features'],
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)

    preds = inference(random_forest, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) > 0