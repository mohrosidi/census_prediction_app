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
from starter.ml.model import compute_slice_metrics, \
    inference, train_model

CWD = os.getcwd()

# Loads config
config_path = os.path.join(CWD, 'starter', 'params.yaml')
with open(config_path, 'r') as fp:
    CONFIG = yaml.safe_load(fp)

MODEL_PARAMS = CONFIG['train_params']

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

@pytest.fixture
def categorical():
    return CONFIG['categorical_features']

@pytest.fixture(scope='function')
def df():
    import pandas as pd

    dataframe = pd.read_csv(DATA_PATH)
    return dataframe

def test_process_data(encoder, binarizer, df, categorical):
    X, y, _, _ = process_data(df,
                              categorical_features=categorical,
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)
    assert isinstance(X, np.ndarray)
    assert len(X) > 0
    assert isinstance(y, np.ndarray)

def test_encoder_artifact(encoder, binarizer, df, categorical):
    _, _, encoder, lb = process_data(df,
                              categorical_features=categorical,
                              label='salary', training=True)
    assert isinstance(lb, LabelBinarizer)
    assert isinstance(encoder, OneHotEncoder)

def test_train(df, categorical):
    X, y, _, _ = process_data(df,
                              categorical_features=categorical,
                              label='salary', training=True, encoder=encoder,
                              lb=binarizer)
    model = train_model(X,y, MODEL_PARAMS)
    assert isinstance(model, RandomForestClassifier)

def test_inference(random_forest, encoder, binarizer, df, categorical):
    X, y, _, _ = process_data(df,
                              categorical_features=categorical,
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)

    preds = inference(random_forest, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) > 0


def test_model_artifact(random_forest):
    assert isinstance(random_forest, RandomForestClassifier)

def test_binarizer_artifact(binarizer):
    assert isinstance(binarizer, LabelBinarizer)

def test_encoder_artifact(encoder):
    assert isinstance(encoder, OneHotEncoder)

def test_compute_slice_metrics(df, random_forest, encoder, binarizer, categorical):
    SLICES = categorical
    for elem in SLICES:
        predictions = compute_slice_metrics(df, elem, random_forest,
                        encoder, binarizer)
        for feature, metrics in predictions.items():
            assert isinstance(feature, str)
            assert isinstance(
                metrics['precision'],
                float) and isinstance(
                metrics['recall'],
                float) and isinstance(
                metrics['fbeta'],
                float)
