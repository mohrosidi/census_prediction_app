"""
Script to train the model

Author : Moh. Rosidi
Date   : August 2021
"""
import os
import json
import yaml
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.model import RFClassifier, train_model, compute_model_metrics, inference, save_artifact, compute_slice_metrics
from ml.data import process_data, preprocess_data

# get current directory
CWD = os.getcwd()

# Set up logging
logging.basicConfig(
    filename=os.path.join(
        CWD,
        'logs',
        'model.log'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    level=logging.INFO,
)

SLICE_LOGGER = logging.getLogger('slice_metrics')
SLICE_LOGGER.setLevel(logging.INFO)
SLICE_LOGGER.addHandler(
    logging.FileHandler(
        filename=os.path.join(
            CWD,
            'logs',
            'slice_output.txt'),
        mode='w',
    ))

# Loads config
with open(os.path.join(CWD, "starter", 'params.yaml'), 'r') as fp:
    CONFIG = yaml.safe_load(fp)

CAT_FEATURES = CONFIG['categorical_features']

# Loads census data
DATA_FILENAME = 'census.csv'
DATA_DIR = os.path.join(
    CWD, 
    'data'
    )
DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

RAW_DF = pd.read_csv(DATA_PATH)
PREPROCESSED_DF = preprocess_data(RAW_DF, dest_path=DATA_DIR)

# Data segregation
TRAIN, TEST = train_test_split(
    PREPROCESSED_DF, 
    test_size=0.20, 
    random_state=CONFIG['random_seed']
    )

# Feature Engineering
X_TRAIN, Y_TRAIN, ENCODER, LABEL = process_data(
    TRAIN, 
    categorical_features=CAT_FEATURES, 
    label="salary", 
    training=True
    )

X_TEST, Y_TEST, _, _ = process_data(
    TEST, 
    categorical_features=CAT_FEATURES, 
    label='salary', training=False, 
    encoder=ENCODER, 
    lb=LABEL)

logging.info(f"Model parameters: {CONFIG['random_forest']}")

# Train and save a model.
MODEL = train_model(X_TRAIN, Y_TRAIN, CONFIG['random_forest'])

# Scoring
Y_TEST_PREDS = inference(MODEL, X_TEST)
PRECISION, RECALL, FBETA = compute_model_metrics(Y_TEST, Y_TEST_PREDS)

logging.info(
    f'Overall model predictions - precision: {PRECISION}, recall: {RECALL}, fbeta: {FBETA}'
    )

# Track the model scores
with open(os.path.join(CWD, "logs", 'scores.json'), "w") as f:
    json.dump({"precision":PRECISION,
    "recall": RECALL,
    "fbeta": FBETA}, 
    f)

# export artifacts
MODEL_DIR = os.path.join(
    CWD, 
    'model')

# model export
MODEL_DEST_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')
save_artifact(MODEL, MODEL_DEST_PATH)

# encoder and labelbinarizer export for inference
save_artifact(ENCODER, os.path.join(MODEL_DIR, 'onehot_encoder.pkl'))
save_artifact(LABEL, os.path.join(MODEL_DIR, 'label_binarizer.pkl'))

# compute metrics based on slice
CLEAN_DF = pd.read_csv(
    os.path.join(
        CWD,
        'data',
        'clean_census.csv'))

for slice in CAT_FEATURES:
    slice_metrics = compute_slice_metrics(CLEAN_DF, slice)

    SLICE_LOGGER.info(f"`{slice}` category")
    for feature_val, metrics in slice_metrics.items():
        SLICE_LOGGER.info(
            f"`{slice}` category -> precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}, fbeta: {metrics['fbeta']:.3f}, numb.rows: {metrics['n_row']} -- {feature_val}.")
    SLICE_LOGGER.info('\n')
