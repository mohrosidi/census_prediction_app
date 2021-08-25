"""
Script to train the model

Author : Moh. Rosidi
Date   : August 2021
"""
import os
import json
import logging
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, \
    inference, save_artifact, compute_slice_metrics
from ml.data import process_data, preprocess_data

CWD = os.getcwd()

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

with open(os.path.join(CWD, "starter", 'params.yaml'), 'r', encoding="UTF-8") as fp:
    CONFIG = yaml.safe_load(fp)

CAT_FEATURES = CONFIG['categorical_features']

DATA_FILENAME = CONFIG['data']
DATA_DIR = os.path.join(
    CWD,
    'data'
)
DATA_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

PREPROCESSED_DF = pd.read_csv(DATA_PATH)

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

logging.info("Model parameters: %s}", CONFIG['train_params'])

MODEL = train_model(X_TRAIN, Y_TRAIN, CONFIG['train_params'])

Y_TEST_PREDS = inference(MODEL, X_TEST)
PRECISION, RECALL, FBETA = compute_model_metrics(Y_TEST, Y_TEST_PREDS)

logging.info(
    'Overall model predictions - precision: %s, recall: %s, fbeta: %s',
    PRECISION, RECALL, FBETA
)

with open(os.path.join(CWD, "logs", 'scores.json'), "w", encoding="UTF-8") as f:
    json.dump({"precision": PRECISION,
               "recall": RECALL,
               "fbeta": FBETA},
              f)

MODEL_DIR = os.path.join(
    CWD,
    'model')

MODEL_DEST_PATH = os.path.join(MODEL_DIR, CONFIG['model_output'])
save_artifact(MODEL, MODEL_DEST_PATH)

save_artifact(ENCODER, os.path.join(MODEL_DIR, CONFIG['encoder_output']))
save_artifact(LABEL, os.path.join(MODEL_DIR, CONFIG['label_binarizer_output']))

CLEAN_DF = pd.read_csv(
    os.path.join(
        CWD,
        'data',
        CONFIG['data']))

for elem in CAT_FEATURES:
    slice_metrics = compute_slice_metrics(CLEAN_DF, elem)

    SLICE_LOGGER.info("`%s` category", elem)
    for feature_val, metrics in slice_metrics.items():
        SLICE_LOGGER.info(
            "`%s` category -> precision: %s, recall: %s, fbeta: %s, numb.rows: %s -- %s.",
                    elem, metrics['precision'], metrics['recall'],
                    metrics['fbeta'], metrics['n_row'], feature_val)
    SLICE_LOGGER.info('\n')
