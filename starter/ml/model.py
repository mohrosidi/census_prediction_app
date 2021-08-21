"""
Training and Scoring Module

Author : Moh. Rosidi
Date   : August 2021
"""
import os
from joblib import load, dump
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from typing import List

from .data import process_data

class RFClassifier:
    """
        Random forest classifier model. Uses the default paths to load the model artifacts
        if none is provided as args.
    """
    _default_model_path = os.path.join(
        os.getcwd(),
        'model',
        'random_forest.pkl')
    _default_binarizer_path = os.path.join(
        os.getcwd(),
        'model',
        'label_binarizer.pkl')
    _default_encoder_path = os.path.join(
        os.getcwd(),
        'model',
        'onehot_encoder.pkl')

    def __init__(
            self,
            model: RandomForestClassifier = None,
            binarizer: LabelBinarizer = None,
            encoder: OneHotEncoder = None):
        # initialise - use default paths if none provided
        self.model = model if model else self._load_artifact(
            RFClassifier._default_model_path)
        self.binarizer = binarizer if binarizer else self._load_artifact(
            RFClassifier._default_binarizer_path)
        self.encoder = encoder if encoder else self._load_artifact(
            RFClassifier._default_encoder_path)

        self._load_categorical_features()

    def inference(self, X: np.array) -> List:
        preds = self.model.predict(X)
        predicted_labels = self.binarizer.inverse_transform(preds)
        return list(predicted_labels)

    def _load_artifact(self, target_path: str):
        return load(target_path)

    def _load_categorical_features(self, config_path: str = None):
        import yaml

        config_path = config_path if config_path else os.path.join(
            os.getcwd(),
            "starter", 
            'config.yaml'
            )

        with open(config_path, 'r') as fp:
            self.CAT_FEATURES = yaml.safe_load(fp)['categorical_features']

def train_model(X_train, y_train, model_params):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model_params: dict
        Model hyperparameters
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.array):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def save_artifact(artifact, dest_path: str):
    """
        Saves the artifact to the dest_path.
        Inputs
        ------
        artifact : Any
            Pickleable/Serializable object
        dest_path : str
            Destination path to save artifact
        Returns
        -------
        success, error_msg : tuple[str, str]
    """
    try:
        dump(artifact, dest_path)
    except Exception as err:
        return False, str(err)
    return True, None


def load_artifact(target_path: str):
    """
        Loads the artifact from the target_path.
        Inputs
        ------
        target_path : str
            Target path the artifact is located
        Returns
        -------
        artifact : Any
    """
    return load(target_path)


def compute_slice_metrics(
        df: pd.DataFrame,
        category: str,
        rf_model: RFClassifier = RFClassifier()):
    """
        Computes model metrics based on data slices
        Inputs
        ------
        df : pd.DataFrame
            Dataframe containing the cleaned data
        category : str
            Dataframe column to slice
        rf_model: RFClassifier
            Random forest model used to perform prediction
        Returns
        -------
        predictions : dict
            Dictionary containing the predictions for each category feature
    """

    predictions = {}
    for cat_feature in df[category].unique():
        filtered_df = df[df[category] == cat_feature]

        X, y, _, _ = process_data(filtered_df,
                                  categorical_features=rf_model.CAT_FEATURES,
                                  label='salary',
                                  training=False,
                                  encoder=rf_model.encoder,
                                  lb=rf_model.binarizer)

        print(f'Predicting slice of {category}: {cat_feature}')
        print(f'Num of rows in dataframe: {len(filtered_df)}')
        y_preds = rf_model.model.predict(X)

        precision, recall, fbeta = compute_model_metrics(y, y_preds)
        predictions[cat_feature] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'n_row': len(filtered_df)}
    return predictions
