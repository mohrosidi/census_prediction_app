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
    Random forest classifier model.
    """
    def __init__(
            self,
            model: RandomForestClassifier = None,
            binarizer: LabelBinarizer = None,
            encoder: OneHotEncoder = None,
            config_path: str = None):
        
        import yaml

        config_path = config_path if config_path else os.path.join(
            os.getcwd(),
            "starter", 
            'params.yaml'
            )

        with open(config_path, 'r') as fp:
            CONFIG = yaml.safe_load(fp)

        self.model =  model if model else load(os.path.join(
            os.getcwd(),
            "starter", 
            CONFIG['model_output']
            )
        )
        self.binarizer = binarizer if binarizer else load(os.path.join(
            os.getcwd(),
            "starter", 
            CONFIG['label_binarizer_output']
            )
        )
        self.encoder = encoder if encoder else load(os.path.join(
            os.getcwd(),
            "starter", 
            CONFIG['encoder_output']
            )
        )

        self.CAT_FEATURES = CONFIG['categorical_features']

    def inference(self, X: np.array) -> List:
        preds = self.model.predict(X)
        predicted_labels = self.binarizer.inverse_transform(preds)
        return list(predicted_labels)
        

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
    """
    dump(artifact, dest_path)

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
        model: RFClassifier = RFClassifier()):
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
                                  categorical_features=model.CAT_FEATURES,
                                  label='salary',
                                  training=False,
                                  encoder=rf_model.encoder,
                                  lb=rf_model.binarizer)

        y_preds = model.model.predict(X)

        precision, recall, fbeta = compute_model_metrics(y, y_preds)
        predictions[cat_feature] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'n_row': len(filtered_df)}
    return predictions
