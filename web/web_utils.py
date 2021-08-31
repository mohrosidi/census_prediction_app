import os
from joblib import load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from pydantic import BaseModel

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
            "params.yaml"
            )

        with open(config_path, 'r') as fp:
            CONFIG = yaml.safe_load(fp)

        self.model =  model if model else load(os.path.join(
            os.getcwd(),
            "model", 
            CONFIG['model_output']
            )
        )
        self.binarizer = binarizer if binarizer else load(os.path.join(
            os.getcwd(),
            "model", 
            CONFIG['label_binarizer_output']
            )
        )
        self.encoder = encoder if encoder else load(os.path.join(
            os.getcwd(),
            "model", 
            CONFIG['encoder_output']
            )
        )

        self.CAT_FEATURES = CONFIG['categorical_features']

    def inference(self, X: np.array):
        preds = self.model.predict(X)
        predicted_labels = self.binarizer.inverse_transform(preds)
        return list(predicted_labels)
        
class CensusObject(BaseModel):
    age: int
    workclass: str
    fnlgt: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": "77516",
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }
