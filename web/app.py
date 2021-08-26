import os
from io import StringIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from starter.ml.model import RFClassifier
from starter.ml.data import process_data
from pydantic import BaseModel

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

description = "CensusAPP API helps you to predict salary category based on demographic data 🚀"

rf_model = RFClassifier()
app = FastAPI()

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


@app.get('/')
def test():
    return "Welcome to census predictor app!"


@app.post('/batch_inference')
async def batch_inference(csv_file: UploadFile = File(...)):

    try:
        str_buf = StringIO(str(csv_file.file.read(), 'utf-8'))
        df = pd.read_csv(str_buf, encoding='utf-8')
        df.columns = [col.strip().replace('-', '_') for col in df.columns]

        X, _, _, _ = process_data(df,
                                  categorical_features=rf_model.CAT_FEATURES,
                                  training=False,
                                  encoder=rf_model.encoder,
                                  lb=rf_model.binarizer)
        y_preds = rf_model.inference(X)

    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None,
                'error': 'Failed to parse uploaded csv.'}

    return {'success': True, 'results': y_preds, 'error': None}


@app.post('/stream_inference')
async def inference(individual: CensusObject):

    try:
        indv_dict = {k: [v] for k, v in individual.dict().items()}
        df = pd.DataFrame.from_dict(indv_dict)

        X, _, _, _ = process_data(df,
                                  categorical_features=rf_model.CAT_FEATURES,
                                  training=False,
                                  encoder=rf_model.encoder,
                                  lb=rf_model.binarizer)
        y_preds = rf_model.inference(X)
    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None,
                'error': 'Failed to perform prediction. \
                    Make sure to specify correct columns and value'}

    return {'success': True, 'results': y_preds, 'error': None}
