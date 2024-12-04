from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from typing import List, Dict

 

app = FastAPI(
    title="Iris Classifier API",
    description="An API to serve predictions for the Iris classification model using MLflow.",
    version="1.0.0",
)

 

mlflow.set_tracking_uri("http://localhost:5000")
MODEL_URI = "models:/IrisClassifier/Production"  
model = mlflow.pyfunc.load_model(MODEL_URI)

 

class PredictionRequest(BaseModel):
    columns: List[str]  
    data: List[List[float]] 

 

@app.post("/predict", summary="Make a prediction", tags=["Prediction"])
async def predict(request: PredictionRequest):
 
    try:
        input_data = pd.DataFrame(request.data, columns=request.columns)

        predictions = model.predict(input_data)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

 

@app.get("/", summary="Root Endpoint", tags=["General"])
async def root():

    return {"message": "Welcome to the Iris Classifier API! Use /docs for API documentation."}