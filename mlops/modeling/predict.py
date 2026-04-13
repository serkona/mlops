from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List
from mlops.config import MODELS_DIR

app = FastAPI(title="Credit Card Fraud Detection API")

try:
    model = joblib.load(MODELS_DIR / "model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
except FileNotFoundError:
    print("Warning: Model or scaler not found at startup.")
    model = None
    scaler = None

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float

@app.post("/predict", response_model=List[PredictionResponse])
def predict(transactions: List[Transaction]):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not transactions:
        return []

    try:
        data = [t.model_dump() for t in transactions]
        df = pd.DataFrame(data)
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = df[cols]
        
        cols_to_scale = ['Time', 'Amount']
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionResponse(prediction=int(pred), fraud_probability=float(prob)))
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
