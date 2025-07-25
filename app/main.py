from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('model.joblib')

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts if a transaction is fraudulent.",
    version="1.0"
)

class Transaction(BaseModel):
    features: list

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]
    return {
        "prediction": int(pred),
        "fraud_probability": float(prob)
    }
