from fastapi import FastAPI
import joblib
import pandas as pd
from src.schemas import CustomerData
from src.database import SessionLocal
from src.models_db import PredictionLog

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


model = joblib.load("models/churn_model.joblib")

config = joblib.load("models/churn_config.joblib")

threshold = config["threshold"]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    input_data = pd.DataFrame([customer.model_dump()])

    churn_probability = model.predict_proba(input_data)[:, 0][0]

    prediction = "Churned" if churn_probability >= threshold else "Stayed"

    db = SessionLocal()

    try:
        prediction_log = PredictionLog(
            prediction=prediction,
            churn_probability=float(churn_probability),
            threshold=float(threshold),
        )

        db.add(prediction_log)
        db.commit()

    finally:
        db.close()

    return {
        "prediction": prediction,
        "churn_probability": round(float(churn_probability), 4),
        "threshold": round(float(threshold), 4),
    }
