from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.alert_system import alert_system

app = FastAPI(title="Network Anomaly Detection API")

# Load model
MODEL_PATH = "model/cic_ids_model.pkl"
model_pipeline = None

def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    else:
        print("Model file not found")

load_model()

class NetworkFlow(BaseModel):
    features: list
    source_ip: str = "Unknown"
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_anomaly: bool
    timestamp: str

@app.get("/")
async def root():
    return {"message": "Network Anomaly Detection API", "status": "active"}

@app.get("/health")
async def health_check():
    model_status = "loaded" if model_pipeline else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(flow: NetworkFlow):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_names = model_pipeline.get("feature_names", [])
        if not feature_names:
            raise ValueError("Model pipeline is missing feature metadata")

        feature_values = list(flow.features)
        if len(feature_values) < len(feature_names):
            feature_values.extend([0.0] * (len(feature_names) - len(feature_values)))
        elif len(feature_values) > len(feature_names):
            feature_values = feature_values[: len(feature_names)]

        x_df = pd.DataFrame([feature_values], columns=feature_names)
        x_df = x_df.replace([np.inf, -np.inf], np.nan)
        x_imputed = model_pipeline["imputer"].transform(x_df)
        x_scaled = model_pipeline["scaler"].transform(x_imputed)

        prediction_encoded = model_pipeline["model"].predict(x_scaled)[0]
        if hasattr(model_pipeline["model"], "predict_proba"):
            probabilities = model_pipeline["model"].predict_proba(x_scaled)[0]
            confidence = probabilities.max()
        else:
            confidence = 1.0
        
        # Decode prediction
        prediction = model_pipeline["label_encoder"].inverse_transform([prediction_encoded])[0]
        
        # Check if it's an anomaly
        is_anomaly = prediction != 'BENIGN'
        
        # Trigger alert if anomaly detected
        if is_anomaly:
            alert_features = {
                'source_ip': flow.source_ip,
                'features': flow.features
            }
            alert_system.trigger_alert(prediction, confidence, alert_features)
        
        response = PredictionResponse(
            prediction=prediction,
            confidence=float(confidence),
            is_anomaly=is_anomaly,
            timestamp=flow.timestamp or datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/alerts")
async def get_recent_alerts(hours: int = 24):
    # This would typically query a database
    # For now, return mock data
    return {
        "alerts": [
            {
                "timestamp": datetime.now().isoformat(),
                "type": "DDoS",
                "source_ip": "192.168.1.100",
                "confidence": 0.95
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)