"""
SOC ML API
FastAPI server

Author: Your Name
"""

from pathlib import Path
import traceback

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ======================================================
# Пути
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# ======================================================
# Загрузка модели
# ======================================================

try:
    model = joblib.load(MODEL_PATH)
    print("[OK] Model loaded")

except Exception as e:
    print(e)
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Scaler loaded")

except Exception:
    scaler = None

try:
    encoder = joblib.load(ENCODER_PATH)
    print("[OK] Label encoder loaded")

except Exception:
    encoder = None


# ======================================================
# FastAPI
# ======================================================

app = FastAPI(
    title="SOC ML API",
    description="IoT Intrusion Detection using Machine Learning",
    version="1.0"
)

# ======================================================
# Модель входных данных
# ======================================================


class PredictionRequest(BaseModel):

    features: list[float]


# ======================================================
# Главная страница
# ======================================================

@app.get("/")
def root():

    return {
        "service": "SOC ML API",
        "status": "running",
        "model_loaded": model is not None
    }


# ======================================================
# Health
# ======================================================

@app.get("/health")
def health():

    return {
        "status": "OK",
        "model": model is not None,
        "scaler": scaler is not None,
        "encoder": encoder is not None
    }


# ======================================================
# Predict
# ======================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Model not loaded"
        )

    try:

        X = np.array(request.features).reshape(1, -1)

        if scaler is not None:
            X = scaler.transform(X)

        prediction = model.predict(X)

        probability = None

        if hasattr(model, "predict_proba"):

            probability = float(
                np.max(model.predict_proba(X))
            )

        predicted_class = prediction[0]

        if encoder is not None:

            predicted_class = encoder.inverse_transform(
                prediction
            )[0]

        response = {

            "prediction": str(predicted_class),

            "confidence": probability

        }

        return response

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ======================================================
# Запуск
# ======================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
