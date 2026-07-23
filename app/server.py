"""
SOC ML API
FastAPI server for CICIoT2023 attack classification

Author: Your Name
"""

from pathlib import Path
import traceback

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# PATHS

# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent


# Директория с обученными объектами
MODELS_DIR = BASE_DIR / "models"


MODEL_PATH = MODELS_DIR / "best_model.pkl"

SCALER_PATH = MODELS_DIR / "scaler.pkl"

ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

FEATURES_PATH = MODELS_DIR / "feature_names.pkl"


# LOAD MODEL COMPONENTS


model = None
scaler = None
encoder = None
feature_names = None


# ------------------------------
# Load ML model
# ------------------------------

try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "[OK] Model loaded:",
        MODEL_PATH
    )

except Exception as e:

    print(
        "[ERROR] Model loading failed:",
        e
    )


# Load scaler

try:

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        "[OK] Scaler loaded:",
        SCALER_PATH
    )

except Exception as e:

    print(
        "[WARNING] Scaler loading failed:",
        e
    )


# Load LabelEncoder

try:

    encoder = joblib.load(
        ENCODER_PATH
    )

    print(
        "[OK] Label encoder loaded:",
        ENCODER_PATH
    )

except Exception as e:

    print(
        "[WARNING] Label encoder loading failed:",
        e
    )


# Load feature names


try:

    feature_names = joblib.load(
        FEATURES_PATH
    )

    print(
        "[OK] Feature names loaded"
    )

    print(
        "Number of features:",
        len(feature_names)
    )

except Exception as e:

    print(
        "[WARNING] Feature names loading failed:",
        e
    )

# FASTAPI APPLICATION


app = FastAPI(

    title="SOC ML Detection API",

    description=(
        "Machine Learning API for "
        "CICIoT2023 network attack classification"
    ),

    version="1.0.0"

)


# REQUEST SCHEMA



class PredictionRequest(BaseModel):

    features: list[float] = Field(

        ...,

        description=(
            "Network traffic features "
            "in the same order as during model training"
        )

    )


# ROOT ENDPOINT


@app.get("/")
def root():

    return {

        "service": "SOC ML Detection API",

        "status": "running",

        "model_loaded": model is not None,

        "scaler_loaded": scaler is not None,

        "encoder_loaded": encoder is not None,

        "features_loaded": feature_names is not None

    }

# HEALTH CHECK



@app.get("/health")
def health():

    return {

        "status": "OK",

        "model": model is not None,

        "scaler": scaler is not None,

        "encoder": encoder is not None,

        "feature_names": feature_names is not None

    }


# MODEL INFORMATION



@app.get("/model-info")
def model_info():

    if model is None:

        raise HTTPException(

            status_code=500,

            detail="Model is not loaded"

        )

    return {

        "model_type": type(model).__name__,

        "number_of_features": (

            len(feature_names)

            if feature_names is not None

            else None

        ),

        "number_of_classes": (

            len(encoder.classes_)

            if encoder is not None

            else None

        ),

        "classes": (

            encoder.classes_.tolist()

            if encoder is not None

            else None

        )

    }

# PREDICTION



@app.post("/predict")
def predict(
    request: PredictionRequest
):


    # Проверка загрузки модели
 

    if model is None:

        raise HTTPException(

            status_code=500,

            detail="Model is not loaded"

        )


    try:

       
        # Получаем входные признаки
       
        features = request.features



        # Проверяем количество признаков
  

        if feature_names is not None:

            expected_features = len(
                feature_names
            )

            received_features = len(
                features
            )


            if received_features != expected_features:

                raise HTTPException(

                    status_code=400,

                    detail={

                        "error": (
                            "Invalid number of features"
                        ),

                        "expected": expected_features,

                        "received": received_features

                    }

                )


   
        # Преобразуем в numpy array
    

        X = np.array(

            features,

            dtype=np.float32

        ).reshape(

            1,

            -1

        )


      
        # Применяем scaler
   

        if scaler is not None:

            X = scaler.transform(

                X

            )


        # Prediction


        prediction = model.predict(

            X

        )


        predicted_class = prediction[0]


        # Probability


        confidence = None


        if hasattr(

            model,

            "predict_proba"

        ):

            probabilities = model.predict_proba(

                X

            )


            confidence = float(

                np.max(

                    probabilities[0]

                )

            )


        # Decode class


        if encoder is not None:

            predicted_label = encoder.inverse_transform(

                [predicted_class]

            )[0]

        else:

            predicted_label = str(

                predicted_class

            )


        # Response

        return {

            "prediction": str(

                predicted_label

            ),

            "class_id": int(

                predicted_class

            ),

            "confidence": confidence,

            "model": type(

                model

            ).__name__

        }


    except HTTPException:

        raise


    except Exception as e:

        traceback.print_exc()


        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# RUN SERVER



if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        "server:app",

        host="127.0.0.1",

        port=8000,

        reload=True

    )
