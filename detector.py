"""
SOC ML Traffic Detector

Автоматическое обнаружение подозрительного
сетевого трафика с использованием обученной
ML-модели на основе CICIoT2023.

Логика:

Network Traffic
        ↓
Feature Validation
        ↓
ML Model
        ↓
Attack Classification
        ↓
Suspicious / Benign
        ↓
SOC Alert
"""


from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# PATHS


BASE_DIR = Path(__file__).resolve().parent.parent


MODELS_DIR = BASE_DIR / "models"


MODEL_PATH = (
    MODELS_DIR
    / "best_model.pkl"
)


SCALER_PATH = (
    MODELS_DIR
    / "scaler.pkl"
)


ENCODER_PATH = (
    MODELS_DIR
    / "label_encoder.pkl"
)


FEATURES_PATH = (
    MODELS_DIR
    / "feature_names.pkl"
)

# SETTINGS



# Название нормального класса
BENIGN_CLASS = "BenignTraffic"


# Порог уверенности модели
CONFIDENCE_THRESHOLD = 0.70


# LOAD MODEL COMPONENTS


print(
    "[INFO] Loading ML components..."
)


model = joblib.load(

    MODEL_PATH

)


encoder = joblib.load(

    ENCODER_PATH

)


feature_names = joblib.load(

    FEATURES_PATH

)

# SCALER



scaler = None


if SCALER_PATH.exists():

    scaler = joblib.load(

        SCALER_PATH

    )


print(

    "[OK] Model:",

    type(model).__name__

)


print(

    "[OK] Number of features:",

    len(feature_names)

)


print(

    "[OK] Number of classes:",

    len(encoder.classes_)

)




def detect_flow(

    flow

):

    """
    Анализирует один сетевой поток.

    Parameters
    ----------
    flow : dict

        Словарь с сетевыми признаками.

    Returns
    -------
    dict

        Результат анализа.
    """

    # Проверяем наличие признаков


    missing_features = [

        feature

        for feature in feature_names

        if feature not in flow

    ]


    if missing_features:

        return {

            "status": "ERROR",

            "error": "Missing features",

            "missing_features": missing_features

        }


    # Формируем признаки


    X = pd.DataFrame(

        [

            [

                flow[feature]

                for feature in feature_names

            ]

        ],

        columns=feature_names

    )

    # Применяем scaler



    if scaler is not None:

        X_model = scaler.transform(

            X

        )

    else:

        X_model = X


    # Получаем предсказание



    prediction = model.predict(

        X_model

    )


    predicted_class_id = int(

        prediction[0]

    )


    predicted_label = encoder.inverse_transform(

        prediction

    )[0]


    # Confidence

    confidence = None


    probabilities = None


    if hasattr(

        model,

        "predict_proba"

    ):


        probabilities = model.predict_proba(

            X_model

        )[0]


        confidence = float(

            np.max(

                probabilities

            )

        )

    # Определяем статус



    if predicted_label == BENIGN_CLASS:

        traffic_status = "BENIGN"

        is_suspicious = False

    else:

        traffic_status = "ATTACK"

        is_suspicious = True


    # Дополнительный анализ confidence

    low_confidence = False


    if confidence is not None:

        if confidence < CONFIDENCE_THRESHOLD:

            low_confidence = True


    # Формируем результат

    result = {

        "status": traffic_status,

        "is_suspicious": is_suspicious,

        "prediction": str(

            predicted_label

        ),

        "class_id": predicted_class_id,

        "confidence": confidence,

        "low_confidence": low_confidence,

        "model": type(

            model

        ).__name__

    }


    return result

# BATCH DETECTION

def detect_csv(

    input_file,

    output_file

):

    """
    Анализирует CSV-файл
    с сетевым трафиком.
    """


    print(

        "[INFO] Loading traffic data..."

    )


    df = pd.read_csv(

        input_file

    )


    print(

        "[INFO] Loaded rows:",

        len(df)

    )

    # Проверяем признаки

    missing_features = [

        feature

        for feature in feature_names

        if feature not in df.columns

    ]


    if missing_features:

        raise ValueError(

            f"Missing features: {missing_features}"

        )

    # Формируем X

    X = df[

        feature_names

    ]

    # Применяем scaler

    if scaler is not None:

        X_model = scaler.transform(

            X

        )

    else:

        X_model = X


    # Prediction


    predictions = model.predict(

        X_model

    )


    
    # Decode labels



    predicted_labels = encoder.inverse_transform(

        predictions

    )


    # Confidence



    if hasattr(

        model,

        "predict_proba"

    ):


        probabilities = model.predict_proba(

            X_model

        )


        confidence = np.max(

            probabilities,

            axis=1

        )


    else:

        confidence = [

            None

            for _ in range(

                len(df)

            )

        ]


    # Add results


    df["prediction"] = predicted_labels


    df["confidence"] = confidence


    df["is_suspicious"] = (

        df["prediction"]

        != BENIGN_CLASS

    )


    df["traffic_status"] = np.where(

        df["is_suspicious"],

        "ATTACK",

        "BENIGN"

    )


    # Save results

    df.to_csv(

        output_file,

        index=False

    )


    print(

        "[OK] Analysis completed"

    )


    print(

        "[OK] Results saved to:",

        output_file

    )


    # Statistics

    print(

        "\n========== DETECTION SUMMARY =========="

    )


    print(

        "Total flows:",

        len(df)

    )


    print(

        "Benign flows:",

        (

            df["traffic_status"]

            == "BENIGN"

        ).sum()

    )


    print(

        "Suspicious flows:",

        (

            df["traffic_status"]

            == "ATTACK"

        ).sum()

    )


    print(

        "\nAttack distribution:"

    )


    print(

        df.loc[

            df["is_suspicious"],

            "prediction"

        ]

        .value_counts()

    )


    print(

        "========================================"

    )


# MAIN


if __name__ == "__main__":


    INPUT_FILE = (

        BASE_DIR

        / "data"

        / "network_traffic.csv"

    )


    OUTPUT_FILE = (

        BASE_DIR

        / "data"

        / "detection_results.csv"

    )


    detect_csv(

        input_file=INPUT_FILE,

        output_file=OUTPUT_FILE

    )