"""
client.py

Client for SOC ML API

Отправляет сетевые признаки
на ML сервер и получает результат
детектирования атаки.
"""


import requests
import json


# =====================================================
# Настройки сервера
# =====================================================


SERVER_URL = "http://127.0.0.1:8000"


PREDICT_ENDPOINT = (
    SERVER_URL +
    "/predict"
)


HEALTH_ENDPOINT = (
    SERVER_URL +
    "/health"
)


# =====================================================
# Проверка сервера
# =====================================================


def check_server():

    try:

        response = requests.get(
            HEALTH_ENDPOINT
        )

        if response.status_code == 200:

            print("[OK] Server available")

            print(
                json.dumps(
                    response.json(),
                    indent=4,
                    ensure_ascii=False
                )
            )

        else:

            print(
                "[ERROR]",
                response.status_code
            )


    except Exception as e:

        print(
            "[ERROR] Server unavailable:",
            e
        )



# =====================================================
# Отправка признаков
# =====================================================


def predict_attack(features):

    payload = {

        "features": features

    }


    try:

        response = requests.post(

            PREDICT_ENDPOINT,

            json=payload

        )


        if response.status_code == 200:

            return response.json()


        else:

            print(
                "Server error:",
                response.text
            )


    except Exception as e:

        print(
            "Connection error:",
            e
        )



# =====================================================
# Красивый вывод результата
# =====================================================


def print_result(result):

    if result is None:

        return


    print("\n========== RESULT ==========")


    print(
        "Prediction:",
        result.get(
            "prediction"
        )
    )


    confidence = result.get(
        "confidence"
    )


    if confidence:

        print(
            "Confidence:",
            round(
                confidence * 100,
                2
            ),
            "%"
        )


    probabilities = result.get(
        "probabilities"
    )


    if probabilities:


        print(
            "\nClass probabilities:"
        )


        for i, value in enumerate(probabilities):

            print(
                f"Class {i}: {value:.4f}"
            )


    print(
        "============================"
    )



# =====================================================
# Тестовый пример
# =====================================================


if __name__ == "__main__":


    # Проверяем сервер

    check_server()



    # Пример сетевого потока CICIoT2023

    test_flow = [

        0.52,      # Flow Duration

        54,        # Header Length

        6,         # Protocol

        1200,      # Rate

        450,       # Packet Size

        40,        # Packet Count

        0.002,     # IAT

        1,         # SYN flag

        0,         # ACK flag

        0

    ]


    result = predict_attack(
        test_flow
    )


    print_result(
        result
    )
