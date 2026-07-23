"""
client.py

Client for SOC ML Detection API.

Отправляет характеристики сетевого трафика
на FastAPI-сервер и получает результат
классификации сетевой активности.

Проект:
CICIoT2023 Network Attack Detection
"""


import json
import requests

# SERVER SETTINGS

SERVER_URL = (
    "http://127.0.0.1:8000"
)


PREDICT_ENDPOINT = (

    SERVER_URL

    + "/predict"

)


HEALTH_ENDPOINT = (

    SERVER_URL

    + "/health"

)


MODEL_INFO_ENDPOINT = (

    SERVER_URL

    + "/model-info"

)


# CHECK SERVER


def check_server():

    """
    Проверяет доступность ML-сервера.
    """

    try:

        response = requests.get(

            HEALTH_ENDPOINT,

            timeout=5

        )


        if response.status_code == 200:

            print(

                "[OK] Server available"

            )


            print(

                json.dumps(

                    response.json(),

                    indent=4,

                    ensure_ascii=False

                )

            )


            return True


        else:

            print(

                "[ERROR] Server returned:",

                response.status_code

            )


            return False


    except requests.exceptions.ConnectionError:

        print(

            "[ERROR] Server unavailable"

        )


        print(

            "Start the server first"

        )


        return False


    except requests.exceptions.Timeout:

        print(

            "[ERROR] Server timeout"

        )


        return False


    except Exception as e:

        print(

            "[ERROR]:",

            e

        )


        return False


# GET MODEL INFORMATION


def get_model_info():

    """
    Получает информацию о загруженной модели.
    """

    try:

        response = requests.get(

            MODEL_INFO_ENDPOINT,

            timeout=5

        )


        if response.status_code == 200:

            return response.json()


        print(

            "[ERROR] Model info error:",

            response.text

        )


        return None


    except Exception as e:

        print(

            "[ERROR] Cannot get model info:",

            e

        )


        return None


# PREDICT ATTACK


def predict_attack(

    features

):

    """
    Отправляет признаки сетевого потока
    на ML-сервер.

    Parameters
    ----------
    features : list[float]

        Признаки сетевого трафика.

        Порядок признаков должен совпадать
        с порядком признаков при обучении модели.

    Returns
    -------
    dict

        Результат предсказания.
    """


    payload = {

        "features": features

    }


    try:

        response = requests.post(

            PREDICT_ENDPOINT,

            json=payload,

            timeout=30

        )


        if response.status_code == 200:

            return response.json()


        else:

            print(

                "[SERVER ERROR]"

            )


            print(

                "Status code:",

                response.status_code

            )


            print(

                "Response:",

                response.text

            )


            return None


    except requests.exceptions.ConnectionError:

        print(

            "[ERROR] Cannot connect to server"

        )


        return None


    except requests.exceptions.Timeout:

        print(

            "[ERROR] Request timeout"

        )


        return None


    except Exception as e:

        print(

            "[ERROR] Request failed:",

            e

        )


        return None


# PRINT RESULT



def print_result(

    result

):

    """
    Красиво выводит результат классификации.
    """


    if result is None:

        return


    print(

        "\n========== SOC ML RESULT =========="

    )


    print(

        "Prediction:",

        result.get(

            "prediction"

        )

    )


    print(

        "Class ID:",

        result.get(

            "class_id"

        )

    )


    confidence = result.get(

        "confidence"

    )


    if confidence is not None:

        print(

            "Confidence:",

            round(

                confidence * 100,

                2

            ),

            "%"

        )


    print(

        "Model:",

        result.get(

            "model"

        )

    )


    print(

        "---------------------------------"

    )

# MAIN



if __name__ == "__main__":


    print(

        "Starting SOC ML Client..."

    )


    # 1. Проверяем сервер



    server_available = check_server()


    if not server_available:

        exit()


    # 2. Получаем информацию о модели



    model_info = get_model_info()


    if model_info is not None:


        print(

            "\n========== MODEL INFO =========="

        )


        print(

            "Model:",

            model_info.get(

                "model_type"

            )

        )


        print(

            "Number of features:",

            model_info.get(

                "number_of_features"

            )

        )


        print(

            "Number of classes:",

            model_info.get(

                "number_of_classes"

            )

        )


        print(

            "-------------------------------------"

        )


    # ---------------------------------

    # 3. Тестовый поток

    # ---------------------------------


    print(

        "\nSending test network flow..."

    )


    # ВАЖНО:
    #
    # Здесь должно быть ровно столько
    # признаков, сколько использовалось
    # при обучении модели.
    #
    # И в том же порядке.


    test_flow = [

        0.52,

        54,

        6,

        1200,

        450,

        40,

        0.002,

        1,

        0,

        0

    ]


    # 4. Отправляем запрос


    result = predict_attack(

        test_flow

    )


    # 5. Выводим результат


    print_result(

        result

    )
