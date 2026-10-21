from typing import Any
import json

import joblib
import pandas as pd

from iris_mlops.config import FEATURES, METRICS_PATH, MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


def load_model_info() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text())


def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
) -> dict[str, Any]:
    model = load_model()
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES
    )
    species = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    return {
        "species": str(species),
        "probabilities": {
            str(clase): round(float(p), 4)
            for clase, p in zip(model.classes_, probabilities)
        }
    }


def main() -> None:
    example = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    print(f"Muestra ejemplo: {example}")
    print(f"Predicción: {predict(**example)}")


if __name__ == "__main__":
    main()