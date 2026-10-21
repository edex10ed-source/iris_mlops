from typing import Any
from datetime import datetime
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# from iris_mlops import __version__
from iris_mlops.config import (
    FEATURES,
    METRICS_PATH,
    MIN_ACCURACY,
    MODELS_PARAMS,
    MODEL_PATH,
)
from iris_mlops.data import load_data, split_data


def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(**MODELS_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }


def save_model(model, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_metadata = {
        "model_version": "0.1.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": type(model).__name__,
        "params": MODELS_PARAMS,
        "features": FEATURES,
        "classes": list(model.classes_),
        "accuracy": metrics["accuracy"]
    }

    METRICS_PATH.write_text(json.dumps(model_metadata, indent=2))


def main() -> None:
    print("1) Cargar datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    Entrenamiento: {len(X_train)} filas | Test: {len(X_test)} filas")

    print("2) Entrenando modelo...")
    model = train_model(X_train, y_train)

    print("3) Evaluando modelo...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"    Accuracy: {metrics['accuracy']:.3f}")

    if metrics["accuracy"] < MIN_ACCURACY:
        raise ValueError(f"El modelo no alcanzó el mínimo: {MIN_ACCURACY}. No se guardará")

    print("4) Guardando el modelo...")
    save_model(model, metrics)
    print(f"    Modelo: {MODEL_PATH}")
    print(f"    Metricas: {METRICS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()