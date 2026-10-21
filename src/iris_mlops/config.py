from pathlib import Path

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "iris.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "iris_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Datos
FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]
TARGET = "species"

# Entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hiperparámetros
MODELS_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "random_state": RANDOM_STATE
}

# Calidad mínima aceptable
MIN_ACCURACY = 0.9