from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from iris_mlops.config import DATA_PATH, FEATURES, RANDOM_STATE, TARGET, TEST_SIZE


def download_data() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = FEATURES + [TARGET]
    df[TARGET] = df[TARGET].map(dict(enumerate(iris.target_names)))

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def validate_data(df: pd.DataFrame) -> None:
    columnas_perdidas = set(FEATURES + [TARGET]) - set(df.columns)
    if columnas_perdidas:
        raise ValueError(f"Faltan columnas: {columnas_perdidas}")

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Hay valores nulos en los datos")

    if (df[FEATURES] <= 0).any().any():
        raise ValueError("Las medidas de la flor deben ser mayores a 0")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        download_data()
    df = pd.read_csv(DATA_PATH)
    validate_data(df)
    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def main() -> None:
    df = download_data()
    validate_data(df)
    print(f"Datos guardados en: {DATA_PATH}")
    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")
    print("Cantidad de flores por especie:")
    print(df[TARGET].value_counts().to_string())


if __name__ == "__main__":
    main()