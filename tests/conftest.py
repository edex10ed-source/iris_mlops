import pytest

from iris_mlops.config import MODEL_PATH
from iris_mlops.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train_main()