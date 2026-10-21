from fastapi.testclient import TestClient

from iris_mlops.api import app


client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_invalid_input():
    sample = {"sepal_length": -5, "sepal_width": 5, "petal_length": 20, "petal_width": 2}
    response = client.post("/predict", json=sample)
    assert response.status_code == 422