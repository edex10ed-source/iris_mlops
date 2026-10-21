from iris_mlops.config import MIN_ACCURACY
from iris_mlops.data import load_data, split_data
from iris_mlops.train import train_model, evaluate_model


def test_model_alcanza_accuracy():
    X_train, X_test, y_train, y_test = split_data(load_data())
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] >= MIN_ACCURACY