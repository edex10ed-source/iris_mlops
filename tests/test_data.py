from iris_mlops.config import FEATURES, TARGET
from iris_mlops.data import load_data, split_data, validate_data


def test_data_shape():
    df = load_data()
    assert df.shape == (150, 5)
    assert list(df.columns) == FEATURES + [TARGET]


def test_data_three_species():
    df = load_data()
    assert set(df[TARGET]) == {"setosa", "versicolor", "virginica"}