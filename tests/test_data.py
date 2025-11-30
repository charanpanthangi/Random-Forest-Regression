import pandas as pd

from app.data import load_data


def test_load_data_shapes():
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)
    assert X.shape[1] == 8  # California Housing has 8 features
