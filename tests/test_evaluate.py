import numpy as np

from app.evaluate import regression_metrics


def test_regression_metrics_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 2.8])

    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mse", "mae", "rmse", "r2"}
    assert metrics["mse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1 <= metrics["r2"] <= 1
