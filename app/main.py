"""End-to-end pipeline for Random Forest Regression on California Housing."""
from __future__ import annotations

from typing import Any

from app.data import load_data
from app.evaluate import regression_metrics
from app.model import build_model
from app.preprocess import split_data
from app.visualize import plot_feature_importance, plot_predictions


def run_pipeline(scale_features: bool = False) -> dict[str, Any]:
    """
    Train, evaluate, and visualize a Random Forest Regressor.

    Parameters
    ----------
    scale_features : bool, optional
        Apply standard scaling. Random forests are scale-invariant, but this
        is useful when comparing against models that do care about feature
        magnitude.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the trained model, metrics, and output paths.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, scale=scale_features)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = regression_metrics(y_test, predictions)

    importance_path = plot_feature_importance(X.columns, model.feature_importances_)
    preds_path = plot_predictions(y_test, predictions)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance_path": importance_path,
        "prediction_plot_path": preds_path,
    }


def print_report(results: dict[str, Any]) -> None:
    """Print evaluation metrics and feature importances in a friendly format."""
    metrics = results["metrics"]
    model = results["model"]

    print("\nRandom Forest Regression Results")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name.upper():4s}: {value:.4f}")

    print("\nTop Feature Importances")
    sorted_pairs = sorted(
        zip(model.feature_names_in_, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for feature, importance in sorted_pairs:
        print(f"{feature}: {importance:.4f}")

    print("\nPlots saved to:")
    print(f"- Feature importance: {results['feature_importance_path']}")
    print(f"- Predicted vs actual: {results['prediction_plot_path']}")


if __name__ == "__main__":
    results = run_pipeline(scale_features=False)
    print_report(results)
