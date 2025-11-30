"""Visualization helpers for Random Forest Regression."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)


def plot_feature_importance(features: Iterable[str], importances: Iterable[float]) -> Path:
    """
    Plot feature importances as a bar chart and save as SVG.

    Parameters
    ----------
    features : Iterable[str]
        Feature names.
    importances : Iterable[float]
        Importance scores from the model.

    Returns
    -------
    Path
        Path to the saved SVG file.
    """
    df = pd.DataFrame({"feature": list(features), "importance": list(importances)})
    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="importance", y="feature", palette="viridis")
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    output_path = EXAMPLES_DIR / "feature_importance.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_predictions(y_true, y_pred) -> Path:
    """
    Plot predicted vs actual scatter and save as SVG.

    Returns
    -------
    Path
        Path to the saved SVG file.
    """
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    plt.tight_layout()

    output_path = EXAMPLES_DIR / "predicted_vs_actual.svg"
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


__all__ = ["plot_feature_importance", "plot_predictions"]
