"""Model construction utilities."""
from __future__ import annotations

from typing import Optional

from sklearn.ensemble import RandomForestRegressor


def build_model(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    max_features: str | int | float | None = 1.0,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Create a configured RandomForestRegressor.

    Random forests average the predictions of many decision trees trained on
    bootstrap samples. Random feature selection at each split makes trees less
    correlated, so the ensemble reduces variance compared with a single tree.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : Optional[int]
        Maximum tree depth; None allows full growth until pure leaves.
    max_features : str | int | float | None
        Number of features to consider at each split; lower values increase
        diversity across trees.
    min_samples_split : int
        Minimum samples required to split an internal node.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Configured model ready to fit.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )


__all__ = ["build_model"]
