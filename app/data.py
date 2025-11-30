"""Data loading utilities for the California Housing dataset."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    dataset = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = dataset.frame.drop(columns=["MedHouseVal"])
    y: pd.Series = dataset.frame["MedHouseVal"]
    return X, y


__all__ = ["load_data"]
