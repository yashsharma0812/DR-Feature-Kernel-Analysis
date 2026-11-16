from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


TARGET = "class"
FEATURES = [
    "quality", "pre_screening", "ma1", "ma2", "ma3", "ma4", "ma5", "ma6",
    "exudate1", "exudate2", "exudate3", "exudate5", "exudate6", "exudate7",
    "exudate8", "macula_opticdisc_distance", "opticdisc_diameter", "am_fm_classification",
]


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: np.ndarray
    groups_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def load_dataset(path: Path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    if list(frame.columns) != FEATURES + [TARGET]:
        raise ValueError("Expected the verified 18-predictor Debrecen schema plus target 'class'.")
    if frame.shape != (1151, 19):
        raise ValueError(f"Expected shape (1151, 19), found {frame.shape}.")
    if frame.isna().any().any():
        raise ValueError("The verified source unexpectedly contains missing values.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain exactly 0 and 1.")
    X, y = frame[FEATURES].copy(), frame[TARGET].astype(int).copy()
    return frame, X, y


def vector_groups(X: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def conflicting_duplicate_groups(X: pd.DataFrame, y: pd.Series) -> int:
    temp = X.copy(); temp["_group"] = vector_groups(X); temp["_target"] = y.to_numpy()
    return int(temp.groupby("_group")._target.nunique().gt(1).sum())


def make_holdout(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> DataSplit:
    groups = vector_groups(X)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    result = DataSplit(
        X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True),
        groups[train_idx], groups[test_idx], train_idx, test_idx,
    )
    if set(result.groups_train) & set(result.groups_test):
        raise AssertionError("Identical predictor vectors crossed the holdout boundary.")
    return result


def cv_splits(X, y, groups, seed=42):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))

