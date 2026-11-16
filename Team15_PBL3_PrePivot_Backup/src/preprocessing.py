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
    "exudate8", "macula_opticdisc_distance", "opticdisc_diameter",
    "am_fm_classification",
]

FEATURE_DESCRIPTIONS = {
    "quality": "Image-quality assessment indicator",
    "pre_screening": "Pre-screening abnormality indicator",
    **{f"ma{i}": f"Microaneurysm detector output at confidence level {i}" for i in range(1, 7)},
    **{f"exudate{i}": "Normalized exudate detector output" for i in [1, 2, 3, 5, 6, 7, 8]},
    "macula_opticdisc_distance": "Normalized macula-to-optic-disc distance",
    "opticdisc_diameter": "Optic-disc diameter measurement",
    "am_fm_classification": "AM/FM-based classifier output",
}


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


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load and validate the preserved Debrecen CSV."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    if list(frame.columns) != FEATURES + [TARGET]:
        raise ValueError(
            "Unexpected Debrecen schema. Expected 18 documented predictors followed by 'class'."
        )
    if frame.shape != (1151, 19):
        raise ValueError(f"Expected 1,151 rows and 19 total columns; found {frame.shape}.")
    if frame.select_dtypes(exclude="number").shape[1]:
        raise TypeError("All Debrecen predictors and the target must be numeric.")
    if frame.isna().any().any():
        raise ValueError("The supplied Debrecen file unexpectedly contains missing values.")
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain exactly 0 (no signs) and 1 (signs of DR).")
    return frame, frame[FEATURES].copy(), frame[TARGET].astype(int).copy()


def duplicate_groups(X: pd.DataFrame) -> np.ndarray:
    """Assign identical feature vectors to one group to prevent copy leakage."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def make_holdout_split(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> DataSplit:
    """Create a reproducible approximately 80/20 stratified group split."""
    groups = duplicate_groups(X)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    split = DataSplit(
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        groups[train_idx], groups[test_idx], train_idx, test_idx,
    )
    if set(split.groups_train).intersection(split.groups_test):
        raise AssertionError("Duplicate feature-vector group leaked across the holdout split.")
    return split


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a presentation-ready data dictionary and quality summary."""
    rows = []
    for feature in FEATURES:
        values = frame[feature]
        rows.append({
            "feature": feature,
            "description": FEATURE_DESCRIPTIONS[feature],
            "dtype": str(values.dtype),
            "missing": int(values.isna().sum()),
            "unique_values": int(values.nunique()),
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
        })
    return pd.DataFrame(rows)

