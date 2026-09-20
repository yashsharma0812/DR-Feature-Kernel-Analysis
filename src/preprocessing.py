from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_split(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> DataSplit:
    # Exact duplicate feature vectors are kept in one partition to prevent a
    # copy in the held-out set from being seen during training. Five folds
    # provide the requested approximately 80/20 stratified holdout.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return DataSplit(X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx])
