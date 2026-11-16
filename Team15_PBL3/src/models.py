from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from . import RANDOM_STATE


MODEL_ORDER = ["Bagging", "Gradient Boosting", "Random Forest"]


@dataclass
class TunedModel:
    name: str
    estimator: object
    best_params: dict
    best_cv_f1: float
    search_results: pd.DataFrame


def initial_models() -> dict[str, object]:
    """Return the three finalized approaches with comparable baseline settings."""
    return {
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            n_estimators=100, bootstrap=True, max_features=1.0,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=2,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, bootstrap=True, max_features="sqrt",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def parameter_grids() -> dict[str, dict]:
    """Use 24 candidates per model for a comparable tuning budget."""
    return {
        "Bagging": {
            "n_estimators": [100, 300],
            "max_samples": [0.8, 1.0],
            "estimator__max_depth": [None, 6, 12],
            "estimator__min_samples_leaf": [1, 3],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.1],
            "max_depth": [1, 2, 3],
            "subsample": [0.8, 1.0],
        },
        "Random Forest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.75],
        },
    }


def tune_models(
    X_train: pd.DataFrame, y_train: pd.Series, groups_train: np.ndarray
) -> dict[str, TunedModel]:
    """Tune all models on training data only using identical grouped folds."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    for name in MODEL_ORDER:
        search = GridSearchCV(
            initial_models()[name], parameter_grids()[name], scoring="f1",
            cv=cv, n_jobs=-1, refit=True, return_train_score=True,
        )
        search.fit(X_train, y_train, groups=groups_train)
        columns = [
            "params", "mean_train_score", "std_train_score",
            "mean_test_score", "std_test_score", "rank_test_score",
            "mean_fit_time", "mean_score_time",
        ]
        results = pd.DataFrame(search.cv_results_)[columns].sort_values("rank_test_score")
        tuned[name] = TunedModel(
            name=name, estimator=clone(search.best_estimator_),
            best_params=search.best_params_, best_cv_f1=float(search.best_score_),
            search_results=results,
        )
    return tuned

