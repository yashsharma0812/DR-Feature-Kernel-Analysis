from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from . import RANDOM_STATE


def build_model(name, n_estimators, params, seed=RANDOM_STATE):
    if name == "Bagging":
        tree = DecisionTreeClassifier(
            max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=seed,
        )
        return BaggingClassifier(
            estimator=tree, n_estimators=n_estimators, max_samples=params.get("max_samples", 1.0),
            max_features=1.0, bootstrap=True, random_state=seed, n_jobs=-1,
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=params.get("learning_rate", .1),
            max_depth=params.get("max_depth", 2), subsample=params.get("subsample", 1.0),
            random_state=seed,
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            max_features=params.get("max_features", "sqrt"), max_samples=params.get("max_samples"),
            bootstrap=True, random_state=seed, n_jobs=-1,
        )
    raise ValueError(name)


def select_non_size_parameters(X, y, groups):
    """Select non-size settings at N=100 using training-only grouped CV."""
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    specifications = {
        "Bagging": (
            BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=100,
                              max_features=1.0, bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1),
            {"estimator__max_depth": [None, 8], "estimator__min_samples_leaf": [1, 3], "max_samples": [.8, 1.0]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            {"learning_rate": [.05, .1], "max_depth": [1, 2], "subsample": [.8, 1.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1),
            {"max_depth": [None, 12], "min_samples_leaf": [1, 3], "max_features": ["sqrt", .5]},
        ),
    }
    selected, rows = {}, []
    for name, (model, grid) in specifications.items():
        search = GridSearchCV(model, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True, return_train_score=True)
        search.fit(X, y, groups=groups)
        params = dict(search.best_params_)
        if name == "Bagging":
            params = {
                "max_depth": params.pop("estimator__max_depth"),
                "min_samples_leaf": params.pop("estimator__min_samples_leaf"), **params,
            }
        selected[name] = params
        rows.append({"model": name, "best_cv_f1": search.best_score_, "parameters": str(params), "candidates": len(search.cv_results_["params"])})
    return selected, pd.DataFrame(rows)

