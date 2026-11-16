from __future__ import annotations

import pandas as pd

from .data_preparation import cv_splits
from .diversity_metrics import diversity_summary, learner_predictions
from .ensemble_models import build_model
from .evaluation import fit_and_measure


MATCHED_TREE_PARAMS = {"max_depth": 12, "min_samples_leaf": 3, "max_samples": .8}


def _evaluate_configuration(model_name, max_features, ensemble_size, X, y, groups, seed=42):
    rows = []
    params = dict(MATCHED_TREE_PARAMS)
    if model_name == "Random Forest":
        params["max_features"] = max_features
    for fold, (fit_idx, val_idx) in enumerate(cv_splits(X, y, groups, seed), start=1):
        model = build_model(model_name, ensemble_size, params, seed)
        measured, _, _ = fit_and_measure(model, X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx])
        predictions = learner_predictions(model, X.iloc[val_idx], model_name)
        diversity = diversity_summary(predictions, y.iloc[val_idx].to_numpy())
        rows.append({
            "model": model_name, "feature_subset": 18 if model_name == "Bagging" else max_features,
            "ensemble_size": ensemble_size, "fold": fold, **diversity, **measured,
        })
    return rows


def run_diversity_experiment(X, y, groups, feature_subsets=(1, 3, 6, 9, 18), ensemble_size=100):
    rows = _evaluate_configuration("Bagging", 18, ensemble_size, X, y, groups)
    for subset in feature_subsets:
        rows.extend(_evaluate_configuration("Random Forest", subset, ensemble_size, X, y, groups))
    folds = pd.DataFrame(rows)
    metrics = ["disagreement", "double_fault", "error_correlation", "learner_accuracy",
               "accuracy", "f1", "roc_auc", "sensitivity", "training_time_seconds"]
    summary = folds.groupby(["model", "feature_subset", "ensemble_size"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    return folds, summary

