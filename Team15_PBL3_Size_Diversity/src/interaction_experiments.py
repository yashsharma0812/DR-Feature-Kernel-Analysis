from __future__ import annotations

import pandas as pd

from . import FEATURE_SUBSETS, INTERACTION_SIZES
from .data_preparation import cv_splits
from .diversity_experiments import MATCHED_TREE_PARAMS
from .diversity_metrics import diversity_summary, learner_predictions
from .ensemble_models import build_model
from .evaluation import fit_and_measure


def run_interaction_experiment(X, y, groups, seed=42):
    """Random-Forest size × feature-subset grid on identical training-only folds."""
    rows = []
    splits = cv_splits(X, y, groups, seed)
    for size in INTERACTION_SIZES:
        for subset in FEATURE_SUBSETS:
            params = {**MATCHED_TREE_PARAMS, "max_features": subset}
            for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
                model = build_model("Random Forest", size, params, seed)
                measured, _, _ = fit_and_measure(model, X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx])
                diversity = diversity_summary(
                    learner_predictions(model, X.iloc[val_idx], "Random Forest"), y.iloc[val_idx].to_numpy()
                )
                rows.append({"ensemble_size": size, "feature_subset": subset, "fold": fold, **diversity, **measured})
    folds = pd.DataFrame(rows)
    metrics = ["disagreement", "double_fault", "learner_accuracy", "accuracy", "f1", "roc_auc",
               "sensitivity", "training_time_seconds", "prediction_ms_per_observation", "train_f1"]
    summary = folds.groupby(["ensemble_size", "feature_subset"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    summary["generalization_gap"] = summary.train_f1_mean - summary.f1_mean
    return folds, summary

