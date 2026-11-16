from __future__ import annotations

import pandas as pd

from . import ENSEMBLE_SIZES, MODEL_ORDER
from .data_preparation import cv_splits
from .ensemble_models import build_model
from .evaluation import fit_and_measure


def run_size_experiment(X, y, groups, selected_params, seeds=(42, 52, 62)):
    """Change only N within each model family; report repeated grouped-CV results."""
    rows = []
    for seed in seeds:
        splits = cv_splits(X, y, groups, seed)
        for model_name in MODEL_ORDER:
            for size in ENSEMBLE_SIZES:
                for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
                    model = build_model(model_name, size, selected_params[model_name], seed)
                    measured, _, _ = fit_and_measure(
                        model, X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx]
                    )
                    rows.append({"model": model_name, "ensemble_size": size, "seed": seed, "fold": fold, **measured})
    folds = pd.DataFrame(rows)
    metrics = ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc",
               "train_f1", "training_time_seconds", "prediction_ms_per_observation"]
    aggregated = folds.groupby(["model", "ensemble_size"])[metrics].agg(["mean", "std"]).reset_index()
    aggregated.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in aggregated.columns]
    aggregated["generalization_gap"] = aggregated.train_f1_mean - aggregated.f1_mean
    return folds, aggregated

