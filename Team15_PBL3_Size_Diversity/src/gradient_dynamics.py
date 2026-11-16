from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .data_preparation import cv_splits
from .evaluation import fit_and_measure, metrics


def run_gradient_configuration_experiment(X, y, groups, seed=42):
    rows = []
    splits = cv_splits(X, y, groups, seed)
    for learning_rate in [.03, .05, .1]:
        for depth in [1, 2, 3]:
            for subsample in [.7, .85, 1.0]:
                for fold, (fit_idx, val_idx) in enumerate(splits, start=1):
                    model = GradientBoostingClassifier(
                        n_estimators=100, learning_rate=learning_rate, max_depth=depth,
                        subsample=subsample, random_state=seed,
                    )
                    measured, _, _ = fit_and_measure(model, X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx])
                    rows.append({"learning_rate": learning_rate, "max_depth": depth, "subsample": subsample,
                                 "fold": fold, **measured})
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["learning_rate", "max_depth", "subsample"])[
        ["accuracy", "f1", "roc_auc", "sensitivity", "train_f1", "training_time_seconds"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    summary["generalization_gap"] = summary.train_f1_mean - summary.f1_mean
    return folds, summary


def staged_learning_curve(X, y, groups, configuration, max_estimators=300, seed=42):
    fit_idx, val_idx = cv_splits(X, y, groups, seed)[0]
    model = GradientBoostingClassifier(
        n_estimators=max_estimators, learning_rate=configuration["learning_rate"],
        max_depth=int(configuration["max_depth"]), subsample=configuration["subsample"],
        random_state=seed,
    ).fit(X.iloc[fit_idx], y.iloc[fit_idx])
    rows = []
    for stage, (train_prob, val_prob) in enumerate(zip(
        model.staged_predict_proba(X.iloc[fit_idx]), model.staged_predict_proba(X.iloc[val_idx])
    ), start=1):
        if stage in {1, 5, 10, 25, 50, 100, 150, 200, 250, 300}:
            train = metrics(y.iloc[fit_idx], (train_prob[:, 1] >= .5).astype(int), train_prob[:, 1])
            valid = metrics(y.iloc[val_idx], (val_prob[:, 1] >= .5).astype(int), val_prob[:, 1])
            rows.append({"stage": stage, "train_f1": train["f1"], "validation_f1": valid["f1"],
                         "train_auc": train["roc_auc"], "validation_auc": valid["roc_auc"]})
    return pd.DataFrame(rows)

