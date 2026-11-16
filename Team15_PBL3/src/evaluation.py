from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from . import RANDOM_STATE


def classification_metrics(y_true, prediction, probability) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall_sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1_score": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def evaluate_model(name, estimator, X_train, y_train, X_test, y_test):
    """Fit once on the full training set and evaluate once on the untouched holdout."""
    model = clone(estimator)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probability = model.predict_proba(X_test)[:, 1]
    prediction = model.predict(X_test)
    prediction_seconds = time.perf_counter() - start
    train_prediction = model.predict(X_train)
    train_probability = model.predict_proba(X_train)[:, 1]
    metrics = classification_metrics(y_test, prediction, probability)
    train_metrics = classification_metrics(y_train, train_prediction, train_probability)
    metrics.update({
        "model": name,
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1_score"],
        "train_roc_auc": train_metrics["roc_auc"],
        "training_time_seconds": training_seconds,
        "prediction_time_seconds": prediction_seconds,
        "prediction_time_ms_per_observation": 1000 * prediction_seconds / len(X_test),
        "serialized_size_kb": len(pickle.dumps(model)) / 1024,
    })
    fpr, tpr, thresholds = roc_curve(y_test, probability)
    roc = pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr, "threshold": thresholds})
    predictions = pd.DataFrame({
        "model": name, "actual": np.asarray(y_test),
        "predicted": prediction, "probability_dr": probability,
    })
    return model, metrics, roc, predictions


def fold_stability(name, estimator, X_train, y_train, groups_train) -> pd.DataFrame:
    """Measure held-out fold performance using duplicate-safe training-only folds."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train), start=1):
        model = clone(estimator).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        pred = model.predict(X_train.iloc[val_idx])
        prob = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        row = classification_metrics(y_train.iloc[val_idx], pred, prob)
        row.update({"model": name, "fold": fold, "n_validation": len(val_idx)})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_importance(name, model, feature_names: list[str]) -> pd.DataFrame:
    if name == "Bagging":
        importance = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    else:
        importance = model.feature_importances_
    return pd.DataFrame({"model": name, "feature": feature_names, "importance": importance})

