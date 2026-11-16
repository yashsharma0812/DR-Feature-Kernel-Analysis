from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


def metrics(y_true, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_and_measure(model, X_fit, y_fit, X_eval, y_eval):
    start = time.perf_counter(); model.fit(X_fit, y_fit); train_time = time.perf_counter() - start
    start = time.perf_counter(); prob = model.predict_proba(X_eval)[:, 1]; pred = model.predict(X_eval); pred_time = time.perf_counter() - start
    result = metrics(y_eval, pred, prob)
    train_pred = model.predict(X_fit); train_prob = model.predict_proba(X_fit)[:, 1]
    result.update({
        "train_f1": f1_score(y_fit, train_pred),
        "train_accuracy": accuracy_score(y_fit, train_pred),
        "training_time_seconds": train_time,
        "prediction_time_seconds": pred_time,
        "prediction_ms_per_observation": 1000 * pred_time / len(X_eval),
    })
    return result, pred, prob

