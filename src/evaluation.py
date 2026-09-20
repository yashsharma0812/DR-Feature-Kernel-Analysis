from time import perf_counter

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y_true, y_pred, scores) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_and_evaluate(model, X_train, y_train, X_test, y_test) -> tuple[dict, np.ndarray, np.ndarray]:
    start = perf_counter(); model.fit(X_train, y_train); train_time = perf_counter() - start
    start = perf_counter(); pred = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    inference_time = perf_counter() - start
    result = binary_metrics(y_test, pred, scores)
    result.update({"training_time_seconds": train_time, "inference_time_seconds": inference_time})
    return result, pred, scores
