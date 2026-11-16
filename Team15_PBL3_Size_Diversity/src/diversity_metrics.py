from __future__ import annotations

import numpy as np


def learner_predictions(model, X, model_name):
    """Return binary predictions with shape learners × observations."""
    if model_name == "Bagging":
        predictions = []
        for estimator, feature_indices in zip(model.estimators_, model.estimators_features_):
            predictions.append(estimator.predict(X.iloc[:, feature_indices].to_numpy()))
        return np.asarray(predictions, dtype=int)
    if model_name == "Random Forest":
        matrix = X.to_numpy()
        return np.asarray([tree.predict(matrix) for tree in model.estimators_], dtype=int)
    raise ValueError("Tree-vote diversity is defined here only for Bagging and Random Forest.")


def diversity_summary(predictions: np.ndarray, y_true: np.ndarray) -> dict:
    """Efficient mean pairwise disagreement, double-fault, and learner accuracy."""
    learners, observations = predictions.shape
    if learners < 2:
        return {"disagreement": np.nan, "double_fault": np.nan,
                "error_correlation": np.nan, "learner_accuracy": np.mean(predictions == y_true)}
    pairs = learners * (learners - 1) / 2
    positive = predictions.sum(axis=0)
    disagreement = np.mean(positive * (learners - positive) / pairs)
    errors = predictions != y_true.reshape(1, -1)
    wrong = errors.sum(axis=0)
    double_fault = np.mean(wrong * (wrong - 1) / 2 / pairs)
    learner_accuracy = np.mean(predictions == y_true.reshape(1, -1))
    # Mean off-diagonal error-indicator correlation; constant learners are excluded.
    valid = errors.std(axis=1) > 0
    if valid.sum() > 1:
        correlation = np.corrcoef(errors[valid].astype(float))
        error_correlation = correlation[np.triu_indices_from(correlation, 1)].mean()
    else:
        error_correlation = np.nan
    return {
        "disagreement": float(disagreement), "double_fault": float(double_fault),
        "error_correlation": float(error_correlation), "learner_accuracy": float(learner_accuracy),
    }

