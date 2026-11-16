from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.evaluation import evaluate_model, feature_importance, fold_stability
from src.models import MODEL_ORDER, parameter_grids, tune_models
from src.preprocessing import FEATURES, feature_summary, load_dataset, make_holdout_split
from src.visualization import plot_eda, plot_model_results


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "debrecen_dr.csv"
RESULTS = ROOT / "results"
EDA_FIGURES = ROOT / "figures" / "eda"
MODEL_FIGURES = ROOT / "figures" / "model_comparison"
RANDOM_STATE = 42


def _native(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def _ensure_dataset():
    """Copy the existing raw Debrecen file without altering its contents."""
    if DATA.exists():
        return
    source = ROOT.parent / "data" / "Retinopathy_Debrecen.csv"
    if not source.exists():
        raise FileNotFoundError("Expected preserved Debrecen CSV in the parent repository data directory.")
    shutil.copy2(source, DATA)


def _conclusion(metrics: pd.DataFrame, folds: pd.DataFrame) -> str:
    ordered = metrics.set_index("model")
    best_accuracy = ordered.accuracy.idxmax()
    best_f1 = ordered.f1_score.idxmax()
    best_auc = ordered.roc_auc.idxmax()
    best_recall = ordered.recall_sensitivity.idxmax()
    fastest_train = ordered.training_time_seconds.idxmin()
    fastest_predict = ordered.prediction_time_seconds.idxmin()
    fold_summary = folds.groupby("model").f1_score.agg(["mean", "std"])
    most_stable = fold_summary["std"].idxmin()
    bag_f1 = ordered.loc["Bagging", "f1_score"]
    rf_f1 = ordered.loc["Random Forest", "f1_score"]
    boost_f1 = ordered.loc["Gradient Boosting", "f1_score"]
    overall = ordered.assign(
        balanced_rank=ordered[["f1_score", "roc_auc", "recall_sensitivity", "specificity"]]
        .rank(ascending=False).mean(axis=1)
    ).balanced_rank.idxmin()
    return f"""# PBL-3 Results and Research Inferences

## Central research question

Which of Bagged Decision Trees, standard Gradient Boosting, and Random Forest provides the most effective and reliable prediction of the Debrecen DR label under the same split and evaluation protocol?

## Held-out results

{metrics[["model", "accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc", "training_time_seconds", "prediction_time_ms_per_observation"]].to_csv(index=False)}

## Evidence-based answers

- Highest held-out accuracy: **{best_accuracy}** ({ordered.loc[best_accuracy, 'accuracy']:.3f}).
- Highest F1: **{best_f1}** ({ordered.loc[best_f1, 'f1_score']:.3f}).
- Highest ROC-AUC: **{best_auc}** ({ordered.loc[best_auc, 'roc_auc']:.3f}).
- Highest sensitivity / lowest false-negative rate: **{best_recall}** ({ordered.loc[best_recall, 'recall_sensitivity']:.3f} sensitivity).
- Lowest cross-validation F1 standard deviation: **{most_stable}** ({fold_summary.loc[most_stable, 'std']:.3f}).
- Fastest final fit: **{fastest_train}**; fastest held-out prediction: **{fastest_predict}** on this machine.
- Random Forest versus ordinary Bagging held-out F1: {rf_f1:.3f} versus {bag_f1:.3f}. This comparison isolates the observed effect of random feature selection only imperfectly because the tuned tree structures may also differ.
- Gradient Boosting versus Bagging held-out F1: {boost_f1:.3f} versus {bag_f1:.3f}. Sequential correction did not automatically guarantee superiority.
- A rank across F1, ROC-AUC, sensitivity, and specificity favors **{overall}**, but the conclusion should also consider stability and cost rather than treating one metric as decisive.

## Limitations

This is one small tabular dataset with image-derived features, five duplicate rows, no raw retinal photographs, one grouped holdout split, no external validation, and no patient demographics or treatment history. The class label indicates dataset-defined signs of DR; these experiments are not clinical validation. Timing is machine-specific, feature importance is model-dependent, and small metric differences may not establish a definitive winner.
"""


def main():
    for directory in [RESULTS, EDA_FIGURES, MODEL_FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
    _ensure_dataset()
    frame, X, y = load_dataset(DATA)
    summary = {
        "source": "UCI Diabetic Retinopathy Debrecen data set",
        "source_url": "https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen",
        "access_date": "2026-11-16",
        "observations": len(frame), "predictors": len(FEATURES),
        "total_columns": frame.shape[1], "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_0": int((y == 0).sum()), "class_1": int((y == 1).sum()),
        "class_ratio_positive_to_negative": float((y == 1).sum() / (y == 0).sum()),
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    }
    (RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    feature_summary(frame).to_csv(RESULTS / "feature_dictionary.csv", index=False)
    frame.describe().T.to_csv(RESULTS / "descriptive_statistics.csv")
    frame.isna().sum().rename("missing_count").to_csv(RESULTS / "missing_values.csv")
    plot_eda(frame, EDA_FIGURES)

    split = make_holdout_split(X, y, RANDOM_STATE)
    split_summary = pd.DataFrame([
        {"partition": "Training", "observations": len(split.X_train), "class_0": int((split.y_train == 0).sum()), "class_1": int((split.y_train == 1).sum())},
        {"partition": "Held-out test", "observations": len(split.X_test), "class_0": int((split.y_test == 0).sum()), "class_1": int((split.y_test == 1).sum())},
    ])
    split_summary.to_csv(RESULTS / "data_split_summary.csv", index=False)
    pd.DataFrame({"row_index": split.train_indices, "partition": "train"}).to_csv(RESULTS / "training_indices.csv", index=False)
    pd.DataFrame({"row_index": split.test_indices, "partition": "test"}).to_csv(RESULTS / "test_indices.csv", index=False)

    tuned = tune_models(split.X_train, split.y_train, split.groups_train)
    all_metrics, all_rocs, all_predictions, all_folds, all_importance = [], [], [], [], []
    best_parameters = {}
    for name in MODEL_ORDER:
        result = tuned[name]
        best_parameters[name] = {
            "parameters": result.best_params,
            "training_cv_best_f1": result.best_cv_f1,
            "candidates": len(result.search_results),
        }
        safe_name = name.lower().replace(" ", "_")
        result.search_results.assign(params=result.search_results.params.astype(str)).to_csv(
            RESULTS / f"{safe_name}_hyperparameter_search.csv", index=False
        )
        fitted, metrics, roc, predictions = evaluate_model(
            name, result.estimator, split.X_train, split.y_train, split.X_test, split.y_test
        )
        folds = fold_stability(name, result.estimator, split.X_train, split.y_train, split.groups_train)
        importance = feature_importance(name, fitted, FEATURES)
        all_metrics.append(metrics); all_rocs.append(roc); all_predictions.append(predictions)
        all_folds.append(folds); all_importance.append(importance)

    metrics = pd.DataFrame(all_metrics)[["model", "accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc", "false_negative_rate", "true_negative", "false_positive", "false_negative", "true_positive", "train_accuracy", "train_f1", "train_roc_auc", "training_time_seconds", "prediction_time_seconds", "prediction_time_ms_per_observation", "serialized_size_kb"]]
    folds = pd.concat(all_folds, ignore_index=True)
    roc_data = pd.concat(all_rocs, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)
    fold_summary = folds.groupby("model").agg(
        cv_mean_f1=("f1_score", "mean"), cv_sd_f1=("f1_score", "std"),
        cv_mean_accuracy=("accuracy", "mean"), cv_sd_accuracy=("accuracy", "std"),
        cv_mean_roc_auc=("roc_auc", "mean"), cv_sd_roc_auc=("roc_auc", "std"),
    ).reset_index()
    metrics = metrics.merge(fold_summary, on="model", validate="one_to_one")
    metrics.to_csv(RESULTS / "model_comparison.csv", index=False)
    folds.to_csv(RESULTS / "cross_validation_results.csv", index=False)
    roc_data.to_csv(RESULTS / "roc_curve_points.csv", index=False)
    predictions.to_csv(RESULTS / "heldout_predictions.csv", index=False)
    importance.to_csv(RESULTS / "feature_importances.csv", index=False)
    (RESULTS / "best_hyperparameters.json").write_text(json.dumps(best_parameters, indent=2, default=_native))
    plot_model_results(metrics, folds, roc_data, importance, MODEL_FIGURES)
    (RESULTS / "results_interpretation.md").write_text(_conclusion(metrics, folds))
    reproducibility = {
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "model_search_candidates_each": {k: int(np.prod([len(v) for v in grid.values()])) for k, grid in parameter_grids().items()},
    }
    (RESULTS / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nSelected hyperparameters:\n", json.dumps(best_parameters, indent=2, default=_native))
    print("\nHeld-out comparison:\n", metrics.to_string(index=False))
    print(f"\nGenerated {len(list(EDA_FIGURES.glob('*.png')))} EDA and {len(list(MODEL_FIGURES.glob('*.png')))} model figures.")


if __name__ == "__main__":
    main()

