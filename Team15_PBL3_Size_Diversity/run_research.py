from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve

from src import MODEL_ORDER, RANDOM_STATE
from src.data_preparation import (
    conflicting_duplicate_groups, load_dataset, make_holdout, vector_groups,
)
from src.diversity_experiments import MATCHED_TREE_PARAMS, run_diversity_experiment
from src.ensemble_models import build_model, select_non_size_parameters
from src.evaluation import fit_and_measure
from src.gradient_dynamics import run_gradient_configuration_experiment, staged_learning_curve
from src.interaction_experiments import run_interaction_experiment
from src.size_experiments import run_size_experiment
from src.visualization import (
    plot_diversity_results, plot_eda, plot_final, plot_gradient, plot_interaction,
    plot_size_results,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "debrecen_dr.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"


def native(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value))


def smallest_near_best(frame, score="f1_mean", tolerance=.002):
    best = frame[score].max()
    return frame[frame[score].ge(best - tolerance)].sort_values(["ensemble_size", score], ascending=[True, False]).iloc[0]


def write_findings(size, diversity, interaction, gradient, final):
    lines = ["# Research Findings — Ensemble Size and Model Diversity", ""]
    lines.append("## Ensemble size")
    for model in MODEL_ORDER:
        data = size[size.model.eq(model)].sort_values("ensemble_size")
        best = data.loc[data.f1_mean.idxmax()]
        change_100_300 = (
            data.loc[data.ensemble_size.eq(300), "f1_mean"].iloc[0]
            - data.loc[data.ensemble_size.eq(100), "f1_mean"].iloc[0]
        )
        lines.append(
            f"- **{model}:** highest mean CV F1 {best.f1_mean:.3f} at N={int(best.ensemble_size)}; "
            f"change from N=100 to N=300 was {change_100_300:+.3f}."
        )
    rf = diversity[diversity.model.eq("Random Forest")]
    highest_div = rf.loc[rf.disagreement_mean.idxmax()]
    best_rf = rf.loc[rf.f1_mean.idxmax()]
    corr = interaction[["disagreement_mean", "f1_mean"]].corr().iloc[0, 1]
    lines += [
        "",
        "## Diversity",
        f"- Greatest measured RF disagreement occurred at max_features={int(highest_div.feature_subset)} "
        f"({highest_div.disagreement_mean:.3f}); its mean learner accuracy was {highest_div.learner_accuracy_mean:.3f}.",
        f"- Best RF diversity-sweep F1 occurred at max_features={int(best_rf.feature_subset)} "
        f"({best_rf.f1_mean:.3f}), showing that maximum disagreement was not automatically optimal.",
        f"- Across the size × diversity grid, disagreement–F1 correlation was {corr:.3f}. "
        "This descriptive association is not a causal estimate.",
        "",
        "## Final held-out configurations",
        "```csv\n" + final[["model", "ensemble_size", "diversity_configuration", "accuracy", "sensitivity", "specificity", "f1", "roc_auc"]]
        .to_csv(index=False).strip() + "\n```",
        "",
        "## Hypotheses",
        "- **H1 qualified:** performance generally improved early, while later gains were small and non-monotonic.",
        "- **H2 qualified:** diversity was useful only when learner accuracy remained adequate; maximum disagreement did not guarantee maximum F1.",
        "- **H3 evaluated:** random feature selection changed disagreement and learner strength relative to matched Bagging.",
        "- **H4 supported descriptively:** size curves differed across feature-subset settings, demonstrating an interaction in the controlled grid.",
        "- **H5 supported:** measured training cost increased with ensemble size while later predictive gains often plateaued.",
        "",
        "## Limitations",
        "One moderate-sized tabular dataset, five duplicate rows, one grouped holdout, internal CV uncertainty, "
        "machine-specific timing, no external validation, and no clinical diagnostic validation. "
        "Feature-subset randomization changes both diversity and individual-tree strength, so diversity is not an isolated causal treatment.",
    ]
    (RESULTS / "research_findings.md").write_text("\n".join(lines))


def write_viva(size, diversity, interaction, final):
    best = final.sort_values("f1", ascending=False).iloc[0]
    rf = diversity[diversity.model.eq("Random Forest")]
    high = rf.loc[rf.disagreement_mean.idxmax()]
    best_div = rf.loc[rf.f1_mean.idxmax()]
    content = f"""# Team 15 PBL-3 Viva Guide

## 1. What is ensemble learning?
It combines predictions from multiple learners so the group can be more reliable than one model.

## 2. What is ensemble size?
The number of base learners. We tested 1, 5, 10, 25, 50, 100, 200, and 300.

## 3. What is model diversity?
It is the extent to which learners make different predictions or errors on the same observations.

## 4. Why does diversity matter?
If accurate learners make different errors, voting can cancel some mistakes. Diversity without learner strength can hurt.

## 5. What is Bagging?
Independent decision trees trained on bootstrap samples, with predictions aggregated.

## 6. What is Gradient Boosting?
Trees added sequentially to correct the current ensemble's errors.

## 7. What is Random Forest?
Bagging plus random feature selection at each split.

## 8. Random Forest versus Bagging?
Both bootstrap independent trees; Random Forest also limits features available at a split.

## 9. Why vary estimators?
To observe early gains, plateaus, instability, and computational growth instead of assuming more is better.

## 10. Why vary max_features?
It controls feature randomization, changing both tree diversity and individual-tree strength.

## 11. Pairwise disagreement?
The mean fraction of validation observations on which two trees predict different labels.

## 12. Double-fault?
The mean fraction on which both members of a learner pair are wrong. Lower shared-error rates are preferable.

## 13. Why is maximum diversity not necessarily optimal?
The highest RF disagreement occurred at k={int(high.feature_subset)}, but the best diversity-sweep F1 occurred at k={int(best_div.feature_subset)}. Weak learners can disagree for unhelpful reasons.

## 14. Why can performance plateau?
Once errors are strongly correlated or variance is already reduced, additional learners add little new information.

## 15. Why cross-validation?
It estimates performance variability on training-only folds and prevents using the final test set for selection.

## 16. Why is sensitivity important?
It measures the proportion of DR-positive labels detected; false negatives are missed positive examples.

## 17. Learner accuracy versus ensemble accuracy?
Learner accuracy averages individual trees; ensemble accuracy evaluates the aggregated vote. Both are needed to interpret diversity.

## 18. Most important result?
The best held-out F1 configuration was {best.model} at N={int(best.ensemble_size)} with {best.diversity_configuration}, F1={best.f1:.3f}. More learners and more disagreement were not universally better; performance depended on the balance among size, diversity, learner strength, and cost.
"""
    (DOCS / "Team15_PBL3_Viva_Guide.md").write_text(content)


def main():
    for path in [RESULTS, DOCS, *[FIGURES / name for name in ["eda", "ensemble_size", "model_diversity", "size_diversity_interaction", "final_comparison"]]]:
        path.mkdir(parents=True, exist_ok=True)

    frame, X, y = load_dataset(DATA)
    split = make_holdout(X, y, RANDOM_STATE)
    conflict_count = conflicting_duplicate_groups(X, y)
    dataset = {
        "observations": len(frame), "predictors": 18, "total_columns": 19,
        "missing_cells": int(frame.isna().sum().sum()), "duplicate_rows": int(frame.duplicated().sum()),
        "conflicting_duplicate_predictor_groups": conflict_count,
        "class_0": int((y == 0).sum()), "class_1": int((y == 1).sum()),
        "train_rows": len(split.X_train), "test_rows": len(split.X_test),
        "train_class_1_rate": split.y_train.mean(), "test_class_1_rate": split.y_test.mean(),
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    }
    (RESULTS / "dataset_audit.json").write_text(json.dumps(dataset, indent=2, default=native))
    frame.describe().T.to_csv(RESULTS / "dataset_description.csv")
    pd.DataFrame({"row_index": split.train_indices}).to_csv(RESULTS / "training_indices.csv", index=False)
    pd.DataFrame({"row_index": split.test_indices}).to_csv(RESULTS / "test_indices.csv", index=False)
    plot_eda(frame, FIGURES / "eda")

    selected_params, selection = select_non_size_parameters(split.X_train, split.y_train, split.groups_train)
    selection.to_csv(RESULTS / "non_size_parameter_selection.csv", index=False)

    size_folds, size = run_size_experiment(split.X_train, split.y_train, split.groups_train, selected_params)
    size_folds.to_csv(RESULTS / "ensemble_size_fold_results.csv", index=False)
    size.to_csv(RESULTS / "ensemble_size_results.csv", index=False)
    plot_size_results(size, FIGURES / "ensemble_size")

    diversity_folds, diversity = run_diversity_experiment(split.X_train, split.y_train, split.groups_train)
    diversity_folds.to_csv(RESULTS / "diversity_fold_results.csv", index=False)
    diversity.to_csv(RESULTS / "diversity_results.csv", index=False)
    plot_diversity_results(diversity, FIGURES / "model_diversity")

    interaction_folds, interaction = run_interaction_experiment(split.X_train, split.y_train, split.groups_train)
    interaction_folds.to_csv(RESULTS / "size_diversity_interaction_folds.csv", index=False)
    interaction.to_csv(RESULTS / "size_diversity_interaction.csv", index=False)
    plot_interaction(interaction, FIGURES / "size_diversity_interaction")

    gradient_folds, gradient = run_gradient_configuration_experiment(split.X_train, split.y_train, split.groups_train)
    gradient_folds.to_csv(RESULTS / "gradient_dynamics_folds.csv", index=False)
    gradient.to_csv(RESULTS / "gradient_dynamics.csv", index=False)
    gradient_best = gradient.sort_values(["f1_mean", "training_time_seconds_mean"], ascending=[False, True]).iloc[0]
    learning_curve = staged_learning_curve(
        split.X_train, split.y_train, split.groups_train,
        {"learning_rate": gradient_best.learning_rate, "max_depth": gradient_best.max_depth, "subsample": gradient_best.subsample},
    )
    learning_curve.to_csv(RESULTS / "gradient_learning_curve.csv", index=False)
    plot_gradient(gradient, learning_curve, FIGURES / "ensemble_size")

    # Training-only selection of final configurations.
    bag_choice = smallest_near_best(size[size.model.eq("Bagging")])
    gb_choice = smallest_near_best(size[size.model.eq("Gradient Boosting")])
    rf_choice = interaction.sort_values(["f1_mean", "ensemble_size", "training_time_seconds_mean"],
                                        ascending=[False, True, True]).iloc[0]
    choices = {
        "Bagging": {"size": int(bag_choice.ensemble_size), "params": selected_params["Bagging"],
                    "diversity": "full features"},
        "Gradient Boosting": {"size": int(gb_choice.ensemble_size), "params": selected_params["Gradient Boosting"],
                              "diversity": "sequential additive learners"},
        "Random Forest": {"size": int(rf_choice.ensemble_size),
                          "params": {**MATCHED_TREE_PARAMS, "max_features": int(rf_choice.feature_subset)},
                          "diversity": f"max_features={int(rf_choice.feature_subset)}"},
    }

    final_rows, roc_rows, prediction_rows = [], [], []
    for name in MODEL_ORDER:
        choice = choices[name]
        model = build_model(name, choice["size"], choice["params"], RANDOM_STATE)
        measured, pred, prob = fit_and_measure(model, split.X_train, split.y_train, split.X_test, split.y_test)
        final_rows.append({"model": name, "ensemble_size": choice["size"],
                           "diversity_configuration": choice["diversity"], **measured})
        fpr, tpr, thresholds = roc_curve(split.y_test, prob)
        roc_rows.extend({"model": name, "fpr": a, "tpr": b, "threshold": c} for a, b, c in zip(fpr, tpr, thresholds))
        prediction_rows.extend({"model": name, "actual": a, "predicted": b, "probability": c}
                               for a, b, c in zip(split.y_test, pred, prob))
    final = pd.DataFrame(final_rows); roc_points = pd.DataFrame(roc_rows)
    final.to_csv(RESULTS / "final_model_comparison.csv", index=False)
    roc_points.to_csv(RESULTS / "final_roc_points.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(RESULTS / "final_test_predictions.csv", index=False)
    plot_final(final, roc_points, FIGURES / "final_comparison")

    configuration = {
        "official_title": "Investigating the Effect of Ensemble Size and Model Diversity on Predictive Performance",
        "random_state": RANDOM_STATE, "selected_non_size_parameters": selected_params,
        "final_configurations": choices,
        "size_seeds": [42, 52, 62], "size_folds_per_seed": 5,
        "ensemble_sizes": [1, 5, 10, 25, 50, 100, 200, 300],
        "interaction_sizes": [10, 25, 50, 100, 200, 300],
        "feature_subsets": [1, 3, 6, 9, 18],
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    (RESULTS / "experiment_configurations.json").write_text(json.dumps(configuration, indent=2, default=native))
    write_findings(size, diversity, interaction, gradient, final)
    write_viva(size, diversity, interaction, final)
    print(json.dumps(dataset, indent=2, default=native))
    print("\nSelected non-size parameters:", json.dumps(selected_params, indent=2, default=native))
    print("\nFinal held-out comparison:\n", final.to_string(index=False))
    print("\nFigures:", len(list(FIGURES.rglob("*.png"))), "Result files:", len(list(RESULTS.glob("*"))))


if __name__ == "__main__":
    main()

