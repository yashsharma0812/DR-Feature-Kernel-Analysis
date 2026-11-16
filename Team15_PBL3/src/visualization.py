from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbl3_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .models import MODEL_ORDER
from .preprocessing import FEATURES


COLORS = {
    "Bagging": "#4C78A8",
    "Gradient Boosting": "#F28E2B",
    "Random Forest": "#59A14F",
}


def configure_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11, "figure.dpi": 110})


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eda(frame: pd.DataFrame, output: Path):
    configure_style()
    counts = frame["class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["No signs of DR (0)", "Signs of DR (1)"], counts, color=["#7E8AA2", "#C44E52"])
    ax.bar_label(bars); ax.set(title="Debrecen Target-Class Distribution", ylabel="Observations", xlabel="Dataset label")
    save(fig, output / "01_class_distribution.png")

    selected = ["ma1", "ma2", "exudate1", "exudate7", "macula_opticdisc_distance", "opticdisc_diameter"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.ravel(), selected):
        sns.histplot(frame, x=feature, hue="class", element="step", stat="density",
                     common_norm=False, palette=["#7E8AA2", "#C44E52"], ax=ax)
        ax.set_title(feature.replace("_", " ").title())
    fig.suptitle("Selected Feature Distributions by DR Label", y=1.01)
    save(fig, output / "02_selected_feature_histograms.png")

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(frame[FEATURES].corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Predictor Pearson-Correlation Heatmap")
    save(fig, output / "03_feature_correlation_heatmap.png")

    long = frame.melt(id_vars="class", value_vars=selected, var_name="feature", value_name="value")
    long["standardized_value"] = long.groupby("feature").value.transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) else 0
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(long, x="feature", y="standardized_value", hue="class",
                palette=["#7E8AA2", "#C44E52"], showfliers=False, ax=ax)
    ax.set(title="Selected Standardized Features by DR Label", xlabel="Feature", ylabel="Standardized value")
    ax.tick_params(axis="x", rotation=20)
    save(fig, output / "04_feature_boxplots_by_target.png")

    missing = frame.isna().sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(missing.index, missing.values, color="#59A14F")
    ax.text(0.5, 0.5, "No missing values detected", transform=ax.transAxes,
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set(title="Missing-Value Audit", xlabel="Missing values", ylabel="Column", xlim=(0, 1))
    save(fig, output / "05_missing_values.png")

    standardized = (frame[FEATURES] - frame[FEATURES].mean()) / frame[FEATURES].std(ddof=0).replace(0, 1)
    standardized["class"] = frame["class"]
    means = standardized.groupby("class")[FEATURES].mean().T
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(means, annot=True, fmt=".2f", center=0, cmap="coolwarm", ax=ax)
    ax.set(title="Standardized Feature Means by DR Label", xlabel="DR class", ylabel="Feature")
    save(fig, output / "06_feature_target_comparison.png")

    outliers = {}
    for feature in FEATURES:
        q1, q3 = frame[feature].quantile([0.25, 0.75]); iqr = q3 - q1
        outliers[feature] = ((frame[feature] < q1 - 1.5 * iqr) | (frame[feature] > q3 + 1.5 * iqr)).mean()
    outlier_frame = pd.Series(outliers).sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=outlier_frame.values, y=outlier_frame.index, color="#4C78A8", ax=ax)
    ax.set(title="Highest IQR Outlier-Flag Rates (Retained)", xlabel="Proportion flagged", ylabel="Feature")
    save(fig, output / "07_outlier_flag_rates.png")


def plot_model_results(metrics, folds, roc_data, importance, output: Path):
    configure_style()
    palette = [COLORS[name] for name in MODEL_ORDER]
    ordered = metrics.set_index("model").loc[MODEL_ORDER].reset_index()
    core = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    long = ordered.melt(id_vars="model", value_vars=core, var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(long, x="metric", y="score", hue="model", hue_order=MODEL_ORDER,
                palette=COLORS, ax=ax)
    ax.set(title="Held-out Test Performance Comparison", xlabel="Metric", ylabel="Score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=18); ax.legend(title="Model", ncol=3)
    save(fig, output / "01_model_performance_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, MODEL_ORDER):
        row = ordered.loc[ordered.model.eq(name)].iloc[0]
        matrix = np.array([[row.true_negative, row.false_positive], [row.false_negative, row.true_positive]])
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap=sns.light_palette(COLORS[name], as_cmap=True),
                    cbar=False, xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"], ax=ax)
        ax.set_title(name)
    fig.suptitle("Held-out Confusion Matrices", y=1.02)
    save(fig, output / "02_confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    for name in MODEL_ORDER:
        data = roc_data.loc[roc_data.model.eq(name)]
        auc = ordered.loc[ordered.model.eq(name), "roc_auc"].iloc[0]
        ax.plot(data.fpr, data.tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Combined Held-out ROC Curves")
    ax.legend()
    save(fig, output / "03_combined_roc_curves.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(folds, x="model", y="f1_score", order=MODEL_ORDER, palette=COLORS, hue="model", legend=False, ax=ax)
    sns.stripplot(folds, x="model", y="f1_score", order=MODEL_ORDER, color="black", size=5, ax=ax)
    ax.set(title="Training-only Cross-validation F1 Distribution", xlabel="Model", ylabel="Fold F1")
    save(fig, output / "04_cross_validation_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(ordered, x="model", y="training_time_seconds", order=MODEL_ORDER, palette=COLORS, hue="model", legend=False, ax=axes[0])
    axes[0].set(title="Final Model Training Time", xlabel="Model", ylabel="Seconds")
    sns.barplot(ordered, x="model", y="prediction_time_ms_per_observation", order=MODEL_ORDER, palette=COLORS, hue="model", legend=False, ax=axes[1])
    axes[1].set(title="Prediction Time per Test Observation", xlabel="Model", ylabel="Milliseconds")
    for ax in axes: ax.tick_params(axis="x", rotation=15)
    save(fig, output / "05_computational_efficiency.png")

    top = importance.groupby("feature").importance.mean().nlargest(10).index
    table = importance.loc[importance.feature.isin(top)].pivot(index="feature", columns="model", values="importance").reindex(columns=MODEL_ORDER)
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set(title="Top Feature Importances Across Ensemble Models", xlabel="Model", ylabel="Feature")
    save(fig, output / "06_feature_importance_comparison.png")

    train_test = ordered.melt(id_vars="model", value_vars=["train_f1", "f1_score"], var_name="split", value_name="f1")
    train_test["split"] = train_test.split.map({"train_f1": "Training", "f1_score": "Held-out test"})
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(train_test, x="model", y="f1", hue="split", order=MODEL_ORDER, ax=ax)
    ax.set(title="Train-versus-Test F1 Comparison", xlabel="Model", ylabel="F1", ylim=(0, 1))
    save(fig, output / "07_train_test_comparison.png")

    stability = folds.groupby("model").agg(cv_mean_f1=("f1_score", "mean"), cv_sd_f1=("f1_score", "std")).reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stability.index, stability.cv_mean_f1, yerr=stability.cv_sd_f1,
           color=palette, capsize=6)
    ax.set(title="Cross-validation Mean F1 ± Standard Deviation", xlabel="Model", ylabel="F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=12)
    save(fig, output / "08_cv_mean_stability.png")

