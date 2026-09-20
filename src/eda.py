from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .visualization import save_figure


def heatmap(matrix, title, path, center=0, cmap="coolwarm"):
    size = min(14, max(8, .55 * len(matrix.columns)))
    fig, ax = plt.subplots(figsize=(size, size * .85))
    sns.heatmap(matrix, cmap=cmap, center=center, square=True, ax=ax,
                cbar_kws={"shrink": .75}, xticklabels=True, yticklabels=True)
    ax.set_title(title); ax.tick_params(axis="x", rotation=55, labelsize=8); ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, path)


def plot_eda(X: pd.DataFrame, y: pd.Series, figures: Path) -> None:
    counts = y.value_counts().sort_index(); pct = y.value_counts(normalize=True).sort_index() * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["DR=0", "DR=1"], counts, color=["#4C78A8", "#E45756"])
    axes[0].set(title="Target Counts", ylabel="Observations", xlabel="Dataset label")
    axes[1].bar(["DR=0", "DR=1"], pct, color=["#4C78A8", "#E45756"])
    axes[1].set(title="Target Percentages", ylabel="Percent (%)", xlabel="Dataset label")
    for ax in axes:
        for p in ax.patches: ax.annotate(f"{p.get_height():.1f}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
    save_figure(fig, figures / "01_target_distribution.png")

    ncols, nrows = 3, int(np.ceil(X.shape[1] / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4*nrows)); axes = np.ravel(axes)
    data = X.assign(class_label=y.map({0: "DR=0", 1: "DR=1"}))
    for ax, col in zip(axes, X.columns):
        sns.histplot(data=data, x=col, hue="class_label", stat="density", common_norm=False,
                     bins=25, element="step", fill=False, ax=ax)
        ax.set_title(f"{col}: P(X | DR)"); ax.set_ylabel("Density")
    for ax in axes[len(X.columns):]: ax.remove()
    fig.suptitle("Class-Conditional Feature Distributions", y=1.002, fontsize=15)
    save_figure(fig, figures / "02_feature_distributions.png")

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.4*nrows)); axes = np.ravel(axes)
    for ax, col in zip(axes, X.columns):
        sns.boxplot(data=data, x="class_label", y=col, hue="class_label", legend=False,
                    palette=["#4C78A8", "#E45756"], ax=ax)
        ax.set_title(f"{col} by class"); ax.set_xlabel("Dataset label")
    for ax in axes[len(X.columns):]: ax.remove()
    fig.suptitle("Class-Conditioned Feature Boxplots (Outliers Retained)", y=1.002, fontsize=15)
    save_figure(fig, figures / "03_feature_boxplots.png")
    heatmap(X.corr("pearson"), "Global Pearson Correlation (Training + Test Descriptive View)", figures / "04_pearson_correlation.png")
    heatmap(X.corr("spearman"), "Global Spearman Correlation (Training + Test Descriptive View)", figures / "05_spearman_correlation.png")


def plot_dependence(matrices, mi, groups, figures: Path):
    mapping = [
        ("dr0_pearson", "Conditional Pearson Correlation — DR=0", "06_conditional_pearson_DR0.png"),
        ("dr1_pearson", "Conditional Pearson Correlation — DR=1", "07_conditional_pearson_DR1.png"),
        ("dr0_spearman", "Conditional Spearman Correlation — DR=0", "08_conditional_spearman_DR0.png"),
        ("dr1_spearman", "Conditional Spearman Correlation — DR=1", "09_conditional_spearman_DR1.png")]
    for key, title, name in mapping: heatmap(matrices[key], title + " (Training Only)", figures / name)
    fig, ax = plt.subplots(figsize=(9, 6)); d = mi.sort_values("mutual_information")
    ax.barh(d.feature, d.mutual_information, color="#59A14F"); ax.set(title="Mutual Information with DR Label (Training Only)", xlabel="Estimated mutual information", ylabel="Feature")
    save_figure(fig, figures / "10_mutual_information.png")
    fig, ax = plt.subplots(figsize=(9, 6)); d = groups.sort_values("dependence_score")
    colors = d.dependence_group.map({"LOW_DEPENDENCE":"#59A14F", "MEDIUM_DEPENDENCE":"#F28E2B", "HIGH_DEPENDENCE":"#E15759"})
    ax.barh(d.feature, d.dependence_score, color=colors); ax.set(title="Class-Conditional Dependence Ranking (Training Only)", xlabel="Composite dependence score", ylabel="Feature")
    save_figure(fig, figures / "11_dependence_ranking.png")
