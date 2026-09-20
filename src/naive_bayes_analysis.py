from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

from .evaluation import fit_and_evaluate
from .visualization import save_figure


def run_naive_bayes(split, groups: pd.DataFrame, tables: Path, figures: Path):
    subsets = {"ALL_FEATURES": split.X_train.columns.tolist()}
    for group in ["LOW_DEPENDENCE", "MEDIUM_DEPENDENCE", "HIGH_DEPENDENCE"]:
        subsets[group] = groups.loc[groups.dependence_group.astype(str).eq(group), "feature"].tolist()
    rows, predictions = [], {}
    for name, cols in subsets.items():
        metrics, pred, _ = fit_and_evaluate(GaussianNB(), split.X_train[cols], split.y_train,
                                             split.X_test[cols], split.y_test)
        metrics.update({"experiment": name, "n_features": len(cols), "features": "|".join(cols)})
        rows.append(metrics); predictions[name] = pred
    results = pd.DataFrame(rows)
    results.to_csv(tables / "naive_bayes_comparison.csv", index=False)
    base = results.loc[results.experiment.eq("ALL_FEATURES")].iloc[0]
    cm = [[base.tn, base.fp], [base.fn, base.tp]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax)
    ax.set(title="Gaussian Naive Bayes — All Features", xlabel="Predicted label", ylabel="True label", xticklabels=["DR=0","DR=1"], yticklabels=["DR=0","DR=1"])
    save_figure(fig, figures / "12_nb_confusion_matrix.png")
    long = results.melt(id_vars="experiment", value_vars=["f1", "recall_sensitivity", "specificity", "roc_auc"], var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(long, x="experiment", y="score", hue="metric", ax=ax)
    ax.set(title="Naive Bayes Performance by Dependence Group", xlabel="Feature subset", ylabel="Test score", ylim=(0,1)); ax.tick_params(axis="x", rotation=15)
    save_figure(fig, figures / "13_nb_dependence_comparison.png")
    return results
