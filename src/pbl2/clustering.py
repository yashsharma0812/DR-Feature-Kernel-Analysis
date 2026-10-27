from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from . import RANDOM_STATE
from .preprocessing import CLUSTER_FEATURES


def _metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    counts = np.bincount(labels)
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "minimum_cluster_size": int(counts.min()),
        "minimum_cluster_fraction": float(counts.min() / len(labels)),
        "cluster_sizes": "|".join(map(str, counts)),
    }


def resampling_stability(
    X: np.ndarray, method: str, k: int, linkage: str = "ward",
    repeats: int = 30, fraction: float = 0.8,
) -> float:
    """Mean ARI between full-data and subsample partitions on shared records."""
    rng = np.random.default_rng(RANDOM_STATE)
    if method == "kmeans":
        reference = KMeans(k, n_init=50, random_state=RANDOM_STATE).fit_predict(X)
    else:
        reference = AgglomerativeClustering(k, linkage=linkage).fit_predict(X)
    scores = []
    for repeat in range(repeats):
        idx = np.sort(rng.choice(len(X), int(fraction * len(X)), replace=False))
        if method == "kmeans":
            labels = KMeans(k, n_init=20, random_state=RANDOM_STATE + repeat + 1).fit_predict(X[idx])
        else:
            labels = AgglomerativeClustering(k, linkage=linkage).fit_predict(X[idx])
        scores.append(adjusted_rand_score(reference[idx], labels))
    return float(np.mean(scores))


def evaluate_clusterings(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(2, 9):
        model = KMeans(k, n_init=50, random_state=RANDOM_STATE).fit(X)
        row = {"method": "K-Means", "linkage": "not_applicable", "k": k,
               "inertia": model.inertia_, **_metrics(X, model.labels_)}
        row["stability_ari"] = resampling_stability(X, "kmeans", k)
        rows.append(row)
    for linkage in ["ward", "complete", "average"]:
        for k in range(2, 7):
            labels = AgglomerativeClustering(k, linkage=linkage).fit_predict(X)
            row = {"method": "Agglomerative", "linkage": linkage, "k": k,
                   "inertia": np.nan, **_metrics(X, labels)}
            row["stability_ari"] = resampling_stability(X, "agglomerative", k, linkage)
            rows.append(row)
    result = pd.DataFrame(rows)
    # Reject degenerate solutions before ranking, then combine four internal/stability diagnostics.
    eligible = result.minimum_cluster_fraction.ge(0.05)
    for col, ascending in [("silhouette", True), ("davies_bouldin", False),
                           ("calinski_harabasz", True), ("stability_ari", True)]:
        values = result.loc[eligible, col]
        scaled = (values - values.min()) / (values.max() - values.min()) if values.max() > values.min() else 1.0
        result.loc[eligible, f"rank_{col}"] = scaled if ascending else 1 - scaled
    rank_cols = [c for c in result if c.startswith("rank_")]
    result["composite_score"] = result[rank_cols].mean(axis=1)
    result.loc[~eligible, "composite_score"] = np.nan
    return result


def fit_selected_models(X: np.ndarray, evaluation: pd.DataFrame):
    k_row = evaluation.loc[evaluation.method.eq("K-Means")].sort_values(
        ["composite_score", "k"], ascending=[False, True]
    ).iloc[0]
    h_row = evaluation.loc[evaluation.method.eq("Agglomerative")].sort_values(
        ["composite_score", "k"], ascending=[False, True]
    ).iloc[0]
    k_model = KMeans(int(k_row.k), n_init=50, random_state=RANDOM_STATE).fit(X)
    h_model = AgglomerativeClustering(int(h_row.k), linkage=h_row.linkage).fit(X)
    h_labels = h_model.fit_predict(X)
    agreement = adjusted_rand_score(k_model.labels_, h_labels)
    return k_model, h_model, k_model.labels_, h_labels, float(agreement), k_row, h_row


def cluster_profiles(baseline: pd.DataFrame, labels: np.ndarray, name: str) -> pd.DataFrame:
    data = baseline.copy()
    data["cluster"] = labels
    profiles = data.groupby("cluster")[CLUSTER_FEATURES].mean()
    profiles.insert(0, "method", name)
    return profiles.reset_index()


def severity_composition(baseline: pd.DataFrame, labels: np.ndarray, name: str) -> pd.DataFrame:
    data = baseline[["severity_state"]].copy()
    data["cluster"] = labels
    counts = pd.crosstab(data.cluster, data.severity_state)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions.reset_index().melt(id_vars="cluster", var_name="severity_state", value_name="proportion").assign(method=name)

