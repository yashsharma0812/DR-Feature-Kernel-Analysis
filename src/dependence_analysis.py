from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def ranked_pairs(matrix: pd.DataFrame, method: str, dr_class: str) -> pd.DataFrame:
    rows = [{"feature_1": a, "feature_2": b, "value": matrix.loc[a, b],
             "absolute_value": abs(matrix.loc[a, b]), "method": method, "class": dr_class}
            for a, b in combinations(matrix.columns, 2)]
    return pd.DataFrame(rows).sort_values("absolute_value", ascending=False).reset_index(drop=True)


def normalized_pairwise_mi(X: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    result = pd.DataFrame(np.eye(X.shape[1]), index=X.columns, columns=X.columns)
    for i, j in combinations(range(X.shape[1]), 2):
        # Symmetrize directional kNN regression MI estimates and normalize to [0,1].
        a = mutual_info_regression(X.iloc[:, [i]], X.iloc[:, j], random_state=random_state)[0]
        b = mutual_info_regression(X.iloc[:, [j]], X.iloc[:, i], random_state=random_state)[0]
        val = (a + b) / 2
        result.iat[i, j] = result.iat[j, i] = val
    off_diag = result.values[~np.eye(len(result), dtype=bool)]
    scale = np.nanmax(off_diag) if len(off_diag) else 1
    if scale > 0:
        result.values[~np.eye(len(result), dtype=bool)] /= scale
    return result


def analyze_dependence(X_train: pd.DataFrame, y_train: pd.Series, tables: Path):
    matrices = {}
    pair_tables = []
    for method in ["pearson", "spearman"]:
        matrices[f"global_{method}"] = X_train.corr(method=method)
        for cls in [0, 1]:
            key = f"dr{cls}_{method}"
            matrices[key] = X_train[y_train.eq(cls)].corr(method=method)
            ranked = ranked_pairs(matrices[key], method, str(cls))
            ranked.to_csv(tables / f"conditional_correlations_DR{cls}_{method}.csv", index=False)
            pair_tables.append(ranked)
    global_pairs = pd.concat([
        ranked_pairs(matrices["global_pearson"], "pearson", "all"),
        ranked_pairs(matrices["global_spearman"], "spearman", "all")])
    global_pairs.query("absolute_value >= 0.70").to_csv(tables / "strong_correlations.csv", index=False)
    # Required aggregate class tables include both correlation diagnostics.
    for cls in [0, 1]:
        pd.concat([p for p in pair_tables if (p["class"] == str(cls)).all()]).to_csv(
            tables / f"conditional_correlations_DR{cls}.csv", index=False)
    mi_target = pd.DataFrame({"feature": X_train.columns,
        "mutual_information": mutual_info_classif(X_train, y_train, random_state=42)})
    mi_target = mi_target.sort_values("mutual_information", ascending=False).reset_index(drop=True)
    mi_target.to_csv(tables / "mutual_information.csv", index=False)
    class_mi = {cls: normalized_pairwise_mi(X_train[y_train.eq(cls)]) for cls in [0, 1]}
    scores = []
    for feature in X_train.columns:
        vals = []
        for cls in [0, 1]:
            for method in ["pearson", "spearman"]:
                row = matrices[f"dr{cls}_{method}"].loc[feature].drop(feature).abs()
                vals.append(row.mean())
            vals.append(class_mi[cls].loc[feature].drop(feature).mean())
        # A class-constant feature has undefined correlation (NaN), which is
        # absence of usable correlation evidence rather than an infinite score.
        scores.append({"feature": feature, "dependence_score": float(np.nanmean(vals))})
    groups = pd.DataFrame(scores).sort_values("dependence_score").reset_index(drop=True)
    groups["dependence_group"] = pd.qcut(groups["dependence_score"], 3,
        labels=["LOW_DEPENDENCE", "MEDIUM_DEPENDENCE", "HIGH_DEPENDENCE"])
    groups["selection_source"] = "training_data_only"
    groups.to_csv(tables / "dependence_groups.csv", index=False)
    return matrices, mi_target, groups, global_pairs, class_mi
