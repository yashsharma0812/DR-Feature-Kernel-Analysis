from pathlib import Path

import numpy as np
import pandas as pd


def descriptive_statistics(X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.columns)
    out["count"] = X.count(); out["mean"] = X.mean(); out["median"] = X.median()
    out["std"] = X.std(); out["variance"] = X.var(); out["minimum"] = X.min()
    out["maximum"] = X.max(); out["range"] = out.maximum - out.minimum
    out["q1"] = X.quantile(.25); out["q3"] = X.quantile(.75); out["iqr"] = out.q3 - out.q1
    out["skewness"] = X.skew(); out["kurtosis"] = X.kurtosis()
    return out


def quality_analysis(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    q1, q3 = X.quantile(.25), X.quantile(.75); iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((X.lt(lower)) | (X.gt(upper))).sum()
    unique = X.nunique()
    near_constant_share = X.apply(lambda s: s.value_counts(normalize=True, dropna=False).iloc[0])
    table = pd.DataFrame({
        "missing": X.isna().sum(), "unique_values": unique,
        "constant": unique.eq(1), "largest_value_share": near_constant_share,
        "near_constant_ge_95pct": near_constant_share.ge(.95), "iqr_outlier_count": outliers,
        "iqr_outlier_percent": 100 * outliers / len(X), "minimum": X.min(), "maximum": X.max(),
    })
    impossible = {}
    for col in ["quality", "pre_screening", "am_fm_classification"]:
        if col in X: impossible[col] = int((~X[col].isin([0, 1])).sum())
    nonnegative = [c for c in X if c.startswith("ma") or c.startswith("exudate") or c in {"macula_opticdisc_distance", "opticdisc_diameter"}]
    impossible.update({c: int((X[c] < 0).sum()) for c in nonnegative})
    counts = y.value_counts(); imbalance = float(counts.max() / counts.min())
    return table, {"invalid_value_counts": impossible, "class_imbalance_ratio": imbalance}


def save_descriptive_tables(X: pd.DataFrame, y: pd.Series, out: Path) -> None:
    descriptive_statistics(X).to_csv(out / "descriptive_statistics.csv")
    descriptive_statistics(X[y.eq(0)]).to_csv(out / "descriptive_statistics_DR0.csv")
    descriptive_statistics(X[y.eq(1)]).to_csv(out / "descriptive_statistics_DR1.csv")
