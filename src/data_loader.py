from pathlib import Path
import json

import pandas as pd


TARGET = "class"
FEATURE_DESCRIPTIONS = {
    "quality": "Image quality assessment (1=sufficient, 0=bad)",
    "pre_screening": "Pre-screening result (1=severe abnormality)",
    **{f"ma{i}": f"Microaneurysm count at confidence level {0.4 + i/10:.1f}" for i in range(1, 7)},
    **{f"exudate{i}": "Normalized exudate detector output" for i in [1, 2, 3, 5, 6, 7, 8]},
    "macula_opticdisc_distance": "Normalized macula-to-optic-disc distance",
    "opticdisc_diameter": "Optic-disc diameter",
    "am_fm_classification": "AM/FM classifier result",
}


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}', found {df.columns.tolist()}")
    if set(df[TARGET].dropna().unique()) != {0, 1}:
        raise ValueError("Target must contain exactly binary values 0 and 1")
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns are unsupported: {non_numeric}")
    return df, df.drop(columns=TARGET), df[TARGET].astype(int)


def dataset_information(df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_counts = df[TARGET].value_counts().sort_index()
    summary = {
        "shape": list(df.shape),
        "observations": len(df),
        "features": df.shape[1] - 1,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "unique_values": df.nunique(dropna=False).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": target_counts.to_dict(),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "target_semantics": {"0": "no signs of DR", "1": "contains signs of DR"},
    }
    (output_dir / "dataset_information.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"Dataset shape: {df.shape}", f"Observations: {len(df)}", f"Features: {df.shape[1]-1}",
        f"Columns: {df.columns.tolist()}", "\nFirst five rows:\n" + df.head().to_string(),
        "\nLast five rows:\n" + df.tail().to_string(), "\nData types:\n" + df.dtypes.to_string(),
        "\nUnique values:\n" + df.nunique(dropna=False).to_string(),
        "\nMissing values:\n" + df.isna().sum().to_string(),
        f"\nDuplicate rows: {df.duplicated().sum()}",
        "\nTarget distribution:\n" + target_counts.to_string(),
        "\nMemory usage (bytes):\n" + df.memory_usage(deep=True).to_string(),
    ]
    report = "\n".join(lines)
    (output_dir / "dataset_information.txt").write_text(report)
    print(report)
    return summary
