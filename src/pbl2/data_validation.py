from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import list_tables, read_table


ANNUAL_VISITS = ["Baseline", "1 year", "2 year", "3 year", "4 year"]
VISIT_ORDER = {visit: i for i, visit in enumerate(ANNUAL_VISITS)}
STATE_ORDER = [
    "No/minimal/mild DR",
    "Moderate NPDR",
    "Moderately severe/severe NPDR",
    "PDR",
]


def severity_state(dro: float) -> str | None:
    """Map documented DR order codes to sufficiently populated clinical states."""
    if dro in {1, 2, 3}:
        return STATE_ORDER[0]
    if dro in {4, 5}:
        return STATE_ORDER[1]
    if dro in {6, 7}:
        return STATE_ORDER[2]
    if dro in {8, 9, 10, 12}:
        return STATE_ORDER[3]
    return None  # Code 90 is ungradable and is not a disease state.


def table_inventory(archive: Path) -> pd.DataFrame:
    rows = []
    for table in list_tables(archive):
        df = read_table(archive, table)
        rows.append({
            "table": table,
            "rows": len(df),
            "columns": len(df.columns),
            "unique_patients": int(df.PtID.nunique()) if "PtID" in df else np.nan,
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_names": "|".join(df.columns),
        })
    return pd.DataFrame(rows)


def build_longitudinal_dataset(archive: Path) -> tuple[pd.DataFrame, dict]:
    """Create one consistently graded annual record per patient-eye-visit."""
    grades = read_table(archive, "AADRSSData.txt")
    visits = read_table(archive, "AAVisitInfo.txt")
    canonical = grades.loc[
        grades.RCTemplateName.eq("UWF_with_Mask") & grades.Visit.isin(ANNUAL_VISITS)
    ].copy()
    duplicate_keys = canonical.duplicated(["PtID", "RCEye", "Visit"], keep=False)
    conflicts = (
        canonical.loc[duplicate_keys]
        .groupby(["PtID", "RCEye", "Visit"]).dro.nunique().gt(1).sum()
    )
    if duplicate_keys.any():
        raise ValueError(f"Canonical annual severity has duplicate keys; conflicts={conflicts}")
    canonical["severity_state"] = canonical.dro.map(severity_state)
    canonical["grade_valid"] = canonical.severity_state.notna()
    annual_visits = visits.loc[visits.Visit.isin(ANNUAL_VISITS)].copy()
    annual_visits["VisitDt"] = pd.to_datetime(annual_visits.VisitDt, errors="coerce")
    if annual_visits.duplicated(["PtID", "Visit"]).any():
        raise ValueError("AAVisitInfo contains duplicate patient/annual-visit keys")
    out = canonical.merge(
        annual_visits[["PtID", "Visit", "VisitDt", "VisitMiss"]],
        on=["PtID", "Visit"], how="left", validate="many_to_one", indicator=True,
    )
    unmatched = int(out._merge.ne("both").sum())
    out = out.drop(columns="_merge")
    out["visit_order"] = out.Visit.map(VISIT_ORDER)
    out = out.sort_values(["PtID", "RCEye", "VisitDt", "visit_order"]).reset_index(drop=True)
    valid = out.loc[out.grade_valid & out.VisitDt.notna()].copy()
    valid["days_since_previous"] = valid.groupby(["PtID", "RCEye"]).VisitDt.diff().dt.days
    sequence_lengths = valid.groupby(["PtID", "RCEye"]).size()
    audit = {
        "raw_severity_rows": len(grades),
        "canonical_annual_rows_including_ungradable": len(out),
        "ungradable_rows_excluded": int((~out.grade_valid).sum()),
        "valid_graded_visits": len(valid),
        "unique_patients": int(valid.PtID.nunique()),
        "unique_patient_eyes": int(sequence_lengths.size),
        "sequences_at_least_2": int(sequence_lengths.ge(2).sum()),
        "sequences_at_least_3": int(sequence_lengths.ge(3).sum()),
        "sequences_at_least_4": int(sequence_lengths.ge(4).sum()),
        "complete_5_visit_sequences": int(sequence_lengths.eq(5).sum()),
        "unmatched_grade_visit_rows": unmatched,
        "missing_dates_in_canonical_grades": int(out.VisitDt.isna().sum()),
        "canonical_template": "UWF_with_Mask",
    }
    return valid, audit


def visit_completeness(archive: Path, longitudinal: pd.DataFrame) -> pd.DataFrame:
    visits = read_table(archive, "AAVisitInfo.txt")
    visits = visits.loc[visits.Visit.isin(ANNUAL_VISITS)].copy()
    visits["VisitDt"] = pd.to_datetime(visits.VisitDt, errors="coerce")
    completed = longitudinal.groupby("Visit").agg(
        valid_graded_eyes=("RCEye", "size"), graded_patients=("PtID", "nunique")
    )
    scheduled = visits.groupby("Visit").agg(
        visit_records=("PtID", "size"), dated_visits=("VisitDt", "count"),
        explicitly_missed=("VisitMiss", "count"),
    )
    return scheduled.join(completed, how="left").reindex(ANNUAL_VISITS).fillna(0).reset_index()

