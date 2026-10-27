from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .data_loader import read_table
from .data_validation import severity_state


CLUSTER_FEATURES = [
    "age", "diabetes_duration", "bmi", "systolic_bp", "diastolic_bp",
    "hba1c", "visual_acuity_letters", "oct_central_thickness",
]


def build_baseline_dataset(archive: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one baseline record per gradable patient-eye without future variables."""
    severity = read_table(archive, "AADRSSData.txt")
    severity = severity.loc[
        severity.Visit.eq("Baseline")
        & severity.RCTemplateName.eq("UWF_with_Mask")
        & severity.dro.ne(90),
        ["PtID", "RCEye", "drSeverity", "dro"],
    ].rename(columns={"RCEye": "Eye", "drSeverity": "original_dr_grade"})
    roster = read_table(archive, "PtRoster.txt")[["PtID", "AgeAsofEnrollDt"]]
    roster["age"] = pd.to_numeric(roster.AgeAsofEnrollDt.replace("90 or older", "90"))
    history = read_table(archive, "AABaselineOcuMedHx.txt")
    history = history[["PtID", "DiabAge", "Height", "Weight"]].merge(
        roster[["PtID", "age"]], on="PtID", validate="one_to_one"
    )
    history.loc[~history.Weight.between(50, 1000), "Weight"] = np.nan
    history.loc[~history.Height.between(40, 90), "Height"] = np.nan
    history["diabetes_duration"] = history.age - history.DiabAge
    history["bmi"] = 703 * history.Weight / history.Height.pow(2)
    history.loc[~history.bmi.between(10, 80), "bmi"] = np.nan
    bp = read_table(archive, "AABldPress.txt").query("Visit == 'Baseline'")
    bp = bp[["PtID", "BldPrSys", "BldPrDia"]].rename(
        columns={"BldPrSys": "systolic_bp", "BldPrDia": "diastolic_bp"}
    )
    lab = read_table(archive, "AALab.txt").query("Visit == 'Baseline'")
    lab["hba1c"] = lab.HgbA1c.fillna(lab.HbA1cLabVal)
    va_raw = read_table(archive, "AAVAReVATest.txt").query("Visit == 'Baseline'")
    va = pd.concat([
        va_raw[["PtID", "EtdrsOD"]].rename(columns={"EtdrsOD": "visual_acuity_letters"}).assign(Eye="OD"),
        va_raw[["PtID", "EtdrsOS"]].rename(columns={"EtdrsOS": "visual_acuity_letters"}).assign(Eye="OS"),
    ], ignore_index=True)
    oct_data = read_table(archive, "RCOCTGradingResult.txt")
    oct_data = oct_data.loc[
        oct_data.Visit.eq("Baseline") & oct_data.GradeType.eq("ST"),
        ["PtID", "OCTScanEye", "OCTCentSubCalc"],
    ].rename(columns={"OCTScanEye": "Eye", "OCTCentSubCalc": "oct_central_thickness"})
    baseline = (
        severity.merge(history[["PtID", "age", "diabetes_duration", "bmi"]], on="PtID", validate="many_to_one")
        .merge(bp, on="PtID", validate="many_to_one")
        .merge(lab[["PtID", "hba1c"]], on="PtID", validate="many_to_one")
        .merge(va, on=["PtID", "Eye"], validate="one_to_one")
        .merge(oct_data, on=["PtID", "Eye"], how="left", validate="one_to_one")
    )
    if baseline.duplicated(["PtID", "Eye"]).any():
        raise ValueError("Baseline patient-eye key is not unique")
    baseline["severity_state"] = baseline.dro.map(severity_state)
    rationale = pd.DataFrame([
        {"feature": f, "role": "clustering_input", "reason": "Baseline numerical measurement; no future information"}
        for f in CLUSTER_FEATURES
    ] + [
        {"feature": "original_dr_grade/dro", "role": "interpretation_only", "reason": "Excluded from inputs to avoid outcome-driven clusters"},
        {"feature": "creatinine/eGFR/urine", "role": "excluded", "reason": "Substantially lower baseline availability"},
        {"feature": "identifiers/dates", "role": "linkage_only", "reason": "Not phenotype variables"},
    ])
    return baseline, rationale


def scale_features(baseline: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(baseline[CLUSTER_FEATURES])
    scaled = scaler.fit_transform(imputed)
    return scaled, imputer, scaler

