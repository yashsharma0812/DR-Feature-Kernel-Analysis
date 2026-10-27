# Protocol AA Data Dictionary Summary

This is a task-focused summary of fields verified against the supplied `DataGlossary.rtf`; it is not a replacement for the full source glossary.

| Source table | Original field(s) | Analysis field | Meaning / role |
|---|---|---|---|
| `PtRoster.txt` | `PtID`, `AgeAsofEnrollDt` | `PtID`, `age` | De-identified patient key; rounded age at enrollment |
| `AABaselineOcuMedHx.txt` | `DiabAge`, `Height`, `Weight` | `diabetes_duration`, `bmi` | Age at diabetes diagnosis; height and weight used for derived duration and BMI |
| `AABldPress.txt` | `BldPrSys`, `BldPrDia` | `systolic_bp`, `diastolic_bp` | Baseline systolic and diastolic blood pressure |
| `AALab.txt` | `HgbA1c`, `HbA1cLabVal` | `hba1c` | HbA1c percentage, using the documented alternate value when the central value is missing |
| `AAVAReVATest.txt` | `EtdrsOD`, `EtdrsOS` | `visual_acuity_letters` | Eye-specific visual-acuity letter score |
| `RCOCTGradingResult.txt` | `OCTCentSubCalc`, `OCTScanEye` | `oct_central_thickness`, `Eye` | Eye-specific OCT central-subfield calculation |
| `AADRSSData.txt` | `RCEye`, `RCTemplateName`, `drSeverity`, `dro` | eye, grading template, original grade, order code | Reading-center DR severity; outcome is excluded from clustering inputs |
| `AAVisitInfo.txt` | `Visit`, `VisitDt`, `VisitMiss` | visit label, date, missed-visit indicator | Chronology and scheduled/completed follow-up linkage |

The processed datasets retain original identifiers and severity fields for reproducible linkage, while public aggregate outputs avoid displaying patient-level histories beyond de-identified examples.
