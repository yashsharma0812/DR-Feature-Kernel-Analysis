# Missing-Value Report

## Baseline clustering inputs

- `age`: 0 / 742 (0.0%)
- `diabetes_duration`: 0 / 742 (0.0%)
- `bmi`: 57 / 742 (7.7%)
- `systolic_bp`: 0 / 742 (0.0%)
- `diastolic_bp`: 0 / 742 (0.0%)
- `hba1c`: 12 / 742 (1.6%)
- `visual_acuity_letters`: 0 / 742 (0.0%)
- `oct_central_thickness`: 1 / 742 (0.1%)

Missing input values are median-imputed only after the eight baseline features are selected. Identifiers and severity are never imputed. The source's 7,152-lb weight value is first marked invalid, which contributes to BMI missingness.

## Longitudinal grading

- Canonical annual severity rows: 3,280.
- Ungradable rows excluded from disease states: 124.
- Canonical grade rows without a linked visit date: 0.
- Valid dated graded visits retained: 3,156.

A missing annual assessment is not forward-filled and is never counted as stable disease. Consecutive *observed* grades are linked with their actual elapsed days.
