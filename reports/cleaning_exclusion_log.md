# Cleaning and Exclusion Log

| Stage | Rule | Effect |
|---|---|---|
| Source preservation | Read `data/Protocol_AA.zip` in place | Original archive unchanged |
| Grading template | Keep `UWF_with_Mask` for annual severity modeling | Avoided mixing multiple reading-center templates |
| Visit scope | Keep Baseline, 1-, 2-, 3-, and 4-year visits | 3,280 canonical rows before grade validity filtering |
| Grade validity | Preserve original `drSeverity`/`dro`; exclude code 90 from states | 124 ungradable rows excluded |
| Date validity | Require a parsable linked date for longitudinal observations | 0 rows excluded for missing date |
| Weight validation | Treat values outside 50–1,000 lb as invalid | One 7,152-lb source value marked missing |
| Height/BMI validation | Height outside 40–90 in or derived BMI outside 10–80 marked missing | No statistically extreme valid value removed merely as an outlier |
| Baseline missingness | Median imputation fitted on selected inputs | Original missingness retained in processed CSV; imputation occurs for modeling |
| Feature leakage | Exclude severity, identifiers, dates, treatment, and future outcomes from clustering | Eight baseline numerical inputs retained |
| Sequence construction | Link only chronologically consecutive observed grades | Missing visits not interpreted as stability |

## Feature-selection rationale

| feature | role | reason |
| --- | --- | --- |
| age | clustering_input | Baseline numerical measurement; no future information |
| diabetes_duration | clustering_input | Baseline numerical measurement; no future information |
| bmi | clustering_input | Baseline numerical measurement; no future information |
| systolic_bp | clustering_input | Baseline numerical measurement; no future information |
| diastolic_bp | clustering_input | Baseline numerical measurement; no future information |
| hba1c | clustering_input | Baseline numerical measurement; no future information |
| visual_acuity_letters | clustering_input | Baseline numerical measurement; no future information |
| oct_central_thickness | clustering_input | Baseline numerical measurement; no future information |
| original_dr_grade/dro | interpretation_only | Excluded from inputs to avoid outcome-driven clusters |
| creatinine/eGFR/urine | excluded | Substantially lower baseline availability |
| identifiers/dates | linkage_only | Not phenotype variables |
