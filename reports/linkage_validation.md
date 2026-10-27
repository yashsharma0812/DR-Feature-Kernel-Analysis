# Patient–Eye–Visit Linkage Validation

- Roster patients: 388.
- Valid longitudinal patients: 388.
- Valid patient-eye sequences: 772.
- Valid graded observations: 3,156.
- Duplicate canonical `PtID`–`RCEye`–`Visit` keys: 0 (enforced by the pipeline).
- Duplicate annual `PtID`–`Visit` date keys: 0 (enforced by the pipeline).
- Unmatched canonical grade-to-visit rows: 0.
- Canonical rows with missing linked dates: 0.
- Chronological ordering: enforced within every patient-eye sequence and covered by an automated test.
- Predictive split leakage: patients, including both eyes, remain wholly in train or test groups.

## Follow-up completeness

| Visit | visit_records | dated_visits | explicitly_missed | valid_graded_eyes | graded_patients |
| --- | --- | --- | --- | --- | --- |
| Baseline | 388 | 388 | 0 | 742 | 381 |
| 1 year | 358 | 345 | 13 | 645 | 335 |
| 2 year | 340 | 330 | 10 | 631 | 321 |
| 3 year | 322 | 307 | 15 | 588 | 301 |
| 4 year | 284 | 284 | 0 | 550 | 283 |

Scheduled visit records and valid eye grades are reported separately. A visit record can contribute two graded eyes, so these columns are not interchangeable denominators.
