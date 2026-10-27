# Protocol AA Dataset Audit

## Source and terms

The source is the DRCR Retina Network Protocol AA public dataset. The original ZIP is read directly and is never modified.

Required attribution:

> The source of the data is the DRCR Retina Network (Unique Federal Award Identification Number (FAIN) UG1EY014231). Protocol AA: Peripheral Diabetic Retinopathy (DR) Lesions on Ultrawide-field Fundus Images and Risk of DR Worsening Over Time [Data file]. Retrieved from https://public.jaeb.org/drcrnet. The analyses content and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the DRCR Retina Network or Jaeb Center for Health Research.

## Verified inventory and linkage

- Tables: 28 pipe-delimited public data tables.
- Roster patients: 388.
- Raw severity rows: 11,628.
- Canonical annual rows before ungradable exclusion: 3,280.
- Ungradable code-90 rows excluded: 124.
- Valid dated graded visits: 3,156.
- Patient-eye sequences: 772 across 388 patients.
- Sequences with at least 2/3/4 observations: 697/640/585.
- Complete five-visit sequences: 462.
- Unmatched grade-to-visit rows: 0.

## Grading decision

`AADRSSData` contains multiple reading-center templates for the same patient-eye-visit. `UWF_with_Mask` was selected before modeling because it is consistently available from baseline through year 4 and masks peripheral information for a comparable central DR severity assessment. Annual keys were unique. Original `drSeverity` and `dro` values are preserved. Code 90 is ungradable and is not treated as a disease state.

## State grouping

- dro 1–3: No/minimal/mild DR
- dro 4–5: Moderate NPDR
- dro 6–7: Moderately severe/severe NPDR
- observed dro 8–12: PDR

The rare no/minimal grades were combined with mild NPDR to avoid an unsupported sparse Markov state. This grouping does not imply equal distances between original grades.

## Data-quality decisions

The source contained one weight value of 7,152 lb, outside the documented/plausible range used here; it was marked missing rather than silently retained. Missing baseline clustering values are median-imputed after feature selection. No outlier was removed solely because it was statistically extreme.
