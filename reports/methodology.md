# PBL-2 Methodology

## Dataset A — Baseline clustering

Analysis unit: patient-eye. Records: 742 from 381 patients. Inputs: age, diabetes_duration, bmi, systolic_bp, diastolic_bp, hba1c, visual_acuity_letters, oct_central_thickness. Severity, identifiers, dates, treatment, and future outcomes are excluded from inputs. Median imputation and z-score scaling precede Euclidean clustering.

K-Means was evaluated for K=2–8. Agglomerative clustering was evaluated with Ward, complete, and average linkage. Metrics include silhouette, Davies–Bouldin, Calinski–Harabasz, cluster size, and 30-repeat 80% subsampling ARI. Solutions with a minimum cluster below 5% were excluded from composite selection to prevent singleton-driven metric inflation.

Selected K-Means: K=2. Selected hierarchical model: ward linkage, K=6. Their adjusted Rand agreement is 0.099.

## Dataset B — Longitudinal sequences

Annual patient-eye observations are ordered using de-identified actual visit dates. Missing visits are not treated as stability. Empirical transition probabilities are row-normalized observed counts, with bootstrap patient-level uncertainty. A grouped patient holdout compares first-order probabilities with second-order probabilities using first-order backoff for unseen contexts.

The Markov analysis is descriptive and predictive, not causal. Unequal intervals, treatment changes, two-eye dependence, attrition, and time-homogeneity assumptions limit interpretation.
