# PBL-2 Presentation Content — 12-Slide Deck

## Slide 1 — Introduction
**Title:** From Cross-sectional Classification to Longitudinal DR Analysis  
**Content:** PBL-1 studied supervised label prediction and kernel geometry. PBL-2 asks how baseline patient-eyes group naturally and how DR states change over time.  
**Visual:** A simple PBL-1 → Unit 3 → Unit 4 research-flow diagram.  
**Speaker note:** The Protocol AA cohort is distinct from Debrecen, so PBL-1 accuracy is not a PBL-2 benchmark.

## Slide 2 — Dataset Selection
**Title:** Why DRCR Retina Network Protocol AA?  
**Content:** Prospective observational data, eye-level grading, and annual follow-up through four years make it appropriate for both clustering and sequential modeling.  
**Technical note:** Use the required DRCR attribution in `dataset_audit.md`; the original ZIP is never modified.

## Slide 3 — Dataset Description
**Title:** Cohort and Analysis Units  
**Numbers:** 388 roster patients; 3,156 valid dated annual grades; 772 patient-eye sequences; 742 gradable baseline patient-eye records from 381 patients for clustering.  
**Visual:** `05_followup_retention.png`.  
**Speaker note:** Distinguish patient visits from graded eyes and scheduled visits from completed assessments.

## Slide 4 — EDA & Data Quality
**Title:** Missingness, Distributions, and Follow-up Intervals  
**Content:** BMI has 57 missing values, HbA1c 12, and OCT thickness 1; median imputation is used for modeling. One implausible 7,152-lb weight is marked missing.  
**Visuals:** `01_baseline_feature_distributions.png`, `02_missingness_heatmap.png`, `06_visit_intervals.png`.  
**Speaker note:** No valid observation was removed merely for being statistically extreme.

## Slide 5 — EDA Insights
**Title:** Severity and Longitudinal Coverage  
**Content:** Code 90 is ungradable, not a disease state. There are 697 sequences with ≥2 grades, 640 with ≥3, and 462 complete five-visit sequences.  
**Visuals:** `04_severity_distribution.png`, `07_example_trajectories.png`, `24_baseline_characteristics_by_severity.png`.

## Slide 6 — Unit 3 Problem Statement and Method
**Title:** Hierarchical vs Partitional Clustering  
**Question:** How do the methods differ in feature-based patient-eye subgroup structure?  
**Method:** Eight baseline-only numerical features; median imputation; z-score scaling; K=2–8; Ward/complete/average linkage; silhouette, Davies–Bouldin, Calinski–Harabasz, size, and resampling stability.  
**Visual:** `09_silhouette_comparison.png`.

## Slide 7 — Unit 3 Results
**Title:** Clustering Results and Interpretation  
**Numbers:** K-Means K=2: silhouette 0.133, stability ARI 0.864. Hierarchical ward K=6: silhouette 0.091, stability ARI 0.318. Cross-method ARI 0.099.  
**Visuals:** `12_cluster_profile_heatmap.png`, `14_clustering_metric_comparison.png`.  
**Inference:** K-Means is more stable here; low agreement shows that subgroup structure is method-sensitive. Clusters are not validated clinical phenotypes.

## Slide 8 — Unit 4 Problem Statement and Markov Concept
**Title:** Sequential DR State Modeling  
**Question:** Does current severity adequately describe the next observed state, and how do transition probabilities shape future distributions?  
**Technical explanation:** First order assumes $P(S_{t+1}|S_t,S_{t-1},\ldots)=P(S_{t+1}|S_t)$.  
**Visual:** `16_sequential_vs_nonsequential.png`.

## Slide 9 — Unit 4 Transition Results
**Title:** What Changes Between Observed Visits?  
**Numbers:** 2,384 transitions: 1,351 stable, 522 progression, 511 regression. The largest probability is No/minimal/mild DR → No/minimal/mild DR (0.661).  
**Visuals:** `17_transition_count_matrix.png`, `18_transition_probability_matrix.png`.

## Slide 10 — Markov Evaluation and Unit Integration
**Title:** Added History and Baseline-Cluster Association  
**Numbers:** First-order holdout accuracy/log loss = 0.554/1.019; second-order with backoff = 0.590/0.998. Baseline clusters link to 2,339 subsequent transitions.  
**Visuals:** `22_markov_order_comparison.png`, `23_transitions_by_baseline_cluster.png`.  
**Speaker note:** The modest second-order improvement does not prove that DR violates a first-order Markov approximation. Cluster differences are unadjusted and non-causal.

## Slide 11 — Result Interpretation, Limitations, and Notebook Link
**Title:** What the Results Do—and Do Not—Show  
**Inferences:** Persistence is common, worsening and improvement both occur, and results vary by clustering choice and transition assumptions.  
**Limitations:** Observational cohort, correlated eyes, attrition, treatment changes, unequal intervals, no external validation.  
**Notebook:** `notebooks/PBL2_Protocol_AA_Analysis.ipynb`  
**GitHub link:** https://github.com/yashsharma0812/DR-Feature-Kernel-Analysis/blob/main/notebooks/PBL2_Protocol_AA_Analysis.ipynb

## Slide 12 — Conclusion and References
**Title:** Conclusion  
**Content:** Clustering provides exploratory baseline profiles; Markov analysis quantifies temporal state dependence; their integration generates hypotheses rather than clinical rules.  
**References:** Protocol AA data attribution from `dataset_audit.md`; Marcus et al. (2022), DOI 10.1001/jamaophthalmol.2022.3131; Silva et al. (2022), DOI 10.1001/jamaophthalmol.2022.3130.  
**Closing line:** PBL-2 extends kernel-based cross-sectional learning into reproducible unsupervised and longitudinal analysis without conflating the two cohorts.
