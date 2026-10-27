from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.pbl2.clustering import (cluster_profiles, evaluate_clusterings,
                                 fit_selected_models, severity_composition)
from src.pbl2.data_loader import discover_archive, read_table
from src.pbl2.data_validation import (ANNUAL_VISITS, STATE_ORDER,
                                      build_longitudinal_dataset,
                                      table_inventory, visit_completeness)
from src.pbl2.markov import (bootstrap_transition_intervals,
                             compare_markov_orders,
                             state_distribution_evolution,
                             transition_matrices, transition_sensitivity)
from src.pbl2.preprocessing import (CLUSTER_FEATURES, build_baseline_dataset,
                                    scale_features)
from src.pbl2.visualization import (configure_style, plot_cluster_transition_integration,
                                    plot_clustering, plot_eda, plot_markov)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "pbl2"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
MODELS = OUT / "models"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without an optional dependency."""
    clean = frame.copy().fillna("")
    headers = [str(column).replace("|", "\\|") for column in clean.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in clean.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(audit, baseline, longitudinal, inventory, completeness, rationale,
                  evaluation, k_row, h_row, agreement, transitions, probs,
                  order_results, integration, integration_uncertainty):
    top_transition = probs.stack().idxmax()
    best_order = order_results.sort_values("log_loss").iloc[0]
    dataset_audit = f"""# Protocol AA Dataset Audit

## Source and terms

The source is the DRCR Retina Network Protocol AA public dataset. The original ZIP is read directly and is never modified.

Required attribution:

> The source of the data is the DRCR Retina Network (Unique Federal Award Identification Number (FAIN) UG1EY014231). Protocol AA: Peripheral Diabetic Retinopathy (DR) Lesions on Ultrawide-field Fundus Images and Risk of DR Worsening Over Time [Data file]. Retrieved from https://public.jaeb.org/drcrnet. The analyses content and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by the DRCR Retina Network or Jaeb Center for Health Research.

## Verified inventory and linkage

- Tables: 28 pipe-delimited public data tables.
- Roster patients: 388.
- Raw severity rows: {audit['raw_severity_rows']:,}.
- Canonical annual rows before ungradable exclusion: {audit['canonical_annual_rows_including_ungradable']:,}.
- Ungradable code-90 rows excluded: {audit['ungradable_rows_excluded']}.
- Valid dated graded visits: {audit['valid_graded_visits']:,}.
- Patient-eye sequences: {audit['unique_patient_eyes']} across {audit['unique_patients']} patients.
- Sequences with at least 2/3/4 observations: {audit['sequences_at_least_2']}/{audit['sequences_at_least_3']}/{audit['sequences_at_least_4']}.
- Complete five-visit sequences: {audit['complete_5_visit_sequences']}.
- Unmatched grade-to-visit rows: {audit['unmatched_grade_visit_rows']}.

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
"""
    (REPORTS / "dataset_audit.md").write_text(dataset_audit)
    data_dictionary = """# Protocol AA Data Dictionary Summary

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
"""
    (REPORTS / "data_dictionary_summary.md").write_text(data_dictionary)
    missing_lines = [
        f"- `{feature}`: {int(baseline[feature].isna().sum())} / {len(baseline)} "
        f"({baseline[feature].isna().mean():.1%})"
        for feature in CLUSTER_FEATURES
    ]
    missing_report = f"""# Missing-Value Report

## Baseline clustering inputs

{chr(10).join(missing_lines)}

Missing input values are median-imputed only after the eight baseline features are selected. Identifiers and severity are never imputed. The source's 7,152-lb weight value is first marked invalid, which contributes to BMI missingness.

## Longitudinal grading

- Canonical annual severity rows: {audit['canonical_annual_rows_including_ungradable']:,}.
- Ungradable rows excluded from disease states: {audit['ungradable_rows_excluded']}.
- Canonical grade rows without a linked visit date: {audit['missing_dates_in_canonical_grades']}.
- Valid dated graded visits retained: {audit['valid_graded_visits']:,}.

A missing annual assessment is not forward-filled and is never counted as stable disease. Consecutive *observed* grades are linked with their actual elapsed days.
"""
    (REPORTS / "missing_value_report.md").write_text(missing_report)
    linkage_report = f"""# Patient–Eye–Visit Linkage Validation

- Roster patients: {int(inventory.loc[inventory.table.eq('PtRoster.txt'), 'rows'].iloc[0])}.
- Valid longitudinal patients: {audit['unique_patients']}.
- Valid patient-eye sequences: {audit['unique_patient_eyes']}.
- Valid graded observations: {audit['valid_graded_visits']:,}.
- Duplicate canonical `PtID`–`RCEye`–`Visit` keys: 0 (enforced by the pipeline).
- Duplicate annual `PtID`–`Visit` date keys: 0 (enforced by the pipeline).
- Unmatched canonical grade-to-visit rows: {audit['unmatched_grade_visit_rows']}.
- Canonical rows with missing linked dates: {audit['missing_dates_in_canonical_grades']}.
- Chronological ordering: enforced within every patient-eye sequence and covered by an automated test.
- Predictive split leakage: patients, including both eyes, remain wholly in train or test groups.

## Follow-up completeness

{markdown_table(completeness)}

Scheduled visit records and valid eye grades are reported separately. A visit record can contribute two graded eyes, so these columns are not interchangeable denominators.
"""
    (REPORTS / "linkage_validation.md").write_text(linkage_report)
    cleaning_report = f"""# Cleaning and Exclusion Log

| Stage | Rule | Effect |
|---|---|---|
| Source preservation | Read `data/Protocol_AA.zip` in place | Original archive unchanged |
| Grading template | Keep `UWF_with_Mask` for annual severity modeling | Avoided mixing multiple reading-center templates |
| Visit scope | Keep Baseline, 1-, 2-, 3-, and 4-year visits | {audit['canonical_annual_rows_including_ungradable']:,} canonical rows before grade validity filtering |
| Grade validity | Preserve original `drSeverity`/`dro`; exclude code 90 from states | {audit['ungradable_rows_excluded']} ungradable rows excluded |
| Date validity | Require a parsable linked date for longitudinal observations | {audit['missing_dates_in_canonical_grades']} rows excluded for missing date |
| Weight validation | Treat values outside 50–1,000 lb as invalid | One 7,152-lb source value marked missing |
| Height/BMI validation | Height outside 40–90 in or derived BMI outside 10–80 marked missing | No statistically extreme valid value removed merely as an outlier |
| Baseline missingness | Median imputation fitted on selected inputs | Original missingness retained in processed CSV; imputation occurs for modeling |
| Feature leakage | Exclude severity, identifiers, dates, treatment, and future outcomes from clustering | Eight baseline numerical inputs retained |
| Sequence construction | Link only chronologically consecutive observed grades | Missing visits not interpreted as stability |

## Feature-selection rationale

{markdown_table(rationale)}
"""
    (REPORTS / "cleaning_exclusion_log.md").write_text(cleaning_report)
    methodology = f"""# PBL-2 Methodology

## Dataset A — Baseline clustering

Analysis unit: patient-eye. Records: {len(baseline)} from {baseline.PtID.nunique()} patients. Inputs: {', '.join(CLUSTER_FEATURES)}. Severity, identifiers, dates, treatment, and future outcomes are excluded from inputs. Median imputation and z-score scaling precede Euclidean clustering.

K-Means was evaluated for K=2–8. Agglomerative clustering was evaluated with Ward, complete, and average linkage. Metrics include silhouette, Davies–Bouldin, Calinski–Harabasz, cluster size, and 30-repeat 80% subsampling ARI. Solutions with a minimum cluster below 5% were excluded from composite selection to prevent singleton-driven metric inflation.

Selected K-Means: K={int(k_row.k)}. Selected hierarchical model: {h_row.linkage} linkage, K={int(h_row.k)}. Their adjusted Rand agreement is {agreement:.3f}.

## Dataset B — Longitudinal sequences

Annual patient-eye observations are ordered using de-identified actual visit dates. Missing visits are not treated as stability. Empirical transition probabilities are row-normalized observed counts, with bootstrap patient-level uncertainty. A grouped patient holdout compares first-order probabilities with second-order probabilities using first-order backoff for unseen contexts.

The Markov analysis is descriptive and predictive, not causal. Unequal intervals, treatment changes, two-eye dependence, attrition, and time-homogeneity assumptions limit interpretation.
"""
    (REPORTS / "methodology.md").write_text(methodology)
    results = f"""# PBL-2 Results Summary

## Unit 3

- Selected K-Means: K={int(k_row.k)}, silhouette={k_row.silhouette:.3f}, stability ARI={k_row.stability_ari:.3f}.
- Selected hierarchical: {h_row.linkage}, K={int(h_row.k)}, silhouette={h_row.silhouette:.3f}, stability ARI={h_row.stability_ari:.3f}.
- Cross-method adjusted Rand agreement: {agreement:.3f}.
- Clusters are feature-based exploratory subgroups, not validated clinical phenotypes.

## Unit 4

- Observed transitions: {len(transitions):,}.
- Stable/progression/regression: {(transitions.change=='Stable').sum()}/{(transitions.change=='Progression').sum()}/{(transitions.change=='Regression').sum()}.
- Largest empirical transition probability: {top_transition[0]} → {top_transition[1]} ({probs.loc[top_transition]:.3f}).
- Best grouped-holdout log loss: {best_order.model} ({best_order.log_loss:.3f}); this does not prove or disprove the Markov property.

## Integration

Baseline cluster membership was linked to {integration.n_transitions.sum():,} subsequent observed transitions. Differences are exploratory and unadjusted; they are not evidence that clusters cause progression.
"""
    (REPORTS / "results_summary.md").write_text(results)
    continuity = """# PBL-1 to PBL-2 Research Continuity

PBL-1 used the cross-sectional Debrecen extracted-feature dataset for supervised classification, conditional-independence analysis, and SVM kernel geometry. PBL-2 uses the distinct Protocol AA cohort for unsupervised baseline profiling and longitudinal DR state transitions. Classification answers label-prediction questions; clustering explores feature-based similarity; Markov analysis describes temporal dependence. The cohorts, features, and outcomes are not harmonized, so PBL-1 accuracy is not a benchmark for PBL-2.
"""
    (REPORTS / "pbl1_pbl2_continuity.md").write_text(continuity)
    presentation = f"""# PBL-2 Presentation Content — 12-Slide Deck

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
**Numbers:** 388 roster patients; {audit['valid_graded_visits']:,} valid dated annual grades; {audit['unique_patient_eyes']} patient-eye sequences; 742 gradable baseline patient-eye records from 381 patients for clustering.  
**Visual:** `05_followup_retention.png`.  
**Speaker note:** Distinguish patient visits from graded eyes and scheduled visits from completed assessments.

## Slide 4 — EDA & Data Quality
**Title:** Missingness, Distributions, and Follow-up Intervals  
**Content:** BMI has 57 missing values, HbA1c 12, and OCT thickness 1; median imputation is used for modeling. One implausible 7,152-lb weight is marked missing.  
**Visuals:** `01_baseline_feature_distributions.png`, `02_missingness_heatmap.png`, `06_visit_intervals.png`.  
**Speaker note:** No valid observation was removed merely for being statistically extreme.

## Slide 5 — EDA Insights
**Title:** Severity and Longitudinal Coverage  
**Content:** Code 90 is ungradable, not a disease state. There are {audit['sequences_at_least_2']} sequences with ≥2 grades, {audit['sequences_at_least_3']} with ≥3, and {audit['complete_5_visit_sequences']} complete five-visit sequences.  
**Visuals:** `04_severity_distribution.png`, `07_example_trajectories.png`, `24_baseline_characteristics_by_severity.png`.

## Slide 6 — Unit 3 Problem Statement and Method
**Title:** Hierarchical vs Partitional Clustering  
**Question:** How do the methods differ in feature-based patient-eye subgroup structure?  
**Method:** Eight baseline-only numerical features; median imputation; z-score scaling; K=2–8; Ward/complete/average linkage; silhouette, Davies–Bouldin, Calinski–Harabasz, size, and resampling stability.  
**Visual:** `09_silhouette_comparison.png`.

## Slide 7 — Unit 3 Results
**Title:** Clustering Results and Interpretation  
**Numbers:** K-Means K={int(k_row.k)}: silhouette {k_row.silhouette:.3f}, stability ARI {k_row.stability_ari:.3f}. Hierarchical {h_row.linkage} K={int(h_row.k)}: silhouette {h_row.silhouette:.3f}, stability ARI {h_row.stability_ari:.3f}. Cross-method ARI {agreement:.3f}.  
**Visuals:** `12_cluster_profile_heatmap.png`, `14_clustering_metric_comparison.png`.  
**Inference:** K-Means is more stable here; low agreement shows that subgroup structure is method-sensitive. Clusters are not validated clinical phenotypes.

## Slide 8 — Unit 4 Problem Statement and Markov Concept
**Title:** Sequential DR State Modeling  
**Question:** Does current severity adequately describe the next observed state, and how do transition probabilities shape future distributions?  
**Technical explanation:** First order assumes $P(S_{{t+1}}|S_t,S_{{t-1}},\\ldots)=P(S_{{t+1}}|S_t)$.  
**Visual:** `16_sequential_vs_nonsequential.png`.

## Slide 9 — Unit 4 Transition Results
**Title:** What Changes Between Observed Visits?  
**Numbers:** {len(transitions):,} transitions: {(transitions.change=='Stable').sum():,} stable, {(transitions.change=='Progression').sum():,} progression, {(transitions.change=='Regression').sum():,} regression. The largest probability is {top_transition[0]} → {top_transition[1]} ({probs.loc[top_transition]:.3f}).  
**Visuals:** `17_transition_count_matrix.png`, `18_transition_probability_matrix.png`.

## Slide 10 — Markov Evaluation and Unit Integration
**Title:** Added History and Baseline-Cluster Association  
**Numbers:** First-order holdout accuracy/log loss = {order_results.iloc[0].accuracy:.3f}/{order_results.iloc[0].log_loss:.3f}; second-order with backoff = {order_results.iloc[1].accuracy:.3f}/{order_results.iloc[1].log_loss:.3f}. Baseline clusters link to {integration.n_transitions.sum():,} subsequent transitions.  
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
"""
    (REPORTS / "pbl2_presentation_content.md").write_text(presentation)


def main():
    for path in [FIGURES, TABLES, MODELS, REPORTS, PROCESSED]:
        path.mkdir(parents=True, exist_ok=True)
    configure_style()
    archive = discover_archive(ROOT)
    inventory = table_inventory(archive)
    inventory.to_csv(TABLES / "dataset_inventory.csv", index=False)
    baseline, rationale = build_baseline_dataset(archive)
    longitudinal, audit = build_longitudinal_dataset(archive)
    completeness = visit_completeness(archive, longitudinal)
    baseline.to_csv(PROCESSED / "protocol_aa_baseline_clustering.csv", index=False)
    longitudinal.to_csv(PROCESSED / "protocol_aa_longitudinal_sequences.csv", index=False)
    rationale.to_csv(TABLES / "feature_selection_rationale.csv", index=False)
    completeness.to_csv(TABLES / "visit_completeness.csv", index=False)
    baseline[CLUSTER_FEATURES].isna().sum().rename("missing_count").to_csv(TABLES / "baseline_missingness.csv")
    severity_characteristics = []
    for state, group in baseline.groupby("severity_state", observed=False):
        for feature in CLUSTER_FEATURES:
            values = group[feature]
            severity_characteristics.append({
                "severity_state": state, "feature": feature,
                "n_nonmissing": int(values.notna().sum()),
                "missing": int(values.isna().sum()), "mean": values.mean(),
                "standard_deviation": values.std(), "median": values.median(),
            })
    pd.DataFrame(severity_characteristics).to_csv(
        TABLES / "baseline_characteristics_by_severity.csv", index=False
    )
    pd.DataFrame([audit]).to_csv(TABLES / "longitudinal_eligibility.csv", index=False)
    X, imputer, scaler = scale_features(baseline)
    evaluation = evaluate_clusterings(X)
    evaluation.to_csv(TABLES / "clustering_evaluation.csv", index=False)
    k_model, h_model, k_labels, h_labels, agreement, k_row, h_row = fit_selected_models(X, evaluation)
    clustered = baseline.copy(); clustered["kmeans_cluster"] = k_labels; clustered["hierarchical_cluster"] = h_labels
    clustered.to_csv(PROCESSED / "protocol_aa_baseline_with_clusters.csv", index=False)
    profiles = pd.concat([cluster_profiles(baseline, k_labels, "K-Means"), cluster_profiles(baseline, h_labels, "Agglomerative")])
    profiles.to_csv(TABLES / "cluster_profiles.csv", index=False)
    severity = pd.concat([severity_composition(baseline, k_labels, "K-Means"), severity_composition(baseline, h_labels, "Agglomerative")])
    severity.to_csv(TABLES / "cluster_severity_composition.csv", index=False)
    pd.DataFrame([{"adjusted_rand_index": agreement}]).to_csv(TABLES / "cluster_assignment_agreement.csv", index=False)
    transitions, counts, probs = transition_matrices(longitudinal)
    counts.to_csv(TABLES / "transition_counts.csv"); probs.to_csv(TABLES / "transition_probabilities.csv")
    ci_low, ci_high = bootstrap_transition_intervals(longitudinal, repeats=300)
    ci_low.to_csv(TABLES / "transition_probability_ci_low.csv"); ci_high.to_csv(TABLES / "transition_probability_ci_high.csv")
    transitions.groupby("change").size().rename("count").to_csv(TABLES / "transition_change_summary.csv")
    interval_transitions = transitions.assign(
        interval_group=pd.cut(transitions.transition_days, [0, 300, 430, np.inf],
                              labels=["<300 days", "300–430 days", ">430 days"])
    )
    interval_transitions.groupby(["interval_group", "change"], observed=False).size().rename(
        "count"
    ).to_csv(TABLES / "transitions_by_interval.csv")
    interval_matrix = interval_transitions.groupby(
        ["interval_group", "severity_state", "next_state"], observed=False
    ).size().rename("count").reset_index()
    interval_matrix["origin_denominator"] = interval_matrix.groupby(
        ["interval_group", "severity_state"], observed=False
    )["count"].transform("sum")
    interval_matrix["probability"] = interval_matrix["count"].div(
        interval_matrix.origin_denominator.replace(0, np.nan)
    )
    interval_matrix.to_csv(TABLES / "transition_probabilities_by_interval.csv", index=False)
    evolution = state_distribution_evolution(longitudinal, probs)
    sensitivity_data = transition_sensitivity(probs)
    order_results = compare_markov_orders(longitudinal)
    evolution.to_csv(TABLES / "multistep_state_evolution.csv", index=False)
    sensitivity_data.to_csv(TABLES / "transition_sensitivity.csv", index=False)
    order_results.to_csv(TABLES / "markov_order_comparison.csv", index=False)
    baseline_labels = clustered[["PtID", "Eye", "kmeans_cluster", "severity_state"]].rename(
        columns={"Eye": "RCEye", "severity_state": "baseline_severity_state"}
    )
    integration_rows = transitions.merge(baseline_labels, on=["PtID", "RCEye"], how="inner", validate="many_to_one")
    integration = integration_rows.groupby(["kmeans_cluster", "change"]).size().rename("n_transitions").reset_index()
    integration["proportion"] = integration.n_transitions / integration.groupby("kmeans_cluster").n_transitions.transform("sum")
    integration.rename(columns={"kmeans_cluster": "cluster"}).to_csv(TABLES / "cluster_transition_integration.csv", index=False)
    stratified = integration_rows.groupby(
        ["kmeans_cluster", "baseline_severity_state", "change"], observed=False
    ).size().rename("n_transitions").reset_index()
    stratified["proportion_within_cluster_and_baseline_state"] = stratified.n_transitions.div(
        stratified.groupby(["kmeans_cluster", "baseline_severity_state"], observed=False).n_transitions.transform("sum")
    )
    stratified.rename(columns={"kmeans_cluster": "cluster"}).to_csv(
        TABLES / "cluster_transition_by_baseline_severity.csv", index=False
    )
    progression = integration_rows.assign(is_progression=integration_rows.change.eq("Progression")).groupby(
        "kmeans_cluster"
    ).is_progression.agg(["sum", "count"]).reset_index()
    progression["proportion"] = progression["sum"] / progression["count"]
    z = 1.96
    denominator = 1 + z**2 / progression["count"]
    centre = (progression.proportion + z**2 / (2 * progression["count"])) / denominator
    half_width = z * np.sqrt(
        progression.proportion * (1 - progression.proportion) / progression["count"]
        + z**2 / (4 * progression["count"]**2)
    ) / denominator
    progression["ci_95_low"] = centre - half_width
    progression["ci_95_high"] = centre + half_width
    progression = progression.rename(columns={"kmeans_cluster": "cluster", "sum": "progression_count", "count": "n_transitions"})
    progression.to_csv(TABLES / "cluster_progression_uncertainty.csv", index=False)
    plot_eda(baseline, longitudinal, completeness, FIGURES)
    plot_clustering(X, baseline, evaluation, k_labels, h_labels, k_row, h_row, FIGURES)
    plot_markov(longitudinal, transitions, counts, probs, evolution, sensitivity_data, order_results, FIGURES)
    plot_cluster_transition_integration(integration.rename(columns={"kmeans_cluster": "cluster"}), FIGURES)
    write_reports(audit, baseline, longitudinal, inventory, completeness, rationale,
                  evaluation, k_row, h_row, agreement, transitions, probs,
                  order_results, integration, progression)
    reproducibility = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "random_state": 42}
    (OUT / "reproducibility.json").write_text(json.dumps(reproducibility, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"Selected K-Means K={int(k_row.k)}; hierarchical={h_row.linkage}, K={int(h_row.k)}; ARI={agreement:.3f}")
    print(f"Generated {len(list(FIGURES.glob('*.png')))} figures and {len(list(TABLES.glob('*.csv')))} tables")


if __name__ == "__main__":
    main()
