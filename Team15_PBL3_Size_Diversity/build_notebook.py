"""Build the revised PBL-3 notebook with actual saved outputs."""
from pathlib import Path
import nbformat as nbf

OUTPUT = Path(__file__).resolve().parent / "notebooks" / "Team15_PBL3_Ensemble_Size_Diversity.ipynb"
def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

def main():
    cells = [
        md("""# Investigating the Effect of Ensemble Size and Model Diversity on Predictive Performance

**Team 15 — Machine Learning PBL-3**

This project studies learner count, measured tree-vote diversity, learner strength, generalization, and computational cost. It replaces the earlier “which classifier is best?” framing and is not clinical validation."""),
        code("""from pathlib import Path
import os, sys, json, numpy as np, pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
from run_research import main as run_research
RESULTS, FIGURES = ROOT/'results', ROOT/'figures'
if os.environ.get('PBL3_REBUILD') == '1' or not (RESULTS/'final_model_comparison.csv').exists():
    run_research()
else:
    print('Using verified saved experiment outputs. Set PBL3_REBUILD=1 to regenerate every model fit.')
print('Project:', ROOT)
print('Results:', len(list(RESULTS.glob('*'))), 'Figures:', len(list(FIGURES.rglob('*.png'))))"""),

        md("""## 1. Title and Research Overview

Bagged Trees, standard Gradient Boosting, and Random Forest are experimental platforms. The primary factors are ensemble size and measured model diversity."""),
        code("""configuration=json.loads((RESULTS/'experiment_configurations.json').read_text())
display(pd.Series({'Title':configuration['official_title'],'Random state':configuration['random_state'],
'Size values':configuration['ensemble_sizes'],'Interaction sizes':configuration['interaction_sizes'],
'Feature subsets':configuration['feature_subsets']}).to_frame('configuration'))"""),

        md("""## 2. Problem Understanding

How do ensemble size and model diversity influence performance, robustness, generalization, and efficiency? More learners can reduce variance, but correlated errors create diminishing returns. Diversity helps only when learners remain sufficiently accurate."""),
        code("""print((ROOT/'docs'/'Experimental_Methodology.md').read_text())"""),

        md("""## 3. Research Objectives and Hypotheses

H1 studies performance plateaus; H2 studies diversity with learner strength; H3 studies RF random features; H4 tests size × diversity interaction; H5 tests computational growth."""),
        code("""display(pd.DataFrame({'Hypothesis':['H1','H2','H3','H4','H5'],
'Factor':['Size','Measured diversity','Random features','Size × diversity','Efficiency'],
'Evidence':['Repeated CV sweep','Disagreement + double-fault','Matched Bagging/RF','6 × 5 RF grid','Measured timings']}))"""),

        md("""## 4. Dataset Selection

The UCI Debrecen data contain image-derived numerical descriptors, not raw photographs. Its 18 predictors and moderate size support controlled tree-ensemble experiments without fabricated demographics."""),
        code("""audit=json.loads((RESULTS/'dataset_audit.json').read_text())
display(pd.Series(audit,name='verified_value').to_frame())"""),

        md("""## 5. Dataset Description

There are 1,151 rows, 18 predictors, one binary target, no missing cells, and five exact duplicate rows. Duplicate predictor groups have no conflicting labels."""),
        code("""display(pd.read_csv(RESULTS/'dataset_description.csv',index_col=0))
print('Classes:',audit['class_0'],audit['class_1'],'| Conflicting duplicate groups:',audit['conflicting_duplicate_predictor_groups'])"""),

        md("""## 6. Exploratory Data Analysis

Microaneurysm features are strongly related, but input correlation is not model diversity. Class distributions overlap and IQR extremes are retained."""),
        code("""for name in ['01_class_distribution.png','02_feature_distributions.png','03_correlation_heatmap.png',
'04_feature_target_heatmap.png','05_outlier_rates.png']:
    display(Image(filename=str(FIGURES/'eda'/name)))"""),

        md("""## 7. Data Preprocessing and Leakage Prevention

A duplicate-grouped approximately 80/20 holdout yields 920 training and 231 test rows. Identical vectors cannot cross the boundary. Selection uses training-only grouped folds."""),
        code("""train_idx=pd.read_csv(RESULTS/'training_indices.csv');test_idx=pd.read_csv(RESULTS/'test_indices.csv')
print('Train:',len(train_idx),'Test:',len(test_idx),'Overlap:',len(set(train_idx.row_index)&set(test_idx.row_index)))
display(pd.read_csv(RESULTS/'non_size_parameter_selection.csv'))"""),

        md("""## 8. Ensemble Learning Approaches

Bagging uses independent bootstrap trees and all predictors; Gradient Boosting adds sequential score components; Random Forest adds random feature selection. Bagging and RF receive matched settings in the controlled diversity study."""),
        code("""display(pd.DataFrame([
['Bagging','Independent bootstrap trees','All 18','Voting'],
['Gradient Boosting','Sequential additive trees','Configured','Additive scores'],
['Random Forest','Independent bootstrap trees','1/3/6/9/18','Voting']],
columns=['Model','Dependence','Features per split','Aggregation']))"""),

        md("""## 9. Experiment A — Effect of Ensemble Size

Only n_estimators changes within each family. Each point has 15 validation measurements: three seeds × five grouped folds."""),
        code("""size=pd.read_csv(RESULTS/'ensemble_size_results.csv')
display(size[['model','ensemble_size','f1_mean','f1_std','roc_auc_mean','sensitivity_mean',
'training_time_seconds_mean','generalization_gap']].round(4))
for name in ['size_vs_accuracy.png','size_vs_f1.png','size_vs_roc_auc.png','size_vs_sensitivity.png']:
    display(Image(filename=str(FIGURES/'ensemble_size'/name)))"""),
        md("""**Interpretation:** Peak mean CV F1 occurred at N=200 for Bagging and Gradient Boosting and N=300 for RF. From N=100 to N=300, changes were approximately 0.000, +0.011, and +0.007 respectively—small, non-monotonic late gains."""),

        md("""## 10. Experiment B — Effect of Model Diversity

Bagging is the full-feature reference. RF varies max_features across 1, 3, 6, 9, and 18 with 100 matched trees. Diversity is measured on validation folds."""),
        code("""diversity=pd.read_csv(RESULTS/'diversity_results.csv')
display(diversity[['model','feature_subset','disagreement_mean','double_fault_mean',
'learner_accuracy_mean','f1_mean','roc_auc_mean']].round(4))
for name in ['01_diversity_measures.png','02_diversity_vs_performance.png','03_learner_strength.png']:
    display(Image(filename=str(FIGURES/'model_diversity'/name)))"""),
        md("""**Interpretation:** One feature produced greatest RF disagreement (0.435) but lower learner accuracy (0.578) and F1 (0.667). Six features produced the best diversity-sweep F1 (0.692). Maximum disagreement was not optimal."""),

        md("""## 11. Diversity Metrics

Disagreement measures different predictions; double-fault measures shared errors. Error correlation is complementary. Learner accuracy is essential because disagreement from weak learners may be harmful."""),
        code("""from src.diversity_metrics import diversity_summary
example=np.array([[0,0,1,1],[0,1,1,0],[1,0,1,0]])
display(pd.Series(diversity_summary(example,np.array([0,0,1,1]))).to_frame('worked_example'))"""),

        md("""## 12. Experiment C — Ensemble Size × Diversity

The central RF grid combines six sizes with five feature subsets while holding other settings constant."""),
        code("""interaction=pd.read_csv(RESULTS/'size_diversity_interaction.csv')
display(interaction[['ensemble_size','feature_subset','disagreement_mean','learner_accuracy_mean',
'f1_mean','roc_auc_mean','training_time_seconds_mean']].round(4))
for name in ['01_f1_heatmap.png','02_disagreement_heatmap.png','03_training_time_heatmap.png',
'04_performance_curves.png','05_disagreement_curves.png']:
    display(Image(filename=str(FIGURES/'size_diversity_interaction'/name)))"""),
        md("""**Interpretation:** Highest grid F1 was 0.696 at N=300 and six features. Highest disagreement was 0.446 at N=10 and one feature, where F1 was 0.648. Size cannot make weak learners irrelevant."""),

        md("""## 13. Gradient Boosting Learning Dynamics

Boosting trees are sequential additive components, so raw voting disagreement is not compared with Bagging/RF. Learning rate, depth, subsampling, and staged curves are analyzed separately."""),
        code("""gradient=pd.read_csv(RESULTS/'gradient_dynamics.csv')
display(gradient.sort_values('f1_mean',ascending=False).head(10).round(4))
display(Image(filename=str(FIGURES/'ensemble_size'/'gradient_configuration_heatmap.png')))
display(Image(filename=str(FIGURES/'ensemble_size'/'gradient_learning_curves.png')))"""),

        md("""## 14. Cross-validation and Robustness

Size points use 15 measurements; diversity and interaction use identical five-fold validation partitions. Standard deviations describe internal variation, not clinical robustness."""),
        code("""folds=pd.read_csv(RESULTS/'ensemble_size_fold_results.csv')
display(folds.groupby(['model','ensemble_size']).f1.agg(['mean','std','min','max']).reset_index().round(4))"""),

        md("""## 15. Computational Efficiency

Training cost rises with learner count while late gains are often small. Prediction timing and train–validation gaps are also measured."""),
        code("""for name in ['size_vs_training_time.png','size_vs_prediction_time.png','size_vs_generalization_gap.png']:
    display(Image(filename=str(FIGURES/'ensemble_size'/name)))"""),

        md("""## 16. Final Test-Set Evaluation

Final configurations are selected from training-only results, fitted on all 920 training rows, and evaluated once on the untouched 231-row test set."""),
        code("""final=pd.read_csv(RESULTS/'final_model_comparison.csv')
display(final[['model','ensemble_size','diversity_configuration','accuracy','precision','sensitivity',
'specificity','f1','roc_auc','training_time_seconds']].round(4))
for name in ['01_final_performance.png','02_final_confusion_matrices.png','03_final_roc.png']:
    display(Image(filename=str(FIGURES/'final_comparison'/name)))"""),

        md("""## 17. Comparative Performance Analysis

Gradient Boosting N=200 has best held-out F1 (0.749). RF selects N=300 with six features. Bagging selects N=25 because it is within 0.002 of the best training-only F1 at lower cost."""),
        code("""display(final.set_index('model')[['ensemble_size','accuracy','sensitivity','specificity','f1','roc_auc',
'training_time_seconds','prediction_ms_per_observation']].round(4))"""),

        md("""## 18. Research Inferences

H1–H3 are qualified. H4 is supported descriptively by differing size curves across feature subsets. H5 is supported by growing cost and late plateaus. No causal or clinical claim is made."""),
        code("""print((RESULTS/'research_findings.md').read_text())"""),

        md("""## 19. Limitations

One moderate dataset, five duplicate rows, one grouped holdout, internal CV uncertainty, no external cohort, machine-specific timing, and no clinical validation. Feature randomization changes both diversity and learner strength. Small differences are not called statistically significant."""),
        code("""print('Interaction disagreement–F1 correlation:',
round(interaction[['disagreement_mean','f1_mean']].corr().iloc[0,1],3))
print('Descriptive association—not causality.')"""),

        md("""## 20. Conclusion

Early size gains were followed by small, non-monotonic improvements. Greatest disagreement did not maximize F1 because learner strength fell. Six features best balanced RF diversity and strength. Gradient Boosting scaled differently and delivered strongest final F1 at N=200. Effective design balances count, diversity, strength, generalization, and cost."""),
        code("""print('Revised PBL-3 complete.')
print('Figures:',len(list(FIGURES.rglob('*.png'))),'Results:',len(list(RESULTS.glob('*'))))
print('Test set was reserved for preselected final configurations.')"""),
    ]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    notebook.metadata.language_info={"name":"python","version":"3.13"}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,OUTPUT)
    print(OUTPUT)

if __name__=="__main__": main()

