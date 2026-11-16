# PBL-3 Pivot Changelog

## Official pivot

The project title changed from **Comparative Analysis of Bagging, Boosting, and Random Forests for Ensemble Prediction** to **Investigating the Effect of Ensemble Size and Model Diversity on Predictive Performance**.

## Preserved

- The complete pre-pivot project is retained in `../Team15_PBL3_PrePivot_Backup/` relative to the repository root.
- The verified UCI Debrecen CSV, team information, duplicate-safe split principle, EDA concepts, classification metrics, and consistent algorithm colors were retained.
- The original `Team15_PBL3/` directory remains intact and is not used as evidence for the revised research question.

## Redesigned

- The central question is no longer “which algorithm is best?”
- Non-size parameters are selected on training-only grouped CV, then held fixed during the primary estimator-count sweep.
- Ensemble size is varied systematically across 1, 5, 10, 25, 50, 100, 200, and 300 learners.
- Bagging and Random Forest use matched tree and bootstrap settings in the controlled diversity study.
- Diversity is measured using disagreement, double-fault, error correlation, and individual learner accuracy.
- A 6 × 5 size–feature-subset grid tests interaction effects.
- Gradient Boosting is analyzed separately through size, learning rate, tree depth, subsampling, and staged learning curves.
- The held-out test set is used only for preselected final configurations.

## Newly implemented

- Repeated-seed grouped-CV ensemble-size experiments.
- Explicit tree-level diversity metrics on validation folds.
- Size × diversity performance and diversity heatmaps.
- Performance-efficiency and generalization-gap analysis.
- Research-hypothesis conclusions, viva guide, revised notebook, tests, and revised presentation pipeline.

## Removed from the revised claim

The earlier three-model held-out comparison is preserved historically but is not treated as sufficient evidence for the faculty-assigned size-and-diversity title.

