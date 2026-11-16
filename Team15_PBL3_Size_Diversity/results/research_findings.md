# Research Findings — Ensemble Size and Model Diversity

## Ensemble size
- **Bagging:** highest mean CV F1 0.671 at N=200; change from N=100 to N=300 was +0.000.
- **Gradient Boosting:** highest mean CV F1 0.702 at N=200; change from N=100 to N=300 was +0.011.
- **Random Forest:** highest mean CV F1 0.680 at N=300; change from N=100 to N=300 was +0.007.

## Diversity
- Greatest measured RF disagreement occurred at max_features=1 (0.435); its mean learner accuracy was 0.578.
- Best RF diversity-sweep F1 occurred at max_features=6 (0.692), showing that maximum disagreement was not automatically optimal.
- Across the size × diversity grid, disagreement–F1 correlation was -0.506. This descriptive association is not a causal estimate.

## Final held-out configurations
```csv
model,ensemble_size,diversity_configuration,accuracy,sensitivity,specificity,f1,roc_auc
Bagging,25,full features,0.6536796536796536,0.6504065040650406,0.6574074074074074,0.6666666666666666,0.7571138211382115
Gradient Boosting,200,sequential additive learners,0.7272727272727273,0.7642276422764228,0.6851851851851852,0.749003984063745,0.8179765130984643
Random Forest,300,max_features=6,0.6796536796536796,0.6585365853658537,0.7037037037037037,0.6864406779661016,0.7848539596507077
```

## Hypotheses
- **H1 qualified:** performance generally improved early, while later gains were small and non-monotonic.
- **H2 qualified:** diversity was useful only when learner accuracy remained adequate; maximum disagreement did not guarantee maximum F1.
- **H3 evaluated:** random feature selection changed disagreement and learner strength relative to matched Bagging.
- **H4 supported descriptively:** size curves differed across feature-subset settings, demonstrating an interaction in the controlled grid.
- **H5 supported:** measured training cost increased with ensemble size while later predictive gains often plateaued.

## Limitations
One moderate-sized tabular dataset, five duplicate rows, one grouped holdout, internal CV uncertainty, machine-specific timing, no external validation, and no clinical diagnostic validation. Feature-subset randomization changes both diversity and individual-tree strength, so diversity is not an isolated causal treatment.