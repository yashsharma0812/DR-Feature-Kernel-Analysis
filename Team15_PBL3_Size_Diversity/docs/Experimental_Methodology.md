# Experimental Methodology

The analysis uses the verified 1,151-row Debrecen dataset with 18 predictors and one binary target. Exact predictor vectors are grouped before an approximately 80/20 stratified holdout so identical vectors cannot cross the boundary. All selection is performed on the 920-row training partition using five group-aware folds.

## Ensemble size

Non-size hyperparameters are selected at 100 estimators using training-only CV. Those settings remain fixed while `n_estimators` changes over 1, 5, 10, 25, 50, 100, 200, and 300. Three random seeds and five folds produce 15 validation measurements per model-size point.

## Diversity

Bagging and Random Forest use depth 12, minimum leaf size 3, 80% bootstrap samples, identical folds, and 100 trees. Random Forest varies `max_features` over 1, 3, 6, 9, and 18; Bagging uses all predictors. Diversity is calculated on validation predictions, not on the test set.

Disagreement is the mean proportion of observations on which a learner pair differs. Double-fault is the mean proportion on which both learners are wrong. Individual learner accuracy is reported because disagreement is useful only in the context of learner strength.

## Interaction

Random Forest is evaluated over six ensemble sizes and five feature-subset sizes with all other settings held constant. This grid is descriptive controlled evidence about random feature selection; it is not a causal estimate of diversity in isolation because `max_features` changes both diversity and learner strength.

## Gradient Boosting

Gradient Boosting is not included in raw tree-vote disagreement comparisons. Its additive regression trees are studied through ensemble size and a separate 3 × 3 × 3 grid of learning rates, depths, and subsample fractions, plus staged learning curves.

## Final evaluation

Final configurations are selected from training-only results. Only those three configurations are fitted on the full training partition and evaluated once on the untouched 231-row test set.

