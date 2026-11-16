# Team 15 PBL-3 Viva Guide

## 1. What is ensemble learning?
It combines predictions from multiple learners so the group can be more reliable than one model.

## 2. What is ensemble size?
The number of base learners. We tested 1, 5, 10, 25, 50, 100, 200, and 300.

## 3. What is model diversity?
It is the extent to which learners make different predictions or errors on the same observations.

## 4. Why does diversity matter?
If accurate learners make different errors, voting can cancel some mistakes. Diversity without learner strength can hurt.

## 5. What is Bagging?
Independent decision trees trained on bootstrap samples, with predictions aggregated.

## 6. What is Gradient Boosting?
Trees added sequentially to correct the current ensemble's errors.

## 7. What is Random Forest?
Bagging plus random feature selection at each split.

## 8. Random Forest versus Bagging?
Both bootstrap independent trees; Random Forest also limits features available at a split.

## 9. Why vary estimators?
To observe early gains, plateaus, instability, and computational growth instead of assuming more is better.

## 10. Why vary max_features?
It controls feature randomization, changing both tree diversity and individual-tree strength.

## 11. Pairwise disagreement?
The mean fraction of validation observations on which two trees predict different labels.

## 12. Double-fault?
The mean fraction on which both members of a learner pair are wrong. Lower shared-error rates are preferable.

## 13. Why is maximum diversity not necessarily optimal?
The highest RF disagreement occurred at k=1, but the best diversity-sweep F1 occurred at k=6. Weak learners can disagree for unhelpful reasons.

## 14. Why can performance plateau?
Once errors are strongly correlated or variance is already reduced, additional learners add little new information.

## 15. Why cross-validation?
It estimates performance variability on training-only folds and prevents using the final test set for selection.

## 16. Why is sensitivity important?
It measures the proportion of DR-positive labels detected; false negatives are missed positive examples.

## 17. Learner accuracy versus ensemble accuracy?
Learner accuracy averages individual trees; ensemble accuracy evaluates the aggregated vote. Both are needed to interpret diversity.

## 18. Most important result?
The best held-out F1 configuration was Gradient Boosting at N=200 with sequential additive learners, F1=0.749. More learners and more disagreement were not universally better; performance depended on the balance among size, diversity, learner strength, and cost.
