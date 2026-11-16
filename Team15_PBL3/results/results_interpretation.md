# PBL-3 Results and Research Inferences

## Central research question

Which of Bagged Decision Trees, standard Gradient Boosting, and Random Forest provides the most effective and reliable prediction of the Debrecen DR label under the same split and evaluation protocol?

## Held-out results

model,accuracy,precision,recall_sensitivity,specificity,f1_score,roc_auc,training_time_seconds,prediction_time_ms_per_observation
Bagging,0.7012987012987013,0.7288135593220338,0.6991869918699187,0.7037037037037037,0.7136929460580913,0.7771002710027101,0.14370912499998667,0.21549729437136594
Gradient Boosting,0.7272727272727273,0.734375,0.7642276422764228,0.6851851851851852,0.749003984063745,0.8179765130984643,0.17214058399986243,0.005270199132858331
Random Forest,0.7056277056277056,0.7350427350427351,0.6991869918699187,0.7129629629629629,0.7166666666666667,0.7884673291177356,0.059032540999851335,0.111175683983887


## Evidence-based answers

- Highest held-out accuracy: **Gradient Boosting** (0.727).
- Highest F1: **Gradient Boosting** (0.749).
- Highest ROC-AUC: **Gradient Boosting** (0.818).
- Highest sensitivity / lowest false-negative rate: **Gradient Boosting** (0.764 sensitivity).
- Lowest cross-validation F1 standard deviation: **Gradient Boosting** (0.025).
- Fastest final fit: **Random Forest**; fastest held-out prediction: **Gradient Boosting** on this machine.
- Random Forest versus ordinary Bagging held-out F1: 0.717 versus 0.714. This comparison isolates the observed effect of random feature selection only imperfectly because the tuned tree structures may also differ.
- Gradient Boosting versus Bagging held-out F1: 0.749 versus 0.714. Sequential correction did not automatically guarantee superiority.
- A rank across F1, ROC-AUC, sensitivity, and specificity favors **Gradient Boosting**, but the conclusion should also consider stability and cost rather than treating one metric as decisive.

## Limitations

This is one small tabular dataset with image-derived features, five duplicate rows, no raw retinal photographs, one grouped holdout split, no external validation, and no patient demographics or treatment history. The class label indicates dataset-defined signs of DR; these experiments are not clinical validation. Timing is machine-specific, feature importance is model-dependent, and small metric differences may not establish a definitive winner.
