# Team 15 PBL-3 — Ensemble Prediction for Diabetic Retinopathy

This project independently compares Bagged Decision Trees, standard Gradient Boosting, and Random Forest on the UCI Debrecen Diabetic Retinopathy dataset. It is isolated from the PBL-1 and PBL-2 notebooks and results.

The verified CSV has 1,151 observations, 18 predictors, and one binary target (19 total columns). It contains image-derived tabular features, not retinal photographs.

## Run

From this directory:

```bash
python run_analysis.py
python -m unittest tests/test_pbl3.py
python build_notebook.py
jupyter execute --inplace notebooks/Team15_PBL3_Ensemble_Comparison.ipynb
python build_presentation.py
```

All three models use the same duplicate-safe stratified holdout and identical training-only cross-validation folds. Results are written to `results/`; 300-DPI plots are separated into `figures/eda/` and `figures/model_comparison/`.

The enhanced 12-slide deck is written to `presentation/Team15_PBL3_Enhanced_Presentation.pptx`. It embeds the verified notebook figures and includes clickable GitHub/Colab links plus a Colab QR code.

## Research integrity

The label represents the dataset-defined presence or absence of DR signs. Results are experimental and are not clinical validation. Timing is machine-specific, and feature importance is model-dependent.

