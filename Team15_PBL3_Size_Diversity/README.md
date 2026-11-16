# Team 15 PBL-3 — Ensemble Size and Model Diversity

Official title: **Investigating the Effect of Ensemble Size and Model Diversity on Predictive Performance**.

This is the revised, isolated PBL-3 research project. The pre-pivot classifier-comparison implementation is preserved separately and is not used as sufficient evidence for this title.

## Run

```bash
python run_research.py
python -m unittest tests/test_experiments.py
python build_notebook.py
jupyter execute --inplace notebooks/Team15_PBL3_Ensemble_Size_Diversity.ipynb
python build_presentation.py
```

The test set is used only for final configurations selected from training-only grouped cross-validation. Gradient Boosting is not treated as directly comparable to independent tree votes for pairwise disagreement.

## Main deliverables

- `notebooks/Team15_PBL3_Ensemble_Size_Diversity.ipynb` — executed research notebook with retained outputs.
- `presentation/Team15_PBL3_Final.pptx` and `.pdf` — 15-slide academic presentation with speaker notes, actual figures, hyperlinks, and a Colab QR code.
- `docs/Experimental_Methodology.md` — leakage-safe experimental protocol.
- `docs/PIVOT_CHANGELOG.md` — audit of retained, redesigned, and new components.
- `docs/Team15_PBL3_Viva_Guide.md` — beginner-friendly viva preparation based on computed results.
- `results/` — fold-level and aggregated CSV outputs plus saved configurations.
- `figures/` — 25 publication-quality visualizations at 300 DPI.

## Reproducibility links

- [Notebook on GitHub](https://github.com/yashsharma0812/DR-Feature-Kernel-Analysis/blob/main/Team15_PBL3_Size_Diversity/notebooks/Team15_PBL3_Ensemble_Size_Diversity.ipynb)
- [Open notebook in Google Colab](https://colab.research.google.com/github/yashsharma0812/DR-Feature-Kernel-Analysis/blob/main/Team15_PBL3_Size_Diversity/notebooks/Team15_PBL3_Ensemble_Size_Diversity.ipynb)
- [Repository](https://github.com/yashsharma0812/DR-Feature-Kernel-Analysis)

The notebook loads verified saved outputs by default. Set `PBL3_REBUILD=1` before executing it to rerun the complete research pipeline.

## Principal finding

Increasing ensemble size produced early gains followed by small or non-monotonic changes. Maximum learner disagreement was not optimal: the Random Forest grid reached its best mean CV F1 at `N=300, max_features=6`, while maximum disagreement occurred at `N=10, max_features=1`. The final Gradient Boosting configuration achieved the strongest held-out F1 and ROC-AUC in this split, but its sequential learners are analyzed separately from independent-vote diversity.

