from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import sklearn, scipy, matplotlib, seaborn


def _metric_line(row):
    return (f"accuracy={row.accuracy:.3f}, precision={row.precision:.3f}, "
            f"recall/sensitivity={row.recall_sensitivity:.3f}, specificity={row.specificity:.3f}, "
            f"F1={row.f1:.3f}, ROC-AUC={row.roc_auc:.3f}")


def write_research_summary(out:Path, info, quality, groups, nb, kernels, sim, eig, gamma, cres, degree, cv, precomputed, mapping):
    top_global=pd.read_csv(out/"tables"/"strong_correlations.csv").head(5)
    c0=pd.read_csv(out/"tables"/"conditional_correlations_DR0.csv").head(5)
    c1=pd.read_csv(out/"tables"/"conditional_correlations_DR1.csv").head(5)
    best=cv.loc[cv.mean_cv_f1.idxmax()]
    def pairs(df): return "; ".join(f"{r.feature_1}-{r.feature_2} ({r.value:.3f}, {r.method})" for _,r in df.iterrows()) or "None above threshold"
    lines=[
        "DATASET OVERVIEW",f"{info['observations']} observations, {info['features']} predictors. Class 0={info['target_distribution'].get(0,info['target_distribution'].get('0'))}; class 1={info['target_distribution'].get(1,info['target_distribution'].get('1'))}. No rows or values were altered.",
        "\nEDA FINDINGS",f"Missing values={sum(info['missing_values'].values())}; duplicates={info['duplicate_rows']}; IQR flags were reported but retained. Class imbalance ratio={quality['class_imbalance_ratio']:.3f}.",
        "\nSTATISTICAL FINDINGS",f"Strongest global |correlation|>=0.70 pairs: {pairs(top_global)}. Correlation is evidence of association, not a complete dependence test.",
        "\nUNIT 1: FEATURE INDEPENDENCE","The global and class-conditional diagnostics show that the predictors are not mutually independent. The scientifically relevant Naive Bayes check is conditional on the class, not merely global correlation.",
        "\nUNIT 1: CONDITIONAL DEPENDENCE",f"Top DR=0 pairs: {pairs(c0)}. Top DR=1 pairs: {pairs(c1)}. Groups are training-only tertiles of a composite of absolute Pearson, absolute Spearman, and normalized kNN pairwise MI evidence.",
        "\nUNIT 1: NAIVE BAYES RESULTS","\n".join(f"{r.experiment}: {_metric_line(r)}" for _,r in nb.iterrows()),
        "\nUNIT 1: INTERPRETATION","Differences across equally sized dependence tertiles are observational: feature identity and predictive information also differ. They do not by themselves establish a causal effect of dependence.",
        "\nUNIT 2: FEATURE-SPACE GEOMETRY","PCA is used only as a two-dimensional visualization. Overlap in this projection is not proof of non-separability in the full feature space.",
        "\nUNIT 2: EXPLICIT FEATURE MAPPING",f"Maximum absolute error between phi(x)^T phi(z) and (x^T z)^2: {mapping.absolute_difference.max():.3e}.",
        "\nUNIT 2: KERNEL TRICK DEMONSTRATION",f"The manually precomputed RBF SVM agreed with the direct RBF SVM on {precomputed.prediction_agreement.iloc[0]:.1%} of test predictions (max decision difference {precomputed.max_abs_decision_difference.iloc[0]:.3e}).",
        "\nUNIT 2: GRAM MATRIX ANALYSIS","; ".join(f"{r.kernel}: symmetry error={r.symmetry_max_abs_error:.2e}, min eigenvalue={r.minimum_eigenvalue:.2e}" for _,r in eig.iterrows())+". Tiny negative eigenvalues may reflect floating-point roundoff.",
        "\nUNIT 2: KERNEL SIMILARITY","; ".join(f"{r.kernel}: same-minus-different={r.difference:.3f}" for _,r in sim.iterrows())+". This is exploratory alignment evidence, not proof.",
        "\nUNIT 2: LINEAR VS POLYNOMIAL VS RBF","\n".join(f"{r.kernel}: {_metric_line(r)}; SV={int(r.support_vectors)} ({r.support_vector_percent:.1f}%)" for _,r in kernels.iterrows()),
        "\nUNIT 2: C ANALYSIS",f"Best tested test F1 (descriptive, not selected) at C={cres.loc[cres.f1.idxmax(),'C']:g}; selection must use training CV.",
        "\nUNIT 2: GAMMA ANALYSIS",f"Best tested test F1 (descriptive, not selected) at gamma={gamma.loc[gamma.f1.idxmax(),'gamma']:g}; train/test gaps are tabulated.",
        "\nUNIT 2: POLYNOMIAL DEGREE ANALYSIS",f"Training-only CV selected degree={int(degree.loc[degree.mean_cv_f1.idxmax(),'degree'])} by mean F1={degree.mean_cv_f1.max():.3f}; the test sweep remains descriptive.",
        "\nUNIT 2: SUPPORT VECTOR ANALYSIS","Support-vector counts and class composition vary by kernel; a lower count is not assumed to imply a better model.",
        "\nUNIT 2: COMPUTATIONAL ANALYSIS",f"Training-only CV selected C={best.C:g}, gamma={best.gamma:g} by mean F1={best.mean_cv_f1:.3f} (mean ROC-AUC={best.mean_cv_roc_auc:.3f}). Baseline timing and prediction metrics are reported together.",
        "\nLIMITATIONS","Single modest-sized extracted-feature dataset; one fixed holdout split; dependence estimators have sampling uncertainty; univariate dependence grouping confounds dependence with feature identity/information; no external or clinical validation; timings are machine- and run-specific.",
        "\nFINAL OBSERVATIONS","These experiments classify the dataset label and demonstrate statistical dependence and kernel geometry. They do not constitute a medical diagnostic system or clinical validation."
    ]
    (out/"summaries"/"research_summary.txt").write_text("\n".join(lines))
    versions={"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scikit-learn":sklearn.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"seaborn":seaborn.__version__}
    (out/"metrics"/"reproducibility.json").write_text(json.dumps({"random_state":42,"test_size":"approximately 0.2 (one of five folds)","stratified":True,"duplicate_feature_grouping":True,"versions":versions},indent=2))


def write_readme(root:Path, info, nb, kernels, cv, degree):
    best=cv.loc[cv.mean_cv_f1.idxmax()]
    kernel_lines="\n".join(f"- {r.kernel}: F1 {r.f1:.3f}, recall {r.recall_sensitivity:.3f}, specificity {r.specificity:.3f}, ROC-AUC {r.roc_auc:.3f}, {int(r.support_vectors)} support vectors." for _,r in kernels.iterrows())
    nb_lines="\n".join(f"- {r.experiment}: F1 {r.f1:.3f}, recall {r.recall_sensitivity:.3f}, specificity {r.specificity:.3f}, ROC-AUC {r.roc_auc:.3f}." for _,r in nb.iterrows())
    text=f'''# Statistical Feature Dependence and Nonlinear Kernel Learning for Diabetic Retinopathy Classification

## Dataset

This research project uses the supplied UCI Diabetic Retinopathy Debrecen extracted-feature CSV: {info['observations']} observations, {info['features']} predictors, and one binary label. The supplied header has seven exudate columns (`exudate1`, `2`, `3`, `5`, `6`, `7`, `8`) and is analyzed exactly as provided. Per UCI metadata, `class=1` contains signs of DR and `class=0` does not. This is experimental classification of a dataset label, not a clinically validated detector.

Source: https://archive.ics.uci.edu/dataset/329/diabetic+retinopathy+debrecen (CC BY 4.0; Antal & Hajdu, 2014).

## Problem Statement

The project investigates statistical assumptions and feature-space geometry rather than merely comparing classifier accuracy.

## Research Questions

The study asks whether retinal image-derived features satisfy Gaussian Naive Bayes' class-conditional independence assumption, how measured dependence relates to its performance, whether linear separation is adequate, and how polynomial/RBF kernels alter geometry, similarity, support-vector structure, generalization, and computation.

## Exploratory Data Analysis

The workflow reports distributions, class-conditioned boxplots, missingness, duplicates, near-constant values, and retained IQR outlier flags.

## Statistical Analysis

Descriptive statistics and global/class-conditional Pearson and Spearman associations are saved as tables and heatmaps. Correlation is not equated with dependence.

## Unit 1

### Naive Bayes

Naive Bayes approximates $P(X_1,\\ldots,X_p|Y)$ by $\\prod_i P(X_i|Y)$. Pearson, Spearman, and pairwise mutual-information evidence are calculated separately within each training class. Composite scores are split into data-driven dependence tertiles without looking at test data.

### Conditional Independence Assumption

The relevant assumption is independence between predictors conditional on the class, not unconditional independence.

### Conditional Dependence Analysis

Within-class correlations and pairwise mutual-information evidence form a reproducible composite ranking.

### Dependence Experiments

Low-, medium-, and high-dependence groups are training-only score tertiles with equal feature counts. Exact duplicate feature vectors are group-constrained to one side of the approximately 80/20 split to prevent duplicate leakage.

### Unit 1 Results

{nb_lines}

These subset comparisons are associations, not controlled causal effects: feature relevance differs along with dependence.

## Unit 2

### Support Vector Machines

Linear, polynomial, and RBF SVMs use the same stratified holdout and training-fitted standardization.

### Feature-Space Geometry

PCA and synthetic circles provide visualization-only geometric demonstrations.

### Kernel Trick

The project demonstrates $K(x,z)=\\phi(x)^T\\phi(z)$ numerically for an explicit quadratic mapping, visualizes a nonlinear lift on synthetic circles, constructs Linear/Polynomial/RBF Gram matrices, checks symmetry and eigenvalues, compares within/between-class similarities, and verifies a precomputed RBF kernel against direct `SVC(kernel="rbf")`.

### Explicit Feature Mapping

The explicit quadratic map is verified numerically against $(x^Tz)^2$ for several pairs.

### Linear Kernel

The linear kernel uses the original standardized dot-product geometry.

### Polynomial Kernel

The polynomial kernel models finite-order feature interactions; degrees 2–5 are evaluated.

### RBF Kernel

The RBF kernel represents local similarity in an implicit infinite-dimensional feature space.

### Gram Matrix Analysis

Manual Gram matrices are checked for symmetry and positive semidefiniteness within numerical tolerance.

### Support Vectors

Counts and percentages are reported by kernel and class because support vectors determine the fitted boundary.

### C

C sensitivity is measured over 0.01–100.

### Gamma

RBF gamma sensitivity is measured over 0.001–10.

### Polynomial Degree

Degree sensitivity is measured over 2–5.

### Performance Comparison

{kernel_lines}

The training-only 5-fold C × gamma search attained its highest mean CV F1 at `C={best.C:g}`, `gamma={best.gamma:g}` (F1 {best.mean_cv_f1:.3f}, ROC-AUC {best.mean_cv_roc_auc:.3f}). C controls violation penalties, gamma controls RBF locality, polynomial degree controls implicit interaction order, and support vectors directly determine the boundary. Training-only degree CV selected `degree={int(degree.loc[degree.mean_cv_f1.idxmax(),'degree'])}` (mean F1 {degree.mean_cv_f1.max():.3f}).

### Computational Trade-offs

Tables and plots report fit/inference time, support-vector percentage, F1, recall and ROC-AUC together. Timings are environment-specific and the dataset is too small for broad scalability claims.

## Limitations

This is one dataset and one held-out split, with no external validation. Dependence estimates have uncertainty. PCA plots are visualization-only. Test sweeps are sensitivity analyses, not hyperparameter selection; only training cross-validation selects settings. No clinical claims are made.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Artifacts are written under `outputs/figures`, `outputs/tables`, `outputs/metrics`, and `outputs/summaries`. The exploratory notebook is in `notebooks/exploratory_analysis.ipynb`.
'''
    (root/"README.md").write_text(text)
