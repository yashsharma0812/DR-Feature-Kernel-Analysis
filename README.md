# Statistical Feature Dependence and Nonlinear Kernel Learning for Diabetic Retinopathy Classification

## Dataset

This research project uses the supplied UCI Diabetic Retinopathy Debrecen extracted-feature CSV: 1151 observations, 18 predictors, and one binary label. The supplied header has seven exudate columns (`exudate1`, `2`, `3`, `5`, `6`, `7`, `8`) and is analyzed exactly as provided. Per UCI metadata, `class=1` contains signs of DR and `class=0` does not. This is experimental classification of a dataset label, not a clinically validated detector.

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

Naive Bayes approximates $P(X_1,\ldots,X_p|Y)$ by $\prod_i P(X_i|Y)$. Pearson, Spearman, and pairwise mutual-information evidence are calculated separately within each training class. Composite scores are split into data-driven dependence tertiles without looking at test data.

### Conditional Independence Assumption

The relevant assumption is independence between predictors conditional on the class, not unconditional independence.

### Conditional Dependence Analysis

Within-class correlations and pairwise mutual-information evidence form a reproducible composite ranking.

### Dependence Experiments

Low-, medium-, and high-dependence groups are training-only score tertiles with equal feature counts. Exact duplicate feature vectors are group-constrained to one side of the approximately 80/20 split to prevent duplicate leakage.

### Unit 1 Results

- ALL_FEATURES: F1 0.503, recall 0.350, specificity 0.954, ROC-AUC 0.733.
- LOW_DEPENDENCE: F1 0.695, recall 1.000, specificity 0.000, ROC-AUC 0.669.
- MEDIUM_DEPENDENCE: F1 0.385, recall 0.244, specificity 0.972, ROC-AUC 0.681.
- HIGH_DEPENDENCE: F1 0.567, recall 0.496, specificity 0.713, ROC-AUC 0.658.

These subset comparisons are associations, not controlled causal effects: feature relevance differs along with dependence.

## Unit 2

### Support Vector Machines

Linear, polynomial, and RBF SVMs use the same stratified holdout and training-fitted standardization.

### Feature-Space Geometry

PCA and synthetic circles provide visualization-only geometric demonstrations.

### Kernel Trick

The project demonstrates $K(x,z)=\phi(x)^T\phi(z)$ numerically for an explicit quadratic mapping, visualizes a nonlinear lift on synthetic circles, constructs Linear/Polynomial/RBF Gram matrices, checks symmetry and eigenvalues, compares within/between-class similarities, and verifies a precomputed RBF kernel against direct `SVC(kernel="rbf")`.

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

- Linear: F1 0.735, recall 0.667, specificity 0.833, ROC-AUC 0.847, 629 support vectors.
- Polynomial: F1 0.771, recall 0.683, specificity 0.898, ROC-AUC 0.835, 641 support vectors.
- RBF: F1 0.714, recall 0.650, specificity 0.806, ROC-AUC 0.819, 721 support vectors.

The training-only 5-fold C × gamma search attained its highest mean CV F1 at `C=100`, `gamma=0.01` (F1 0.730, ROC-AUC 0.822). C controls violation penalties, gamma controls RBF locality, polynomial degree controls implicit interaction order, and support vectors directly determine the boundary. Training-only degree CV selected `degree=3` (mean F1 0.695).

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
