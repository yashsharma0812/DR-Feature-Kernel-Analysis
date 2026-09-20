"""Idempotently add the complete result sections to the executed EDA notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


NOTEBOOK = Path(__file__).with_name("exploratory_analysis.ipynb")
MARKER = "## 8. PCA feature-space geometry"


def result_cells():
    return [
        new_markdown_cell(
            MARKER + "\n\n"
            "PCA is used only for visualization. Overlap in two principal components is not proof "
            "that the complete feature space is nonlinearly separable."
        ),
        new_code_cell(
            "pca_variance = pd.read_csv(TABLES / 'pca_explained_variance.csv')\n"
            "display(pca_variance)\n"
            "display(Image(filename=str(FIGURES / '14_pca_feature_geometry.png')))\n"
            "display(Image(filename=str(FIGURES / '15_linear_svm_pca.png')))"
        ),
        new_markdown_cell(
            "## 9. Synthetic nonlinear geometry and decision boundaries\n\n"
            "The synthetic dataset isolates the geometric effect of kernel choice; it is not part of "
            "the real DR performance evaluation."
        ),
        new_code_cell(
            "for filename in [\n"
            "    '18_synthetic_linear_boundary.png',\n"
            "    '19_synthetic_polynomial_boundary.png',\n"
            "    '20_synthetic_rbf_boundary.png',\n"
            "]:\n"
            "    display(Image(filename=str(FIGURES / filename)))"
        ),
        new_markdown_cell("## 10. Class-sorted kernel similarity structure"),
        new_code_cell(
            "similarity = pd.read_csv(TABLES / 'kernel_similarity_analysis.csv')\n"
            "display(similarity)\n"
            "display(Image(filename=str(FIGURES / '24_rbf_kernel_sorted_by_class.png')))\n"
            "display(Image(filename=str(FIGURES / '25_kernel_similarity_comparison.png')))"
        ),
        new_markdown_cell(
            "## 11. Linear and nonlinear boundaries on the PCA projection\n\n"
            "These boundaries are fitted in the two-dimensional visualization projection and are not "
            "the full-dimensional models used for the reported test metrics."
        ),
        new_code_cell(
            "for filename in [\n"
            "    '26_linear_svm_boundary.png',\n"
            "    '27_polynomial_svm_boundary.png',\n"
            "    '28_rbf_svm_boundary.png',\n"
            "]:\n"
            "    display(Image(filename=str(FIGURES / filename)))"
        ),
        new_markdown_cell(
            "## 12. RBF gamma sensitivity\n\n"
            "Small gamma gives broad influence and smoother behavior; large gamma gives local influence. "
            "Train/test gaps, not appearance alone, provide evidence about overfitting."
        ),
        new_code_cell(
            "gamma_results = pd.read_csv(TABLES / 'gamma_analysis.csv')\n"
            "display(gamma_results[[\n"
            "    'gamma', 'train_accuracy', 'accuracy', 'precision', 'recall_sensitivity',\n"
            "    'specificity', 'train_f1', 'f1', 'train_roc_auc', 'roc_auc',\n"
            "    'support_vectors', 'training_time_seconds'\n"
            "]])\n"
            "display(Image(filename=str(FIGURES / '29_gamma_boundary_comparison.png')))\n"
            "display(Image(filename=str(FIGURES / '30_gamma_performance.png')))\n"
            "display(Image(filename=str(FIGURES / '31_gamma_support_vectors.png')))"
        ),
        new_markdown_cell(
            "## 13. C regularization sensitivity\n\n"
            "Small C applies stronger regularization and tolerates more violations; large C penalizes "
            "training errors more heavily."
        ),
        new_code_cell(
            "c_results = pd.read_csv(TABLES / 'C_analysis.csv')\n"
            "display(c_results[[\n"
            "    'C', 'train_accuracy', 'accuracy', 'precision', 'recall_sensitivity',\n"
            "    'specificity', 'train_f1', 'f1', 'train_roc_auc', 'roc_auc',\n"
            "    'support_vectors', 'training_time_seconds'\n"
            "]])\n"
            "display(Image(filename=str(FIGURES / '32_C_performance.png')))\n"
            "display(Image(filename=str(FIGURES / '33_C_support_vectors.png')))"
        ),
        new_markdown_cell(
            "## 14. Polynomial degree sensitivity\n\n"
            "Degree controls the order of implicit feature interactions. The CV columns are learned "
            "from training data only; holdout scores remain descriptive."
        ),
        new_code_cell(
            "degree_results = pd.read_csv(TABLES / 'polynomial_degree_analysis.csv')\n"
            "display(degree_results[[\n"
            "    'degree', 'train_accuracy', 'accuracy', 'precision', 'recall_sensitivity',\n"
            "    'specificity', 'train_f1', 'f1', 'roc_auc', 'support_vectors',\n"
            "    'mean_cv_f1', 'std_cv_f1', 'mean_cv_roc_auc', 'std_cv_roc_auc'\n"
            "]])\n"
            "display(Image(filename=str(FIGURES / '34_polynomial_degree_analysis.png')))"
        ),
        new_markdown_cell("## 15. Support vectors and predictive–computational trade-offs"),
        new_code_cell(
            "support_results = pd.read_csv(TABLES / 'support_vector_analysis.csv')\n"
            "display(support_results)\n"
            "display(kernel_results[[\n"
            "    'kernel', 'f1', 'roc_auc', 'recall_sensitivity', 'support_vector_percent',\n"
            "    'training_time_seconds', 'inference_time_seconds'\n"
            "]])\n"
            "display(Image(filename=str(FIGURES / '39_performance_vs_computation.png')))"
        ),
        new_markdown_cell("## 16. Consolidated model comparison"),
        new_code_cell(
            "nb_baseline = nb_results.loc[nb_results['experiment'].eq('ALL_FEATURES')].copy()\n"
            "nb_baseline['model'] = 'Gaussian Naive Bayes'\n"
            "svm_models = kernel_results.copy()\n"
            "svm_models['model'] = svm_models['kernel'] + ' SVM'\n"
            "columns = ['model', 'accuracy', 'precision', 'recall_sensitivity', 'specificity', 'f1', 'roc_auc']\n"
            "comparison = pd.concat([nb_baseline[columns], svm_models[columns]], ignore_index=True)\n"
            "display(comparison.round({c: 3 for c in columns[1:]}))\n"
            "best_f1 = comparison.loc[comparison['f1'].idxmax()]\n"
            "best_auc = comparison.loc[comparison['roc_auc'].idxmax()]\n"
            "print(f\"Highest holdout F1: {best_f1['model']} ({best_f1['f1']:.3f})\")\n"
            "print(f\"Highest holdout ROC-AUC: {best_auc['model']} ({best_auc['roc_auc']:.3f})\")"
        ),
        new_markdown_cell("## 17. Research-question answers generated from the experimental outputs"),
        new_code_cell(
            "top_global = strong.sort_values('absolute_value', ascending=False).iloc[0]\n"
            "top_dr0 = conditional_dr0.sort_values('absolute_value', ascending=False).iloc[0]\n"
            "top_dr1 = conditional_dr1.sort_values('absolute_value', ascending=False).iloc[0]\n"
            "best_rbf = cv.loc[cv['mean_cv_f1'].idxmax()]\n"
            "best_degree = degree_results.loc[degree_results['mean_cv_f1'].idxmax()]\n"
            "print('RQ1–RQ3: Features are strongly dependent globally and conditionally.')\n"
            "print(f\"  Global maximum: {top_global.feature_1}–{top_global.feature_2}, "
            "{top_global.method}={top_global.value:.4f}\")\n"
            "print(f\"  DR=0 maximum: {top_dr0.feature_1}–{top_dr0.feature_2}, "
            "{top_dr0.method}={top_dr0.value:.4f}\")\n"
            "print(f\"  DR=1 maximum: {top_dr1.feature_1}–{top_dr1.feature_2}, "
            "{top_dr1.method}={top_dr1.value:.4f}\")\n"
            "print('RQ4–RQ5: Naive Bayes behavior changed substantially across dependence groups, but the effect is confounded by feature identity and information.')\n"
            "print('RQ6–RQ7: The linear SVM was competitive; nonlinear flexibility did not uniformly improve every metric.')\n"
            "print('RQ8–RQ9: Explicit mapping, Gram matrices, and precomputed-kernel equivalence verify similarity-space operation.')\n"
            "print(f\"RQ10: Training CV selected RBF C={best_rbf.C:g}, gamma={best_rbf.gamma:g}; "
            "polynomial degree={int(best_degree.degree)}.\")\n"
            "print('RQ11: Polynomial SVM gave the best holdout F1, while Linear SVM gave the best ROC-AUC; RBF used the most support vectors.')"
        ),
        new_markdown_cell(
            "## Final limitations and presentation guidance\n\n"
            "- Results come from one extracted-feature dataset and one grouped holdout split.\n"
            "- Dependence estimates have sampling uncertainty.\n"
            "- Dependence-group comparisons also change feature identity and predictive information.\n"
            "- PCA figures are visualization-only.\n"
            "- Timing results are specific to this machine and run.\n"
            "- There is no external or clinical validation.\n\n"
            "For a presentation, use the consolidated table and selected figures as the main narrative; "
            "retain the full notebook as the evidence-rich appendix."
        ),
    ]


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    caution_index = next(
        (i for i, cell in enumerate(notebook.cells)
         if cell.cell_type == "markdown" and "## Interpretation caution" in cell.source),
        len(notebook.cells),
    )
    marker_index = next(
        (i for i, cell in enumerate(notebook.cells)
         if cell.cell_type == "markdown" and MARKER in cell.source),
        None,
    )
    if marker_index is not None:
        del notebook.cells[marker_index:caution_index]
        caution_index = marker_index
    notebook.cells[caution_index:caution_index] = result_cells()
    nbformat.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
