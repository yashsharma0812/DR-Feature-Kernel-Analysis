from __future__ import annotations

from pathlib import Path

import pandas as pd
import qrcode
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "presentation"
OUTPUT = OUT_DIR / "Team15_PBL3_Final.pptx"
QR_PATH = OUT_DIR / "Team15_PBL3_Colab_QR.png"

REPO_URL = "https://github.com/yashsharma0812/DR-Feature-Kernel-Analysis"
NOTEBOOK_URL = (
    REPO_URL
    + "/blob/main/Team15_PBL3_Size_Diversity/notebooks/"
    + "Team15_PBL3_Ensemble_Size_Diversity.ipynb"
)
COLAB_URL = (
    "https://colab.research.google.com/github/yashsharma0812/"
    "DR-Feature-Kernel-Analysis/blob/main/Team15_PBL3_Size_Diversity/notebooks/"
    "Team15_PBL3_Ensemble_Size_Diversity.ipynb"
)

NAVY = RGBColor(13, 32, 56)
NAVY_2 = RGBColor(26, 52, 82)
TEAL = RGBColor(17, 139, 145)
CYAN = RGBColor(96, 210, 205)
BLUE = RGBColor(63, 109, 169)
ORANGE = RGBColor(238, 139, 53)
GREEN = RGBColor(73, 151, 94)
RED = RGBColor(190, 75, 79)
WHITE = RGBColor(255, 255, 255)
LIGHT = RGBColor(244, 248, 251)
MID = RGBColor(218, 228, 237)
DARK = RGBColor(37, 48, 62)
GREY = RGBColor(91, 106, 123)


def text(slide, x, y, w, h, value, size=18, color=DARK, bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = str(value)
    r.font.name = "Aptos"
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=15, color=DARK, gap=7):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"•  {item}"
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(gap)
    return box


def panel(slide, x, y, w, h, fill=LIGHT, line=MID, rounded=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    return shape


def picture(slide, path, x, y, w, h, border=True):
    path = Path(path)
    if border:
        panel(slide, x, y, w, h, WHITE, MID, False)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(
        str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
        Inches(pw), Inches(ph)
    )


def base_slide(prs, number, section, title_value):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    text(slide, 0.47, 0.18, 8.2, 0.22, section.upper(), 9.5, TEAL, True)
    text(slide, 0.47, 0.48, 12.2, 0.48, title_value, 24, NAVY, True)
    mark = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.47), Inches(1.06), Inches(1.05), Inches(0.045))
    mark.fill.solid(); mark.fill.fore_color.rgb = TEAL; mark.line.fill.background()
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(7.12), Inches(12.53), Inches(0.015))
    line.fill.solid(); line.fill.fore_color.rgb = MID; line.line.fill.background()
    text(slide, 0.47, 7.19, 10.5, 0.15,
         "TEAM 15 · 21CSC305P — MACHINE LEARNING · SRM IST", 8.3, GREY)
    text(slide, 11.65, 7.17, 1.05, 0.18, f"{number:02d} / 15", 8.5, GREY, True, PP_ALIGN.RIGHT)
    return slide


def note(slide, value):
    slide.notes_slide.notes_text_frame.text = value.strip()


def card(slide, x, y, w, h, value, label, color=TEAL):
    panel(slide, x, y, w, h, WHITE, MID)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.07), Inches(h))
    accent.fill.solid(); accent.fill.fore_color.rgb = color; accent.line.fill.background()
    text(slide, x + 0.17, y + 0.10, w - 0.25, 0.42, value, 22, color, True)
    text(slide, x + 0.17, y + 0.59, w - 0.25, 0.25, label, 9.5, GREY)


def table(slide, x, y, w, h, headers, rows, font_size=10, widths=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    tab = shape.table
    if widths:
        for i, width in enumerate(widths):
            tab.columns[i].width = Inches(width)
    for c, header in enumerate(headers):
        cell = tab.cell(0, c)
        cell.text = str(header)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY_2
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        for run in cell.text_frame.paragraphs[0].runs:
            run.font.name = "Aptos"; run.font.size = Pt(font_size); run.font.bold = True; run.font.color.rgb = WHITE
    for r, row in enumerate(rows, 1):
        for c, value in enumerate(row):
            cell = tab.cell(r, c)
            cell.text = str(value)
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            cell.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.name = "Aptos"; run.font.size = Pt(font_size); run.font.color.rgb = DARK
    return tab


def hyperlink(slide, x, y, w, h, label, url, size=11):
    box = text(slide, x, y, w, h, label, size, BLUE, True)
    box.text_frame.paragraphs[0].runs[0].hyperlink.address = url
    return box


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qrcode.make(COLAB_URL).save(QR_PATH)

    final = pd.read_csv(RESULTS / "final_model_comparison.csv")
    sizes = pd.read_csv(RESULTS / "ensemble_size_results.csv")
    diversity = pd.read_csv(RESULTS / "diversity_results.csv")
    interaction = pd.read_csv(RESULTS / "size_diversity_interaction.csv")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1 — Title
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    text(slide, 0.65, 0.35, 8.5, 0.24, "TEAM 15 · PBL-3 · 21CSC305P — MACHINE LEARNING", 10, CYAN, True)
    text(slide, 0.68, 1.05, 8.4, 2.05,
         "Investigating the Effect of\nEnsemble Size and Model Diversity\non Predictive Performance",
         29, WHITE, True)
    text(slide, 0.7, 3.33, 8.1, 0.55,
         "A controlled study using retinal image-derived features for diabetic retinopathy classification",
         15.5, RGBColor(207, 220, 235))
    for i, (label, color) in enumerate([("SIZE", BLUE), ("DIVERSITY", TEAL), ("PERFORMANCE", ORANGE)]):
        x = 0.72 + i * 2.35
        panel(slide, x, 4.15, 2.05, 0.55, NAVY_2, color)
        text(slide, x, 4.28, 2.05, 0.22, label, 10.5, WHITE, True, PP_ALIGN.CENTER)
    panel(slide, 9.35, 0.72, 3.25, 5.95, NAVY_2, RGBColor(66, 91, 119))
    text(slide, 9.68, 1.0, 2.6, 0.27, "TEAM MEMBERS", 11.5, CYAN, True)
    members = [
        "Himanish Srivastava\nRA2411003010946",
        "Aymen Aamir\nRA2411003010951",
        "Yash Sharma\nRA2411003010957",
        "Hannah Imtiaz\nRA2411003010967",
    ]
    for i, member in enumerate(members):
        text(slide, 9.68, 1.49 + i * 0.83, 2.62, 0.58, member, 11.5, WHITE, i == 2)
    text(slide, 9.68, 5.07, 2.6, 0.27, "FACULTY", 10.5, CYAN, True)
    text(slide, 9.68, 5.46, 2.6, 0.65, "Dr. Aishwarya R\nSRM Institute of Science and Technology", 10.5, WHITE)
    text(slide, 0.7, 6.35, 8.1, 0.32, "Academic Year 2026–27 · Python / scikit-learn / Jupyter", 10.5, RGBColor(207, 220, 235))
    text(slide, 11.58, 7.08, 1.0, 0.18, "01 / 15", 8.5, RGBColor(207, 220, 235), True, PP_ALIGN.RIGHT)
    note(slide, "Introduce the revised faculty-assigned project title. Emphasize that this is not a simple contest between three classifiers. The investigation isolates learner count and learner diversity as experimental factors, using the three ensemble families as platforms. The dataset contains derived numerical descriptors, not retinal photographs. Limit claims to predictive evidence on this dataset.")

    # 2 — Problem
    slide = base_slide(prs, 2, "Problem understanding", "When Do Larger and More Diverse Ensembles Help?")
    panel(slide, 0.55, 1.42, 5.55, 4.95, LIGHT, MID)
    text(slide, 0.84, 1.72, 4.95, 0.35, "CENTRAL RESEARCH QUESTION", 11, TEAL, True)
    text(slide, 0.84, 2.18, 4.95, 1.55,
         "How do ensemble size and model diversity influence predictive performance, robustness, generalization, and computational efficiency?",
         20, NAVY, True)
    bullets(slide, 0.84, 4.02, 4.95, 1.55, [
        "Size = number of base learners",
        "Diversity = differences in learner predictions/errors",
        "Performance = quality, stability, generalization and cost",
    ], 13.5)
    factors = [
        ("INDEPENDENT FACTOR 1", "Ensemble size", "N = 1, 5, 10, 25, 50, 100, 200, 300", BLUE),
        ("INDEPENDENT FACTOR 2", "Model diversity", "Controlled through random feature-subset size", TEAL),
        ("OUTCOMES", "Predictive evidence", "F1, AUC, sensitivity, gaps, variability and time", ORANGE),
    ]
    for i, (tag, head, body, color) in enumerate(factors):
        y = 1.42 + i * 1.66
        panel(slide, 6.45, y, 6.15, 1.36, WHITE, color)
        text(slide, 6.72, y + 0.18, 2.1, 0.2, tag, 9, color, True)
        text(slide, 6.72, y + 0.49, 2.25, 0.3, head, 16, NAVY, True)
        text(slide, 9.1, y + 0.37, 3.15, 0.58, body, 12.5, DARK)
    note(slide, "Define the two factors before discussing algorithms. Size is directly controlled with n_estimators. Diversity is measured rather than assumed, and is manipulated through max_features in matched bootstrap tree ensembles. Outcomes extend beyond one score: we examine predictive metrics, cross-validation variability, generalization gaps, and measured runtime. The central question is conditional: benefits may plateau or depend on diversity.")

    # 3 — Objectives and hypotheses
    slide = base_slide(prs, 3, "Research design", "Objectives and Testable Hypotheses")
    hypotheses = [
        ("H1", "Size", "Initial gains followed by diminishing returns or a plateau", BLUE),
        ("H2", "Diversity", "Diversity may help when individual learners remain accurate", TEAL),
        ("H3", "Random features", "Feature randomization may increase diversity and generalization", GREEN),
        ("H4", "Interaction", "The effect of size may depend on diversity configuration", ORANGE),
        ("H5", "Efficiency", "Cost rises even when late predictive gains become small", RED),
    ]
    for i, (h, head, body, color) in enumerate(hypotheses):
        y = 1.34 + i * 1.06
        panel(slide, 0.6, y, 12.05, 0.85, WHITE, MID)
        panel(slide, 0.72, y + 0.12, 0.65, 0.58, color, color)
        text(slide, 0.72, y + 0.27, 0.65, 0.2, h, 12, WHITE, True, PP_ALIGN.CENTER)
        text(slide, 1.62, y + 0.19, 2.0, 0.25, head, 14.5, NAVY, True)
        text(slide, 3.66, y + 0.18, 8.55, 0.42, body, 13.2, DARK)
    panel(slide, 0.65, 6.72, 12.0, 0.25, NAVY_2, NAVY_2)
    text(slide, 0.85, 6.745, 11.6, 0.16, "Hypotheses are evaluated empirically and may be supported, rejected, or qualified.", 9.5, WHITE, True, PP_ALIGN.CENTER)
    note(slide, "Present H1 through H5 as hypotheses, not predetermined findings. H1 and H5 motivate the estimator-count sweep and timing measurements. H2 and H3 motivate explicit disagreement, double-fault and learner-strength measurements. H4 motivates the two-dimensional Random Forest grid. The evaluation is descriptive and controlled; no statistical-significance or causal claim is made.")

    # 4 — Dataset selection
    slide = base_slide(prs, 4, "Dataset selection", "Why the UCI Debrecen DR Dataset?")
    picture(slide, FIG / "eda/01_class_distribution.png", 0.58, 1.42, 6.35, 4.95)
    panel(slide, 7.25, 1.42, 5.35, 4.95, LIGHT, MID)
    text(slide, 7.55, 1.73, 4.75, 0.35, "SELECTION RATIONALE", 11, TEAL, True)
    bullets(slide, 7.55, 2.18, 4.65, 2.5, [
        "Moderate sample size makes controlled grids computationally feasible",
        "18 numerical image-derived predictors suit tree ensembles",
        "Binary target supports sensitivity, specificity, F1 and ROC-AUC",
        "Near-balanced classes reduce trivial majority-class behavior",
        "Same domain as PBL-1, but a new research protocol",
    ], 13.2)
    panel(slide, 7.55, 5.15, 4.75, 0.85, NAVY_2, NAVY_2)
    text(slide, 7.82, 5.35, 4.2, 0.35, "UCI source · Dataset ID 329", 13, WHITE, True, PP_ALIGN.CENTER)
    note(slide, "The official UCI dataset has 1,151 observations, 18 numerical predictors and one binary target. The inputs are retinal image-derived descriptors rather than raw fundus images. Its moderate size enables repeated cross-validation across a controlled grid. It is useful for algorithmic investigation but does not establish clinical validity or external generalizability.")

    # 5 — Dataset description
    slide = base_slide(prs, 5, "Dataset description", "Verified Schema, Quality Checks and Leakage-Safe Split")
    card(slide, 0.55, 1.36, 1.9, 1.0, "1,151", "observations", BLUE)
    card(slide, 2.62, 1.36, 1.9, 1.0, "18 + 1", "predictors + target", TEAL)
    card(slide, 4.69, 1.36, 1.9, 1.0, "0", "missing values", GREEN)
    card(slide, 6.76, 1.36, 1.9, 1.0, "5", "exact duplicate rows", ORANGE)
    card(slide, 8.83, 1.36, 1.9, 1.0, "0", "conflicting groups", GREEN)
    card(slide, 10.9, 1.36, 1.9, 1.0, "920 / 231", "train / test", RED)
    picture(slide, FIG / "eda/03_correlation_heatmap.png", 0.58, 2.68, 6.25, 3.82)
    panel(slide, 7.12, 2.68, 5.5, 3.82, LIGHT, MID)
    text(slide, 7.43, 2.96, 4.8, 0.35, "REPRODUCIBLE EVALUATION", 11, TEAL, True)
    bullets(slide, 7.43, 3.42, 4.75, 2.35, [
        "Stratified 80:20 holdout; seed 42",
        "Duplicate predictor vectors grouped before splitting",
        "Identical held-out observations for all final models",
        "Five-fold StratifiedGroupKFold on training data",
        "No test-set access during configuration selection",
        "No unnecessary scaling for tree ensembles",
    ], 12.8)
    note(slide, "The audit verified 18 predictors plus one target, no missing values, five exact duplicate rows and no predictor-duplicate groups with conflicting labels. Group-aware stratification prevents identical predictor vectors from leaking between training and test sets. The fixed split contains 920 training and 231 test observations. All exploratory selection occurs only through grouped cross-validation on the training portion.")

    # 6 — EDA
    slide = base_slide(prs, 6, "Exploratory data analysis", "EDA Insights That Matter for Tree Ensembles")
    picture(slide, FIG / "eda/02_feature_distributions.png", 0.55, 1.38, 6.12, 3.05)
    picture(slide, FIG / "eda/05_outlier_rates.png", 6.88, 1.38, 5.82, 3.05)
    panel(slide, 0.65, 4.72, 12.0, 1.64, LIGHT, MID)
    text(slide, 0.94, 4.95, 2.05, 0.25, "KEY INSIGHTS", 11, TEAL, True)
    bullets(slide, 2.77, 4.89, 9.45, 1.05, [
        "Several predictors are skewed and contain IQR-defined outliers; tree splits are comparatively robust to monotonic scale differences.",
        "Correlated inputs can create alternative useful split paths, but input correlation is not model diversity.",
        "No missing-value imputation is required; class balance supports stratified evaluation.",
    ], 12.2, gap=4)
    note(slide, "The feature panels show heterogeneous ranges, skew and visible outliers. Tree-based models are a reasonable fit because they do not require feature scaling and can model nonlinear thresholds. Correlation among predictors may influence which variables trees choose, but it must not be confused with diversity among trained learners. Diversity is measured later from validation predictions and errors.")

    # 7 — Approaches
    slide = base_slide(prs, 7, "Implementation", "Three Ensemble Mechanisms as Experimental Platforms")
    models = [
        ("BAGGING", "Parallel bootstrap trees", "All 18 features available at each split", "Independent votes reduce variance", BLUE),
        ("GRADIENT BOOSTING", "Sequential additive trees", "Each stage corrects residual structure", "Learning rate and depth control dynamics", ORANGE),
        ("RANDOM FOREST", "Parallel bootstrap trees", "Random feature subset at each split", "Feature randomization changes strength + diversity", GREEN),
    ]
    for i, (name, mechanism, change, purpose, color) in enumerate(models):
        x = 0.55 + i * 4.25
        panel(slide, x, 1.45, 3.75, 4.75, WHITE, color)
        panel(slide, x, 1.45, 3.75, 0.72, color, color, False)
        text(slide, x + 0.15, 1.69, 3.45, 0.22, name, 12, WHITE, True, PP_ALIGN.CENTER)
        text(slide, x + 0.25, 2.55, 3.25, 0.62, mechanism, 18, NAVY, True, PP_ALIGN.CENTER)
        text(slide, x + 0.3, 3.49, 3.15, 0.68, change, 13, DARK, False, PP_ALIGN.CENTER)
        panel(slide, x + 0.35, 4.52, 3.05, 0.92, LIGHT, MID)
        text(slide, x + 0.53, 4.73, 2.7, 0.48, purpose, 11.5, color, True, PP_ALIGN.CENTER)
    text(slide, 0.75, 6.5, 11.8, 0.3,
         "Controlled size sweep: change only n_estimators after training-only selection of non-size settings.",
         12.5, NAVY, True, PP_ALIGN.CENTER)
    note(slide, "Bagging and Random Forest are matched bootstrap tree ensembles, which makes their diversity comparison meaningful. Random Forest changes the feature candidates at each split. Gradient Boosting is sequential: its component trees contribute additive scores and are not independent voting classifiers. Therefore raw tree-level disagreement is not compared directly with Bagging and Random Forest; Gradient Boosting is analyzed through size, learning rate, depth, subsampling and learning curves.")

    # 8 — Experiment A
    slide = base_slide(prs, 8, "Experiment A", "Effect of Ensemble Size: Early Gains, Then Small Changes")
    picture(slide, FIG / "ensemble_size/size_vs_f1.png", 0.55, 1.36, 8.05, 4.95)
    panel(slide, 8.88, 1.36, 3.75, 4.95, LIGHT, MID)
    text(slide, 9.17, 1.68, 3.15, 0.3, "CV F1 PEAKS", 11, TEAL, True)
    peak_rows = []
    for model in ["Bagging", "Gradient Boosting", "Random Forest"]:
        sub = sizes[sizes["model"] == model]
        best = sub.loc[sub["f1_mean"].idxmax()]
        peak_rows.append((model, f"N={int(best['ensemble_size'])}", f"{best['f1_mean']:.3f}"))
    table(slide, 9.14, 2.15, 3.22, 1.62, ["Model", "Peak", "F1"], peak_rows, 9.3, [1.45, 0.9, 0.87])
    text(slide, 9.17, 4.12, 3.05, 0.3, "N=100 → N=300", 11, TEAL, True)
    changes = []
    for model in ["Bagging", "Gradient Boosting", "Random Forest"]:
        sub = sizes[sizes["model"] == model].set_index("ensemble_size")
        changes.append((model, f"{sub.loc[300, 'f1_mean'] - sub.loc[100, 'f1_mean']:+.3f}"))
    table(slide, 9.14, 4.56, 3.22, 1.2, ["Model", "Δ F1"], changes, 9.5, [2.08, 1.14])
    text(slide, 9.1, 5.91, 3.28, 0.25, "H1: qualified support", 12, NAVY, True, PP_ALIGN.CENTER)
    note(slide, "This figure reports grouped five-fold CV means across three seeds while holding non-size settings fixed. Bagging peaks near N=200 at F1 0.671, Gradient Boosting near N=200 at 0.702, and Random Forest near N=300 at 0.680. From N=100 to N=300, changes are approximately 0.000, +0.011 and +0.007 respectively. Curves show diminishing and sometimes non-monotonic changes, so H1 is supported only in a qualified sense.")

    # 9 — Experiment B
    slide = base_slide(prs, 9, "Experiment B", "Measured Diversity Must Be Interpreted With Learner Strength")
    picture(slide, FIG / "model_diversity/02_diversity_vs_performance.png", 0.55, 1.36, 7.1, 4.95)
    drows = []
    labels = {"Bagging": "Bagging full"}
    for _, row in diversity.iterrows():
        if row["model"] == "Bagging":
            label = "Bagging full"
        else:
            label = f"RF k={int(row['feature_subset'])}"
        drows.append((label, f"{row['disagreement_mean']:.3f}", f"{row['double_fault_mean']:.3f}",
                      f"{row['learner_accuracy_mean']:.3f}", f"{row['f1_mean']:.3f}"))
    table(slide, 7.88, 1.42, 4.72, 3.55,
          ["Config", "Dis.", "DF", "Learner", "F1"], drows, 8.7, [1.38, .78, .78, .9, .88])
    panel(slide, 7.9, 5.24, 4.68, 1.03, NAVY_2, NAVY_2)
    text(slide, 8.15, 5.44, 4.18, 0.46,
         "Highest disagreement: RF k=1\nBest F1: RF k=6",
         12.5, WHITE, True, PP_ALIGN.CENTER)
    note(slide, "At N=100 under matched tree and bootstrap settings, RF k=1 has the highest disagreement, about 0.435, but lower individual-learner accuracy, about 0.578, and F1 about 0.667. RF k=6 achieves the best F1, about 0.692, with lower disagreement, about 0.404, and stronger learners. The double-fault measure is also reported because shared errors matter. This supports the qualified H2 interpretation: diversity helps only alongside sufficient learner strength.")

    # 10 — Experiment C
    slide = base_slide(prs, 10, "Experiment C", "Ensemble Size × Diversity: The Best Setting Is Not the Most Diverse")
    picture(slide, FIG / "size_diversity_interaction/01_f1_heatmap.png", 0.55, 1.37, 6.25, 4.95)
    picture(slide, FIG / "size_diversity_interaction/02_disagreement_heatmap.png", 6.98, 1.37, 5.68, 4.95)
    best = interaction.loc[interaction["f1_mean"].idxmax()]
    most = interaction.loc[interaction["disagreement_mean"].idxmax()]
    panel(slide, 2.02, 6.42, 9.3, 0.43, NAVY_2, NAVY_2)
    text(slide, 2.18, 6.52, 9.0, 0.2,
         f"Best F1 {best['f1_mean']:.3f}: N={int(best['ensemble_size'])}, k={int(best['feature_subset'])}  ·  Highest disagreement {most['disagreement_mean']:.3f}: N={int(most['ensemble_size'])}, k={int(most['feature_subset'])}",
         10.5, WHITE, True, PP_ALIGN.CENTER)
    note(slide, "The integrative grid varies six ensemble sizes and five max_features values using the same grouped CV folds. The best mean F1 is about 0.696 at N=300, k=6. Maximum disagreement is about 0.446 at N=10, k=1, where F1 is only about 0.648. The descriptive disagreement–F1 correlation is negative in this grid, about -0.510, but this is not causal evidence because feature randomization changes both diversity and learner strength. H4 is evaluated through the changing surface rather than a single ranking.")

    # 11 — Comparative performance
    slide = base_slide(prs, 11, "Comparative performance", "Final Training-Selected Configurations on the Untouched Test Set")
    picture(slide, FIG / "final_comparison/01_final_performance.png", 0.55, 1.36, 6.3, 3.45)
    rows = []
    for _, r in final.iterrows():
        config = f"N={int(r['ensemble_size'])}"
        if r["model"] == "Random Forest":
            config += ", k=6"
        elif r["model"] == "Bagging":
            config += ", full"
        else:
            config += ", sequential"
        rows.append((r["model"], config, f"{r['accuracy']:.3f}", f"{r['sensitivity']:.3f}",
                     f"{r['specificity']:.3f}", f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}"))
    table(slide, 7.05, 1.42, 5.58, 2.32,
          ["Model", "Selected", "Acc", "Sens", "Spec", "F1", "AUC"], rows, 8.5,
          [1.2, 1.4, .56, .61, .61, .56, .64])
    picture(slide, FIG / "ensemble_size/size_vs_training_time.png", 0.55, 4.98, 6.3, 1.42)
    panel(slide, 7.05, 4.05, 5.58, 2.35, LIGHT, MID)
    text(slide, 7.33, 4.32, 4.95, 0.3, "EVIDENCE-BASED INTERPRETATION", 11, TEAL, True)
    bullets(slide, 7.33, 4.76, 4.92, 1.25, [
        "Gradient Boosting achieved the strongest held-out metrics in this split.",
        "RF N=300, k=6 balanced diversity and learner strength.",
        "Larger ensembles increased measured training cost while late F1 gains were small.",
    ], 11.8, gap=4)
    note(slide, "Final configurations were selected using training-only CV and evaluated once on the untouched 231-observation test set. Gradient Boosting N=200 achieved F1 0.749 and AUC 0.818; Random Forest N=300, k=6 achieved F1 0.686 and AUC 0.785; Bagging N=25 achieved F1 0.667 and AUC 0.757. These are split-specific predictive results, not clinical claims. Runtime values are measured on the execution environment and should be compared relatively.")

    # 12 — Inferences
    slide = base_slide(prs, 12, "Result interpretation", "What the Experiments Reveal")
    findings = [
        ("01", "Larger is not automatically better", "Most gains occur early; late changes are small and sometimes non-monotonic.", BLUE),
        ("02", "Maximum diversity is not optimal", "RF k=1 maximized disagreement but weakened individual learners and F1.", TEAL),
        ("03", "Balance matters", "RF k=6 offered the best strength–diversity balance in the controlled grid.", GREEN),
        ("04", "Boosting scales differently", "Sequential correction produced the strongest held-out result; raw tree disagreement is not comparable.", ORANGE),
        ("05", "Efficiency constrains complexity", "Training cost rose with N even after performance approached a plateau.", RED),
    ]
    for i, (num, head, body, color) in enumerate(findings):
        y = 1.34 + i * 1.06
        panel(slide, 0.58, y, 12.08, 0.86, WHITE, MID)
        text(slide, 0.8, y + 0.19, 0.65, 0.35, num, 15, color, True, PP_ALIGN.CENTER)
        text(slide, 1.62, y + 0.17, 3.15, 0.32, head, 14, NAVY, True)
        text(slide, 4.93, y + 0.16, 7.2, 0.46, body, 12.2, DARK)
    text(slide, 0.75, 6.73, 11.85, 0.2,
         "Limitations: moderate dataset size · one domain · descriptive comparisons · no external clinical validation",
         10.3, GREY, True, PP_ALIGN.CENTER)
    note(slide, "Connect each result back to H1–H5. H1 and H2 receive qualified support. H3 is evaluated through matched Bagging/RF settings: feature randomization changes both diversity and strength. H4 is supported descriptively by the interaction surface. H5 is supported by measured growth in training cost alongside small late gains. Limitations include one moderate-sized dataset, no external validation, descriptive rather than inferential comparisons, and hardware-dependent timing.")

    # 13 — Reproducibility
    slide = base_slide(prs, 13, "Notebook and reproducibility", "Inspect, Re-run and Verify the Complete Study")
    panel(slide, 0.58, 1.42, 8.25, 4.95, LIGHT, MID)
    text(slide, 0.9, 1.75, 7.55, 0.3, "EXECUTED NOTEBOOK", 11, TEAL, True)
    text(slide, 0.9, 2.2, 7.5, 0.65, "Team15_PBL3_Ensemble_Size_Diversity.ipynb", 18, NAVY, True)
    bullets(slide, 0.9, 3.02, 7.4, 1.34, [
        "20 research sections · 21 executed code cells · retained outputs",
        "Saved split indices, experiment configurations, CSVs and 300-DPI figures",
        "Full rebuild available with PBL3_REBUILD=1",
        "Five automated tests pass",
    ], 12.8)
    hyperlink(slide, 0.92, 4.62, 7.35, 0.28, "Open notebook on GitHub", NOTEBOOK_URL, 12)
    hyperlink(slide, 0.92, 5.08, 7.35, 0.28, "Open directly in Google Colab", COLAB_URL, 12)
    hyperlink(slide, 0.92, 5.54, 7.35, 0.28, "Open project repository", REPO_URL, 12)
    panel(slide, 9.15, 1.42, 3.45, 4.95, WHITE, MID)
    picture(slide, QR_PATH, 9.55, 1.9, 2.65, 2.65, False)
    text(slide, 9.45, 4.88, 2.85, 0.65, "SCAN TO OPEN\nIN GOOGLE COLAB", 12, NAVY, True, PP_ALIGN.CENTER)
    text(slide, 9.45, 5.68, 2.85, 0.35, "Links are embedded and clickable", 9.5, GREY, False, PP_ALIGN.CENTER)
    note(slide, "Use the QR code or clickable links to inspect the executed notebook and repository. The notebook loads verified saved outputs by default for a fast presentation run. Setting the PBL3_REBUILD environment variable to 1 performs the complete experiment suite again. Split indices and configurations are saved, and tests cover key data and experiment invariants. Repository links become externally reachable once this local pivot is pushed to GitHub.")

    # 14 — Conclusion
    slide = base_slide(prs, 14, "Conclusion", "Answer to the Revised Research Question")
    panel(slide, 0.62, 1.4, 12.0, 1.2, NAVY_2, NAVY_2)
    text(slide, 0.98, 1.72, 11.3, 0.55,
         "Effective ensembles require enough learners and useful diversity—without sacrificing individual learner strength.",
         20, WHITE, True, PP_ALIGN.CENTER)
    conclusions = [
        ("SIZE", "Performance generally stabilized by moderate-to-large N; extra learners delivered small late gains.", BLUE),
        ("DIVERSITY", "Random feature selection increased disagreement most at k=1, but k=6 produced better predictive balance.", TEAL),
        ("INTERACTION", "The best observed RF grid point was N=300, k=6—not the maximum-diversity configuration.", GREEN),
        ("TRADE-OFF", "Gradient Boosting N=200 led held-out performance, while all families incurred increasing cost with N.", ORANGE),
    ]
    for i, (head, body, color) in enumerate(conclusions):
        x = 0.62 + (i % 2) * 6.1
        y = 2.98 + (i // 2) * 1.55
        panel(slide, x, y, 5.9, 1.24, WHITE, color)
        text(slide, x + 0.24, y + 0.18, 1.25, 0.25, head, 11, color, True)
        text(slide, x + 1.48, y + 0.16, 4.1, 0.68, body, 12.4, DARK)
    text(slide, 0.75, 6.45, 11.85, 0.3,
         "Research conclusion: optimize the balance of learner count, diversity, strength and cost—not tree count alone.",
         13.5, NAVY, True, PP_ALIGN.CENTER)
    note(slide, "Conclude with the main research answer. More learners can improve stability and performance, but benefits diminish. Diversity is valuable only when the component learners retain useful predictive strength. In the controlled Random Forest grid, N=300 and k=6 gave the best mean F1; maximum disagreement at N=10, k=1 did not. Gradient Boosting produced the strongest final held-out metrics but follows a different sequential mechanism. These findings are dataset-specific and require external validation.")

    # 15 — References
    slide = base_slide(prs, 15, "References", "Data, Methods and Software")
    refs = [
        "Antal, B. & Hajdu, A. (2014). An ensemble-based system for automatic screening of diabetic retinopathy. Knowledge-Based Systems, 60, 20–27.",
        "UCI Machine Learning Repository. Diabetic Retinopathy Debrecen Dataset (Dataset 329). https://archive.ics.uci.edu/dataset/329/diabetic",
        "Breiman, L. (1996). Bagging Predictors. Machine Learning, 24, 123–140.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.",
        "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189–1232.",
        "Kuncheva, L. I. & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles and their relationship with ensemble accuracy. Machine Learning, 51, 181–207.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.",
        "Project artifacts: executed notebook, saved CSV results, figures, methodology, changelog and viva guide in the linked repository.",
    ]
    panel(slide, 0.6, 1.38, 12.05, 5.48, LIGHT, MID)
    bullets(slide, 0.9, 1.68, 11.45, 4.85, refs, 11.4, gap=5)
    note(slide, "These references cover the dataset, the three ensemble mechanisms, explicit classifier-diversity measures and the primary software library. The repository contains the exact configurations and actual outputs used in every result slide. End by inviting questions about controlled variables, diversity measurement, leakage prevention, plateau behavior or the strength–diversity trade-off.")

    prs.save(OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    build()
