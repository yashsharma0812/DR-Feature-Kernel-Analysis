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
EDA = ROOT / "figures" / "eda"
MODELS = ROOT / "figures" / "model_comparison"
RESULTS = ROOT / "results"
PRESENTATION = ROOT / "presentation"
OUTPUT = PRESENTATION / "Team15_PBL3_Enhanced_Presentation.pptx"
QR_PATH = PRESENTATION / "pbl3_colab_qr.png"

NOTEBOOK_URL = "https://github.com/yashsharma0812/DR-Feature-Kernel-Analysis/blob/main/Team15_PBL3/Team15_PBL3_Ensemble_Comparison.ipynb"
COLAB_URL = "https://colab.research.google.com/github/yashsharma0812/DR-Feature-Kernel-Analysis/blob/main/Team15_PBL3/Team15_PBL3_Ensemble_Comparison.ipynb"
REPO_URL = "https://github.com/yashsharma0812/DR-Feature-Kernel-Analysis"

NAVY = RGBColor(15, 31, 55)
NAVY_2 = RGBColor(27, 50, 83)
BLUE = RGBColor(76, 120, 168)
ORANGE = RGBColor(242, 142, 43)
GREEN = RGBColor(89, 161, 79)
RED = RGBColor(196, 78, 82)
TEAL = RGBColor(41, 139, 143)
LIGHT = RGBColor(244, 247, 251)
MID = RGBColor(219, 228, 239)
DARK = RGBColor(35, 42, 52)
GREY = RGBColor(96, 108, 124)
WHITE = RGBColor(255, 255, 255)


def add_text(slide, x, y, w, h, text, size=18, color=DARK, bold=False,
             align=PP_ALIGN.LEFT, font="Aptos", valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_bullets(slide, x, y, w, h, bullets, size=17, color=DARK, spacing=6):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for index, item in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = item
        paragraph.level = 0
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(spacing)
        paragraph.bullet = True
    return box


def add_panel(slide, x, y, w, h, fill=LIGHT, line=MID, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    return shape


def add_picture_fit(slide, path: Path, x, y, w, h, border=True):
    if border:
        add_panel(slide, x, y, w, h, WHITE, MID, False)
    with Image.open(path) as image:
        iw, ih = image.size
    ratio = min(w / iw, h / ih)
    pw, ph = iw * ratio, ih * ratio
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    return slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))


def add_footer(slide, number):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.35), Inches(7.12), Inches(12.63), Inches(0.02))
    line.fill.solid(); line.fill.fore_color.rgb = MID; line.line.fill.background()
    add_text(slide, 0.42, 7.18, 9.6, 0.2,
             "21CSC305P — Machine Learning · SRM Institute of Science and Technology",
             8.5, GREY)
    add_text(slide, 11.65, 7.16, 1.1, 0.22, f"{number:02d} / 12", 9, GREY, True, PP_ALIGN.RIGHT)


def add_header(slide, section, title, number):
    add_text(slide, 0.45, 0.2, 8.0, 0.28, section.upper(), 10, TEAL, True)
    add_text(slide, 0.45, 0.52, 12.2, 0.52, title, 25, NAVY, True)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(1.08), Inches(1.0), Inches(0.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    add_footer(slide, number)


def add_metric_card(slide, x, y, w, h, value, label, color=BLUE):
    panel = add_panel(slide, x, y, w, h, WHITE, MID)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h))
    accent.fill.solid(); accent.fill.fore_color.rgb = color; accent.line.fill.background()
    add_text(slide, x + 0.18, y + 0.12, w - 0.3, h * 0.5, value, 23, color, True)
    add_text(slide, x + 0.18, y + h * 0.57, w - 0.3, h * 0.3, label, 10, GREY)


def add_table(slide, x, y, w, h, data, headers, col_widths=None, font_size=11):
    rows, cols = len(data) + 1, len(headers)
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    if col_widths:
        for idx, width in enumerate(col_widths):
            table.columns[idx].width = Inches(width)
    for col, header in enumerate(headers):
        cell = table.cell(0, col)
        cell.text = str(header)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY_2
        for run in cell.text_frame.paragraphs[0].runs:
            run.font.bold = True; run.font.color.rgb = WHITE; run.font.size = Pt(font_size)
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    for row_index, row in enumerate(data, start=1):
        for col, value in enumerate(row):
            cell = table.cell(row_index, col)
            cell.text = str(value)
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE if row_index % 2 else LIGHT
            for run in cell.text_frame.paragraphs[0].runs:
                run.font.color.rgb = DARK; run.font.size = Pt(font_size)
            cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER if col else PP_ALIGN.LEFT
    return table


def add_link(slide, x, y, w, h, label, url, color=BLUE):
    box = add_text(slide, x, y, w, h, label, 10.5, color, True)
    paragraph = box.text_frame.paragraphs[0]
    if paragraph.runs:
        paragraph.runs[0].hyperlink.address = url
    return box


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    background.fill.solid()
    background.fill.fore_color.rgb = WHITE
    return slide


def build():
    PRESENTATION.mkdir(parents=True, exist_ok=True)
    qrcode.make(COLAB_URL).save(QR_PATH)
    metrics = pd.read_csv(RESULTS / "model_comparison.csv").set_index("model")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1 — Title
    slide = new_slide(prs)
    slide.background.fill.fore_color.rgb = NAVY
    add_text(slide, 0.55, 0.35, 8.5, 0.25, "TEAM 15 · PBL-3 · 21CSC305P — MACHINE LEARNING", 10, RGBColor(117, 218, 214), True)
    add_text(slide, 0.6, 1.05, 8.7, 1.6,
             "Comparative Analysis of Bagging,\nBoosting, and Random Forests",
             30, WHITE, True)
    add_text(slide, 0.62, 2.75, 8.2, 0.55,
             "Ensemble Prediction of Diabetic Retinopathy Using Retinal Image-Derived Features",
             16, RGBColor(211, 222, 238))
    for idx, (name, color) in enumerate([("Bagging", BLUE), ("Gradient Boosting", ORANGE), ("Random Forest", GREEN)]):
        x = 0.7 + idx * 2.55
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(3.48), Inches(0.17), Inches(0.17))
        dot.fill.solid(); dot.fill.fore_color.rgb = color; dot.line.fill.background()
        add_text(slide, x + 0.25, 3.43, 2.2, 0.28, name, 11, WHITE, True)
    add_panel(slide, 9.35, 0.75, 3.3, 5.8, NAVY_2, RGBColor(63, 89, 124))
    add_text(slide, 9.7, 1.1, 2.6, 0.3, "TEAM MEMBERS", 12, RGBColor(117, 218, 214), True)
    members = [
        "Himanish Srivastava\nRA2411003010946",
        "Aymen Aamir\nRA2411003010951",
        "Yash Sharma\nRA2411003010957",
        "Hannah Imtiaz\nRA2411003010967",
    ]
    for i, member in enumerate(members):
        add_text(slide, 9.7, 1.6 + i * 0.85, 2.6, 0.62, member, 12, WHITE, i == 2)
    add_text(slide, 0.65, 6.32, 8.3, 0.55,
             "Faculty: Dr. Aishwarya R  ·  SRM Institute of Science and Technology  ·  Academic Year 2026–27",
             10.5, RGBColor(211, 222, 238))
    add_text(slide, 11.55, 7.05, 1.1, 0.22, "01 / 12", 9, RGBColor(211, 222, 238), True, PP_ALIGN.RIGHT)

    # 2 — Continuity
    slide = new_slide(prs); add_header(slide, "Research continuity", "Where We Left Off and What PBL-3 Adds", 2)
    stages = [
        ("PBL-1", "Supervised foundations", "GNB conditional independence\nSVM kernel geometry\nDebrecen dataset", BLUE),
        ("PBL-2", "Profiles and sequences", "Clustering\nMarkov transitions\nProtocol AA", TEAL),
        ("PBL-3", "Ensemble prediction", "Bagging vs Boosting vs RF\nNew independent protocol\nDebrecen dataset", ORANGE),
    ]
    for i, (tag, title, body, color) in enumerate(stages):
        x = 0.55 + i * 4.25
        add_panel(slide, x, 1.45, 3.7, 3.35, LIGHT, MID)
        add_text(slide, x + 0.22, 1.67, 0.9, 0.3, tag, 12, color, True)
        add_text(slide, x + 0.22, 2.08, 3.2, 0.55, title, 19, NAVY, True)
        add_text(slide, x + 0.22, 2.82, 3.2, 1.35, body, 14, DARK)
        if i < 2:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 3.72), Inches(2.75), Inches(0.5), Inches(0.55))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = MID; arrow.line.fill.background()
    add_panel(slide, 0.7, 5.15, 11.95, 1.35, NAVY_2, NAVY_2)
    add_text(slide, 1.0, 5.38, 11.3, 0.35,
             "PBL-3 asks: Which ensemble strategy provides the most effective and reliable prediction?",
             18, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, 1.0, 5.85, 11.3, 0.35,
             "Same research domain; independent split, tuning, outputs, notebook, and conclusions.",
             12, RGBColor(204, 220, 237), False, PP_ALIGN.CENTER)

    # 3 — Motivation
    slide = new_slide(prs); add_header(slide, "Introduction", "Research Motivation and Evaluation Framework", 3)
    add_panel(slide, 0.55, 1.4, 5.55, 4.85, LIGHT, MID)
    add_text(slide, 0.82, 1.68, 5.0, 0.48, "Central research question", 13, TEAL, True)
    add_text(slide, 0.82, 2.18, 4.95, 1.25,
             "Which of the three ensemble approaches provides the most effective and reliable prediction of the Debrecen DR label?",
             20, NAVY, True)
    add_bullets(slide, 0.82, 3.62, 4.95, 2.1, [
        "Identical observations and folds",
        "Sensitivity and false negatives matter",
        "Generalization and stability matter",
        "Computational cost is measured",
    ], 14)
    dimensions = [
        ("Predictive\nquality", BLUE), ("Sensitivity", RED), ("CV\nstability", TEAL),
        ("Generalization", ORANGE), ("Efficiency", GREEN),
    ]
    for i, (label, color) in enumerate(dimensions):
        x = 6.45 + (i % 3) * 2.05
        y = 1.55 + (i // 3) * 2.05
        add_panel(slide, x, y, 1.72, 1.55, WHITE, color)
        add_text(slide, x + 0.12, y + 0.48, 1.48, 0.65, label, 14, color, True, PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, 6.45, 5.55, 5.9, 0.55,
             "Debrecen features → shared protocol → three ensembles → evidence-based comparison",
             13, NAVY, True, PP_ALIGN.CENTER)

    # 4 — Dataset with class chart
    slide = new_slide(prs); add_header(slide, "Dataset", "UCI Debrecen Diabetic Retinopathy Dataset", 4)
    add_metric_card(slide, 0.55, 1.42, 1.8, 1.1, "1,151", "observations", BLUE)
    add_metric_card(slide, 2.5, 1.42, 1.8, 1.1, "18", "predictors", TEAL)
    add_metric_card(slide, 4.45, 1.42, 1.8, 1.1, "0", "missing values", GREEN)
    add_metric_card(slide, 6.4, 1.42, 1.8, 1.1, "5", "duplicate rows", ORANGE)
    add_picture_fit(slide, EDA / "01_class_distribution.png", 0.6, 2.85, 6.25, 3.75)
    add_panel(slide, 7.2, 2.85, 5.45, 3.75, LIGHT, MID)
    add_text(slide, 7.48, 3.1, 4.9, 0.4, "Why this dataset?", 17, NAVY, True)
    add_bullets(slide, 7.48, 3.58, 4.85, 2.3, [
        "Image-derived numerical descriptors",
        "Binary label supports sensitivity/specificity",
        "Suitable for tree ensembles",
        "Moderate size enables reproducible tuning",
        "Tabular data—not raw retinal photographs",
    ], 13.5)
    add_text(slide, 7.48, 6.05, 4.8, 0.3, "Verified schema: 18 inputs + 1 target", 11, TEAL, True)

    # 5 — Data validation
    slide = new_slide(prs); add_header(slide, "Data preparation", "Validation, Split and Leakage Prevention", 5)
    split_data = [["Training", "920", "432", "488"], ["Held-out test", "231", "108", "123"]]
    add_table(slide, 0.55, 1.55, 6.1, 1.55, split_data, ["Partition", "Rows", "Class 0", "Class 1"],
              [2.3, 1.2, 1.2, 1.2], 12)
    add_picture_fit(slide, EDA / "05_missing_values.png", 7.05, 1.42, 5.55, 3.0)
    add_panel(slide, 0.55, 3.45, 6.1, 2.75, LIGHT, MID)
    add_text(slide, 0.82, 3.72, 5.5, 0.38, "Leakage controls", 16, NAVY, True)
    add_bullets(slide, 0.82, 4.18, 5.4, 1.65, [
        "Duplicate vectors kept in one partition",
        "No train–test index overlap",
        "Tuning uses training folds only",
        "Test set evaluated once after selection",
    ], 13.5)
    add_panel(slide, 7.05, 4.72, 5.55, 1.48, NAVY_2, NAVY_2)
    add_text(slide, 7.35, 4.98, 4.95, 0.35, "No scaling or imputation required", 16, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, 7.35, 5.42, 4.95, 0.35, "Random state = 42  ·  Positive class = 1", 11, RGBColor(207, 220, 235), False, PP_ALIGN.CENTER)

    # 6 — EDA visual wall
    slide = new_slide(prs); add_header(slide, "Exploratory analysis", "Feature Structure and Target-Linked Patterns", 6)
    add_picture_fit(slide, EDA / "03_feature_correlation_heatmap.png", 0.45, 1.35, 4.0, 4.65)
    add_picture_fit(slide, EDA / "06_feature_target_comparison.png", 4.63, 1.35, 3.65, 4.65)
    add_picture_fit(slide, EDA / "02_selected_feature_histograms.png", 8.47, 1.35, 4.4, 4.65)
    add_panel(slide, 0.62, 6.15, 12.05, 0.65, NAVY_2, NAVY_2)
    add_text(slide, 0.88, 6.31, 11.5, 0.3,
             "Insight: correlated detector families and overlapping class distributions motivate multivariate ensembles; association is not causation.",
             11.5, WHITE, True, PP_ALIGN.CENTER)

    # 7 — Model distinction
    slide = new_slide(prs); add_header(slide, "Problem statement", "Bagging, Boosting and Random Forest", 7)
    cards = [
        ("BAGGING", BLUE, ["Bootstrap rows", "Independent trees", "All features at splits", "Variance reduction"]),
        ("GRADIENT BOOSTING", ORANGE, ["Sequential trees", "Corrects current errors", "Learning-rate control", "Loss optimization"]),
        ("RANDOM FOREST", GREEN, ["Bootstrap rows", "Independent trees", "Random features at splits", "Tree decorrelation"]),
    ]
    for i, (title, color, points) in enumerate(cards):
        x = 0.55 + i * 4.25
        add_panel(slide, x, 1.5, 3.75, 4.45, WHITE, color)
        ribbon = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.5), Inches(3.75), Inches(0.68))
        ribbon.fill.solid(); ribbon.fill.fore_color.rgb = color; ribbon.line.fill.background()
        add_text(slide, x + 0.15, 1.68, 3.45, 0.25, title, 13, WHITE, True, PP_ALIGN.CENTER)
        add_bullets(slide, x + 0.32, 2.48, 3.05, 2.65, points, 14)
    add_panel(slide, 1.2, 6.15, 10.9, 0.62, NAVY_2, NAVY_2)
    add_text(slide, 1.45, 6.31, 10.4, 0.28,
             "Critical distinction: Random Forest is Bagging plus random feature selection; Boosting is sequential.",
             12, WHITE, True, PP_ALIGN.CENTER)

    # 8 — Methodology
    slide = new_slide(prs); add_header(slide, "Methodology", "Controlled Tuning and Selected Configurations", 8)
    steps = [("920", "training rows"), ("5", "grouped folds"), ("24", "candidates/model"), ("F1", "selection metric"), ("231", "test rows")]
    for i, (value, label) in enumerate(steps):
        x = 0.55 + i * 2.5
        add_metric_card(slide, x, 1.4, 2.05, 1.0, value, label, [BLUE, TEAL, ORANGE, RED, GREEN][i])
        if i < 4:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.07), Inches(1.73), Inches(0.38), Inches(0.32))
            arrow.fill.solid(); arrow.fill.fore_color.rgb = MID; arrow.line.fill.background()
    config_data = [
        ["Bagging", "300", "unrestricted", "leaf=3; sample=0.8", "0.685"],
        ["Gradient Boosting", "200", "2", "lr=0.1; sample=0.8", "0.718"],
        ["Random Forest", "100", "16", "leaf=1; max_features=√p", "0.703"],
    ]
    add_table(slide, 0.65, 2.85, 12.0, 2.35, config_data,
              ["Model", "Estimators", "Depth", "Other selected settings", "Best CV F1"],
              [2.25, 1.35, 1.25, 4.45, 1.45], 11)
    add_panel(slide, 1.2, 5.62, 10.9, 0.72, LIGHT, MID)
    add_text(slide, 1.5, 5.83, 10.3, 0.28,
             "The untouched test set is excluded from all hyperparameter decisions.",
             13, NAVY, True, PP_ALIGN.CENTER)

    # 9 — Generalization visuals
    slide = new_slide(prs); add_header(slide, "Model behaviour", "Stability and Generalization", 9)
    add_picture_fit(slide, MODELS / "07_train_test_comparison.png", 0.55, 1.38, 6.05, 4.55)
    add_picture_fit(slide, MODELS / "08_cv_mean_stability.png", 6.75, 1.38, 5.95, 4.55)
    add_panel(slide, 0.8, 6.12, 11.75, 0.67, NAVY_2, NAVY_2)
    add_text(slide, 1.05, 6.29, 11.25, 0.28,
             "Gradient Boosting has the smallest train–test gap and lowest CV F1 SD (0.025); RF and Bagging show stronger overfitting signals.",
             11.5, WHITE, True, PP_ALIGN.CENTER)

    # 10 — Results visual comparison
    slide = new_slide(prs); add_header(slide, "Results", "Held-out Comparison on the Same 231 Observations", 10)
    add_picture_fit(slide, MODELS / "01_model_performance_comparison.png", 0.45, 1.3, 7.35, 3.1)
    add_picture_fit(slide, MODELS / "02_confusion_matrices.png", 0.45, 4.55, 7.35, 2.1)
    add_picture_fit(slide, MODELS / "03_combined_roc_curves.png", 8.0, 1.3, 4.9, 5.35)
    add_panel(slide, 8.15, 5.7, 4.55, 0.78, NAVY_2, NAVY_2)
    add_text(slide, 8.35, 5.88, 4.15, 0.4,
             "Gradient Boosting: F1 0.749 · AUC 0.818 · FN 29",
             11, WHITE, True, PP_ALIGN.CENTER)

    # 11 — Inference, reproducibility, links
    slide = new_slide(prs); add_header(slide, "Inference & reproducibility", "Interpretation, Efficiency and Project Access", 11)
    add_picture_fit(slide, MODELS / "05_computational_efficiency.png", 0.45, 1.35, 6.25, 3.25)
    add_panel(slide, 0.55, 4.78, 6.05, 1.62, LIGHT, MID)
    add_text(slide, 0.82, 4.98, 5.5, 0.3, "Verified notebook", 13, NAVY, True)
    add_text(slide, 0.82, 5.35, 5.5, 0.75,
             "34 cells · 17 executed code cells · outputs under all 17\n0 errors · 15 figures at 300 DPI · 5 tests passed",
             11.5, DARK)
    add_panel(slide, 6.95, 1.35, 5.85, 5.05, WHITE, MID)
    add_text(slide, 7.25, 1.62, 3.2, 0.3, "Open the complete project", 15, NAVY, True)
    qr = slide.shapes.add_picture(str(QR_PATH), Inches(10.45), Inches(1.58), Inches(1.85), Inches(1.85))
    qr.click_action.hyperlink.address = COLAB_URL
    add_text(slide, 10.35, 3.48, 2.05, 0.4, "Scan for Colab", 9.5, GREY, True, PP_ALIGN.CENTER)
    add_link(slide, 7.25, 2.25, 3.0, 0.5, "▶ Notebook on GitHub", NOTEBOOK_URL)
    add_link(slide, 7.25, 2.86, 3.0, 0.5, "▶ Open in Google Colab", COLAB_URL, ORANGE)
    add_link(slide, 7.25, 3.47, 3.0, 0.5, "▶ GitHub repository", REPO_URL, GREEN)
    add_text(slide, 7.25, 4.3, 5.0, 1.55,
             "Inference\nGradient Boosting leads accuracy, sensitivity, F1 and ROC-AUC, reducing false negatives from 37 to 29. Timing is machine-specific; results are not clinical validation.",
             11.5, DARK, False)

    # 12 — Conclusion
    slide = new_slide(prs); add_header(slide, "Conclusion", "From Individual Trees to Evidence-Based Ensemble Selection", 12)
    add_panel(slide, 0.55, 1.4, 6.15, 4.95, LIGHT, MID)
    add_text(slide, 0.85, 1.72, 5.55, 0.36, "Final conclusions", 17, NAVY, True)
    add_bullets(slide, 0.85, 2.2, 5.4, 3.55, [
        "Gradient Boosting leads: accuracy 0.727, sensitivity 0.764, F1 0.749, AUC 0.818.",
        "It produces the fewest false negatives and lowest CV F1 variation.",
        "Random Forest only modestly improves on ordinary Bagging.",
        "Random feature selection is not decisive in this experiment.",
        "Findings are experimental—not clinical diagnostic validation.",
    ], 13.2, spacing=8)
    add_picture_fit(slide, MODELS / "06_feature_importance_comparison.png", 6.95, 1.4, 5.8, 4.35)
    add_text(slide, 7.1, 5.92, 5.45, 0.35,
             "Feature importance is model-dependent association, not causal evidence.",
             10.5, GREY, False, PP_ALIGN.CENTER)
    add_panel(slide, 0.9, 6.5, 11.55, 0.42, NAVY_2, NAVY_2)
    add_text(slide, 1.15, 6.58, 11.05, 0.22,
             "THANK YOU  ·  From Individual Learners to Reliable Ensemble Prediction",
             11, WHITE, True, PP_ALIGN.CENTER)

    prs.save(OUTPUT)
    print(OUTPUT)
    print("slides", len(prs.slides))


if __name__ == "__main__":
    build()

