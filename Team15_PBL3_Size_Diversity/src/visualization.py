from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbl3_pivot_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import MODEL_ORDER
from .data_preparation import FEATURES


COLORS = {"Bagging": "#4C78A8", "Gradient Boosting": "#F28E2B", "Random Forest": "#59A14F"}


def configure():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11})


def save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_eda(frame, output):
    configure(); output = Path(output)
    counts = frame["class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5)); bars=ax.bar(["No signs (0)", "Signs (1)"], counts, color=["#7E8AA2","#C44E52"])
    ax.bar_label(bars); ax.set(title="Debrecen Class Distribution", ylabel="Observations", xlabel="Dataset label")
    save(fig, output/"01_class_distribution.png")
    selected=["ma1","ma2","exudate1","exudate7","macula_opticdisc_distance","opticdisc_diameter"]
    fig, axes=plt.subplots(2,3,figsize=(14,8))
    for ax, feature in zip(axes.ravel(),selected):
        sns.histplot(frame,x=feature,hue="class",element="step",stat="density",common_norm=False,ax=ax)
        ax.set_title(feature.replace("_"," ").title())
    fig.suptitle("Selected Feature Distributions by DR Label",y=1.01); save(fig,output/"02_feature_distributions.png")
    fig,ax=plt.subplots(figsize=(12,9)); sns.heatmap(frame[FEATURES].corr(),cmap="coolwarm",center=0,vmin=-1,vmax=1,ax=ax)
    ax.set_title("Predictor Correlation Heatmap"); save(fig,output/"03_correlation_heatmap.png")
    standardized=(frame[FEATURES]-frame[FEATURES].mean())/frame[FEATURES].std(ddof=0).replace(0,1); standardized["class"]=frame["class"]
    means=standardized.groupby("class")[FEATURES].mean().T
    fig,ax=plt.subplots(figsize=(9,8)); sns.heatmap(means,annot=True,fmt=".2f",center=0,cmap="coolwarm",ax=ax)
    ax.set(title="Standardized Feature Means by DR Label",xlabel="DR class",ylabel="Feature"); save(fig,output/"04_feature_target_heatmap.png")
    outliers={}
    for feature in FEATURES:
        q1,q3=frame[feature].quantile([.25,.75]); iqr=q3-q1
        outliers[feature]=((frame[feature]<q1-1.5*iqr)|(frame[feature]>q3+1.5*iqr)).mean()
    rates=pd.Series(outliers).sort_values(ascending=False).head(12)
    fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(x=rates.values,y=rates.index,color="#4C78A8",ax=ax)
    ax.set(title="Highest IQR Outlier-Flag Rates (Retained)",xlabel="Proportion flagged",ylabel="Feature"); save(fig,output/"05_outlier_rates.png")


def _line_with_band(data, metric, ylabel, title, path):
    fig,ax=plt.subplots(figsize=(9,5.5))
    for model in MODEL_ORDER:
        d=data[data.model.eq(model)].sort_values("ensemble_size")
        ax.plot(d.ensemble_size,d[f"{metric}_mean"],"o-",label=model,color=COLORS[model])
        low=d[f"{metric}_mean"]-d[f"{metric}_std"].fillna(0); high=d[f"{metric}_mean"]+d[f"{metric}_std"].fillna(0)
        ax.fill_between(d.ensemble_size,low,high,color=COLORS[model],alpha=.12)
    ax.set(title=title,xlabel="Number of estimators",ylabel=ylabel); ax.legend(); save(fig,path)


def plot_size_results(results, output):
    configure(); output=Path(output)
    for metric,label in [("accuracy","Accuracy"),("f1","F1-score"),("roc_auc","ROC-AUC"),("sensitivity","Sensitivity")]:
        _line_with_band(results,metric,label,f"Ensemble Size vs {label}",output/f"size_vs_{metric}.png")
    _line_with_band(results,"training_time_seconds","Training time (seconds)","Ensemble Size vs Training Time",output/"size_vs_training_time.png")
    _line_with_band(results,"prediction_ms_per_observation","Prediction time (ms/observation)","Ensemble Size vs Prediction Time",output/"size_vs_prediction_time.png")
    fig,ax=plt.subplots(figsize=(9,5.5))
    for model in MODEL_ORDER:
        d=results[results.model.eq(model)].sort_values("ensemble_size")
        ax.plot(d.ensemble_size,d.generalization_gap,"o-",label=model,color=COLORS[model])
    ax.set(title="Train–Validation F1 Gap vs Ensemble Size",xlabel="Number of estimators",ylabel="F1 gap");ax.legend()
    save(fig,output/"size_vs_generalization_gap.png")


def plot_diversity_results(summary, output):
    configure(); output=Path(output)
    rf=summary[summary.model.eq("Random Forest")].sort_values("feature_subset")
    bag=summary[summary.model.eq("Bagging")].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(rf.feature_subset,rf.disagreement_mean,"o-",color=COLORS["Random Forest"],label="RF")
    axes[0].axhline(bag.disagreement_mean,color=COLORS["Bagging"],ls="--",label="Bagging baseline")
    axes[0].set(title="Prediction Disagreement",xlabel="Features available per RF split",ylabel="Mean pairwise disagreement");axes[0].legend()
    axes[1].plot(rf.feature_subset,rf.double_fault_mean,"o-",color=COLORS["Random Forest"],label="RF")
    axes[1].axhline(bag.double_fault_mean,color=COLORS["Bagging"],ls="--",label="Bagging baseline")
    axes[1].set(title="Double-Fault Measure",xlabel="Features available per RF split",ylabel="Mean double-fault");axes[1].legend()
    save(fig,output/"01_diversity_measures.png")
    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.scatter(rf.disagreement_mean,rf.f1_mean,s=90,c=rf.feature_subset,cmap="viridis")
    for row in rf.itertuples(): ax.annotate(f"k={int(row.feature_subset)}",(row.disagreement_mean,row.f1_mean),xytext=(5,5),textcoords="offset points")
    ax.scatter([bag.disagreement_mean],[bag.f1_mean],marker="X",s=130,color=COLORS["Bagging"],label="Bagging")
    ax.set(title="Diversity and Ensemble F1",xlabel="Mean pairwise disagreement",ylabel="CV F1");ax.legend()
    save(fig,output/"02_diversity_vs_performance.png")
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(rf.feature_subset,rf.learner_accuracy_mean,"o-",label="Individual learner accuracy",color="#B279A2")
    ax.plot(rf.feature_subset,rf.f1_mean,"o-",label="Ensemble F1",color=COLORS["Random Forest"])
    ax.set(title="Learner Strength and Ensemble Performance",xlabel="Features available per RF split",ylabel="Score");ax.legend()
    save(fig,output/"03_learner_strength.png")


def plot_interaction(summary, output):
    configure(); output=Path(output)
    for metric,title,name in [
        ("f1_mean","CV F1: Size × Feature Subset","01_f1_heatmap.png"),
        ("disagreement_mean","Disagreement: Size × Feature Subset","02_disagreement_heatmap.png"),
        ("training_time_seconds_mean","Training Time: Size × Feature Subset","03_training_time_heatmap.png"),
    ]:
        pivot=summary.pivot(index="feature_subset",columns="ensemble_size",values=metric)
        fig,ax=plt.subplots(figsize=(9,5));sns.heatmap(pivot,annot=True,fmt=".3f",cmap="YlGnBu",ax=ax)
        ax.set(title=title,xlabel="Ensemble size",ylabel="Features per split");save(fig,output/name)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for subset,d in summary.groupby("feature_subset"):
        d=d.sort_values("ensemble_size")
        axes[0].plot(d.ensemble_size,d.f1_mean,"o-",label=f"k={subset}")
        axes[1].plot(d.ensemble_size,d.roc_auc_mean,"o-",label=f"k={subset}")
    axes[0].set(title="F1 vs Size Across Diversity Settings",xlabel="Ensemble size",ylabel="CV F1")
    axes[1].set(title="ROC-AUC vs Size Across Diversity Settings",xlabel="Ensemble size",ylabel="CV ROC-AUC")
    axes[1].legend(ncol=2,fontsize=8);save(fig,output/"04_performance_curves.png")
    fig,ax=plt.subplots(figsize=(9,5.5))
    for subset,d in summary.groupby("feature_subset"):
        d=d.sort_values("ensemble_size");ax.plot(d.ensemble_size,d.disagreement_mean,"o-",label=f"k={subset}")
    ax.set(title="Disagreement vs Ensemble Size",xlabel="Ensemble size",ylabel="Mean disagreement");ax.legend(ncol=2)
    save(fig,output/"05_disagreement_curves.png")


def plot_gradient(summary, learning_curve, output):
    configure(); output=Path(output)
    best_subsample=summary.groupby("subsample").f1_mean.mean().idxmax()
    data=summary[summary.subsample.eq(best_subsample)]
    pivot=data.pivot(index="max_depth",columns="learning_rate",values="f1_mean")
    fig,ax=plt.subplots(figsize=(7,5));sns.heatmap(pivot,annot=True,fmt=".3f",cmap="Oranges",ax=ax)
    ax.set(title=f"Gradient Boosting F1 (subsample={best_subsample:g})",xlabel="Learning rate",ylabel="Tree depth")
    save(fig,output/"gradient_configuration_heatmap.png")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(learning_curve.stage,learning_curve.train_f1,"o-",label="Train")
    axes[0].plot(learning_curve.stage,learning_curve.validation_f1,"o-",label="Validation")
    axes[0].set(title="Gradient Boosting F1 Learning Curve",xlabel="Estimators",ylabel="F1");axes[0].legend()
    axes[1].plot(learning_curve.stage,learning_curve.train_auc,"o-",label="Train")
    axes[1].plot(learning_curve.stage,learning_curve.validation_auc,"o-",label="Validation")
    axes[1].set(title="Gradient Boosting ROC-AUC Learning Curve",xlabel="Estimators",ylabel="ROC-AUC");axes[1].legend()
    save(fig,output/"gradient_learning_curves.png")


def plot_final(metrics, roc_points, output):
    configure(); output=Path(output)
    core=["accuracy","precision","sensitivity","specificity","f1","roc_auc"]
    long=metrics.melt(id_vars="model",value_vars=core,var_name="metric",value_name="score")
    fig,ax=plt.subplots(figsize=(11,5.5));sns.barplot(long,x="metric",y="score",hue="model",hue_order=MODEL_ORDER,palette=COLORS,ax=ax)
    ax.set(title="Final Held-out Performance",xlabel="Metric",ylabel="Score",ylim=(0,1));ax.legend(ncol=3)
    save(fig,output/"01_final_performance.png")
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,model in zip(axes,MODEL_ORDER):
        row=metrics[metrics.model.eq(model)].iloc[0];matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]])
        sns.heatmap(matrix,annot=True,fmt=".0f",cbar=False,cmap="Blues",ax=ax)
        ax.set(title=model,xlabel="Predicted",ylabel="Actual")
    save(fig,output/"02_final_confusion_matrices.png")
    fig,ax=plt.subplots(figsize=(7,6))
    for model in MODEL_ORDER:
        d=roc_points[roc_points.model.eq(model)];auc=metrics.loc[metrics.model.eq(model),"roc_auc"].iloc[0]
        ax.plot(d.fpr,d.tpr,label=f"{model} (AUC={auc:.3f})",color=COLORS[model])
    ax.plot([0,1],[0,1],"--",color="grey");ax.set(title="Final Held-out ROC Curves",xlabel="False-positive rate",ylabel="True-positive rate");ax.legend()
    save(fig,output/"03_final_roc.png")

