from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbl2_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.decomposition import PCA

from .data_validation import ANNUAL_VISITS, STATE_ORDER
from .preprocessing import CLUSTER_FEATURES


PALETTE = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B279A2", "#76B7B2"]


def configure_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11})


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eda(baseline, longitudinal, completeness, figures: Path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13)); axes = axes.ravel()
    for ax, col in zip(axes, CLUSTER_FEATURES):
        sns.histplot(baseline[col], kde=True, ax=ax, color=PALETTE[0]); ax.set_title(col.replace("_", " ").title())
    axes[-1].axis("off"); fig.suptitle("Protocol AA Baseline Feature Distributions", y=1.01)
    save(fig, figures / "01_baseline_feature_distributions.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(baseline[CLUSTER_FEATURES].isna().T, cbar=False, yticklabels=True, xticklabels=False, cmap=["white", "#E15759"], ax=ax)
    ax.set(title="Baseline Missingness Pattern", xlabel="Patient-eye records", ylabel="Feature")
    save(fig, figures / "02_missingness_heatmap.png")

    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(baseline[CLUSTER_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Baseline Feature Pearson Correlation")
    save(fig, figures / "03_baseline_correlation.png")

    fig, ax = plt.subplots(figsize=(9, 5)); order=STATE_ORDER
    counts=longitudinal.severity_state.value_counts().reindex(order,fill_value=0)
    ax.bar(counts.index,counts.values,color=PALETTE[:4]);ax.set(title="Valid DR Severity-State Observations",xlabel="Grouped DR severity state",ylabel="Graded visits");ax.tick_params(axis="x",rotation=18)
    save(fig, figures / "04_severity_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5));d=completeness.set_index("Visit").reindex(ANNUAL_VISITS)
    ax.plot(d.index,d.dated_visits,"o-",label="Dated patient visits");ax.plot(d.index,d.graded_patients,"s-",label="Patients with valid grade");ax.set(title="Annual Follow-up Retention",xlabel="Scheduled annual visit",ylabel="Patients");ax.legend()
    save(fig, figures / "05_followup_retention.png")

    fig, ax = plt.subplots(figsize=(8, 5));sns.histplot(longitudinal.days_since_previous.dropna(),bins=35,kde=True,ax=ax,color=PALETTE[1]);ax.axvline(365,color="black",ls="--",label="365 days");ax.set(title="Intervals Between Consecutive Graded Visits",xlabel="Days",ylabel="Intervals");ax.legend()
    save(fig, figures / "06_visit_intervals.png")

    seq=longitudinal.groupby(["PtID","RCEye"]).filter(lambda x: len(x)>=5)
    keys=seq[["PtID","RCEye"]].drop_duplicates().head(8)
    fig, ax=plt.subplots(figsize=(10,6));mapping={s:i for i,s in enumerate(STATE_ORDER)}
    for sequence_number, row in enumerate(keys.itertuples(index=False), start=1):
        one=seq[(seq.PtID==row.PtID)&(seq.RCEye==row.RCEye)].sort_values("visit_order")
        ax.plot(one.visit_order,one.severity_state.map(mapping),"o-",alpha=.8,
                label=f"Example sequence {sequence_number}")
    ax.set(xticks=range(5),xticklabels=ANNUAL_VISITS,yticks=range(4),yticklabels=STATE_ORDER,title="Example Patient-Eye DR State Trajectories",xlabel="Visit",ylabel="Severity state");ax.legend(ncol=2,fontsize=8)
    save(fig, figures / "07_example_trajectories.png")

    # Standardization makes differently scaled clinical measurements comparable
    # without implying that the grouped severity codes are equally spaced.
    standardized = baseline[CLUSTER_FEATURES].copy()
    standardized = (standardized - standardized.mean()) / standardized.std(ddof=0)
    standardized["severity_state"] = baseline.severity_state
    severity_profiles = (
        standardized.groupby("severity_state", observed=False)[CLUSTER_FEATURES]
        .mean().reindex(STATE_ORDER)
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(severity_profiles, annot=True, fmt=".2f", center=0,
                cmap="coolwarm", ax=ax)
    ax.set(title="Baseline Characteristics by DR Severity State",
           xlabel="Standardized baseline feature mean", ylabel="Baseline DR state")
    save(fig, figures / "24_baseline_characteristics_by_severity.png")


def plot_clustering(X, baseline, evaluation, k_labels, h_labels, k_row, h_row, figures: Path):
    km=evaluation[evaluation.method.eq("K-Means")]
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(km.k,km.inertia,"o-");ax.set(title="K-Means Elbow Analysis",xlabel="K",ylabel="Within-cluster sum of squares (inertia)")
    save(fig,figures/"08_kmeans_elbow.png")
    fig,ax=plt.subplots(figsize=(8,5))
    for (method,linkage),d in evaluation.groupby(["method","linkage"]):ax.plot(d.k,d.silhouette,"o-",label=f"{method}: {linkage}")
    ax.set(title="Silhouette Score Across Clustering Configurations",xlabel="Number of clusters",ylabel="Silhouette score");ax.legend(fontsize=8)
    save(fig,figures/"09_silhouette_comparison.png")
    fig,ax=plt.subplots(figsize=(12,6));Z=scipy_linkage(X,method=str(h_row.linkage));dendrogram(Z,truncate_mode="lastp",p=35,leaf_rotation=90,leaf_font_size=8,ax=ax);ax.set(title=f"Hierarchical Dendrogram — {h_row.linkage.title()} Linkage",xlabel="Merged groups",ylabel="Distance")
    save(fig,figures/"10_hierarchical_dendrogram.png")
    coords=PCA(n_components=2).fit_transform(X);fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,labels,title in [(axes[0],k_labels,f"K-Means (K={int(k_row.k)})"),(axes[1],h_labels,f"Agglomerative {h_row.linkage} (K={int(h_row.k)})")]:
        ax.scatter(coords[:,0],coords[:,1],c=labels,cmap="tab10",s=24,alpha=.75);ax.set(title=title+" — PCA Visualization Only",xlabel="PC1",ylabel="PC2")
    save(fig,figures/"11_cluster_pca_projection.png")
    scaled=pd.DataFrame(X,columns=CLUSTER_FEATURES);scaled["K-Means"]=k_labels;scaled["Agglomerative"]=h_labels
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,col,title in [(axes[0],"K-Means","K-Means Standardized Profiles"),(axes[1],"Agglomerative","Hierarchical Standardized Profiles")]:
        sns.heatmap(scaled.groupby(col)[CLUSTER_FEATURES].mean(),annot=True,fmt=".2f",center=0,cmap="coolwarm",ax=ax);ax.set_title(title);ax.set_xlabel("Feature")
    save(fig,figures/"12_cluster_profile_heatmap.png")
    sizes=pd.DataFrame({"K-Means":pd.Series(k_labels).value_counts(),"Agglomerative":pd.Series(h_labels).value_counts()}).fillna(0)
    fig,ax=plt.subplots(figsize=(8,5));sizes.plot(kind="bar",ax=ax,color=PALETTE[:2]);ax.set(title="Selected Cluster Size Comparison",xlabel="Cluster label",ylabel="Patient-eye records");ax.tick_params(axis="x",rotation=0)
    save(fig,figures/"13_cluster_size_comparison.png")
    selected=evaluation.loc[(evaluation.method.eq("K-Means")&evaluation.k.eq(k_row.k))|(evaluation.method.eq("Agglomerative")&evaluation.linkage.eq(h_row.linkage)&evaluation.k.eq(h_row.k))]
    long=selected.melt(id_vars=["method","linkage"],value_vars=["silhouette","stability_ari"],var_name="metric",value_name="value")
    fig,ax=plt.subplots(figsize=(8,5));sns.barplot(long,x="method",y="value",hue="metric",ax=ax);ax.set(title="Selected Clustering Quality and Stability",xlabel="Method",ylabel="Score")
    save(fig,figures/"14_clustering_metric_comparison.png")
    temp=baseline[["severity_state"]].copy();temp["K-Means"]=k_labels;temp["Agglomerative"]=h_labels
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,col in zip(axes,["K-Means","Agglomerative"]):
        tab=pd.crosstab(temp[col],temp.severity_state,normalize="index").reindex(columns=STATE_ORDER,fill_value=0);tab.plot(kind="bar",stacked=True,ax=ax,color=PALETTE[:4]);ax.set(title=f"DR Severity Composition — {col}",xlabel="Cluster",ylabel="Proportion");ax.legend(fontsize=7)
    save(fig,figures/"15_severity_composition_by_cluster.png")


def plot_markov(longitudinal, transitions, counts, probs, evolution, sensitivity, order_results, figures: Path):
    example=longitudinal.sort_values(["PtID","RCEye","VisitDt"]).head(12)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    sns.scatterplot(data=example,x="dro",y="visit_order",hue="severity_state",ax=axes[0],legend=False);axes[0].set(title="Non-sequential View: Rows Without Linked History",xlabel="Original DR order code",ylabel="Visit order")
    key=longitudinal.groupby(["PtID","RCEye"]).size().sort_values(ascending=False).index[0];one=longitudinal.set_index(["PtID","RCEye"]).loc[key].sort_values("visit_order");axes[1].plot(one.visit_order,one.severity_state.map({s:i for i,s in enumerate(STATE_ORDER)}),"o-");axes[1].set(xticks=range(5),xticklabels=ANNUAL_VISITS,yticks=range(4),yticklabels=STATE_ORDER,title="Sequential View: Chronologically Linked History",xlabel="Visit",ylabel="State")
    save(fig,figures/"16_sequential_vs_nonsequential.png")
    for matrix,name,title,fmt in [(counts,"17_transition_count_matrix.png","Observed Transition Counts","g"),(probs,"18_transition_probability_matrix.png","Empirical Transition Probabilities",".3f")]:
        fig,ax=plt.subplots(figsize=(8,6));sns.heatmap(matrix,annot=True,fmt=fmt,cmap="Blues",ax=ax);ax.set(title=title,xlabel="Next observed state",ylabel="Current state");save(fig,figures/name)
    dist=pd.crosstab(longitudinal.Visit,longitudinal.severity_state,normalize="index").reindex(index=ANNUAL_VISITS,columns=STATE_ORDER,fill_value=0)
    fig,ax=plt.subplots(figsize=(9,5));dist.plot(kind="bar",stacked=True,ax=ax,color=PALETTE[:4]);ax.set(title="Observed DR State Distribution Across Visits",xlabel="Visit",ylabel="Proportion");ax.legend(fontsize=8)
    save(fig,figures/"19_observed_state_by_visit.png")
    fig,ax=plt.subplots(figsize=(9,5));sns.lineplot(evolution,x="step",y="probability",hue="state",marker="o",ax=ax);ax.set(title="Modeled Multi-step State Distribution",xlabel="Transition step",ylabel="Probability")
    save(fig,figures/"20_multistep_state_evolution.png")
    fig,ax=plt.subplots(figsize=(10,5));sns.lineplot(sensitivity,x="step",y="probability",hue="state",style="scenario",markers=True,dashes=True,ax=ax);ax.set(title="Transition Sensitivity from Moderate NPDR",xlabel="Modeled step",ylabel="Probability");ax.legend(fontsize=7,ncol=2)
    save(fig,figures/"21_transition_sensitivity.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));axes[0].bar(order_results.model,order_results.accuracy,color=PALETTE[:2]);axes[0].set(title="Grouped Holdout Accuracy",ylabel="Accuracy");axes[0].tick_params(axis="x",rotation=15);axes[1].bar(order_results.model,order_results.log_loss,color=PALETTE[:2]);axes[1].set(title="Grouped Holdout Log Loss",ylabel="Log loss (lower is better)");axes[1].tick_params(axis="x",rotation=15)
    save(fig,figures/"22_markov_order_comparison.png")


def plot_cluster_transition_integration(summary: pd.DataFrame, figures: Path):
    fig,ax=plt.subplots(figsize=(9,5));sns.barplot(summary,x="cluster",y="proportion",hue="change",ax=ax);ax.set(title="Observed Transition Types by Baseline K-Means Cluster",xlabel="Baseline cluster",ylabel="Proportion of transitions");ax.legend(title="Observed change")
    save(fig,figures/"23_transitions_by_baseline_cluster.png")

