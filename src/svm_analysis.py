from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluation import binary_metrics, fit_and_evaluate
from .kernel_analysis import _decision_plot, rbf_kernel
from .visualization import save_figure


def _pipeline(kernel, **kwargs):
    return Pipeline([("scaler",StandardScaler()),("svm",SVC(kernel=kernel,**kwargs))])


def _support_info(model, n_train):
    svc=model.named_steps["svm"] if hasattr(model,"named_steps") else model
    return {"sv_dr0":int(svc.n_support_[0]),"sv_dr1":int(svc.n_support_[1]),
            "support_vectors":int(svc.n_support_.sum()),"support_vector_percent":100*svc.n_support_.sum()/n_train}


def _evaluate_with_train(model, split):
    metrics,pred,scores=fit_and_evaluate(model,split.X_train,split.y_train,split.X_test,split.y_test)
    train_pred=model.predict(split.X_train); train_score=model.decision_function(split.X_train)
    metrics.update({"train_accuracy":accuracy_score(split.y_train,train_pred),"train_f1":f1_score(split.y_train,train_pred),"train_roc_auc":roc_auc_score(split.y_train,train_score)})
    metrics.update(_support_info(model,len(split.X_train)))
    return metrics,pred,scores


def pca_geometry(split, figures: Path, tables: Path):
    scaler=StandardScaler().fit(split.X_train); Xt=scaler.transform(split.X_train); Xv=scaler.transform(split.X_test)
    pca=PCA(n_components=2).fit(Xt); Zt=pca.transform(Xt); Zv=pca.transform(Xv)
    variance=pd.DataFrame({"component":["PC1","PC2"],"explained_variance_ratio":pca.explained_variance_ratio_})
    variance["cumulative_explained_variance_ratio"]=variance.explained_variance_ratio.cumsum();variance.to_csv(tables/"pca_explained_variance.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,5));sc=ax.scatter(Zt[:,0],Zt[:,1],c=split.y_train,cmap="coolwarm",s=25,alpha=.75);ax.set(title="2D PCA Projection — Visualization Only",xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})");ax.legend(*sc.legend_elements(),title="DR label")
    save_figure(fig,figures/"14_pca_feature_geometry.png")
    linear=SVC(kernel="linear",C=1).fit(Zt,split.y_train);fig,ax=plt.subplots(figsize=(7,5));_decision_plot(ax,linear,Zt,np.asarray(split.y_train),"2D PCA Linear SVM — Visualization Only")
    w=linear.coef_[0];margin=2/np.linalg.norm(w);ax.text(.02,.02,f"Margin width = {margin:.3f}",transform=ax.transAxes,bbox={"facecolor":"white","alpha":.8})
    save_figure(fig,figures/"15_linear_svm_pca.png")
    for kernel,kw,name,title in [("linear",{},"26_linear_svm_boundary.png","Linear"),("poly",{"degree":3,"coef0":1},"27_polynomial_svm_boundary.png","Polynomial"),("rbf",{},"28_rbf_svm_boundary.png","RBF")]:
        m=SVC(kernel=kernel,C=1,**kw).fit(Zt,split.y_train);fig,ax=plt.subplots(figsize=(7,5));_decision_plot(ax,m,Zt,np.asarray(split.y_train),f"{title} SVM on 2D PCA — Visualization Only");save_figure(fig,figures/name)
    return variance,Zt


def baseline_kernels(split,tables:Path,figures:Path):
    specs={"Linear":_pipeline("linear",C=1),"Polynomial":_pipeline("poly",C=1,degree=3,coef0=1),"RBF":_pipeline("rbf",C=1)}
    rows=[]
    for name,m in specs.items():
        result,_,_=_evaluate_with_train(m,split);result["kernel"]=name;rows.append(result)
    out=pd.DataFrame(rows);out.to_csv(tables/"kernel_baseline_comparison.csv",index=False)
    out[["kernel","sv_dr0","sv_dr1","support_vectors","support_vector_percent"]].to_csv(tables/"support_vector_analysis.csv",index=False)
    long=out.melt(id_vars="kernel",value_vars=["f1","recall_sensitivity","specificity","roc_auc"],var_name="metric",value_name="score")
    fig,ax=plt.subplots(figsize=(9,5));sns.barplot(long,x="kernel",y="score",hue="metric",ax=ax);ax.set(title="Real Dataset Kernel Performance",xlabel="Kernel",ylabel="Test score",ylim=(0,1));save_figure(fig,figures/"37_kernel_performance_comparison.png")
    fig,ax=plt.subplots(figsize=(8,5));out.plot(x="kernel",y=["sv_dr0","sv_dr1"],kind="bar",stacked=True,ax=ax,color=["#4C78A8","#E45756"]);ax.set(title="Support Vectors by Kernel and Class",xlabel="Kernel",ylabel="Support-vector count");ax.tick_params(axis="x",rotation=0);save_figure(fig,figures/"38_support_vector_comparison.png")
    return out


def sensitivity_experiments(split,tables:Path,figures:Path):
    gamma_rows=[]
    for g in [.001,.01,.1,1,10]:
        r,_,_=_evaluate_with_train(_pipeline("rbf",C=1,gamma=g),split);r["gamma"]=g;gamma_rows.append(r)
    gamma=pd.DataFrame(gamma_rows);gamma.to_csv(tables/"gamma_analysis.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].semilogx(gamma.gamma,gamma.f1,"o-",label="Test F1");axes[0].semilogx(gamma.gamma,gamma.train_f1,"o--",label="Train F1");axes[0].semilogx(gamma.gamma,gamma.roc_auc,"s-",label="Test ROC-AUC");axes[0].set(title="RBF Gamma: Generalization",xlabel="Gamma",ylabel="Score",ylim=(0,1));axes[0].legend();axes[1].semilogx(gamma.gamma,gamma.training_time_seconds,"o-",color="#B279A2");axes[1].set(title="RBF Gamma: Training Time",xlabel="Gamma",ylabel="Seconds");save_figure(fig,figures/"30_gamma_performance.png")
    fig,ax=plt.subplots(figsize=(7,4.5));ax.semilogx(gamma.gamma,gamma.support_vectors,"o-");ax.set(title="RBF Gamma vs Support Vectors",xlabel="Gamma",ylabel="Support-vector count");save_figure(fig,figures/"31_gamma_support_vectors.png")
    c_rows=[]
    for c in [.01,.1,1,10,100]:
        r,_,_=_evaluate_with_train(_pipeline("rbf",C=c,gamma="scale"),split);r["C"]=c;c_rows.append(r)
    cres=pd.DataFrame(c_rows);cres.to_csv(tables/"C_analysis.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5));ax.semilogx(cres.C,cres.f1,"o-",label="Test F1");ax.semilogx(cres.C,cres.train_f1,"o--",label="Train F1");ax.semilogx(cres.C,cres.roc_auc,"s-",label="Test ROC-AUC");ax.set(title="RBF Regularization (C) Performance",xlabel="C",ylabel="Score",ylim=(0,1));ax.legend();save_figure(fig,figures/"32_C_performance.png")
    fig,ax=plt.subplots(figsize=(7,4.5));ax.semilogx(cres.C,cres.support_vectors,"o-");ax.set(title="C vs Support Vectors",xlabel="C",ylabel="Support-vector count");save_figure(fig,figures/"33_C_support_vectors.png")
    degree_rows=[]
    degree_cv=StratifiedKFold(5,shuffle=True,random_state=42)
    for d in [2,3,4,5]:
        r,_,_=_evaluate_with_train(_pipeline("poly",C=1,degree=d,coef0=1),split);r["degree"]=d
        cv_scores=cross_validate(_pipeline("poly",C=1,degree=d,coef0=1),split.X_train,split.y_train,
                                 cv=degree_cv,scoring={"f1":"f1","roc_auc":"roc_auc"},n_jobs=1)
        r["mean_cv_f1"]=cv_scores["test_f1"].mean();r["std_cv_f1"]=cv_scores["test_f1"].std()
        r["mean_cv_roc_auc"]=cv_scores["test_roc_auc"].mean();r["std_cv_roc_auc"]=cv_scores["test_roc_auc"].std()
        degree_rows.append(r)
    degree=pd.DataFrame(degree_rows);degree.to_csv(tables/"polynomial_degree_analysis.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].plot(degree.degree,degree.f1,"o-",label="Test F1");axes[0].plot(degree.degree,degree.train_f1,"o--",label="Train F1");axes[0].plot(degree.degree,degree.roc_auc,"s-",label="Test ROC-AUC");axes[0].set(title="Polynomial Degree Performance",xlabel="Degree",ylabel="Score",ylim=(0,1));axes[0].legend();axes[1].plot(degree.degree,degree.support_vectors,"o-");axes[1].set(title="Degree vs Support Vectors",xlabel="Degree",ylabel="Support-vector count");save_figure(fig,figures/"34_polynomial_degree_analysis.png")
    return gamma,cres,degree


def cross_validation_surface(split,tables:Path,figures:Path):
    rows=[];cv=StratifiedKFold(5,shuffle=True,random_state=42)
    for c in [.1,1,10,100]:
        for g in [.001,.01,.1,1]:
            scores=cross_validate(_pipeline("rbf",C=c,gamma=g),split.X_train,split.y_train,cv=cv,scoring={"f1":"f1","roc_auc":"roc_auc"},n_jobs=1)
            rows.append({"C":c,"gamma":g,"mean_cv_f1":scores["test_f1"].mean(),"std_cv_f1":scores["test_f1"].std(),"mean_cv_roc_auc":scores["test_roc_auc"].mean(),"std_cv_roc_auc":scores["test_roc_auc"].std()})
    out=pd.DataFrame(rows);out.to_csv(tables/"C_gamma_cross_validation.csv",index=False)
    for metric,name,title in [("mean_cv_f1","35_C_gamma_F1_heatmap.png","Training CV Mean F1"),("mean_cv_roc_auc","36_C_gamma_ROCAUC_heatmap.png","Training CV Mean ROC-AUC")]:
        fig,ax=plt.subplots(figsize=(7,5));sns.heatmap(out.pivot(index="C",columns="gamma",values=metric),annot=True,fmt=".3f",cmap="viridis",ax=ax);ax.set_title(f"RBF C × Gamma — {title}");save_figure(fig,figures/name)
    return out


def precomputed_experiment(split,tables:Path):
    scaler=StandardScaler().fit(split.X_train);Xt=scaler.transform(split.X_train);Xv=scaler.transform(split.X_test)
    gamma=1/(Xt.shape[1]*Xt.var());Ktr=rbf_kernel(Xt,Xt,gamma);Kte=rbf_kernel(Xv,Xt,gamma)
    manual=SVC(kernel="precomputed",C=1);t=perf_counter();manual.fit(Ktr,split.y_train);fit_time=perf_counter()-t;t=perf_counter();pm=manual.predict(Kte);sm=manual.decision_function(Kte);infer=perf_counter()-t
    direct=SVC(kernel="rbf",C=1,gamma=gamma).fit(Xt,split.y_train);pdirect=direct.predict(Xv);sdirect=direct.decision_function(Xv)
    row=binary_metrics(split.y_test,pm,sm);row.update({"gamma":gamma,"prediction_agreement":float(np.mean(pm==pdirect)),"max_abs_decision_difference":float(np.max(abs(sm-sdirect))),"support_vectors_precomputed":int(manual.n_support_.sum()),"support_vectors_direct":int(direct.n_support_.sum()),"training_time_seconds":fit_time,"inference_time_seconds":infer})
    out=pd.DataFrame([row]);out.to_csv(tables/"precomputed_kernel_comparison.csv",index=False);return out


def computational_plot(baselines,figures):
    fig,axes=plt.subplots(2,2,figsize=(11,9));pairs=[("training_time_seconds","f1","F1 vs Training Time"),("training_time_seconds","roc_auc","ROC-AUC vs Training Time"),("support_vector_percent","f1","F1 vs Support-Vector %"),("support_vector_percent","recall_sensitivity","Recall vs Support-Vector %")]
    for ax,(x,y,title) in zip(axes.ravel(),pairs):
        ax.scatter(baselines[x],baselines[y],s=65)
        for _,r in baselines.iterrows():ax.annotate(r.kernel,(r[x],r[y]),xytext=(4,4),textcoords="offset points")
        ax.set(title=title,xlabel=x.replace("_"," ").title(),ylabel=y.replace("_"," ").title())
    save_figure(fig,figures/"39_performance_vs_computation.png")
