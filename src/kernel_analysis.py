from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .eda import heatmap
from .visualization import save_figure


def linear_kernel(X, Z): return X @ Z.T
def polynomial_kernel(X, Z, gamma=0.1, coef0=1.0, degree=2): return (gamma * (X @ Z.T) + coef0) ** degree
def rbf_kernel(X, Z, gamma=0.1):
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(Z**2, axis=1)[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(distances, 0))


def explicit_quadratic_map(X):
    return np.column_stack([X[:, 0]**2, np.sqrt(2)*X[:, 0]*X[:, 1], X[:, 1]**2])


def explicit_mapping_verification(tables: Path) -> pd.DataFrame:
    pairs = [(np.array([1., 2.]), np.array([3., -1.])),
             (np.array([.5, -2.]), np.array([1.5, 4.])),
             (np.array([-1.2, .7]), np.array([2.1, -3.]))]
    rows = []
    for x, z in pairs:
        explicit = float(explicit_quadratic_map(x[None])[0] @ explicit_quadratic_map(z[None])[0])
        kernel = float((x @ z)**2)
        rows.append({"x": x.tolist(), "z": z.tolist(), "explicit_mapped_dot_product": explicit,
                     "polynomial_kernel_result": kernel, "absolute_difference": abs(explicit-kernel)})
    out = pd.DataFrame(rows); out.to_csv(tables / "explicit_mapping_verification.csv", index=False)
    print("\nExplicit feature-map verification:\n", out.to_string(index=False))
    return out


def _decision_plot(ax, model, X, y, title):
    pad=.5; xx, yy=np.meshgrid(np.linspace(X[:,0].min()-pad,X[:,0].max()+pad,250), np.linspace(X[:,1].min()-pad,X[:,1].max()+pad,250))
    grid=np.c_[xx.ravel(),yy.ravel()]; zz=model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx,yy,zz,levels=[-1e9,0,1e9],alpha=.16,colors=["#4C78A8","#E45756"])
    ax.contour(xx,yy,zz,levels=[-1,0,1],colors=["#777","#111","#777"],linestyles=["--","-","--"],linewidths=[1,1.5,1])
    ax.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm",s=24,edgecolor="white",linewidth=.25)
    ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=75,facecolors="none",edgecolors="black",label="Support vectors")
    ax.set(title=title,xlabel="$x_1$",ylabel="$x_2$"); ax.legend(loc="best",fontsize=8)


def synthetic_demonstrations(figures: Path):
    X,y=make_circles(n_samples=350,noise=.08,factor=.42,random_state=42); X=StandardScaler().fit_transform(X)
    fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm",edgecolor="white",s=28)
    ax.set(title="Synthetic Nonlinear Dataset (Geometry Demonstration Only)",xlabel="$x_1$",ylabel="$x_2$"); save_figure(fig,figures/"16_synthetic_nonlinear_dataset.png")
    z=X[:,0]**2+X[:,1]**2; fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111,projection="3d")
    ax.scatter(X[:,0],X[:,1],z,c=y,cmap="coolwarm",s=20); ax.set(title="Explicit Nonlinear Lift: $z=x_1^2+x_2^2$",xlabel="$x_1$",ylabel="$x_2$",zlabel="$z$")
    save_figure(fig,figures/"17_explicit_mapping_3d.png")
    specs=[("linear",{},"18_synthetic_linear_boundary.png","Linear Kernel"),("poly",{"degree":2,"coef0":1},"19_synthetic_polynomial_boundary.png","Polynomial Kernel"),("rbf",{"gamma":1},"20_synthetic_rbf_boundary.png","RBF Kernel")]
    for kernel,kw,name,title in specs:
        m=SVC(kernel=kernel,C=1,**kw).fit(X,y); fig,ax=plt.subplots(figsize=(6,5)); _decision_plot(ax,m,X,y,f"Synthetic Circles — {title}"); save_figure(fig,figures/name)
    return X,y


def gram_analysis(X_train, y_train, tables: Path, figures: Path):
    scaler=StandardScaler().fit(X_train); Xs=scaler.transform(X_train); n=min(300,len(Xs)); Xs=Xs[:n]; ys=np.asarray(y_train)[:n]
    kernels={"Linear":linear_kernel(Xs,Xs),"Polynomial":polynomial_kernel(Xs,Xs),"RBF":rbf_kernel(Xs,Xs)}
    names={"Linear":"21_linear_kernel_matrix.png","Polynomial":"22_polynomial_kernel_matrix.png","RBF":"23_rbf_kernel_matrix.png"}
    eigrows=[]; simrows=[]
    for name,K in kernels.items():
        heatmap(pd.DataFrame(K),f"{name} Gram Matrix (n={n})",figures/names[name],center=None,cmap="viridis")
        eig=np.linalg.eigvalsh((K+K.T)/2)
        eigrows.append({"kernel":name,"symmetry_max_abs_error":float(np.max(abs(K-K.T))),"minimum_eigenvalue":float(eig.min()),"negative_eigenvalues_below_minus_1e-8":int((eig < -1e-8).sum()),"maximum_eigenvalue":float(eig.max())})
        tri=np.triu_indices(n,1); same=ys[tri[0]]==ys[tri[1]]; vals=K[tri]
        simrows.append({"kernel":name,"mean_same_class_similarity":vals[same].mean(),"mean_different_class_similarity":vals[~same].mean(),"difference":vals[same].mean()-vals[~same].mean()})
    pd.DataFrame(eigrows).to_csv(tables/"kernel_matrix_eigenvalues.csv",index=False)
    sim=pd.DataFrame(simrows); sim.to_csv(tables/"kernel_similarity_analysis.csv",index=False)
    order=np.argsort(ys); Ks=kernels["RBF"][np.ix_(order,order)]; split=int((ys[order]==0).sum())
    fig,ax=plt.subplots(figsize=(8,7)); sns.heatmap(Ks,cmap="viridis",ax=ax,cbar_kws={"label":"RBF similarity"}); ax.axhline(split,color="white",lw=2);ax.axvline(split,color="white",lw=2);ax.set(title="RBF Gram Matrix Sorted by DR Class",xlabel="Sorted observation",ylabel="Sorted observation")
    save_figure(fig,figures/"24_rbf_kernel_sorted_by_class.png")
    rng=np.random.default_rng(42); pair_idx=np.array(list(combinations(range(n),2))); pair_idx=pair_idx[rng.choice(len(pair_idx),min(8000,len(pair_idx)),replace=False)]
    records=[]
    for name,K in kernels.items():
        for i,j in pair_idx: records.append({"kernel":name,"pair_type":"Same class" if ys[i]==ys[j] else "Different class","similarity":K[i,j]})
    pair_df=pd.DataFrame(records); pair_df.to_csv(tables/"kernel_similarity_pairs.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,5)); sns.boxplot(pair_df,x="kernel",y="similarity",hue="pair_type",showfliers=False,ax=ax);ax.set(title="Kernel Similarity: Same-Class vs Different-Class Pairs",xlabel="Kernel",ylabel="Pairwise similarity")
    save_figure(fig,figures/"25_kernel_similarity_comparison.png")
    return sim, pd.DataFrame(eigrows)


def representative_gamma_boundaries(X,y,figures):
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,g in zip(axes,[.01,.1,10]):
        m=SVC(kernel="rbf",gamma=g,C=1).fit(X,y);_decision_plot(ax,m,X,y,f"RBF: gamma={g}")
    fig.suptitle("Gamma Controls RBF Boundary Locality (Synthetic Data)");save_figure(fig,figures/"29_gamma_boundary_comparison.png")
