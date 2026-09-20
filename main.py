import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbl_ml_matplotlib")

from src.data_loader import dataset_information, load_dataset
from src.dependence_analysis import analyze_dependence
from src.eda import plot_dependence, plot_eda
from src.kernel_analysis import (explicit_mapping_verification, gram_analysis,
    representative_gamma_boundaries, synthetic_demonstrations)
from src.naive_bayes_analysis import run_naive_bayes
from src.preprocessing import make_split
from src.research_reporting import write_readme, write_research_summary
from src.statistical_analysis import quality_analysis, save_descriptive_tables
from src.svm_analysis import (baseline_kernels, computational_plot,
    cross_validation_surface, pca_geometry, precomputed_experiment,
    sensitivity_experiments)
from src.visualization import configure_style


ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"Retinopathy_Debrecen.csv"
OUT=ROOT/"outputs";FIG=OUT/"figures";TAB=OUT/"tables";MET=OUT/"metrics";SUM=OUT/"summaries"


def main():
    for p in [FIG,TAB,MET,SUM]:p.mkdir(parents=True,exist_ok=True)
    configure_style();df,X,y=load_dataset(DATA);info=dataset_information(df,SUM)
    quality_table,quality=quality_analysis(X,y);quality_table.to_csv(TAB/"data_quality_analysis.csv")
    (MET/"data_quality_summary.json").write_text(json.dumps(quality,indent=2))
    save_descriptive_tables(X,y,TAB);plot_eda(X,y,FIG)
    split=make_split(X,y);matrices,mi,groups,_,_=analyze_dependence(split.X_train,split.y_train,TAB);plot_dependence(matrices,mi,groups,FIG)
    nb=run_naive_bayes(split,groups,TAB,FIG)
    pca_geometry(split,FIG,TAB);mapping=explicit_mapping_verification(TAB);synthetic_X,synthetic_y=synthetic_demonstrations(FIG)
    sim,eig=gram_analysis(split.X_train,split.y_train,TAB,FIG);representative_gamma_boundaries(synthetic_X,synthetic_y,FIG)
    kernels=baseline_kernels(split,TAB,FIG);gamma,cres,degree=sensitivity_experiments(split,TAB,FIG)
    cv=cross_validation_surface(split,TAB,FIG);precomputed=precomputed_experiment(split,TAB);computational_plot(kernels,FIG)
    write_research_summary(OUT,info,quality,groups,nb,kernels,sim,eig,gamma,cres,degree,cv,precomputed,mapping)
    write_readme(ROOT,info,nb,kernels,cv,degree)
    print(f"\nComplete: {len(list(FIG.glob('*.png')))} figures and {len(list(TAB.glob('*.csv')))} CSV tables generated.")


if __name__=="__main__":main()
