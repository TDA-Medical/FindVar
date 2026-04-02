"""
Phase 4: 생물학적 해석 + 분류 성능 검증
=======================================
1. GO/KEGG Pathway Enrichment (TDA Top 200)
2. 분류 성능: TDA 유전자 vs 유클리드 유전자 vs 전체 유전자
3. TDA+유클리드 결합 시 부가가치 확인
4. TDA-only 유전자(유클리드 ns)만으로의 분류 성능
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from pathlib import Path
import warnings
import gseapy as gp

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 설정
# ============================================================
DATA_PATH = Path("f:/coding/TDA/Data-preprocessing/data_preprocessing/cleaned_tcga_tpm_for_TAE.csv")
GENE_IMP_PATH = Path("f:/coding/TDA/phase3_gene_traceback/results/gene_importance_full.csv")
LATENT_PATH = Path("f:/coding/TDA/Data-preprocessing/TAE/results/latent/woutSMOTE/latent_32d_cosine.csv")
OUTPUT_DIR = Path("f:/coding/TDA/phase4_biological_interpretation/results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 데이터 로드
# ============================================================
print("=" * 70)
print("Phase 4: 생물학적 해석 + 분류 검증")
print("=" * 70)

print("\n데이터 로드...")
df_latent = pd.read_csv(LATENT_PATH)
y = df_latent["Target"].values
print(f"  샘플: {len(y)} (종양 {sum(y==1)}, 정상 {sum(y==0)})")

df_genes = pd.read_csv(GENE_IMP_PATH)
print(f"  유전자 중요도: {len(df_genes)} 유전자")

# 원본 발현 데이터 로드
print("  원본 발현 데이터 로드 중...")
df_expr = pd.read_csv(DATA_PATH, index_col=0)
print(f"  발현 데이터: {df_expr.shape}")

# ============================================================
# 1. Pathway Enrichment Analysis
# ============================================================
print("\n[1] Pathway Enrichment Analysis...")

# TDA Top 200 유전자
tda_top200 = df_genes.nsmallest(200, "tda_rank")["gene"].tolist()
# unknown_ 제거
tda_top200 = [g for g in tda_top200 if not g.startswith("unknown_")]
print(f"  TDA Top 200 유전자 (유효): {len(tda_top200)}개")

# 유클리드 Top 200 유전자
if "euclidean_rank" in df_genes.columns:
    euc_top200 = df_genes.nsmallest(200, "euclidean_rank")["gene"].tolist()
    euc_top200 = [g for g in euc_top200 if not g.startswith("unknown_")]
    print(f"  유클리드 Top 200 유전자 (유효): {len(euc_top200)}개")

# TDA-only 유전자 (유클리드 ns인 것)
if "P_Value" in df_genes.columns:
    tda_only_ns = df_genes[(df_genes["tda_rank"] <= 200) & (df_genes["P_Value"] > 0.05)]["gene"].tolist()
    tda_only_ns = [g for g in tda_only_ns if not g.startswith("unknown_")]
    print(f"  TDA-only (유클리드 ns) 유전자: {len(tda_only_ns)}개")

# GO Enrichment - TDA Top 200
print("\n  GO Enrichment (TDA Top 200)...")
try:
    enr_tda = gp.enrichr(
        gene_list=tda_top200,
        gene_sets=["GO_Biological_Process_2023", "GO_Molecular_Function_2023", "KEGG_2021_Human"],
        organism="human",
        outdir=None,
        no_plot=True,
    )
    df_enr_tda = enr_tda.results
    df_enr_tda_sig = df_enr_tda[df_enr_tda["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
    print(f"  유의미한 pathway (adj.p<0.05): {len(df_enr_tda_sig)}개")
    if len(df_enr_tda_sig) > 0:
        print("\n  [TDA Top 200 — 상위 20 Pathways]")
        for _, row in df_enr_tda_sig.head(20).iterrows():
            print(f"    {row['Gene_set']}: {row['Term']}")
            print(f"      p_adj={row['Adjusted P-value']:.2e}, genes={row['Overlap']}, "
                  f"involved: {row['Genes'][:80]}...")
    df_enr_tda_sig.to_csv(OUTPUT_DIR / "enrichment_tda_top200.csv", index=False)
except Exception as e:
    print(f"  GO Enrichment 실패: {e}")
    df_enr_tda_sig = pd.DataFrame()

# GO Enrichment - 유클리드 Top 200
print("\n  GO Enrichment (유클리드 Top 200)...")
try:
    enr_euc = gp.enrichr(
        gene_list=euc_top200,
        gene_sets=["GO_Biological_Process_2023", "GO_Molecular_Function_2023", "KEGG_2021_Human"],
        organism="human",
        outdir=None,
        no_plot=True,
    )
    df_enr_euc = enr_euc.results
    df_enr_euc_sig = df_enr_euc[df_enr_euc["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
    print(f"  유의미한 pathway (adj.p<0.05): {len(df_enr_euc_sig)}개")
    if len(df_enr_euc_sig) > 0:
        print("\n  [유클리드 Top 200 — 상위 10 Pathways]")
        for _, row in df_enr_euc_sig.head(10).iterrows():
            print(f"    {row['Gene_set']}: {row['Term']}")
            print(f"      p_adj={row['Adjusted P-value']:.2e}, overlap={row['Overlap']}")
    df_enr_euc_sig.to_csv(OUTPUT_DIR / "enrichment_euclidean_top200.csv", index=False)
except Exception as e:
    print(f"  GO Enrichment 실패: {e}")
    df_enr_euc_sig = pd.DataFrame()

# ============================================================
# 2. 분류 성능 비교
# ============================================================
print("\n[2] 분류 성능 비교...")

# 유전자 세트 정의
gene_sets = {}

# TDA Top N
for n in [20, 50, 100, 200]:
    top_genes = df_genes.nsmallest(n, "tda_rank")["gene"].tolist()
    top_genes = [g for g in top_genes if g in df_expr.columns]
    if len(top_genes) > 0:
        gene_sets[f"TDA_top{n}"] = top_genes

# 유클리드 Top N
if "euclidean_rank" in df_genes.columns:
    for n in [20, 50, 100, 200]:
        top_genes = df_genes.nsmallest(n, "euclidean_rank")["gene"].tolist()
        top_genes = [g for g in top_genes if g in df_expr.columns]
        if len(top_genes) > 0:
            gene_sets[f"Euclidean_top{n}"] = top_genes

# TDA-only (유클리드 ns)
if len(tda_only_ns) > 0:
    tda_only_in_data = [g for g in tda_only_ns if g in df_expr.columns]
    if len(tda_only_in_data) > 0:
        gene_sets["TDA_only_ns"] = tda_only_in_data

# TDA + 유클리드 결합
if "TDA_top100" in gene_sets and "Euclidean_top100" in gene_sets:
    combined = list(set(gene_sets["TDA_top100"] + gene_sets["Euclidean_top100"]))
    gene_sets["Combined_top100"] = combined

# Latent 32d (기준선)
gene_sets["Latent_32d"] = [f"z{i}" for i in range(32)]

print(f"  평가할 유전자 세트: {len(gene_sets)}개")
for name, genes in gene_sets.items():
    print(f"    {name}: {len(genes)} features")

# 분류기 정의
classifiers = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for set_name, genes in gene_sets.items():
    # 데이터 준비
    if set_name == "Latent_32d":
        X_set = df_latent[genes].values
    else:
        X_set = df_expr[genes].values

    for clf_name, clf in classifiers.items():
        # AUC
        auc_scores = cross_val_score(clf, X_set, y, cv=cv, scoring="roc_auc")
        # F1
        f1_scores = cross_val_score(clf, X_set, y, cv=cv, scoring="f1")
        # Accuracy
        acc_scores = cross_val_score(clf, X_set, y, cv=cv, scoring="accuracy")

        results.append({
            "gene_set": set_name,
            "n_features": len(genes),
            "classifier": clf_name,
            "auc_mean": auc_scores.mean(),
            "auc_std": auc_scores.std(),
            "f1_mean": f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "acc_mean": acc_scores.mean(),
            "acc_std": acc_scores.std(),
        })

    print(f"  {set_name} 완료")

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_DIR / "classification_results.csv", index=False)

# 결과 요약 (최고 분류기 기준)
print("\n  [분류 성능 요약 (AUC 기준, 최고 분류기)]")
best_per_set = df_results.loc[df_results.groupby("gene_set")["auc_mean"].idxmax()]
best_per_set = best_per_set.sort_values("auc_mean", ascending=False)
print(best_per_set[["gene_set", "n_features", "classifier", "auc_mean", "auc_std", "f1_mean"]].to_string(index=False))

# ============================================================
# 3. 시각화
# ============================================================
print("\n[3] 시각화 생성...")

# 분류 성능 비교 차트
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics = [("auc_mean", "auc_std", "AUC"), ("f1_mean", "f1_std", "F1"), ("acc_mean", "acc_std", "Accuracy")]

for ax, (mean_col, std_col, title) in zip(axes, metrics):
    # RandomForest 기준
    rf = df_results[df_results["classifier"] == "RandomForest"].sort_values(mean_col, ascending=True)
    colors = []
    for name in rf["gene_set"]:
        if "TDA_only" in name:
            colors.append("#FF6F00")
        elif "TDA" in name and "Combined" not in name:
            colors.append("#D32F2F")
        elif "Euclidean" in name:
            colors.append("#1565C0")
        elif "Combined" in name:
            colors.append("#4CAF50")
        else:
            colors.append("#78909C")

    ax.barh(range(len(rf)), rf[mean_col], xerr=rf[std_col], color=colors, capsize=3)
    ax.set_yticks(range(len(rf)))
    ax.set_yticklabels(rf["gene_set"], fontsize=8)
    ax.set_xlabel(title)
    ax.set_title(f"{title} (RandomForest, 5-fold CV)")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "classification_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Pathway 비교: TDA vs 유클리드
if len(df_enr_tda_sig) > 0 and len(df_enr_euc_sig) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # TDA pathways
    top_tda = df_enr_tda_sig.head(15)
    axes[0].barh(range(len(top_tda)), -np.log10(top_tda["Adjusted P-value"]), color="#D32F2F")
    axes[0].set_yticks(range(len(top_tda)))
    axes[0].set_yticklabels([t[:60] for t in top_tda["Term"]], fontsize=7)
    axes[0].set_xlabel("-log10(Adjusted P-value)")
    axes[0].set_title("TDA Top 200: Enriched Pathways")
    axes[0].invert_yaxis()

    # Euclidean pathways
    top_euc = df_enr_euc_sig.head(15)
    axes[1].barh(range(len(top_euc)), -np.log10(top_euc["Adjusted P-value"]), color="#1565C0")
    axes[1].set_yticks(range(len(top_euc)))
    axes[1].set_yticklabels([t[:60] for t in top_euc["Term"]], fontsize=7)
    axes[1].set_xlabel("-log10(Adjusted P-value)")
    axes[1].set_title("Euclidean Top 200: Enriched Pathways")
    axes[1].invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pathway_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# TDA vs Euclidean pathway overlap
if len(df_enr_tda_sig) > 0 and len(df_enr_euc_sig) > 0:
    tda_terms = set(df_enr_tda_sig["Term"])
    euc_terms = set(df_enr_euc_sig["Term"])
    both_terms = tda_terms & euc_terms
    tda_only_terms = tda_terms - euc_terms
    euc_only_terms = euc_terms - tda_terms

    print(f"\n  [Pathway 겹침 분석]")
    print(f"    TDA-only pathways: {len(tda_only_terms)}")
    print(f"    Euclidean-only pathways: {len(euc_only_terms)}")
    print(f"    Both: {len(both_terms)}")
    if len(both_terms) > 0:
        print(f"    겹치는 pathways: {list(both_terms)[:5]}")

    pathway_summary = {
        "tda_sig_pathways": len(tda_terms),
        "euc_sig_pathways": len(euc_terms),
        "overlap": len(both_terms),
        "tda_only": len(tda_only_terms),
        "euc_only": len(euc_only_terms),
    }
    pd.DataFrame([pathway_summary]).to_csv(OUTPUT_DIR / "pathway_overlap_summary.csv", index=False)

print(f"\n{'=' * 70}")
print("Phase 4 완료!")
print(f"{'=' * 70}")
