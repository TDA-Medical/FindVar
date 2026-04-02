"""
Phase 5: 논문용 시각화 + 종합 정리
==================================
논문 Figure 구성:
  Fig 1. 파이프라인 개요 (텍스트로 대체)
  Fig 2. Persistence Diagram 비교 (종양 vs 정상)
  Fig 3. Permutation Test + H1 Count Test (통계 검증)
  Fig 4. Top 유전자 + TDA vs 유클리드 비교
  Fig 5. Pathway 비교
  Fig 6. 분류 성능 비교
  Fig 7. Latent Space PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from ripser import ripser
from persim import plot_diagrams
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path("f:/coding/TDA/phase5_visualization_paper/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# 데이터 로드
LATENT_PATH = Path("f:/coding/TDA/Data-preprocessing/TAE/results/latent/woutSMOTE/latent_32d_cosine.csv")
GENE_IMP_PATH = Path("f:/coding/TDA/phase3_gene_traceback/results/gene_importance_full.csv")
PERM_PATH = Path("f:/coding/TDA/phase2_persistent_homology/results/permutation_test_results.csv")
H1_PATH = Path("f:/coding/TDA/phase2_persistent_homology/results/h1_count_test_results.csv")
CLF_PATH = Path("f:/coding/TDA/phase4_biological_interpretation/results/classification_results.csv")
ENR_TDA_PATH = Path("f:/coding/TDA/phase4_biological_interpretation/results/enrichment_tda_top200.csv")
ENR_EUC_PATH = Path("f:/coding/TDA/phase4_biological_interpretation/results/enrichment_euclidean_top200.csv")

df_latent = pd.read_csv(LATENT_PATH)
X = df_latent.drop(columns=["Target"]).values
y = df_latent["Target"].values
X_tumor = X[y == 1]
X_normal = X[y == 0]

df_genes = pd.read_csv(GENE_IMP_PATH)
df_clf = pd.read_csv(CLF_PATH)

print("=" * 60)
print("Phase 5: 논문용 시각화 생성")
print("=" * 60)

# ============================================================
# Figure 2: Persistence Diagram (종양 vs 정상)
# ============================================================
print("\nFig 2: Persistence Diagram...")

# 종양 서브샘플 (113개, 크기 매칭)
idx_t = np.random.choice(len(X_tumor), size=113, replace=False)
dgms_tumor = ripser(X_tumor[idx_t], maxdim=1)["dgms"]
dgms_normal = ripser(X_normal, maxdim=1)["dgms"]
dgms_all = ripser(X, maxdim=1)["dgms"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, dgms, title, n in [
    (axes[0], dgms_all, "All Samples", len(X)),
    (axes[1], dgms_tumor, "Tumor (n=113, subsampled)", 113),
    (axes[2], dgms_normal, "Normal (n=113)", 113),
]:
    plot_diagrams(dgms, ax=ax, show=False)
    ax.set_title(title, fontweight="bold")

    # H1 카운트 표시
    h1_count = len(dgms[1])
    h1_sig = sum(1 for b, d in dgms[1] if d - b > 0.01)
    ax.text(0.02, 0.98, f"H0: {len(dgms[0])} features\nH1: {h1_sig} features",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_persistence_diagrams.png")
fig.savefig(OUTPUT_DIR / "fig2_persistence_diagrams.pdf")
plt.close(fig)

# ============================================================
# Figure 3: 통계 검증 (H1 Count Test)
# ============================================================
print("Fig 3: Statistical Validation...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 32d_cosine H1 count test 재실행
tumor_h1_counts = []
for i in range(300):
    idx = np.random.choice(len(X_tumor), size=113, replace=False)
    dgms = ripser(X_tumor[idx], maxdim=1)["dgms"]
    h1_count = sum(1 for b, d in dgms[1] if d - b > 0.01)
    tumor_h1_counts.append(h1_count)

normal_dgms = ripser(X_normal, maxdim=1)["dgms"]
normal_h1 = sum(1 for b, d in normal_dgms[1] if d - b > 0.01)
tumor_h1 = np.array(tumor_h1_counts)

# Panel A: H1 count distribution
ax = axes[0]
ax.hist(tumor_h1, bins=25, alpha=0.8, color="#1565C0", edgecolor="white",
        label=f"Tumor subsamples (n=113)\nmean={tumor_h1.mean():.1f}")
ax.axvline(normal_h1, color="#D32F2F", linewidth=2.5, linestyle="--",
           label=f"Normal (n=113) = {normal_h1}")
p_val = np.mean(tumor_h1 <= normal_h1)
ax.set_title(f"(A) H1 Feature Count\np < 0.001", fontweight="bold")
ax.set_xlabel("H1 Feature Count")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8)

# Panel B: H0 mean persistence 비교
tumor_h0_pers = []
normal_h0_pers_boot = []
for i in range(300):
    idx = np.random.choice(len(X_tumor), size=113, replace=False)
    dgms = ripser(X_tumor[idx], maxdim=1)["dgms"]
    lifetimes = dgms[0][:, 1] - dgms[0][:, 0]
    tumor_h0_pers.append(np.mean(lifetimes[np.isfinite(lifetimes)]))

    idx_n = np.random.choice(len(X_normal), size=113, replace=True)
    dgms_n = ripser(X_normal[idx_n], maxdim=1)["dgms"]
    lifetimes_n = dgms_n[0][:, 1] - dgms_n[0][:, 0]
    normal_h0_pers_boot.append(np.mean(lifetimes_n[np.isfinite(lifetimes_n)]))

ax = axes[1]
ax.hist(tumor_h0_pers, bins=25, alpha=0.7, color="#1565C0", edgecolor="white",
        label=f"Tumor (mean={np.mean(tumor_h0_pers):.3f})")
ax.hist(normal_h0_pers_boot, bins=25, alpha=0.7, color="#D32F2F", edgecolor="white",
        label=f"Normal (mean={np.mean(normal_h0_pers_boot):.3f})")
ax.set_title("(B) H0 Mean Persistence\n(Bootstrap, n=113 each)", fontweight="bold")
ax.set_xlabel("Mean Persistence")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8)

# Panel C: Effect summary
ax = axes[2]
categories = ["H1 Count\n(Tumor)", "H1 Count\n(Normal)", "H0 Pers.\n(Tumor)", "H0 Pers.\n(Normal)"]
means = [tumor_h1.mean(), normal_h1, np.mean(tumor_h0_pers), np.mean(normal_h0_pers_boot)]
stds = [tumor_h1.std(), 0, np.std(tumor_h0_pers), np.std(normal_h0_pers_boot)]
colors_bar = ["#1565C0", "#D32F2F", "#1565C0", "#D32F2F"]

# 두 그룹으로 나눠서 그리기
ax2 = ax.twinx()
x = [0, 1]
ax.bar(x, means[:2], yerr=stds[:2], width=0.6, color=colors_bar[:2], capsize=4, alpha=0.8)
ax.set_ylabel("H1 Feature Count", color="#333")
ax.set_ylim(0, max(means[:2]) * 1.5)

x2 = [2.5, 3.5]
ax2.bar(x2, means[2:], yerr=stds[2:], width=0.6, color=colors_bar[2:], capsize=4, alpha=0.8)
ax2.set_ylabel("H0 Mean Persistence", color="#333")

ax.set_xticks([0, 1, 2.5, 3.5])
ax.set_xticklabels(categories, fontsize=8)
ax.set_title("(C) Tumor vs Normal Summary", fontweight="bold")

# Significance annotations
ax.annotate("***", xy=(0.5, max(means[:2]) * 1.2), ha="center", fontsize=14, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_statistical_validation.png")
fig.savefig(OUTPUT_DIR / "fig3_statistical_validation.pdf")
plt.close(fig)

# ============================================================
# Figure 4: TDA 유전자 + TDA vs 유클리드
# ============================================================
print("Fig 4: Gene Discovery...")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

# Panel A: Top 20 genes
ax1 = fig.add_subplot(gs[0])
top20 = df_genes.nsmallest(20, "tda_rank")

# 유클리드에서 ns인 유전자 하이라이트
colors_gene = []
for _, row in top20.iterrows():
    if "P_Value" in row and pd.notna(row["P_Value"]) and row["P_Value"] > 0.05:
        colors_gene.append("#FF6F00")  # TDA-only
    else:
        colors_gene.append("#1565C0")

ax1.barh(range(20), top20["tda_importance_norm"].values, color=colors_gene)
ax1.set_yticks(range(20))
ax1.set_yticklabels(top20["gene"].values, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("TDA Importance (normalized)")
ax1.set_title("(A) Top 20 TDA Genes\n(Orange = Euclidean non-significant)", fontweight="bold")

# Panel B: TDA rank vs Euclidean rank
ax2 = fig.add_subplot(gs[1])
if "euclidean_rank" in df_genes.columns:
    TOP_N = 200
    mask_tda = df_genes["tda_rank"] <= TOP_N
    mask_euc = df_genes["euclidean_rank"] <= TOP_N

    # Background
    bg = df_genes[~mask_tda & ~mask_euc]
    ax2.scatter(bg["euclidean_rank"], bg["tda_rank"], c="#E8E8E8", s=3, alpha=0.2, rasterized=True)

    # TDA-only
    tda_only = df_genes[mask_tda & ~mask_euc]
    ax2.scatter(tda_only["euclidean_rank"], tda_only["tda_rank"],
                c="#FF6F00", s=15, alpha=0.7, label=f"TDA-only ({len(tda_only)})", zorder=3)

    # Euclidean-only
    euc_only = df_genes[~mask_tda & mask_euc]
    ax2.scatter(euc_only["euclidean_rank"], euc_only["tda_rank"],
                c="#1565C0", s=15, alpha=0.7, label=f"Euclidean-only ({len(euc_only)})", zorder=3)

    # Both
    both = df_genes[mask_tda & mask_euc]
    if len(both) > 0:
        ax2.scatter(both["euclidean_rank"], both["tda_rank"],
                    c="#4CAF50", s=20, alpha=0.8, label=f"Both ({len(both)})", zorder=4)

    ax2.axhline(TOP_N, color="gray", linestyle=":", alpha=0.4)
    ax2.axvline(TOP_N, color="gray", linestyle=":", alpha=0.4)

    # 핵심 유전자 라벨
    for gene in ["EFCAB3", "RPRM", "HSPB9", "PGC", "ACTB"]:
        row = df_genes[df_genes["gene"] == gene]
        if len(row) > 0:
            row = row.iloc[0]
            ax2.annotate(gene, (row["euclidean_rank"], row["tda_rank"]),
                        fontsize=7, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                        xytext=(5, 5), textcoords="offset points")

    ax2.set_xlabel("Euclidean Rank (|Point-Biserial Corr|)")
    ax2.set_ylabel("TDA Rank")
    ax2.set_title(f"(B) TDA vs Euclidean: Top {TOP_N}\n(Overlap = {len(both)})", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(-500, len(df_genes) + 500)
    ax2.set_ylim(-500, len(df_genes) + 500)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_gene_discovery.png")
fig.savefig(OUTPUT_DIR / "fig4_gene_discovery.pdf")
plt.close(fig)

# ============================================================
# Figure 5: Pathway 비교
# ============================================================
print("Fig 5: Pathway Comparison...")

df_enr_tda = pd.read_csv(ENR_TDA_PATH) if ENR_TDA_PATH.exists() else pd.DataFrame()
df_enr_euc = pd.read_csv(ENR_EUC_PATH) if ENR_EUC_PATH.exists() else pd.DataFrame()

if len(df_enr_tda) > 0 and len(df_enr_euc) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # TDA pathways
    top_tda = df_enr_tda.head(12)
    y_pos = range(len(top_tda))
    axes[0].barh(y_pos, -np.log10(top_tda["Adjusted P-value"]), color="#D32F2F", alpha=0.85)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([t[:55] + "..." if len(t) > 55 else t for t in top_tda["Term"]], fontsize=8)
    axes[0].set_xlabel("-log10(Adjusted P-value)")
    axes[0].set_title("(A) TDA Top 200: Enriched Pathways", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].axvline(-np.log10(0.05), color="gray", linestyle=":", alpha=0.5)

    # Euclidean pathways
    top_euc = df_enr_euc.head(12)
    y_pos = range(len(top_euc))
    axes[1].barh(y_pos, -np.log10(top_euc["Adjusted P-value"]), color="#1565C0", alpha=0.85)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([t[:55] + "..." if len(t) > 55 else t for t in top_euc["Term"]], fontsize=8)
    axes[1].set_xlabel("-log10(Adjusted P-value)")
    axes[1].set_title("(B) Euclidean Top 200: Enriched Pathways", fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].axvline(-np.log10(0.05), color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_pathway_comparison.png")
    fig.savefig(OUTPUT_DIR / "fig5_pathway_comparison.pdf")
    plt.close(fig)

# ============================================================
# Figure 6: 분류 성능
# ============================================================
print("Fig 6: Classification Performance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: AUC 비교 (LogisticRegression 기준)
ax = axes[0]
lr = df_clf[df_clf["classifier"] == "LogisticRegression"].sort_values("auc_mean", ascending=True)
color_map = {
    "TDA_only_ns": "#FF6F00", "TDA_top20": "#EF5350", "TDA_top50": "#E53935",
    "TDA_top100": "#C62828", "TDA_top200": "#B71C1C",
    "Euclidean_top20": "#42A5F5", "Euclidean_top50": "#1E88E5",
    "Euclidean_top100": "#1565C0", "Euclidean_top200": "#0D47A1",
    "Combined_top100": "#4CAF50", "Latent_32d": "#78909C",
}
colors_clf = [color_map.get(name, "#999") for name in lr["gene_set"]]

ax.barh(range(len(lr)), lr["auc_mean"], xerr=lr["auc_std"],
        color=colors_clf, capsize=3, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(lr)))
ax.set_yticklabels(lr["gene_set"].values, fontsize=8)
ax.set_xlabel("AUC (5-fold CV)")
ax.set_title("(A) Classification: AUC\n(Logistic Regression)", fontweight="bold")
ax.set_xlim(0.9, 1.005)
ax.axvline(0.993, color="#FF6F00", linestyle=":", alpha=0.6, label="TDA-only ns = 0.993")
ax.legend(fontsize=7)

# Panel B: Feature 수 vs AUC
ax = axes[1]
best = df_clf.loc[df_clf.groupby("gene_set")["auc_mean"].idxmax()]
for _, row in best.iterrows():
    name = row["gene_set"]
    color = color_map.get(name, "#999")
    marker = "D" if "TDA_only" in name else "o" if "TDA" in name else "s" if "Euclidean" in name else "^"
    ax.scatter(row["n_features"], row["auc_mean"], c=color, s=100, marker=marker,
               edgecolors="black", linewidth=0.5, zorder=3)
    ax.annotate(name, (row["n_features"], row["auc_mean"]),
                fontsize=6, ha="left", xytext=(5, 3), textcoords="offset points")

ax.set_xlabel("Number of Features")
ax.set_ylabel("Best AUC (5-fold CV)")
ax.set_title("(B) Features vs Performance", fontweight="bold")
ax.set_ylim(0.93, 1.005)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig6_classification.png")
fig.savefig(OUTPUT_DIR / "fig6_classification.pdf")
plt.close(fig)

# ============================================================
# Figure 7: Latent Space PCA
# ============================================================
print("Fig 7: Latent Space PCA...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Panel A: 종양 vs 정상
ax = axes[0]
ax.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="#FFCDD2", s=5, alpha=0.4, label="Tumor (1,102)")
ax.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="#1565C0", s=20, alpha=0.8, label="Normal (113)",
           edgecolors="black", linewidth=0.3)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("(A) 32d Cosine Latent Space", fontweight="bold")
ax.legend(fontsize=9)

# Panel B: Top latent 차원의 종양/정상 분포
ax = axes[1]
lat_analysis = pd.read_csv(Path("f:/coding/TDA/phase3_gene_traceback/results/latent_dimension_analysis.csv"))
top5 = lat_analysis.nlargest(5, "cohens_d")

positions = []
for i, (_, row) in enumerate(top5.iterrows()):
    dim = row["dim_idx"]
    t_vals = X_tumor[:, int(dim)]
    n_vals = X_normal[:, int(dim)]

    bp_t = ax.boxplot([t_vals], positions=[i * 3], widths=0.8,
                       patch_artist=True, showfliers=False)
    bp_n = ax.boxplot([n_vals], positions=[i * 3 + 1], widths=0.8,
                       patch_artist=True, showfliers=False)

    bp_t["boxes"][0].set_facecolor("#FFCDD2")
    bp_n["boxes"][0].set_facecolor("#BBDEFB")

ax.set_xticks([i * 3 + 0.5 for i in range(5)])
ax.set_xticklabels([row["latent_dim"] for _, row in top5.iterrows()], fontsize=9)
ax.set_xlabel("Latent Dimension (Top 5 by Cohen's d)")
ax.set_ylabel("Value")
ax.set_title("(B) Tumor vs Normal: Top Latent Dimensions", fontweight="bold")

# 범례
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#FFCDD2", label="Tumor"),
                   Patch(facecolor="#BBDEFB", label="Normal")], fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig7_latent_space.png")
fig.savefig(OUTPUT_DIR / "fig7_latent_space.pdf")
plt.close(fig)

# ============================================================
# Summary Figure: 전체 파이프라인 결과 한눈에
# ============================================================
print("Summary Figure...")

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# (1,1) PH diagram - tumor vs normal
ax = fig.add_subplot(gs[0, 0])
plot_diagrams(dgms_tumor, ax=ax, show=False)
h1_t = sum(1 for b, d in dgms_tumor[1] if d - b > 0.01)
ax.set_title(f"Tumor PH (H1={h1_t})", fontweight="bold")

ax = fig.add_subplot(gs[0, 1])
plot_diagrams(dgms_normal, ax=ax, show=False)
h1_n = sum(1 for b, d in dgms_normal[1] if d - b > 0.01)
ax.set_title(f"Normal PH (H1={h1_n})", fontweight="bold")

# (1,3) H1 count test
ax = fig.add_subplot(gs[0, 2])
ax.hist(tumor_h1, bins=25, alpha=0.8, color="#1565C0", edgecolor="white")
ax.axvline(normal_h1, color="#D32F2F", linewidth=2.5, linestyle="--")
ax.set_title(f"H1 Count Test (p<0.001)", fontweight="bold")
ax.set_xlabel("H1 Features")

# (2,1) Top 15 genes
ax = fig.add_subplot(gs[1, 0])
top15 = df_genes.nsmallest(15, "tda_rank")
colors_g = ["#FF6F00" if pd.notna(row.get("P_Value")) and row["P_Value"] > 0.05 else "#1565C0"
            for _, row in top15.iterrows()]
ax.barh(range(15), top15["tda_importance_norm"].values, color=colors_g)
ax.set_yticks(range(15))
ax.set_yticklabels(top15["gene"].values, fontsize=7)
ax.invert_yaxis()
ax.set_title("Top 15 TDA Genes", fontweight="bold")
ax.set_xlabel("TDA Importance")

# (2,2) Pathway
ax = fig.add_subplot(gs[1, 1])
if len(df_enr_tda) > 0:
    top6 = df_enr_tda.head(6)
    ax.barh(range(len(top6)), -np.log10(top6["Adjusted P-value"]), color="#D32F2F", alpha=0.85)
    ax.set_yticks(range(len(top6)))
    ax.set_yticklabels([t[:40] + "..." if len(t) > 40 else t for t in top6["Term"]], fontsize=7)
    ax.set_xlabel("-log10(adj. p)")
    ax.invert_yaxis()
ax.set_title("TDA Enriched Pathways", fontweight="bold")

# (2,3) Classification
ax = fig.add_subplot(gs[1, 2])
sets_show = ["TDA_only_ns", "TDA_top100", "Euclidean_top100", "Latent_32d"]
clf_show = df_clf[(df_clf["classifier"] == "LogisticRegression") &
                   (df_clf["gene_set"].isin(sets_show))].sort_values("auc_mean")
colors_c = [color_map.get(n, "#999") for n in clf_show["gene_set"]]
ax.barh(range(len(clf_show)), clf_show["auc_mean"], xerr=clf_show["auc_std"],
        color=colors_c, capsize=3)
ax.set_yticks(range(len(clf_show)))
labels = [f"{row['gene_set']} ({row['n_features']})" for _, row in clf_show.iterrows()]
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("AUC")
ax.set_xlim(0.9, 1.005)
ax.set_title("Classification (AUC)", fontweight="bold")

plt.savefig(OUTPUT_DIR / "summary_figure.png")
plt.savefig(OUTPUT_DIR / "summary_figure.pdf")
plt.close(fig)

print(f"\n모든 Figure 저장 완료: {OUTPUT_DIR}")
print(f"\n{'=' * 60}")
print("Phase 5 완료!")
print(f"{'=' * 60}")
