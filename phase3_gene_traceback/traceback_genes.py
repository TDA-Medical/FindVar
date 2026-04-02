"""
Phase 3: H1 루프 → Latent 차원 → 유전자 역추적 (수정 버전)
============================================================
접근법 변경:
  - 루프 참여도 기반 latent 차원 비교 → 효과 크기가 너무 작음
  - 대신: 종양 vs 정상의 latent 차원 차이 + 디코더 가중치 조합으로 유전자 식별
  - 핵심 논리: "종양에서 H1 루프가 더 많다" → "종양/정상 latent 차이가 루프를 유발"
    → "그 차이를 만드는 유전자가 TDA 발견 유전자"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ripser import ripser
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 설정
# ============================================================
LATENT_PATH = Path("f:/coding/TDA/Data-preprocessing/TAE/results/latent/woutSMOTE/latent_32d_cosine.csv")
MODEL_PATH = Path("f:/coding/TDA/Data-preprocessing/TAE/models/weights/tae_dim32_cosine.pth")
DATA_PATH = Path("f:/coding/TDA/Data-preprocessing/data_preprocessing/cleaned_tcga_tpm_for_TAE.csv")
BRCA_STATS_PATH = Path("f:/coding/TDA/Data-preprocessing/data_analysis/output_brca_patients/00_BRCA_All_23368_Genes_Statistics.csv")
OUTPUT_DIR = Path("f:/coding/TDA/phase3_gene_traceback/results")
OUTPUT_DIR.mkdir(exist_ok=True)


class TopologicalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, input_dim), nn.ReLU()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


print("=" * 70)
print("Phase 3: 유전자 역추적 (32d_cosine)")
print("=" * 70)

# ============================================================
# 데이터 로드
# ============================================================
df_latent = pd.read_csv(LATENT_PATH)
X = df_latent.drop(columns=["Target"]).values
y = df_latent["Target"].values
X_tumor = X[y == 1]
X_normal = X[y == 0]
print(f"\nLatent 데이터: {X.shape[0]} 샘플, {X.shape[1]} 차원")
print(f"종양: {len(X_tumor)}, 정상: {len(X_normal)}")

# ============================================================
# 1단계: 종양 vs 정상 Latent 차원 비교
# ============================================================
print("\n[1단계] 종양 vs 정상 Latent 차원 비교...")

latent_results = []
for dim in range(X.shape[1]):
    t_mean = np.mean(X_tumor[:, dim])
    n_mean = np.mean(X_normal[:, dim])
    pooled_std = np.sqrt((np.var(X_tumor[:, dim]) * len(X_tumor) + np.var(X_normal[:, dim]) * len(X_normal))
                         / (len(X_tumor) + len(X_normal)))
    cohens_d = abs(t_mean - n_mean) / (pooled_std + 1e-10)
    stat, p_val = stats.mannwhitneyu(X_tumor[:, dim], X_normal[:, dim], alternative="two-sided")

    latent_results.append({
        "latent_dim": f"z{dim}", "dim_idx": dim,
        "tumor_mean": t_mean, "normal_mean": n_mean,
        "diff": t_mean - n_mean, "cohens_d": cohens_d,
        "mann_whitney_p": p_val,
    })

df_lat = pd.DataFrame(latent_results).sort_values("cohens_d", ascending=False)
df_lat["p_adjusted"] = np.minimum(df_lat["mann_whitney_p"] * 32, 1.0)
df_lat["significant"] = df_lat["p_adjusted"].apply(
    lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
)

sig_dims = df_lat[df_lat["p_adjusted"] < 0.05]
print(f"\n  유의미한 차원 (Bonferroni p<0.05): {len(sig_dims)}개")
print(df_lat[["latent_dim", "cohens_d", "diff", "p_adjusted", "significant"]].head(15).to_string(index=False))

# ============================================================
# 2단계: TAE 디코더로 Latent → 유전자 매핑
# ============================================================
print("\n[2단계] TAE 디코더 가중치 분석...")

# 모델 로드 (checkpoint에서 input_dim 추론)
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
input_dim = state["decoder.6.bias"].shape[0]
print(f"  모델 input_dim: {input_dim}")

model = TopologicalAutoencoder(input_dim=input_dim, latent_dim=32)
model.load_state_dict(state)
model.eval()
print("  모델 로드 완료")

# 원본 데이터에서 유전자 이름 로드
df_original = pd.read_csv(DATA_PATH, nrows=1)
gene_cols = [c for c in df_original.columns if c != df_original.columns[0]]

# 모델과 CSV의 유전자 수가 다를 수 있으므로 맞춤
if len(gene_cols) < input_dim:
    print(f"  주의: CSV 유전자({len(gene_cols)}) < 모델 input({input_dim})")
    gene_names = gene_cols + [f"unknown_{i}" for i in range(input_dim - len(gene_cols))]
elif len(gene_cols) > input_dim:
    gene_names = gene_cols[:input_dim]
else:
    gene_names = gene_cols
print(f"  유전자 수: {len(gene_names)}")

# 디코더 Jacobian: 종양 평균 latent 주변에서 수치 미분
print("  디코더 Jacobian 계산...")
z_mean = torch.FloatTensor(X_tumor.mean(axis=0)).unsqueeze(0)
epsilon = 0.01

jacobian = np.zeros((input_dim, 32))
with torch.no_grad():
    x_base = model.decoder(z_mean).numpy().flatten()
    for dim in range(32):
        z_plus = z_mean.clone()
        z_plus[0, dim] += epsilon
        x_plus = model.decoder(z_plus).numpy().flatten()
        jacobian[:, dim] = (x_plus - x_base) / epsilon

print(f"  Jacobian shape: {jacobian.shape}")

# ============================================================
# 3단계: TDA 유전자 중요도 계산
# ============================================================
print("\n[3단계] TDA 유전자 중요도 계산...")

# 방법: 각 유전자의 TDA 중요도 = sum(|Jacobian[:, d]| * Cohen's d(d)) for all d
# 즉, 종양/정상 차이가 큰 latent 차원에서 민감하게 반응하는 유전자가 중요
all_cohens_d = df_lat.sort_values("dim_idx")["cohens_d"].values

gene_importance = np.zeros(input_dim)
for dim in range(32):
    gene_importance += np.abs(jacobian[:, dim]) * all_cohens_d[dim]

# 정규화
gene_importance_norm = gene_importance / (gene_importance.max() + 1e-10)

df_genes = pd.DataFrame({
    "gene": gene_names,
    "tda_importance": gene_importance,
    "tda_importance_norm": gene_importance_norm,
})
df_genes["tda_rank"] = df_genes["tda_importance"].rank(ascending=False).astype(int)
df_genes = df_genes.sort_values("tda_importance", ascending=False)

print(f"\n  [TDA 유전자 중요도 Top 30]")
print(df_genes.head(30)[["gene", "tda_importance", "tda_rank"]].to_string(index=False))

# ============================================================
# 4단계: 교차 검증 — TDA vs 유클리드
# ============================================================
print("\n[4단계] 기존 유클리드 분석과 교차 검증...")

df_brca = pd.read_csv(BRCA_STATS_PATH)
print(f"  BRCA 통계: {len(df_brca)} 유전자")

# 병합
df_merged = df_genes.merge(df_brca, left_on="gene", right_on="Gene", how="left")

# Abs PB Corr 기반 유클리드 랭킹
if "PB_Corr" in df_merged.columns:
    df_merged["abs_pb"] = df_merged["PB_Corr"].abs()
    df_merged["euclidean_rank"] = df_merged["abs_pb"].rank(ascending=False, na_option="bottom").astype(int)

# 카테고리 분류
# A: TDA에서도 유클리드에서도 중요 (양쪽 Top 200)
# B: TDA에서만 중요 (TDA Top 200, 유클리드 Bottom)
# C: 유클리드에서만 중요 (유클리드 Top 200, TDA Bottom)
if "euclidean_rank" in df_merged.columns:
    TOP_N = 200
    df_merged["category"] = "D_neither"
    mask_tda = df_merged["tda_rank"] <= TOP_N
    mask_euc = df_merged["euclidean_rank"] <= TOP_N

    df_merged.loc[mask_tda & mask_euc, "category"] = "A_both"
    df_merged.loc[mask_tda & ~mask_euc, "category"] = "B_tda_only"
    df_merged.loc[~mask_tda & mask_euc, "category"] = "C_euclidean_only"

    cat_counts = df_merged["category"].value_counts()
    print(f"\n  [카테고리 분류 (Top {TOP_N} 기준)]")
    for cat, cnt in cat_counts.items():
        print(f"    {cat}: {cnt}개")

    # B 카테고리 (TDA-only) 상세
    tda_only = df_merged[df_merged["category"] == "B_tda_only"].sort_values("tda_rank")
    print(f"\n  [B: TDA에서만 발견된 유전자 Top 20]")
    cols_show = ["gene", "tda_rank", "tda_importance_norm", "euclidean_rank", "PB_Corr", "P_Value"]
    cols_show = [c for c in cols_show if c in tda_only.columns]
    print(tda_only[cols_show].head(20).to_string(index=False))

    # A 카테고리 (양쪽 모두) 상세
    both = df_merged[df_merged["category"] == "A_both"].sort_values("tda_rank")
    print(f"\n  [A: 양쪽 모두에서 발견된 유전자 Top 20]")
    print(both[cols_show].head(20).to_string(index=False))

# ============================================================
# 결과 저장
# ============================================================
print(f"\n{'=' * 70}")
print("결과 저장...")

df_merged.sort_values("tda_rank").to_csv(OUTPUT_DIR / "gene_importance_full.csv", index=False)
df_merged[df_merged["tda_rank"] <= 100].sort_values("tda_rank").to_csv(OUTPUT_DIR / "gene_importance_top100.csv", index=False)
df_lat.to_csv(OUTPUT_DIR / "latent_dimension_analysis.csv", index=False)

if "category" in df_merged.columns:
    df_merged[df_merged["category"] == "B_tda_only"].sort_values("tda_rank").to_csv(OUTPUT_DIR / "tda_only_genes.csv", index=False)
    df_merged[df_merged["category"] == "A_both"].sort_values("tda_rank").to_csv(OUTPUT_DIR / "both_methods_genes.csv", index=False)

# ============================================================
# 시각화
# ============================================================
print("시각화 생성...")

# 1. Latent 차원 중요도
fig, ax = plt.subplots(figsize=(14, 5))
sorted_lat = df_lat.sort_values("dim_idx")
colors = ["#D32F2F" if p < 0.05 else "#90A4AE" for p in sorted_lat["p_adjusted"]]
ax.bar(sorted_lat["latent_dim"], sorted_lat["cohens_d"], color=colors)
ax.set_xlabel("Latent Dimension")
ax.set_ylabel("Cohen's d (Tumor vs Normal)")
ax.set_title("Latent Dimension: Tumor vs Normal Difference\n(Red = Bonferroni p < 0.05)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "latent_dimension_importance.png", dpi=150)
plt.close(fig)

# 2. Top 30 유전자
fig, ax = plt.subplots(figsize=(14, 6))
top30 = df_genes.head(30)
ax.barh(range(30), top30["tda_importance_norm"].values, color="#1565C0")
ax.set_yticks(range(30))
ax.set_yticklabels(top30["gene"].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("TDA Importance (normalized)")
ax.set_title("Top 30 Genes Contributing to Tumor-Specific Topological Structure")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "top30_genes.png", dpi=150)
plt.close(fig)

# 3. TDA rank vs Euclidean rank scatter
if "euclidean_rank" in df_merged.columns:
    fig, ax = plt.subplots(figsize=(8, 8))
    cat_colors = {"A_both": "#4CAF50", "B_tda_only": "#FF6F00",
                  "C_euclidean_only": "#1565C0", "D_neither": "#E0E0E0"}
    for cat, color in cat_colors.items():
        sub = df_merged[df_merged["category"] == cat]
        ax.scatter(sub["euclidean_rank"], sub["tda_rank"],
                   c=color, s=8 if cat == "D_neither" else 20,
                   alpha=0.3 if cat == "D_neither" else 0.7,
                   label=f"{cat} ({len(sub)})", edgecolors="none")

    # TDA-only 유전자 중 상위 5개 라벨
    tda_top5 = df_merged[df_merged["category"] == "B_tda_only"].nsmallest(5, "tda_rank")
    for _, row in tda_top5.iterrows():
        ax.annotate(row["gene"], (row["euclidean_rank"], row["tda_rank"]),
                    fontsize=7, fontweight="bold")

    ax.set_xlabel("Euclidean Rank (|Point-Biserial Corr|)")
    ax.set_ylabel("TDA Rank")
    ax.set_title(f"TDA Rank vs Euclidean Rank (Top {TOP_N})")
    ax.axhline(TOP_N, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(TOP_N, color="gray", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_xlim(0, len(df_merged))
    ax.set_ylim(0, len(df_merged))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "tda_vs_euclidean_rank.png", dpi=150)
    plt.close(fig)

# 4. Venn-style summary
if "category" in df_merged.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["A_both", "B_tda_only", "C_euclidean_only"]
    cat_labels = ["Both Methods", "TDA Only", "Euclidean Only"]
    cat_vals = [len(df_merged[df_merged["category"] == c]) for c in cats]
    cat_cols = ["#4CAF50", "#FF6F00", "#1565C0"]
    bars = ax.bar(cat_labels, cat_vals, color=cat_cols)
    for bar, val in zip(bars, cat_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", fontweight="bold")
    ax.set_ylabel("Number of Genes")
    ax.set_title(f"Gene Discovery: TDA vs Euclidean (Top {TOP_N} each)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "discovery_comparison.png", dpi=150)
    plt.close(fig)

# 5. PCA latent space (종양/정상)
fig, ax = plt.subplots(figsize=(8, 6))
pca = PCA(n_components=2)
X_all_2d = pca.fit_transform(X)
ax.scatter(X_all_2d[y == 1, 0], X_all_2d[y == 1, 1], c="#FFCDD2", s=5, alpha=0.5, label="Tumor")
ax.scatter(X_all_2d[y == 0, 0], X_all_2d[y == 0, 1], c="#1565C0", s=15, alpha=0.8, label="Normal")
ax.set_title("32d_cosine Latent Space (PCA)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "latent_pca.png", dpi=150)
plt.close(fig)

print("\n모든 결과 저장 완료!")
print(f"\n{'=' * 70}")
print("Phase 3 완료!")
print(f"{'=' * 70}")
