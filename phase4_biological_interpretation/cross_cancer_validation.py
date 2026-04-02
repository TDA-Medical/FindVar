"""
Cross-Cancer Validation: H2C Gene Panel
========================================
TCGA 전체 데이터(10,005 샘플, 33개 암종)에서 H2C 37개 유전자의
종양/정상 분류 성능을 검증한다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path("f:/coding/TDA/phase4_biological_interpretation/results")
RAW_DIR = Path("f:/coding/TDA/Data-preprocessing/GSE62944_RAW")

# ============================================================
# 데이터 로드
# ============================================================
print("=" * 60)
print("Cross-Cancer Validation: H2C Gene Panel")
print("=" * 60)

print("\n[1] 데이터 로드...")
print("  종양 데이터 로드 중 (1.4GB, 시간 소요)...")
df_tumor = pd.read_csv(RAW_DIR / "GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz",
                        sep='\t', index_col=0).T
print(f"  종양: {df_tumor.shape}")

print("  정상 데이터 로드 중...")
df_normal = pd.read_csv(RAW_DIR / "GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz",
                         sep='\t', index_col=0).T
print(f"  정상: {df_normal.shape}")

# 라벨
df_tumor["_target"] = 1
df_normal["_target"] = 0
df_all = pd.concat([df_tumor, df_normal], axis=0)
y_all = df_all["_target"].values
df_all = df_all.drop(columns=["_target"])

# 바코드에서 암종 추출 (TSS 코드 → 프로젝트 코드)
# TCGA 바코드: TCGA-XX-YYYY-ZZA -> sample type ZZ (01=tumor, 11=normal)
barcodes = df_all.index.tolist()
sample_types = []
for b in barcodes:
    parts = b.split('-')
    if len(parts) >= 4:
        sample_types.append(parts[3][:2])
    else:
        sample_types.append("??")

df_all["_sample_type"] = sample_types
df_all["_barcode"] = barcodes

# 임상 데이터에서 암종 매핑
clin_path = Path("f:/coding/TDA/Data-preprocessing/raw_data/GSE62944_Clinical_Variables.txt")
df_clin = pd.read_csv(clin_path, sep='\t', index_col=0, low_memory=False).T

# disease_code가 비어있으면 바코드에서 프로젝트 추출 시도
# TCGA TSS code → cancer type 매핑 (주요 암종)
# 대안: histological_type 사용
if 'histological_type' in df_clin.columns:
    hist_map = df_clin['histological_type'].to_dict()
    df_all["_histological"] = df_all["_barcode"].map(hist_map)

# TSS코드 기반 매핑 대신, 바코드 처음 12자로 환자 매칭
# 실제로는 TCGA 프로젝트 코드를 알아야 함
# 간단한 방법: 바코드의 TSS(2글자)로 암종 클러스터링
tss_to_cancer = {}
if 'histological_type' in df_clin.columns:
    for barcode, row in df_clin.iterrows():
        if isinstance(barcode, str) and barcode.startswith('TCGA-'):
            tss = barcode.split('-')[1]
            hist = row.get('histological_type', '')
            if isinstance(hist, str) and hist != '[Not Available]':
                tss_to_cancer[tss] = hist

# TSS → 암종 매핑
tss_codes = [b.split('-')[1] if len(b.split('-')) > 1 else '??' for b in barcodes]
cancer_types = [tss_to_cancer.get(t, 'Unknown') for t in tss_codes]
df_all["_cancer"] = cancer_types
df_all["_tss"] = tss_codes

print(f"\n  전체: {len(df_all)} 샘플")
print(f"  종양: {sum(y_all==1)}, 정상: {sum(y_all==0)}")

# 암종별 분포 (TSS 기반 그룹핑)
# BRCA TSS 코드 식별 (BRCA 전처리 데이터의 바코드에서)
brca_data = pd.read_csv("f:/coding/TDA/Data-preprocessing/data_preprocessing/cleaned_tcga_tpm_for_TAE.csv",
                         usecols=[0], nrows=5)
brca_barcodes = brca_data.iloc[:, 0].tolist()
brca_tss = set(b.split('-')[1] for b in brca_barcodes if isinstance(b, str))

# 더 정확하게: 전체 BRCA 바코드
brca_full = pd.read_csv("f:/coding/TDA/Data-preprocessing/data_preprocessing/cleaned_tcga_tpm_for_TAE.csv",
                          usecols=[0])
brca_all_barcodes = set(brca_full.iloc[:, 0].tolist())
brca_all_tss = set(b.split('-')[1] for b in brca_all_barcodes if isinstance(b, str))

df_all["_is_brca"] = df_all["_barcode"].isin(brca_all_barcodes) | df_all["_tss"].isin(brca_all_tss)

n_brca = df_all["_is_brca"].sum()
n_other = (~df_all["_is_brca"]).sum()
print(f"  BRCA 추정: {n_brca}, 기타 암종: {n_other}")

# ============================================================
# H2C 유전자 확인
# ============================================================
print("\n[2] H2C 유전자 매칭...")

gene_imp = pd.read_csv("f:/coding/TDA/phase3_gene_traceback/results/gene_importance_full.csv")
h2c_genes = gene_imp[(gene_imp["tda_rank"] <= 200) & (gene_imp["P_Value"] > 0.05)]["gene"].tolist()
h2c_genes = [g for g in h2c_genes if not g.startswith("unknown_")]

# 유전자 매칭 (발현 데이터에 있는 것만)
gene_cols = [c for c in df_all.columns if not c.startswith("_")]
h2c_available = [g for g in h2c_genes if g in gene_cols]
print(f"  H2C 유전자: {len(h2c_genes)}개 중 {len(h2c_available)}개 매칭")

# 유클리드 Top 37도 비교
euc_top37 = gene_imp.nsmallest(37, "euclidean_rank")["gene"].tolist()
euc_available = [g for g in euc_top37 if g in gene_cols]
print(f"  유클리드 Top 37: {len(euc_available)}개 매칭")

# ============================================================
# 분류 검증
# ============================================================
print("\n[3] Cross-Cancer 분류 검증...")

# log1p 변환
X_all = df_all[gene_cols].values.astype(np.float32)
X_all = np.log1p(X_all)

# 유전자 인덱스
h2c_idx = [gene_cols.index(g) for g in h2c_available]
euc_idx = [gene_cols.index(g) for g in euc_available]

results = []

# 데이터셋 정의
datasets = {
    "All_cancers": np.ones(len(df_all), dtype=bool),  # 전체
    "BRCA_only": df_all["_is_brca"].values,
    "Non_BRCA": (~df_all["_is_brca"]).values,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for ds_name, mask in datasets.items():
    X_ds = X_all[mask]
    y_ds = y_all[mask]

    if sum(y_ds == 0) < 5 or sum(y_ds == 1) < 5:
        print(f"  {ds_name}: 샘플 부족, 건너뜀")
        continue

    print(f"\n  [{ds_name}] 종양={sum(y_ds==1)}, 정상={sum(y_ds==0)}")

    for gene_set_name, gene_idx in [("H2C_37", h2c_idx), ("Euclidean_top37", euc_idx)]:
        X_sub = X_ds[:, gene_idx]

        for clf_name, clf in [
            ("LogReg", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])),
            ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("GB", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]:
            try:
                auc = cross_val_score(clf, X_sub, y_ds, cv=cv, scoring="roc_auc")
                f1 = cross_val_score(clf, X_sub, y_ds, cv=cv, scoring="f1")
                results.append({
                    "dataset": ds_name, "gene_set": gene_set_name,
                    "classifier": clf_name, "n_tumor": int(sum(y_ds == 1)),
                    "n_normal": int(sum(y_ds == 0)),
                    "auc_mean": auc.mean(), "auc_std": auc.std(),
                    "f1_mean": f1.mean(), "f1_std": f1.std(),
                })
            except Exception as e:
                print(f"    {gene_set_name}/{clf_name}: 실패 - {e}")

    print(f"  {ds_name} 완료")

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_DIR / "cross_cancer_results.csv", index=False)

# 요약 출력
print(f"\n{'=' * 60}")
print("[결과 요약] (최고 분류기 기준)")
best = df_results.loc[df_results.groupby(["dataset", "gene_set"])["auc_mean"].idxmax()]
print(best[["dataset", "gene_set", "n_tumor", "n_normal", "classifier", "auc_mean", "auc_std"]].to_string(index=False))

# ============================================================
# 시각화
# ============================================================
print("\n[4] 시각화...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for i, ds_name in enumerate(["All_cancers", "BRCA_only", "Non_BRCA"]):
    ax = axes[i]
    sub = df_results[(df_results["dataset"] == ds_name) & (df_results["classifier"] == "GB")]
    if len(sub) == 0:
        sub = df_results[df_results["dataset"] == ds_name]

    if len(sub) > 0:
        colors = ["#FF6F00" if "H2C" in gs else "#1565C0" for gs in sub["gene_set"]]
        ax.bar(range(len(sub)), sub["auc_mean"], yerr=sub["auc_std"],
               color=colors, capsize=5, edgecolor="white")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["gene_set"].values, rotation=15, fontsize=9)
        ax.set_ylabel("AUC")
        n_t = sub["n_tumor"].iloc[0]
        n_n = sub["n_normal"].iloc[0]
        ax.set_title(f"{ds_name}\n(Tumor={n_t}, Normal={n_n})", fontweight="bold")
        ax.set_ylim(0.5, 1.05)

        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(j, row["auc_mean"] + row["auc_std"] + 0.01,
                    f"{row['auc_mean']:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "cross_cancer_validation.png", dpi=200, bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "cross_cancer_validation.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\n{'=' * 60}")
print("Cross-Cancer Validation 완료!")
print(f"{'=' * 60}")
