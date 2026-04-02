"""
Phase 1: Persistent Homology 탐색적 분석
========================================
목표: latent 표현에서 종양/정상 간 위상적 차이가 실제로 존재하는지 확인

입력: TAE latent CSV (woutSMOTE)
출력: persistence diagram 비교, Wasserstein distance 정량화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams, wasserstein, bottleneck
from pathlib import Path
import time

# ============================================================
# 설정
# ============================================================
BASE_DIR = Path("f:/coding/TDA/Data-preprocessing/TAE/results/latent/woutSMOTE")
OUTPUT_DIR = Path("f:/coding/TDA/phase1_tda_setup/results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 분석할 latent 파일들 (대표적인 것들만)
LATENT_FILES = {
    "16d_cosine": "latent_16d_cosine.csv",
    "32d_cosine": "latent_32d_cosine.csv",
    "64d_cosine": "latent_64d_cosine.csv",
    "16d_pearson": "latent_16d_pearson.csv",
    "16d_euclidean": "latent_16d_euclidean.csv",
}

MAX_DIM = 1  # H0, H1 계산 (H2는 계산 비용이 크므로 일단 제외)


def load_latent(filepath):
    """Latent CSV 로드, 특징과 라벨 분리"""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Target"]).values
    y = df["Target"].values
    return X, y


def compute_ph(X, maxdim=1):
    """Persistent Homology 계산"""
    result = ripser(X, maxdim=maxdim)
    return result["dgms"]


def count_significant_features(dgm, threshold=0.1):
    """persistence가 threshold 이상인 feature 수"""
    if len(dgm) == 0:
        return 0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    # inf 제거 (H0의 마지막 connected component)
    finite_mask = np.isfinite(lifetimes)
    return np.sum(lifetimes[finite_mask] > threshold)


def summarize_diagram(dgm, name=""):
    """Persistence diagram 요약 통계"""
    lifetimes = dgm[:, 1] - dgm[:, 0]
    finite_mask = np.isfinite(lifetimes)
    finite_lifetimes = lifetimes[finite_mask]

    if len(finite_lifetimes) == 0:
        return {"name": name, "count": 0}

    return {
        "name": name,
        "count": len(finite_lifetimes),
        "mean_persistence": float(np.mean(finite_lifetimes)),
        "max_persistence": float(np.max(finite_lifetimes)),
        "std_persistence": float(np.std(finite_lifetimes)),
        "median_persistence": float(np.median(finite_lifetimes)),
    }


# ============================================================
# 메인 분석
# ============================================================
print("=" * 70)
print("Phase 1: Persistent Homology 탐색적 분석")
print("=" * 70)

all_results = []

for label, filename in LATENT_FILES.items():
    filepath = BASE_DIR / filename
    if not filepath.exists():
        print(f"  [SKIP] {filename} not found")
        continue

    print(f"\n{'─' * 50}")
    print(f"분석 중: {label} ({filename})")
    print(f"{'─' * 50}")

    X, y = load_latent(filepath)
    X_tumor = X[y == 1]
    X_normal = X[y == 0]

    print(f"  전체: {X.shape[0]} 샘플, {X.shape[1]} 차원")
    print(f"  종양: {X_tumor.shape[0]}, 정상: {X_normal.shape[0]}")

    # --- 전체 데이터 PH ---
    t0 = time.time()
    dgms_all = compute_ph(X, maxdim=MAX_DIM)
    t_all = time.time() - t0
    print(f"  전체 PH 계산: {t_all:.1f}초")

    # --- 종양 서브그룹 PH ---
    t0 = time.time()
    dgms_tumor = compute_ph(X_tumor, maxdim=MAX_DIM)
    t_tumor = time.time() - t0
    print(f"  종양 PH 계산: {t_tumor:.1f}초")

    # --- 정상 서브그룹 PH ---
    t0 = time.time()
    dgms_normal = compute_ph(X_normal, maxdim=MAX_DIM)
    t_normal = time.time() - t0
    print(f"  정상 PH 계산: {t_normal:.1f}초")

    # --- 요약 통계 ---
    for dim in range(MAX_DIM + 1):
        h_label = f"H{dim}"
        stats_tumor = summarize_diagram(dgms_tumor[dim], f"{label}_tumor_{h_label}")
        stats_normal = summarize_diagram(dgms_normal[dim], f"{label}_normal_{h_label}")

        print(f"\n  [{h_label}] 종양: {stats_tumor['count']} features, "
              f"평균 persistence={stats_tumor.get('mean_persistence', 0):.4f}, "
              f"최대={stats_tumor.get('max_persistence', 0):.4f}")
        print(f"  [{h_label}] 정상: {stats_normal['count']} features, "
              f"평균 persistence={stats_normal.get('mean_persistence', 0):.4f}, "
              f"최대={stats_normal.get('max_persistence', 0):.4f}")

        # --- Wasserstein & Bottleneck distance ---
        w_dist = wasserstein(dgms_tumor[dim], dgms_normal[dim])
        b_dist = bottleneck(dgms_tumor[dim], dgms_normal[dim])
        print(f"  [{h_label}] Wasserstein distance: {w_dist:.4f}")
        print(f"  [{h_label}] Bottleneck distance:  {b_dist:.4f}")

        all_results.append({
            "latent": label,
            "homology": h_label,
            "tumor_features": stats_tumor["count"],
            "normal_features": stats_normal["count"],
            "tumor_mean_pers": stats_tumor.get("mean_persistence", 0),
            "normal_mean_pers": stats_normal.get("mean_persistence", 0),
            "tumor_max_pers": stats_tumor.get("max_persistence", 0),
            "normal_max_pers": stats_normal.get("max_persistence", 0),
            "wasserstein": w_dist,
            "bottleneck": b_dist,
        })

    # --- Persistence Diagram 시각화 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_diagrams(dgms_all, ax=axes[0], show=False)
    axes[0].set_title(f"{label} - All ({X.shape[0]} samples)")

    plot_diagrams(dgms_tumor, ax=axes[1], show=False)
    axes[1].set_title(f"{label} - Tumor ({X_tumor.shape[0]} samples)")

    plot_diagrams(dgms_normal, ax=axes[2], show=False)
    axes[2].set_title(f"{label} - Normal ({X_normal.shape[0]} samples)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"ph_diagram_{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  저장: ph_diagram_{label}.png")

# ============================================================
# 결과 요약 테이블
# ============================================================
print("\n" + "=" * 70)
print("결과 요약")
print("=" * 70)

df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))
df_results.to_csv(OUTPUT_DIR / "ph_comparison_summary.csv", index=False)
print(f"\n저장: ph_comparison_summary.csv")

# --- 요약 바 차트 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wasserstein distance 비교
h0_data = df_results[df_results["homology"] == "H0"]
h1_data = df_results[df_results["homology"] == "H1"]

x = np.arange(len(h0_data))
width = 0.35

axes[0].bar(x - width/2, h0_data["wasserstein"].values, width, label="H0", color="#2196F3")
axes[0].bar(x + width/2, h1_data["wasserstein"].values, width, label="H1", color="#FF5722")
axes[0].set_xticks(x)
axes[0].set_xticklabels(h0_data["latent"].values, rotation=45, ha="right")
axes[0].set_ylabel("Wasserstein Distance")
axes[0].set_title("Tumor vs Normal: Wasserstein Distance")
axes[0].legend()

# Bottleneck distance 비교
axes[1].bar(x - width/2, h0_data["bottleneck"].values, width, label="H0", color="#2196F3")
axes[1].bar(x + width/2, h1_data["bottleneck"].values, width, label="H1", color="#FF5722")
axes[1].set_xticks(x)
axes[1].set_xticklabels(h0_data["latent"].values, rotation=45, ha="right")
axes[1].set_ylabel("Bottleneck Distance")
axes[1].set_title("Tumor vs Normal: Bottleneck Distance")
axes[1].legend()

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "distance_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("저장: distance_comparison.png")

print("\n" + "=" * 70)
print("Phase 1 탐색 완료!")
print("=" * 70)
