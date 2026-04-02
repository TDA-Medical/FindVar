"""
Phase 2: Persistent Homology 심층 분석 (최적화 버전)
====================================================
핵심 최적화: Permutation test에서 양쪽 모두 113개로 서브샘플링하여
  - 샘플 크기 효과를 통제
  - 계산 시간을 대폭 단축 (1215 샘플 PH → 113 샘플 PH)

분석 내용:
  1. Permutation test (size-matched): p-value 산출
  2. Bootstrap stability: 정상 샘플 PH 안정성
  3. Size-matched comparison: 종양 vs 정상, 종양 vs 종양 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein, bottleneck
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 설정
# ============================================================
BASE_DIR = Path("f:/coding/TDA/Data-preprocessing/TAE/results/latent/woutSMOTE")
OUTPUT_DIR = Path("f:/coding/TDA/phase2_persistent_homology/results")
OUTPUT_DIR.mkdir(exist_ok=True)

LATENT_FILES = {
    "16d_cosine": "latent_16d_cosine.csv",
    "32d_cosine": "latent_32d_cosine.csv",
    "64d_cosine": "latent_64d_cosine.csv",
}

N_PERMUTATIONS = 500
N_BOOTSTRAP = 200
N_SIZE_MATCH = 200
MAX_DIM = 1
SUBSAMPLE_SIZE = 113  # 정상 샘플 수에 맞춤


def load_latent(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Target"]).values
    y = df["Target"].values
    return X, y


def compute_ph(X, maxdim=1):
    return ripser(X, maxdim=maxdim)["dgms"]


def compute_distances(dgms1, dgms2, maxdim=1):
    result = {}
    for dim in range(maxdim + 1):
        result[f"H{dim}_wasserstein"] = wasserstein(dgms1[dim], dgms2[dim])
        result[f"H{dim}_bottleneck"] = bottleneck(dgms1[dim], dgms2[dim])
    return result


def count_h1_features(dgms, threshold=0.01):
    """H1에서 threshold 이상 persistence를 가진 feature 수"""
    if len(dgms[1]) == 0:
        return 0
    lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
    return int(np.sum(lifetimes > threshold))


# ============================================================
# 1. Size-matched Permutation Test
# ============================================================
def permutation_test_matched(X_tumor, X_normal, n_perm=500):
    """
    크기 매칭된 Permutation test:
    - 종양에서 113개 서브샘플 vs 정상 113개
    - 라벨을 셔플하여 null distribution 생성
    """
    n = len(X_normal)
    print(f"\n  [Permutation Test] 크기 매칭 ({n} vs {n}), {n_perm}회...")

    # 관찰된 거리 (여러 번 서브샘플하여 평균)
    obs_dists = []
    for _ in range(20):
        idx_t = np.random.choice(len(X_tumor), size=n, replace=False)
        dgms_t = compute_ph(X_tumor[idx_t])
        dgms_n = compute_ph(X_normal)
        obs_dists.append(compute_distances(dgms_t, dgms_n))

    observed = {}
    for k in obs_dists[0].keys():
        observed[k] = float(np.mean([d[k] for d in obs_dists]))

    # Null distribution: 종양+정상을 합친 후 랜덤 분할
    X_combined = np.vstack([X_tumor, X_normal])
    null_dists = {k: [] for k in observed.keys()}
    null_h1_counts = {"group_a": [], "group_b": []}
    t0 = time.time()

    for i in range(n_perm):
        # 전체에서 랜덤으로 2그룹 추출 (각 113개)
        perm_idx = np.random.permutation(len(X_combined))
        group_a = X_combined[perm_idx[:n]]
        group_b = X_combined[perm_idx[n:2*n]]

        dgms_a = compute_ph(group_a)
        dgms_b = compute_ph(group_b)
        dists = compute_distances(dgms_a, dgms_b)
        for k, v in dists.items():
            null_dists[k].append(v)

        null_h1_counts["group_a"].append(count_h1_features(dgms_a))
        null_h1_counts["group_b"].append(count_h1_features(dgms_b))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_perm - i - 1)
            print(f"    {i+1}/{n_perm} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    # p-value
    results = {}
    for k in observed.keys():
        null_arr = np.array(null_dists[k])
        p_value = float(np.mean(null_arr >= observed[k]))
        results[k] = {
            "observed": observed[k],
            "null_mean": float(np.mean(null_arr)),
            "null_std": float(np.std(null_arr)),
            "p_value": p_value,
            "null_distribution": null_arr,
        }
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"    {k}: obs={observed[k]:.4f}, null={np.mean(null_arr):.4f}+/-{np.std(null_arr):.4f}, p={p_value:.4f} {sig}")

    results["_null_h1_counts"] = null_h1_counts
    return results


# ============================================================
# 2. H1 Feature Count Test
# ============================================================
def h1_count_test(X_tumor, X_normal, n_iter=200):
    """
    종양 서브샘플(113개)과 정상(113개)의 H1 feature 수 비교
    종양에 정말 더 많은 루프가 있는지 검증
    """
    n = len(X_normal)
    print(f"\n  [H1 Count Test] 종양 서브샘플 vs 정상, {n_iter}회...")

    # 정상의 H1 count
    dgms_normal = compute_ph(X_normal)
    normal_h1 = count_h1_features(dgms_normal)

    tumor_h1_counts = []
    for i in range(n_iter):
        idx = np.random.choice(len(X_tumor), size=n, replace=False)
        dgms = compute_ph(X_tumor[idx])
        tumor_h1_counts.append(count_h1_features(dgms))

    tumor_h1 = np.array(tumor_h1_counts)
    p_value = float(np.mean(tumor_h1 <= normal_h1))  # 종양이 정상보다 적거나 같을 확률

    print(f"    정상 H1 features: {normal_h1}")
    print(f"    종양 H1 features: {tumor_h1.mean():.1f} +/- {tumor_h1.std():.1f} (min={tumor_h1.min()}, max={tumor_h1.max()})")
    print(f"    p-value (tumor <= normal): {p_value:.4f}")

    return {
        "normal_h1": normal_h1,
        "tumor_h1_mean": float(tumor_h1.mean()),
        "tumor_h1_std": float(tumor_h1.std()),
        "tumor_h1_min": int(tumor_h1.min()),
        "tumor_h1_max": int(tumor_h1.max()),
        "p_value": p_value,
        "tumor_h1_counts": tumor_h1,
    }


# ============================================================
# 3. Bootstrap Stability
# ============================================================
def bootstrap_stability(X_normal, n_boot=200):
    print(f"\n  [Bootstrap] 정상 {len(X_normal)}개, {n_boot}회 리샘플링...")
    stats = []
    for i in range(n_boot):
        idx = np.random.choice(len(X_normal), size=len(X_normal), replace=True)
        dgms = compute_ph(X_normal[idx])
        for dim in range(MAX_DIM + 1):
            lifetimes = dgms[dim][:, 1] - dgms[dim][:, 0]
            finite = lifetimes[np.isfinite(lifetimes)]
            stats.append({
                "iteration": i, "homology": f"H{dim}",
                "n_features": len(finite),
                "mean_pers": float(np.mean(finite)) if len(finite) > 0 else 0,
                "max_pers": float(np.max(finite)) if len(finite) > 0 else 0,
            })
    df = pd.DataFrame(stats)
    for dim in ["H0", "H1"]:
        sub = df[df["homology"] == dim]
        cv = sub["n_features"].std() / sub["n_features"].mean() * 100 if sub["n_features"].mean() > 0 else 0
        print(f"    {dim}: features={sub['n_features'].mean():.1f}+/-{sub['n_features'].std():.1f} (CV={cv:.1f}%)")
    return df


# ============================================================
# 메인
# ============================================================
print("=" * 70)
print("Phase 2: Persistent Homology 심층 분석")
print("=" * 70)

all_perm = {}
all_h1 = {}
all_boot = {}

for label, filename in LATENT_FILES.items():
    filepath = BASE_DIR / filename
    print(f"\n{'=' * 70}")
    print(f"분석: {label}")
    print(f"{'=' * 70}")

    X, y = load_latent(filepath)
    X_tumor, X_normal = X[y == 1], X[y == 0]
    print(f"  종양: {len(X_tumor)}, 정상: {len(X_normal)}")

    all_perm[label] = permutation_test_matched(X_tumor, X_normal, N_PERMUTATIONS)
    all_h1[label] = h1_count_test(X_tumor, X_normal, N_SIZE_MATCH)
    all_boot[label] = bootstrap_stability(X_normal, N_BOOTSTRAP)


# ============================================================
# 결과 저장
# ============================================================
print(f"\n{'=' * 70}")
print("결과 저장...")

# Permutation results CSV
rows = []
for label, results in all_perm.items():
    for metric in ["H0_wasserstein", "H0_bottleneck", "H1_wasserstein", "H1_bottleneck"]:
        d = results[metric]
        rows.append({
            "latent": label, "metric": metric,
            "observed": d["observed"], "null_mean": d["null_mean"],
            "null_std": d["null_std"], "p_value": d["p_value"],
            "significant": "***" if d["p_value"] < 0.001 else "**" if d["p_value"] < 0.01 else "*" if d["p_value"] < 0.05 else "ns",
        })
df_perm = pd.DataFrame(rows)
df_perm.to_csv(OUTPUT_DIR / "permutation_test_results.csv", index=False)
print("\n[Permutation Test]")
print(df_perm.to_string(index=False))

# H1 count results CSV
rows_h1 = []
for label, data in all_h1.items():
    rows_h1.append({"latent": label, **{k: v for k, v in data.items() if k != "tumor_h1_counts"}})
df_h1 = pd.DataFrame(rows_h1)
df_h1.to_csv(OUTPUT_DIR / "h1_count_test_results.csv", index=False)
print("\n[H1 Count Test]")
print(df_h1.to_string(index=False))

# Bootstrap CSV
rows_boot = []
for label, df in all_boot.items():
    for dim in ["H0", "H1"]:
        sub = df[df["homology"] == dim]
        rows_boot.append({
            "latent": label, "homology": dim,
            "features_mean": sub["n_features"].mean(),
            "features_std": sub["n_features"].std(),
            "cv_pct": sub["n_features"].std() / sub["n_features"].mean() * 100 if sub["n_features"].mean() > 0 else 0,
            "mean_pers_mean": sub["mean_pers"].mean(),
            "mean_pers_std": sub["mean_pers"].std(),
        })
df_boot = pd.DataFrame(rows_boot)
df_boot.to_csv(OUTPUT_DIR / "bootstrap_stability_results.csv", index=False)
print("\n[Bootstrap Stability]")
print(df_boot.to_string(index=False))

# ============================================================
# 시각화
# ============================================================
# 1. Permutation null distributions
fig, axes = plt.subplots(len(LATENT_FILES), 4, figsize=(20, 4 * len(LATENT_FILES)))
for row, (label, results) in enumerate(all_perm.items()):
    for col, metric in enumerate(["H0_wasserstein", "H0_bottleneck", "H1_wasserstein", "H1_bottleneck"]):
        ax = axes[row, col]
        null = results[metric]["null_distribution"]
        obs = results[metric]["observed"]
        p = results[metric]["p_value"]
        ax.hist(null, bins=40, alpha=0.7, color="#78909C", edgecolor="white")
        ax.axvline(obs, color="#D32F2F", linewidth=2.5, linestyle="--",
                   label=f"Observed={obs:.2f}")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{label} | {metric}\np={p:.4f} {sig}", fontsize=10)
        ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "permutation_null_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. H1 count comparison
fig, axes = plt.subplots(1, len(LATENT_FILES), figsize=(6 * len(LATENT_FILES), 4.5))
for i, (label, data) in enumerate(all_h1.items()):
    ax = axes[i]
    ax.hist(data["tumor_h1_counts"], bins=30, alpha=0.7, color="#1976D2",
            label=f"Tumor subsample (n=113)\nmean={data['tumor_h1_mean']:.1f}", edgecolor="white")
    ax.axvline(data["normal_h1"], color="#D32F2F", linewidth=2.5, linestyle="--",
               label=f"Normal (n=113) = {data['normal_h1']}")
    ax.set_title(f"{label} | H1 Feature Count\np={data['p_value']:.4f}")
    ax.set_xlabel("H1 Feature Count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "h1_count_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 3. Bootstrap stability
fig, axes = plt.subplots(len(LATENT_FILES), 2, figsize=(12, 4 * len(LATENT_FILES)))
for row, (label, df) in enumerate(all_boot.items()):
    for col, dim in enumerate(["H0", "H1"]):
        ax = axes[row, col]
        sub = df[df["homology"] == dim]
        ax.hist(sub["n_features"], bins=25, alpha=0.7, color="#43A047", edgecolor="white")
        cv = sub["n_features"].std() / sub["n_features"].mean() * 100
        ax.set_title(f"{label} | Normal Bootstrap {dim}\n"
                     f"mean={sub['n_features'].mean():.1f}, CV={cv:.1f}%")
        ax.set_xlabel("Feature Count")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "bootstrap_stability.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 4. Summary bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = list(all_perm.keys())
x = np.arange(len(labels))
w = 0.35

for ax_idx, metric_base in enumerate(["wasserstein", "bottleneck"]):
    ax = axes[ax_idx]
    h0_vals = [all_perm[l][f"H0_{metric_base}"]["observed"] for l in labels]
    h0_null = [all_perm[l][f"H0_{metric_base}"]["null_mean"] for l in labels]
    h1_vals = [all_perm[l][f"H1_{metric_base}"]["observed"] for l in labels]
    h1_null = [all_perm[l][f"H1_{metric_base}"]["null_mean"] for l in labels]

    bars1 = ax.bar(x - w/2, h0_vals, w*0.45, label="H0 observed", color="#1565C0")
    bars2 = ax.bar(x - w/2 + w*0.45, h0_null, w*0.45, label="H0 null mean", color="#90CAF9")
    bars3 = ax.bar(x + w/2, h1_vals, w*0.45, label="H1 observed", color="#C62828")
    bars4 = ax.bar(x + w/2 + w*0.45, h1_null, w*0.45, label="H1 null mean", color="#FFCDD2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Observed vs Null: {metric_base.title()} Distance")
    ax.legend(fontsize=8)
    ax.set_ylabel(f"{metric_base.title()} Distance")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "observed_vs_null_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n모든 시각화 저장 완료!")
print(f"\n{'=' * 70}")
print("Phase 2 완료!")
print(f"{'=' * 70}")
