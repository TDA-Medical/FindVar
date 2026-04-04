"""Task 2 — TCGA clinical survival analysis (BRCA).

Implements validation_plan.md Task 2, using:
- Prepped cohort table from prepare_tcga_survival_cohort.py
- Gene list + weights from phase3_gene_traceback/results/gene_importance_full.csv
- Expression fetched on-demand from UCSC Xena (TCGA BRCA HiSeqV2)

Outputs (default):
- FindVar/phase7_survival_analysis/results/cox_results.csv
- FindVar/phase7_survival_analysis/results/km_results.csv
- FindVar/phase7_survival_analysis/results/score_method_comparison.csv
- FindVar/phase7_survival_analysis/results/fig_kaplan_meier.pdf

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xenaPython import xenaQuery as xq


DEFAULT_HOST = "https://tcga.xenahubs.net"
DEFAULT_EXPR_DATASET = "TCGA.BRCA.sampleMap/HiSeqV2"


def tcga_sample_to_xena_sample(sample: str) -> str:
    # Convert e.g. TCGA-XX-YYYY-01A -> TCGA-XX-YYYY-01
    # Works for 01A/01B/11A/etc.
    parts = sample.split("-")
    if len(parts) < 4:
        return sample
    sample_type = parts[3][:2]
    return "-".join(parts[:3] + [sample_type])


def stage_to_group(stage: str) -> str:
    if not isinstance(stage, str) or stage.strip() == "":
        return "Unknown"
    s = stage.strip().upper()
    # Examples: Stage IIA, Stage II, STAGE IA
    if "STAGE" in s:
        s = s.replace("STAGE", "").strip()
    if s.startswith("IV"):
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return "Unknown"


def parse_any_yes(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).lower()
    if "yes" in s:
        return 1.0
    if "no" in s:
        return 0.0
    return np.nan


def load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"sample", "OS", "OS.time", "patient_barcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cohort missing required columns: {sorted(missing)}")

    df = df.copy()
    df["OS"] = pd.to_numeric(df["OS"], errors="coerce")
    df["OS.time"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df = df.dropna(subset=["OS", "OS.time"])
    df = df[df["OS.time"] > 0]

    df["sample_xena"] = df["sample"].astype(str).map(tcga_sample_to_xena_sample)
    return df


def load_gene_importance(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize columns
    for col in ["gene", "tda_rank", "P_Value", "tda_importance_norm", "euclidean_rank"]:
        if col not in df.columns:
            raise ValueError(f"gene importance file missing column '{col}'")
    return df


def get_h2c_genes(df_imp: pd.DataFrame) -> list[str]:
    genes = df_imp[(df_imp["tda_rank"] <= 200) & (df_imp["P_Value"] > 0.05)]["gene"].astype(str).tolist()
    genes = [g for g in genes if not g.startswith("unknown_")]
    # keep stable order by tda_rank
    genes = (
        df_imp[(df_imp["tda_rank"] <= 200) & (df_imp["P_Value"] > 0.05) & (~df_imp["gene"].astype(str).str.startswith("unknown_"))]
        .sort_values("tda_rank")["gene"].astype(str).tolist()
    )
    return genes


def get_euclidean_top37(df_imp: pd.DataFrame) -> list[str]:
    genes = df_imp.nsmallest(37, "euclidean_rank")["gene"].astype(str).tolist()
    genes = [g for g in genes if not g.startswith("unknown_")]
    return genes[:37]


def fetch_expression(host: str, dataset: str, samples: list[str], genes: list[str]) -> pd.DataFrame:
    # Xena gene-values API returns a list of dicts: {gene, scores: [[...]]}
    # Some genes may be missing or return empty score vectors; handle robustly.
    resp = xq.dataset_gene_values(host, dataset, samples, genes)

    df = pd.DataFrame(index=samples)
    for g in genes:
        df[g] = np.nan

    for item in resp:
        gene = item.get("gene")
        scores = item.get("scores")
        if gene is None or scores is None:
            continue

        row = scores[0] if isinstance(scores, list) and len(scores) > 0 else None
        if row is None:
            continue

        row_list = list(row)
        if len(row_list) != len(samples):
            # Skip malformed responses (e.g., empty score list)
            continue

        gene = str(gene)
        if gene not in df.columns:
            continue
        df[gene] = row_list

    # Coerce to numeric (some hubs may return values as strings)
    df = df.apply(pd.to_numeric, errors="coerce")
    # Guard against non-finite values (dropna won't remove +/-inf)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def drop_all_nan_genes(expr: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in expr.columns if expr[c].isna().all()]
    if missing:
        expr = expr.drop(columns=missing)
    return expr, missing


def score_pca(expr: pd.DataFrame) -> pd.Series:
    X = expr.values.astype(float)
    # Standardize with a minimum std to avoid exploding values for near-constant genes
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    min_std = 1e-3
    std = np.where(std < min_std, min_std, std)
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    # Use 'full' SVD to avoid noisy matmul RuntimeWarnings seen on some macOS/numpy builds
    pca = PCA(n_components=1, svd_solver="full", random_state=42)
    s = pca.fit_transform(Xs).ravel()
    return pd.Series(s, index=expr.index, name="score_pca")


def score_weighted(expr: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(expr.columns).astype(float)
    X = expr.values.astype(float)
    s = np.dot(X, w.values)
    return pd.Series(s, index=expr.index, name="score_weighted")


def score_cox_beta(expr: pd.DataFrame, durations: pd.Series, events: pd.Series, penalizer: float) -> tuple[pd.Series, pd.Series]:
    # Fit multivariate Cox on gene expressions to derive beta coefficients.
    X = expr.copy().astype(float)
    # Standardize genes for stability
    X.loc[:, :] = StandardScaler().fit_transform(X.values)

    df = X.copy()
    df["T"] = durations.values
    df["E"] = events.values

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="T", event_col="E")
    betas = cph.params_.copy()
    risk = X.dot(betas)
    return risk.rename("score_cox_beta"), betas


@dataclass
class KMResult:
    gene_set: str
    score_method: str
    n: int
    n_high: int
    n_low: int
    events: int
    p_logrank: float


def km_logrank(durations: pd.Series, events: pd.Series, score: pd.Series) -> tuple[KMResult, dict[str, pd.DataFrame]]:
    # Median split
    median = float(score.median())
    grp_high = score >= median

    d_high = durations[grp_high]
    e_high = events[grp_high]
    d_low = durations[~grp_high]
    e_low = events[~grp_high]

    lr = logrank_test(d_high, d_low, event_observed_A=e_high, event_observed_B=e_low)
    return (
        KMResult(
            gene_set="",
            score_method=score.name or "score",
            n=len(score),
            n_high=int(grp_high.sum()),
            n_low=int((~grp_high).sum()),
            events=int(events.sum()),
            p_logrank=float(lr.p_value),
        ),
        {
            "high": pd.DataFrame({"T": d_high, "E": e_high}),
            "low": pd.DataFrame({"T": d_low, "E": e_low}),
        },
    )


def fit_cox_univariate(durations: pd.Series, events: pd.Series, score: pd.Series, penalizer: float = 0.0) -> dict[str, object]:
    df = pd.DataFrame({"T": durations, "E": events, "score": score.astype(float)})
    df = df.dropna()
    df["score_z"] = (df["score"] - df["score"].mean()) / (df["score"].std(ddof=0) + 1e-12)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df[["T", "E", "score_z"]], duration_col="T", event_col="E")

    row = cph.summary.loc["score_z"]
    return {
        "n": int(df.shape[0]),
        "events": int(df["E"].sum()),
        "hr": float(np.exp(row["coef"])),
        "ci_lower": float(np.exp(row["coef lower 95%"])),
        "ci_upper": float(np.exp(row["coef upper 95%"])),
        "p": float(row["p"]),
        "concordance": float(cph.concordance_index_),
    }


def build_multivariate_design(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Age
    age_cols = ["age_at_diagnosis.diagnoses", "age_at_index.demographic"]
    age = None
    for c in age_cols:
        if c in df.columns:
            age = pd.to_numeric(df[c], errors="coerce")
            break
    if age is not None:
        # Heuristic: if values are large, they are days -> convert to years
        if float(age.dropna().median()) > 150:
            out["age_years"] = age / 365.25
        else:
            out["age_years"] = age

    # Stage
    stage_col = "ajcc_pathologic_stage.diagnoses"
    if stage_col in df.columns:
        stage_group = df[stage_col].map(stage_to_group)
        dummies = pd.get_dummies(stage_group, prefix="stage", dummy_na=False)
        # drop unknown baseline if present
        if "stage_Unknown" in dummies.columns:
            dummies = dummies.drop(columns=["stage_Unknown"])
        out = pd.concat([out, dummies], axis=1)

    # Treatment (binary: any yes)
    tr_col = "treatment_or_therapy.treatments.diagnoses"
    if tr_col in df.columns:
        out["any_treatment_yes"] = df[tr_col].map(parse_any_yes)

    return out


def fit_cox_multivariate(durations: pd.Series, events: pd.Series, score: pd.Series, covariates: pd.DataFrame, penalizer: float) -> dict[str, object]:
    df = pd.DataFrame({"T": durations, "E": events, "score": score.astype(float)}, index=score.index)
    df = pd.concat([df, covariates], axis=1)
    df = df.dropna(subset=["T", "E", "score"])  # covariates can be partially missing

    df["score_z"] = (df["score"] - df["score"].mean()) / (df["score"].std(ddof=0) + 1e-12)

    # Drop raw score to avoid duplication
    cols = ["T", "E", "score_z"] + [c for c in covariates.columns if c in df.columns]
    df_model = df[cols].dropna()  # multivariate requires complete cases

    if df_model.shape[0] < 50:
        raise ValueError("Too few complete cases for multivariate Cox")

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_model, duration_col="T", event_col="E")

    row = cph.summary.loc["score_z"]
    return {
        "n": int(df_model.shape[0]),
        "events": int(df_model["E"].sum()),
        "hr": float(np.exp(row["coef"])),
        "ci_lower": float(np.exp(row["coef lower 95%"])),
        "ci_upper": float(np.exp(row["coef upper 95%"])),
        "p": float(row["p"]),
        "concordance": float(cph.concordance_index_),
        "covariates": json.dumps([c for c in df_model.columns if c not in {"T", "E", "score_z"}]),
    }


def plot_km(ax, durations: pd.Series, events: pd.Series, score: pd.Series, title: str) -> float:
    median = float(score.median())
    grp_high = score >= median

    kmf = KaplanMeierFitter()

    kmf.fit(durations[~grp_high], event_observed=events[~grp_high], label=f"Low (n={int((~grp_high).sum())})")
    kmf.plot(ax=ax, ci_show=False)

    kmf.fit(durations[grp_high], event_observed=events[grp_high], label=f"High (n={int(grp_high.sum())})")
    kmf.plot(ax=ax, ci_show=False)

    lr = logrank_test(durations[grp_high], durations[~grp_high], event_observed_A=events[grp_high], event_observed_B=events[~grp_high])
    p = float(lr.p_value)

    ax.set_title(f"{title}\nlog-rank p={p:.3g}")
    ax.set_xlabel("Time (OS.time)")
    ax.set_ylabel("Survival probability")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=Path("FindVar/phase7_survival_analysis/data/brca_survival_cohort.tsv"))
    parser.add_argument("--gene-importance", type=Path, default=Path("FindVar/phase3_gene_traceback/results/gene_importance_full.csv"))
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--expr-dataset", type=str, default=DEFAULT_EXPR_DATASET)
    parser.add_argument("--outdir", type=Path, default=Path("FindVar/phase7_survival_analysis/results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cox-penalizer", type=float, default=0.1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_cohort(args.cohort)
    df_imp = load_gene_importance(args.gene_importance)

    h2c_genes = get_h2c_genes(df_imp)
    euc_genes = get_euclidean_top37(df_imp)

    # Ensure H2C gene count expected (plan: 37)
    # If not exactly 37, keep the first 37 by rank to match the plan output size.
    if len(h2c_genes) > 37:
        h2c_genes = h2c_genes[:37]

    durations = df["OS.time"].astype(float)
    events = df["OS"].astype(int)

    # Filter to samples available in expression dataset
    dataset_samples = set(xq.dataset_samples(args.host, args.expr_dataset))
    df = df[df["sample_xena"].isin(dataset_samples)].copy()
    df = df.set_index("sample_xena", drop=False)
    durations = df["OS.time"].astype(float)
    events = df["OS"].astype(int)

    # Determine gene universe for the expression dataset (for random control sampling)
    dataset_fields = set(xq.dataset_field(args.host, args.expr_dataset))
    # remove non-gene fields if present
    dataset_fields.discard("sampleID")

    # Random control genes sampled from the expression-universe (excluding unknown_ and excluding union sets)
    candidate_genes = [g for g in df_imp["gene"].astype(str).tolist() if not g.startswith("unknown_")]
    exclude = set(h2c_genes) | set(euc_genes)
    candidate_genes = [g for g in candidate_genes if g not in exclude and g in dataset_fields]
    if len(candidate_genes) < 37:
        raise ValueError(f"Not enough candidate genes in dataset for Random_37 (n={len(candidate_genes)})")
    rand_genes = list(np.random.choice(candidate_genes, size=37, replace=False))

    # Fetch expression
    samples = df.index.astype(str).tolist()
    expr_h2c = fetch_expression(args.host, args.expr_dataset, samples, h2c_genes)
    expr_euc = fetch_expression(args.host, args.expr_dataset, samples, euc_genes)
    expr_rand = fetch_expression(args.host, args.expr_dataset, samples, rand_genes)

    # Drop genes that are entirely missing in this dataset (and persist for traceability)
    expr_h2c, missing_h2c = drop_all_nan_genes(expr_h2c)
    expr_euc, missing_euc = drop_all_nan_genes(expr_euc)
    expr_rand, missing_rand = drop_all_nan_genes(expr_rand)

    (pd.Series(missing_h2c, name="gene").to_csv(outdir / "missing_genes_h2c.csv", index=False))
    (pd.Series(missing_euc, name="gene").to_csv(outdir / "missing_genes_euclidean.csv", index=False))
    (pd.Series(missing_rand, name="gene").to_csv(outdir / "missing_genes_random.csv", index=False))

    # If H2C gene set shrank, record the effective list
    h2c_genes_eff = [g for g in h2c_genes if g in expr_h2c.columns]
    euc_genes_eff = [g for g in euc_genes if g in expr_euc.columns]
    rand_genes_eff = [g for g in rand_genes if g in expr_rand.columns]

    # Drop samples with any missing expression within each effective set
    expr_h2c = expr_h2c[h2c_genes_eff].dropna(axis=0, how="any")
    expr_euc = expr_euc[euc_genes_eff].dropna(axis=0, how="any")
    expr_rand = expr_rand[rand_genes_eff].dropna(axis=0, how="any")

    # Use common samples across sets for fair comparison
    common_samples = expr_h2c.index.intersection(expr_euc.index).intersection(expr_rand.index)
    df = df.loc[common_samples].copy()
    durations = df["OS.time"].astype(float)
    events = df["OS"].astype(int)

    expr_h2c = expr_h2c.loc[common_samples]
    expr_euc = expr_euc.loc[common_samples]
    expr_rand = expr_rand.loc[common_samples]

    # Update effective gene lists after filtering
    h2c_genes = h2c_genes_eff
    euc_genes = euc_genes_eff
    rand_genes = rand_genes_eff

    # Score methods for H2C
    weights_h2c = df_imp.set_index("gene")["tda_importance_norm"]

    scores_h2c = {
        "pca": score_pca(expr_h2c),
        "weighted": score_weighted(expr_h2c, weights_h2c),
    }

    # Cox-beta based score may fail; handle gracefully
    betas_h2c = None
    try:
        s_beta, betas = score_cox_beta(expr_h2c, durations, events, penalizer=args.cox_penalizer)
        scores_h2c["cox_beta"] = s_beta
        betas_h2c = betas
    except Exception:
        pass

    # For Euclidean and Random: use PCA + (optional) weighted if euclidean weights exist
    scores_euc = {"pca": score_pca(expr_euc)}
    scores_rand = {"pca": score_pca(expr_rand)}

    # Compare H2C score methods by log-rank p (OS)
    comparison_rows = []
    for method, score in scores_h2c.items():
        uni = fit_cox_univariate(durations, events, score)
        km, _ = km_logrank(durations, events, score)
        comparison_rows.append(
            {
                "gene_set": "H2C_37",
                "score_method": method,
                "n": uni["n"],
                "events": uni["events"],
                "cox_hr_perSD": uni["hr"],
                "cox_p": uni["p"],
                "cox_concordance": uni["concordance"],
                "km_logrank_p": km.p_logrank,
            }
        )
    df_cmp = pd.DataFrame(comparison_rows).sort_values("km_logrank_p")
    df_cmp.to_csv(outdir / "score_method_comparison.csv", index=False)

    best_method = df_cmp.iloc[0]["score_method"] if len(df_cmp) else "pca"
    h2c_score = scores_h2c[str(best_method)]

    # Choose Euclidean score method to match if possible; fallback to PCA
    euc_score = scores_euc.get(str(best_method), scores_euc["pca"])
    rand_score = scores_rand["pca"]

    # KM figure: (A) H2C, (B) Euclidean
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    p_h2c = plot_km(axes[0], durations, events, h2c_score, title=f"H2C_37 ({best_method})")
    p_euc = plot_km(axes[1], durations, events, euc_score, title="Euclidean_top37 (pca)")
    plt.tight_layout()
    fig.savefig(outdir / "fig_kaplan_meier.pdf", bbox_inches="tight")
    plt.close(fig)

    # Cox results
    cov = build_multivariate_design(df)

    rows = []
    for gene_set, score_name, score in [
        ("H2C_37", str(best_method), h2c_score),
        ("Euclidean_top37", "pca", euc_score),
        ("Random_37", "pca", rand_score),
    ]:
        uni = fit_cox_univariate(durations, events, score, penalizer=args.cox_penalizer)
        rows.append(
            {
                "endpoint": "OS",
                "model": "univariate",
                "gene_set": gene_set,
                "score_method": score_name,
                **uni,
            }
        )

        try:
            multi = fit_cox_multivariate(durations, events, score, cov, penalizer=args.cox_penalizer)
            rows.append(
                {
                    "endpoint": "OS",
                    "model": "multivariate",
                    "gene_set": gene_set,
                    "score_method": score_name,
                    **multi,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "endpoint": "OS",
                    "model": "multivariate",
                    "gene_set": gene_set,
                    "score_method": score_name,
                    "n": np.nan,
                    "events": np.nan,
                    "hr": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p": np.nan,
                    "concordance": np.nan,
                    "covariates": json.dumps([]),
                    "error": str(e),
                }
            )

    df_cox = pd.DataFrame(rows)
    df_cox.to_csv(outdir / "cox_results.csv", index=False)

    # KM summary table
    km_rows = []
    for gene_set, score_name, score in [
        ("H2C_37", str(best_method), h2c_score),
        ("Euclidean_top37", "pca", euc_score),
        ("Random_37", "pca", rand_score),
    ]:
        km, _ = km_logrank(durations, events, score)
        km_rows.append(
            {
                "endpoint": "OS",
                "gene_set": gene_set,
                "score_method": score_name,
                "n": km.n,
                "events": km.events,
                "n_high": km.n_high,
                "n_low": km.n_low,
                "logrank_p": km.p_logrank,
            }
        )
    pd.DataFrame(km_rows).to_csv(outdir / "km_results.csv", index=False)

    # Persist selected gene lists for traceability
    pd.Series(h2c_genes, name="gene").to_csv(outdir / "genes_h2c_37.csv", index=False)
    pd.Series(euc_genes, name="gene").to_csv(outdir / "genes_euclidean_top37.csv", index=False)
    pd.Series(rand_genes, name="gene").to_csv(outdir / "genes_random_37.csv", index=False)

    print("[survival_analysis] done")
    print(f"  samples used: n={len(df)} events={int(events.sum())}")
    print(f"  best H2C score method: {best_method}")
    print(f"  KM log-rank p: H2C={p_h2c:.3g}, Euclidean={p_euc:.3g}")


if __name__ == "__main__":
    main()
