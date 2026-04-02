# FindVar

**TDA 기반 암 바이오마커 발견 파이프라인**

TCGA-BRCA RNA-seq 데이터에 Topological Data Analysis(TDA)를 적용하여, 기존 유클리드 통계에서 발견할 수 없었던 암 관련 바이오마커 유전자 조합(**H2C Gene Panel**)을 식별한 프로젝트입니다.

---

## 핵심 발견

| 발견 | 내용 |
|------|------|
| **H1 루프 구조** | 종양은 정상 대비 2.5배 더 많은 H1 루프를 가짐 (p < 0.001) |
| **유전자 겹침 0%** | TDA Top 200과 유클리드 Top 200이 완전히 다른 유전자 세트 |
| **H2C Gene Panel** | 유클리드에서 비유의미(p>0.05)한 37개 유전자로 AUC=0.993 달성 |
| **Pathway 직교성** | TDA: 세포침습/골격 vs 유클리드: 대사/이온채널 (Pathway 겹침 0) |

---

## 프로젝트 구조

```
FindVar/
├── README.md                                ← 이 문서
├── plan.md                                  ← 전체 분석 계획
├── result.md                                ← 종합 결과 정리 (논문 작업용)
│
├── phase1_tda_setup/                        ← Phase 1: TDA 탐색적 분석
│   ├── verify_install.py                    │  라이브러리 설치 검증
│   ├── explore_ph.py                        │  Persistent Homology 탐색
│   ├── PHASE1_REPORT.md                     │  분석 보고서
│   └── results/
│       ├── ph_comparison_summary.csv        │  PH 비교 요약 테이블
│       ├── ph_diagram_*.png                 │  Persistence Diagram (5개 설정)
│       └── distance_comparison.png          │  Wasserstein/Bottleneck 비교
│
├── phase2_persistent_homology/              ← Phase 2: 통계 검증
│   ├── analyze_ph.py                        │  Permutation test + Bootstrap
│   ├── PHASE2_REPORT.md                     │  분석 보고서
│   └── results/
│       ├── permutation_test_results.csv     │  Permutation p-value 테이블
│       ├── h1_count_test_results.csv        │  H1 count test (핵심 결과)
│       ├── bootstrap_stability_results.csv  │  Bootstrap 안정성
│       ├── permutation_null_distributions.png
│       ├── h1_count_comparison.png          │  ★ H1 count: 종양 vs 정상
│       ├── observed_vs_null_comparison.png
│       └── bootstrap_stability.png
│
├── phase3_gene_traceback/                   ← Phase 3: 유전자 역추적
│   ├── traceback_genes.py                   │  디코더 Jacobian 기반 역추적
│   ├── PHASE3_REPORT.md                     │  분석 보고서
│   └── results/
│       ├── gene_importance_full.csv         │  전체 20,876 유전자 TDA 랭킹
│       ├── gene_importance_top100.csv       │  Top 100 상세
│       ├── tda_only_genes.csv               │  TDA-only 200개 유전자
│       ├── both_methods_genes.csv           │  양쪽 발견 유전자 (0개)
│       ├── latent_dimension_analysis.csv    │  32개 latent 차원 분석
│       ├── top30_genes.png                  │  Top 30 유전자 바 차트
│       ├── tda_vs_euclidean_rank.png        │  ★ TDA vs 유클리드 산점도
│       ├── discovery_comparison.png         │  발견 유전자 벤 다이어그램
│       ├── latent_dimension_importance.png
│       └── latent_pca.png
│
├── phase4_biological_interpretation/        ← Phase 4: Pathway + 분류 검증
│   ├── pathway_and_validation.py            │  GO/KEGG + 분류 성능
│   ├── PHASE4_REPORT.md                     │  분석 보고서
│   └── results/
│       ├── enrichment_tda_top200.csv        │  TDA Pathway enrichment
│       ├── enrichment_euclidean_top200.csv  │  유클리드 Pathway enrichment
│       ├── classification_results.csv       │  전체 분류 성능 결과
│       ├── pathway_overlap_summary.csv      │  Pathway 겹침 요약
│       ├── classification_comparison.png    │  ★ 분류 성능 비교
│       └── pathway_comparison.png           │  ★ Pathway 비교
│
└── phase5_visualization_paper/              ← Phase 5: 논문용 시각화
    ├── generate_figures.py                  │  Figure 생성 스크립트
    └── figures/
        ├── fig2_persistence_diagrams.pdf    │  Persistence Diagram
        ├── fig3_statistical_validation.pdf  │  통계 검증
        ├── fig4_gene_discovery.pdf          │  유전자 발견
        ├── fig5_pathway_comparison.pdf      │  Pathway 비교
        ├── fig6_classification.pdf          │  분류 성능
        ├── fig7_latent_space.pdf            │  Latent Space
        ├── summary_figure.pdf              │  전체 요약
        └── *.png                            │  (PNG 버전 동봉)
```

---

## 분석 파이프라인

```
TCGA-BRCA RNA-seq (1,215 samples × 20,862 genes)
  │
  ├─ [전처리] log1p → GPU ComBat → 유전자 필터링 (Data-preprocessing 리포)
  │
  ├─ [TAE] Topological Autoencoder (32d cosine latent)
  │
  ├─ [Phase 1] Persistent Homology 탐색 → 종양/정상 차이 확인
  │
  ├─ [Phase 2] Size-matched permutation test → H1 루프 p < 0.001
  │
  ├─ [Phase 3] 디코더 Jacobian → 유전자 역추적 → TDA vs 유클리드 겹침 0%
  │
  ├─ [Phase 4] Pathway enrichment + 분류 검증 → H2C AUC=0.993
  │
  └─ [Phase 5] 논문용 Figure 생성 (PDF 벡터)
```

---

## H2C Gene Panel

유클리드 통계에서 **완전히 비유의미(p > 0.05)**했으나, TDA에서 핵심으로 식별된 37개 유전자.

대표 유전자:

| 유전자 | TDA 순위 | 유클리드 P-value | 기능 |
|--------|---------|-----------------|------|
| EFCAB3 | 8 | 0.791 | Ca2+ 결합 도메인 |
| PGC | 11 | 0.908 | Pepsinogen C |
| RPRM | 13 | 0.206 | p53 표적, G2 체크포인트 |
| RPRML | 14 | 0.333 | Reprimo-like |
| HSPB9 | 18 | 0.924 | 소형 열충격단백질 |

전체 목록: `phase3_gene_traceback/results/tda_only_genes.csv`

---

## 실행 환경

| 항목 | 값 |
|------|-----|
| Python | 3.12.13 (conda: tda) |
| PyTorch | 2.11.0+cu126 |
| ripser | 0.6.14 |
| persim | 0.3.8 |
| gudhi | 3.12.0 |
| scikit-learn | 1.8.0 |
| gseapy | 1.1.13 |

---

## 관련 리포지터리

| 리포 | 내용 |
|------|------|
| [Data-preprocessing](https://github.com/TDA-Medical/Data-preprocessing) | 전처리 + TAE 학습 |
| **FindVar** (이 리포) | TDA 분석 + H2C 발견 |
