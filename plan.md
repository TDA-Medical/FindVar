# TDA 분석 계획서

## 프로젝트 목표

TCGA-BRCA 유전자 발현 데이터에 Topological Data Analysis(TDA)를 적용하여,
기존 유클리드 기반 통계 분석에서 발견하지 못한 **암 관련 잠재적 바이오마커(유전자 조합)**를 식별한다.

---

## 현재 완료된 작업

| 단계 | 상태 | 산출물 |
|------|------|--------|
| 데이터 전처리 (log1p + GPU ComBat) | 완료 | `data_preprocessing/cleaned_tcga_tpm_for_TAE.csv` (1,215 샘플 x 20,862 유전자) |
| 탐색적 데이터 분석 (EDA) | 완료 | `data_analysis/output*/` (T-test, 상관관계, 카이제곱 등) |
| Bad genes 식별 | 완료 | `data_analysis/output_bad_genes/` (2,020개 bad genes, HLA/RPL 패밀리 농축) |
| TAE 학습 | 완료 | `TAE/models/` (16d/32d/64d, cosine/pearson/euclidean/sinkhorn 변형) |
| Latent 추출 + SMOTE | 완료 | `TAE/results/woutSMOTE/`, `TAE/results/wSMOTE/` |

### 기존 분석(유클리드)에서 이미 발견된 것
- 개별 유전자 수준: EGFR, FIGF, ESR1, PGR 등의 차별 발현
- 호르몬 의존성 확인 (ESR1 17배, PGR 10.9배)
- TSS(조직 출처)가 강력한 교란요인
- BRCA1/2는 "Not Bad"로 분류됨

---

## 앞으로 할 작업

### Phase 1: TDA 기반 분석 환경 구축

**1-1. 라이브러리 선정 및 설치**
- `ripser` — Vietoris-Rips Persistent Homology 계산 (빠르고 메모리 효율적)
- `persim` — Persistence diagram 비교 (Wasserstein/Bottleneck distance)
- `gudhi` — 고급 TDA 기능 (Alpha complex, Simplex tree 등)
- `giotto-tda` — scikit-learn 호환 TDA 파이프라인 (선택)

**1-2. 입력 데이터 결정**
- 주 분석 대상: `TAE/results/woutSMOTE/latent_*.csv` (SMOTE 전 원본 latent)
- 비교군: 원본 고차원 데이터 (`cleaned_tcga_tpm_for_TAE.csv`)에서 서브샘플링
- 차원별(16d/32d/64d) 및 메트릭별(cosine/pearson/euclidean/sinkhorn) 비교

---

### Phase 2: Persistent Homology 계산

**2-1. 전체 데이터에 대한 PH 계산**
- Vietoris-Rips complex 구성
- H0 (연결 성분), H1 (루프/구멍), H2 (빈 공간) 계산
- 각 latent 설정(차원 x 메트릭)별로 persistence diagram 생성

**2-2. 종양 vs 정상 서브그룹 PH**
- 종양 샘플(1,105개)과 정상 샘플(113개)을 분리하여 각각 PH 계산
- 두 그룹의 persistence diagram 비교
- Wasserstein distance / Bottleneck distance로 위상적 차이 정량화

**2-3. 안정성 검증**
- Bootstrap 리샘플링으로 persistence feature의 안정성 확인
- 다양한 필트레이션 파라미터에 대한 민감도 분석

---

### Phase 3: 위상적 특징에서 유전자로 역추적

> 이 단계가 핵심: "위상적으로 중요한 특징"을 만드는 "유전자 조합"을 찾아내는 것

**3-1. Persistence Feature → Latent Dimension 매핑**
- 종양/정상 간 차이가 큰 persistence feature 식별
- 해당 feature를 만드는 latent 차원(z0~z15 등) 추적
- 어떤 latent 차원 조합이 위상적 구조 차이에 기여하는지 분석

**3-2. Latent Dimension → 원본 유전자 매핑**
- TAE 디코더의 가중치를 분석하여 latent → gene 기여도 계산
- 각 latent 차원에 가장 크게 기여하는 유전자 Top-K 추출
- 위상적으로 중요한 latent 차원과 연결되는 유전자 조합 도출

**3-3. 기존 분석과의 교차 검증**
- TDA에서 발견한 유전자 조합이 기존 T-test/상관관계에서는 어떠했는지 확인
- **"유클리드에서 유의미하지 않았으나 TDA에서 중요한 유전자"** 를 특별히 주목
- 이 유전자들이 bad genes 목록과 어떻게 관련되는지 확인

---

### Phase 4: 생물학적 해석 및 명명

**4-1. Pathway 분석**
- 발견한 유전자 조합에 대해 Gene Ontology (GO) enrichment 분석
- KEGG pathway 매핑
- 기존 암 관련 경로(p53, PI3K-Akt, MAPK 등)와의 연관성 확인

**4-2. 바이오마커 명명**
- 발견한 유전자 조합/패턴에 이름 부여
  - **H2C (Hwang-Hwang-Choi) Gene Panel**: 37개 유전자
- 명명 기준: 팀원 성씨 (황 2명 + 최 1명)

**4-3. 분류 성능 검증 (선택)**
- 발견한 유전자 조합만으로 종양/정상 분류 수행
- 기존 유전자 세트(EGFR, BRCA1 등) 대비 성능 비교
- TDA 기반 특징의 부가 가치 입증

---

### Phase 5: 시각화 및 논문 준비

**5-1. 핵심 시각화**
- Persistence diagram / Barcode plot (종양 vs 정상 비교)
- Mapper 그래프 (데이터의 위상적 네트워크 구조)
- 유전자 기여도 히트맵 (latent ↔ gene 매핑)
- 기존 분석 vs TDA 분석 벤 다이어그램

**5-2. 논문 스토리라인**
```
① 기존 유클리드 분석에서는 이런 유전자들이 발견되었다 (배경)
② 같은 데이터에 TAE + TDA를 적용하니 ①에서 안 보이던 패턴이 나왔다 (방법론)
③ 이 패턴을 구성하는 유전자 조합을 "H2C"로 명명한다 (발견)
④ 이 조합의 생물학적 의미를 pathway 분석으로 뒷받침한다 (해석)
⑤ 분류 성능도 기존 대비 향상됨을 보인다 (검증)
```

---

## 예상 산출물

| 산출물 | 위치 (예정) |
|--------|-------------|
| TDA 분석 스크립트 | `tda_analysis/` |
| Persistence diagram CSV/이미지 | `tda_analysis/results/` |
| 유전자 역추적 결과 | `tda_analysis/results/gene_mapping/` |
| Pathway 분석 결과 | `tda_analysis/results/pathway/` |
| 최종 시각화 | `tda_analysis/figures/` |

---

## 기술적 고려사항

- **메모리**: 1,215 샘플의 pairwise distance matrix는 ~11.8MB (float64)로 관리 가능
- **계산 시간**: Ripser는 16~64d latent에서 수 초~수 분 내 완료 예상
- **메트릭 선택**: TAE 학습 시 사용한 메트릭(cosine/pearson)과 TDA 필트레이션 메트릭을 일치시키는 것이 구조 보존에 유리
- **정상 샘플 수 (113개)**: 서브그룹 분석 시 통계적 검정력 한계를 인지하고, bootstrap으로 보완

---

## 우선순위

1. **Phase 2-1, 2-2** — 먼저 PH를 계산해서 종양/정상 간 위상적 차이가 실제로 존재하는지 확인
2. **Phase 3** — 차이가 확인되면 유전자 역추적 (프로젝트의 핵심 기여)
3. **Phase 4** — 생물학적 해석으로 발견의 의미 부여
4. **Phase 5** — 논문화
