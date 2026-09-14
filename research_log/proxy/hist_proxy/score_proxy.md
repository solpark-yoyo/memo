# Score Proxy Registry — score 기반 memorization metric 논문 요약

**score function $s_\theta = \nabla_x \log p_t(x)$ 기반(또는 그 미분·등가 변환) memorization
metric을 제시한 논문**을 registry-entry 형식으로 축적.

**스크리닝 조건**: → **score proxy survey**: `skills/proxy-his/rules/score_proxy.md` 참조

---

## 판정표 — ① score based metric 제시 = Yes 논문만 (2026-08-24)

| 논문 | ② 연관 논리 | ③ 메인트랙 | ④ VRAM·speed 장점 | ① metric (짧은 설명) |
|---|---|---|---|---|
| **Jeon et al. 2025** (ICML 2025 Spotlight) | Yes — sharp log-density | Yes | Yes — HVP·Arnoldi $O(m^2d)$ | $\|H^\Delta s^\Delta\|^2$, 첫 step AUC 0.998 |
| **Asthana & Belagiannis 2026** (ICLR 2026) | Yes — 이론 명시(aniso 정렬+iso norm) | Yes | Yes — UNet 4회 | $s^\Delta$ cos(t≈0)+norm(t≈T) |
| **Shin et al. 2026** (ICML 2026) | Yes — emb 근처 score 급상승 | Yes | Yes — 저해상도 latent | score 선적분 + cond Jacobian (초록 수준) |
| p-Laplace 2025 (SSVM 2025) | Yes — 음의 flux + error bound | No — 비메인트랙 | No — O(N) score query | $\Delta_1\log p$ ball 경계 MC, no-prompt AUC 0.913 |

**표 외 (① No)**: Wen 2025(ICLR)·Han 2025(NeurIPS) — ε 표기(score와 아핀 등가, 별도 표기)
→ 아래 "참고 — ε 계열" 섹션

---

## 1. [Jeon et al., 2025] Sharpness of Probability Landscapes — score Hessian metric (SAIL)  `[SUMMARIZED]`

### 스크리닝 (yes/no)

1. **score based metric 제시: Yes** — metric이 score 차이 $s^\Delta$와 그 Hessian-vector
   product $H^\Delta s^\Delta$로 정의됨 (log-density sharpness의 명시적 score 미분)
2. **memorization 연관 논리: Yes** — memorization = 학습 분포의 "날카로운 국소 첨두(sharp
   peaks)"; metric 값 크면 memorized. Wen metric을 sharpness 관점에서 이론적으로 재해석·증폭
3. **탑티어 컨퍼 메인 트랙 게재: Yes** — ICML 2025 (Spotlight)
4. **VRAM·speed 장점: Yes** — Hessian-vector product(autodiff) + Arnoldi 반복
   $O(m^2d)$ (전체 고유분해 $O(d^3)$ 대비) "minimal overhead" 주장. 단 명시적 VRAM/wall-clock
   수치는 미확인

### 설명

**metric (detection)**

$$x_t \xrightarrow{\text{ⓐ}} s_\theta^\Delta \Rightarrow d_{\text{sharp}} = \big\|H_\theta^\Delta(x_t)\, s_\theta^\Delta(x_t)\big\|^2$$

- $s_\theta^\Delta(x_t) = s_\theta(x_t,c) - s_\theta(x_t)$ (guidance score 차이),
  $H_\theta^\Delta = H_\theta(x_t,c) - H_\theta(x_t)$ (log-density Hessian 차이)
- ⓐ: **UNet (Denoiser)** — cond/uncond 2회 + autodiff HVP (score 표기 명시)
- (결과식) **Calculation** — 값 크면 memorized (sharp peak). Wen metric($\|s^\Delta\|$, 2제곱)을
  4제곱 증폭으로 민감도 향상
- 검증: **첫 step ($t{=}T{-}1$)** 측정만으로 AUC 0.998 (generations 4개)

**완화 (SAIL — Sharpness-Aware Initialization for Latent diffusion)**

$$x_T \leftarrow x_T - \nabla_{x_T}\Big[\big\|s_\theta^\Delta(x_T + \delta s_\theta^\Delta(x_T)) - s_\theta^\Delta(x_T)\big\|^2 + \alpha\|x_T\|^2\Big]$$

- update 변수 = **x_T(initial noise)** — smooth 영역으로 초기 노이즈 이동 (Adam lr 0.05,
  threshold 조기중단으로 가우시안 정합 유지)

### 비교

우리 ①(Tweedie domain gap)은 $s$와 $x_0$ 추정 간 괴리, 본 metric은 $s$의 **곡률(Hessian)** —
같은 score family의 상위 미분 신호. 첫 step 단발 측정 구조가 우리 ① 측정 프로토콜과 동일선상

---

## 2. [Asthana & Belagiannis, 2026] — guidance score 정렬+norm metric M(x_T, c)  `[SUMMARIZED]`

### 스크리닝 (yes/no)

1. **score based metric 제시: Yes** — guidance 벡터를 **score** $s_\theta$ 표기로 정의하고
   cos 정렬·norm으로 결합
2. **memorization 연관 논리: Yes** — 이론 명시: t≈0에서 guidance score와 uncond score의
   **이방성 정렬** + t≈T에서 **등방성 노름**으로 memorization 특징화
3. **탑티어 컨퍼 메인 트랙 게재: Yes** — ICLR 2026
4. **VRAM·speed 장점: Yes** — trajectory 없이 **UNet 4회 호출** (denoising rollout 불필요)

### 설명

$$M(x_T, c) = \gamma_1 \cos\big(s_\theta^\Delta|_{t\approx0},\ s_\theta|_{t\approx0}\big) + \gamma_2 \big\|s_\theta^\Delta|_{t\approx T}\big\|$$

- $s_\theta^\Delta = s_\theta(x,t,c) - s_\theta(x,t)$ · **UNet (Denoiser)** 동일 $x_T$ 2개
  timestep × cond/uncond = 4회 · **Calculation** 결합 후 logistic regressor 임계 판정
- 완화: prompt embedding $c$ 최적화 (x_T 고정)

### 비교

score 방향(cos) 항은 우리 체계에 없는 신호 차원 — ①·③(norm 계열)과 직교 보완. 측정 비용이
가장 저렴(UNet 4회)해 재검증 후보

---

## 3. [Shin et al., 2026] Text Embedding Interpolation — score 선적분 + Jacobian  `[SUMMARIZED · 초록 수준]`

### 스크리닝 (yes/no)

1. **score based metric 제시: Yes** — cond/uncond **score 차이**를 "unconditional → prompt
   embedding 선형 보간 경로의 conditional score **선적분(linear integral)**"로 해석하고,
   prompt embedding에서 평가한 **conditional score Jacobian**으로 급상승 측정
2. **memorization 연관 논리: Yes** — score trajectory가 prompt embedding 근처에서 급격히
   상승하는 현상 = memorization 징후
3. **탑티어 컨퍼 메인 트랙 게재: Yes** — ICML 2026
4. **VRAM·speed 장점: Yes** — 축소 latent resolution에서도 안정 작동 → 저해상도 평가로
   **메모리 효율적 감지** 주장

### 설명

- 수식·AUC 등 상세는 미검증 (초록/포스터 페이지 기준 — 원문 확인 후 갱신 필요)
- 관점: ε가 아닌 **score를 적분 경로로 해석** — interpolation 축에서의 관성(선적분)이라는
  신규 정식화

### 비교

"text embedding 축을 따른 score 민감도(Jacobian)"는 우리 registry에 없는 축 — ①·③이
timestep 축, 본 metric이 embedding 축. 저해상도 평가 아이디어는 eps_trajectory 비용 절감에 차용 가능

---

## 4. [p-Laplace, 2025] — $\Delta_p \log p$ 국소 곡률 metric  `[SUMMARIZED · 비메인트랙]`

### 스크리닝 (yes/no)

1. **score based metric 제시: Yes** — estimated score $\hat s$로 log-probability의 p-Laplace
   연산자 $\Delta_p u = \nabla\cdot(|\nabla u|^{p-2}\nabla u)$, $\nabla u = s$를 Monte Carlo
   ball-경계 적분으로 계산 (p=1이 강건·최적)
2. **memorization 연관 논리: Yes** — memorized point = 국소 확률 첨두 → gradient 내향 →
   **음의 flux** ($\Delta_1 u < 0$). 이론적 error bound 제시
3. **탑티어 컨퍼 메인 트랙 게재: No** — SSVM 2025 (전문 소규모 컨퍼런스, top-tier 아님)
4. **VRAM·speed 장점: No** — 샘플당 N(=64)개 구면 점에서 score query — 500 steps 기준
   ≈32,000회 평가로 감지 목적 기준 고비용 (prompt 불필요한 post-generation regime은 독자적)

### 설명

- **No-prompt regime** AUC 0.913 (Wen baseline 0.502 → +81%), with-prompt 0.958 · SD1.4,
  Webster 500 memorized prompts · 생성 후(post-hoc) 감지

### 비교

prompt·CFG 없이 score만으로 감지한다는 점이 독자적 — 우리 ①(tweedie re-forward)도 prompt
참조 없이 측정 가능하므로 no-prompt 프로토콜 비교 상대

---

## 참고 — ε 계열 (score와 아핀 등가, score 표기 아님)

- **[Wen et al., 2025 ICLR]** $\|\epsilon_\theta(x_t,e_p)-\epsilon_\theta(x_t,e_\emptyset)\|_2$ —
  ① **No** (ε 예측 노이션; $\epsilon_\theta \propto -s_\theta$ 아핀 등가이나 논문이 score
  프레임으로 제시하지 않음) ② Yes ③ Yes ④ Yes (추가 UNet 호출 0회, 첫 step AUC 0.960·0.2s)
- **[Han et al., 2025 NeurIPS]** 동일 신호의 x_T 단발 버전 — ① No ② Yes ③ Yes ④ 부분
  (명시 수치 없음)
- 상세 요약은 `x_t_proxy.md` 참조

---

## 관리 규칙

- 논문 1개 = 항목 1개, 제목: `## [저자 et al., 연도] 축약명 — metric명 [상태]`
- **각 항목 맨 처음 4조건 yes/no 판정** → 이후 설명 (수식 체인+ⓛ라벨+UNet/Calculation →
  완화/사용 방식 → 우리 registry와 비교 1~2줄)
- **연도·venue는 제목과 상태 첫 줄에 모두 표기**
- 상태: `SUMMARIZED` / `PENDING` / `LOCAL` / `비메인트랙`·`초록 수준` 등 한정자 명시
- 탑티어 컨퍼 메인 트랙만 대표 등재 (예외는 한정자 명시)
- 사실만 기록, 추정은 "추정" 명시
