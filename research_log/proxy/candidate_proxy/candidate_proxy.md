# Candidate Proxy Registry — memo_proxy 후보군

후보 proxy의 정의·측정 경로·평가 상태를 관리하는 원천 파일.

**관리 규칙**
- 새 후보는 여기에 등록 → 상태 `PENDING` — **등록 형식은 `memo/.claude/skills/proxy-check/rules/registry-entry.md` 템플릿(① 기준)을 따른다**
- 평가는 `/proxy-check` 스킬이 수행 — 판정 후 이 파일의 상태·근거 보고서 링크를 갱신
- 상태: `PENDING` / `GOOD` / `CONDITIONAL` / `REJECT` (기준표는 SKILL.md 참조)

---

## ① memo_proxy — Tweedie domain gap  `[GOOD 2026-08-22]`

### 상태

- `GOOD` (2026-08-22) — 20셋 중 19셋 분리 (step 비율 평균 94%, 전형 비율 5~20x).
  예외 plot12(1.06x)는 약기억 프롬pt 가능성
- 근거: [proxy_report_memo_proxy_260822.md](evi/proxy_report_memo_proxy_260822.md)

### 수식

$$x_T \xrightarrow{\text{\textcircled{a}}} x_t \xrightarrow{\text{\textcircled{b}}} \epsilon_{\text{cfg}} \xrightarrow{\text{\textcircled{c}}} \hat{x}_0 \xrightarrow{\text{\textcircled{d}}} x_s \xrightarrow{\text{\textcircled{e}}} \epsilon_s \Rightarrow \mathcal{L} = \frac{\|\epsilon_{\text{ref}} - \epsilon_s\|^2}{D}, \quad \epsilon_{\text{ref}} \sim \mathcal{N}(0, I)$$

- ⓐ: **Multi UNet (Denoising Step)** — $x_T \in \mathbb{R}^{1\times4\times64\times64}$에서 DDIM denoising 연쇄로 $x_t \in \mathbb{R}^{1\times4\times64\times64}$ 도달 (init_steps만큼 UNet 반복 호출; gradient가 이 chain을 타고 $x_T$까지 전파 — AdjointDPM의 대상 구간)
- ⓑ: **UNet (Denoiser)** — $\epsilon_{\text{cfg}} \in \mathbb{R}^{1\times4\times64\times64}$, $\epsilon_{\text{cfg}}(x_t, t) = \epsilon_{\text{uc}} + w_{\text{cfg}} \cdot (\epsilon_c - \epsilon_{\text{uc}})$ ($w_{\text{cfg}}$ = cfg, staged CFG 시 `cfg_eff_at`)
- ⓒ: **Calculation (Tweedie's Formula)** — $\hat{x}_0 \in \mathbb{R}^{1\times4\times64\times64}$, $\hat{x}_0 = \left( x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_{\text{cfg}} \right) \big/ \sqrt{\bar{\alpha}_t}$
- ⓓ: **Calculation (DDIM FP)** — $x_s \in \mathbb{R}^{1\times4\times64\times64}$, $x_s = \sqrt{\bar{\alpha}_s}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_s}\,\epsilon$ (고정 중간 step $s$ = schedule 50% 지점, $\epsilon$ = 실제 주입 noise)
- ⓔ: **UNet (Denoiser)** — $\epsilon_s \in \mathbb{R}^{1\times4\times64\times64}$, $\epsilon_s = \epsilon_{\text{cfg}}(x_s, s)$ (forwarded point에서 재평가)
- (결과식) **Calculation (Memo_proxy)** — $\mathcal{L} \in \mathbb{R}$ (스칼라), $\mathcal{L} = \|\epsilon_{\text{ref}} - \epsilon_s\|^2 / D$, $\epsilon_{\text{ref}} \sim \mathcal{N}(0,I)$는 $x_T$와 독립, $D = 4 \times 64 \times 64$

**memo_proxy 의미**: tweedie domain gap

**memo_proxy 해석**: memo_proxy가 클수록 $x_t$에서 추정한 $\hat{x}_0$가 tweedie domain gap이 크다는 것을 의미한다.

### grad 업데이트 대상

$$\boldsymbol{x}_T \xrightarrow[\text{UNet}]{\text{a}} \boldsymbol{x}_t \xrightarrow[\text{UNet}]{\text{b}} \boldsymbol{\epsilon}_{\text{cfg}} \xrightarrow[\text{Calculation}]{\text{c}} \hat{\boldsymbol{x}}_0 \xrightarrow[\text{Calculation}]{\text{d}} \boldsymbol{x}_s \xrightarrow[\text{UNet}]{\text{e}} \boldsymbol{\epsilon}_s \Rightarrow \mathcal{L}$$

$$\boldsymbol{x}_T \leftarrow \boldsymbol{x}_T - \nabla_{\boldsymbol{x}_T} \mathcal{L}$$

### 1-1. [Twd_gap] grad 업데이트 대상 — optimization: latent x_t

**상태**: `GOOD` (2026-08-24) — 측정 체인 ①과 동일(loss·chain 공유) → ① 개형 근거 재사용
(19/20셋 분리, 비율 5~20x). grad 도달점(x_t)은 감지력과 무관.
근거: [proxy_report_memo_proxy_xt_260824.md](evi/proxy_report_memo_proxy_xt_260824.md)

**v4 (2026-08-24)** — **DDIM denoising 먼저**, 이후 Adam update. align 제거:

$$z_{t-1} \leftarrow \text{DDIM}(z_t) \qquad (\eta{=}0,\ \text{no grad})$$

$$z_{t-1} \leftarrow \text{Adam}\big(z_{t-1},\ \nabla_{z_{t-1}} \mathcal{L}\big) \qquad \big(\mathcal{L} = \text{memo\_head}(z_{t-1}) = \|\epsilon_{\text{ref}} - \epsilon_s\|^2 / D\big)$$

- loss 도 update 도 **z_{t-1} 기준** (DDIM 으로 미리 간 상태에서 head 계산 → Adam 갱신)
- update norm ≈ lr·√(B·D) (Adam) → 갱신된 z_{t-1} 에서 gap 전진
- ⓐ($x_T \to x_t$)는 여전히 no_grad — **중간 latent 영역에서만 최적화**, $x_T$ 불변
- align loss 제거 (memo loss만) · backward UNet 2회(head 만) · AdjointDPM 불필요

---

## ② compact spectrum energy — block_energy  `[REJECT 2026-08-22]`

### 상태

- `REJECT` (2026-08-22) — 3+1 개형 검증 11/20셋 분리 (비율 0.88~1.20x, 방향 역전 존재),
  집단 통계(AUC 0.480~0.610)와 수렴 — 저주파 집중은 일반 이미지 형성 현상
- 근거: [proxy_report_block_energy_260822.md](evi/proxy_report_block_energy_260822.md)
- 경향: text 곡선 20/20 하강 vs memo 6 상승 + 14 하강 (기울기 기준 분리 10/20) —
  방향성 있는 신호는 존재하나 검출기 기준 미달

### 수식

$$x_t \xrightarrow{\text{\textcircled{a}}} y \xrightarrow{\text{\textcircled{b}}} \{E_p\} \Rightarrow \mathcal{L} = \frac{\text{mean}_p(E_p)}{B}, \quad 1 = \text{white Gaussian}$$

- ⓐ: **Calculation (f_r_to_c)** — $x_t \in \mathbb{R}^{1\times4\times64\times64}$를 flatten($N = 4 \times 64 \times 64$) 후
  rfft 해 $y \in \mathbb{C}^{N/2}$ 산출, $y = f_{r \to c}(x_t)$ — rfft(ortho) 후
  $f[0] \leftarrow (f[0] + j\,f[-1])/\sqrt{2}$ 로 Hermitian 제거
- ⓑ: **Calculation (Block Energy)** — $\{E_p\} \in \mathbb{R}^{P}$ ($P = 512$ blocks),
  $E_p = \|y[pB:(p+1)B]\|^2$, block size $B = 16$
- (결과식) **Calculation (block_energy)** — $\mathcal{L} \in \mathbb{R}$ (스칼라),
  $\mathcal{L} = \text{mean}_p(E_p)/B$, **1 = white Gaussian 기준**.
  대역 집계: $\text{band}_{\text{low}} = \text{mean}(E_0..E_{49})/B$,
  $\text{band}_{\text{high}} = \text{mean}(E_{461}..E_{511})/B$ ($\text{freq\_ratio} = 0.1$)

**compact spectrum energy 의미**: latent의 compact spectral domain에서 block별 energy 분포

**compact spectrum energy 해석**: proxy가 1에서 벗어날수록 $x_t$의 스펙트럼이 white Gaussian에서 벗어나 특정 대역에 energy가 집중됨을 의미한다.

### grad 업데이트 대상

$$\boldsymbol{x}_t \xrightarrow[\text{Calculation}]{\text{a}} \boldsymbol{y} \xrightarrow[\text{Calculation}]{\text{b}} \{\boldsymbol{E}_p\} \Rightarrow \mathcal{L}$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_t - \nabla_{\boldsymbol{x}_t} \mathcal{L}$$

---

## ③ **[TEST]** eps_cfg energy  `[GOOD 2026-08-22]`

### 상태

- `GOOD` (2026-08-22) — 10/10셋 분리 (step 비율 평균 99%, 비율 1.23~2.05x)
- 근거: [proxy_report_eps_l2_260822.md](evi/proxy_report_eps_l2_260822.md)

### 수식

$$x_t \xrightarrow{\text{\textcircled{a}}} \epsilon_{\text{cfg}} \Rightarrow \mathcal{L} = \frac{\|\epsilon_{\text{cfg}}\|^2}{D}, \quad 1 = \text{white Gaussian}$$

- ⓐ: **UNet (Denoiser)** — $\epsilon_{\text{cfg}} \in \mathbb{R}^{1\times4\times64\times64}$,
  $\epsilon_{\text{cfg}}(x_t, t) = \epsilon_{\text{uc}} + w_{\text{cfg}} \cdot (\epsilon_c - \epsilon_{\text{uc}})$
  ($w_{\text{cfg}}$ = cfg, `--loss_cfg`로 독립 조절 가능)
- (결과식) **Calculation (eps_cfg energy)** — $\mathcal{L} \in \mathbb{R}$ (스칼라),
  $\mathcal{L} = \|\epsilon_{\text{cfg}}\|^2 / D$, $D = 4 \times 64 \times 64$
  (**1 = white Gaussian 기준**)

**eps_cfg energy 의미**: CFG 결합 noise prediction의 총 에너지 (주파수 분해 없이 raw)

**eps_cfg energy 해석**: eps_cfg energy가 클수록 $\epsilon_{\text{cfg}}$가 white Gaussian에서 크게 벗어난 구조적 신호를 담고 있음을 의미한다.

### grad 업데이트 대상

$$\boldsymbol{x}_t \xrightarrow[\text{UNet}]{\text{a}} \boldsymbol{\epsilon}_{\text{cfg}} \Rightarrow \mathcal{L}$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_t - \nabla_{\boldsymbol{x}_t} \mathcal{L}$$

---

## ④ eps spectrum energy — x_t → score → spectrum  `[GOOD 2026-08-24]`

### 상태

- `GOOD` (2026-08-24) — 5/5셋 전 step 구간 분리(100%) · 곡선-mean pairwise AUC 1.000
  (text n=15: 0.0529±0.0012 / memo n=5: 0.0748±0.0180, ratio 1.16~2.11x)
- 근거: [proxy_report_eps_spectral_260824.md](evi/proxy_report_eps_spectral_260824.md)
- 측정: **이미 구현** — `eps_trajectory.py --proxy_trend --signal eps` ·
  shell `shells/others/run_eps_spectrum_eps_energy.sh` (기본 SIGNAL=eps,
  scale = mean(E_p)/B, 1 = white Gaussian) ·
  기존 결과: `eps_trajectory/results_proxy_trend/signal=eps/ddim/CFG=7.5_NFE=50/seed=42/batch=5/plot_num=5`
- 최적화: `optimize_xt_spectral.py --loss_type eps` **이미 구현 존재**
  (`shells/eval_chen/run_memo_spectral_eps.sh`)

### 수식

$$x_t \xrightarrow{\text{\textcircled{a}}} \epsilon_{\text{cfg}} \xrightarrow{\text{\textcircled{b}}} y \xrightarrow{\text{\textcircled{c}}} \{E_p\} \Rightarrow \mathcal{L} = \frac{\text{mean}_p(E_p)}{B}, \quad 1 = \text{white Gaussian 기준}$$

- ⓐ: **UNet (Denoiser)** — $\epsilon_{\text{cfg}} \in \mathbb{R}^{1\times4\times64\times64}$,
  $\epsilon_{\text{cfg}}(x_t, t) = \epsilon_{\text{uc}} + w_{\text{cfg}} \cdot (\epsilon_c - \epsilon_{\text{uc}})$
- ⓑ: **Calculation (f_r_to_c)** — flatten 후 rfft(ortho) → $y \in \mathbb{C}^{N/2}$
  (② block_energy와 동일 변환, 입력만 x_t → ε_cfg)
- ⓒ: **Calculation (Block Energy)** — $E_p = \|y[pB:(p+1)B]\|^2$, block $B=16$, $P=512$
- (결과식) **Calculation** — $\mathcal{L} = \text{mean}_p(E_p)/B$ 스칼라.
  **1 = white Gaussian 기준**: ε_cfg가 표준정규(정상 denoiser)면 flat spectrum → 1,
  memorized 방향 성분이 끼면 특정 대역 $E_p$ 집중

**eps spectrum energy 의미**: score(ε) 공간의 주파수별 energy 분포

**eps spectrum energy 해석**: proxy가 1에서 벗어날수록 ε_cfg의 스펙트럼이 white Gaussian에서
벗어나 구조적(memorized) 신호가 특정 대역에 집중됨을 의미한다.

### grad 업데이트 대상

$$\boldsymbol{x}_t \xrightarrow[\text{UNet}]{\text{a}} \boldsymbol{\epsilon}_{\text{cfg}} \xrightarrow[\text{Calculation}]{\text{b}} \boldsymbol{y} \xrightarrow[\text{Calculation}]{\text{c}} \{\boldsymbol{E}_p\} \Rightarrow \mathcal{L}$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_t - \nabla_{\boldsymbol{x}_t} \mathcal{L}$$

- UNet forward가 grad chain에 포함 (x_t까지 backward — `optimize_xt_spectral.py` eps 갈래)

### 비교

② (x_t 직접 spectral, REJECT)와 ③ (ε raw energy, GOOD)의 **결합** — ③가 입증한 ε 공간의
판별력을 주파수 축으로 분해. ②가 REJECT된 이유(저주파 집중 = 일반 이미지 형성 현상)가
ε 공간에서는 다르게 나타나는지가 평가 포인트

---

## ⑤ eps_ref MSE — x_t score 와 ε_ref 의 MSE  `[GOOD 2026-08-24]`

### 상태

- `GOOD` (2026-08-24) — 20/20셋 방향 일치 (ratio 1.01~1.46x), 곡선-mean **AUC 0.9958**
  (text 1.846±0.021 / memo 2.156±0.261), step 분리 17/20셋 94~100%
- 측정: `eps_trajectory.py --measure eps_ref_mse` (신규 구현 — ①의 ε_ref·③의
  eps_l2_sig 재사용) · shell `shells/others/run_eps_trajectory_eps_ref_mse.sh`
- 근거: [proxy_report_eps_ref_mse_260824.md](evi/proxy_report_eps_ref_mse_260824.md)
- 최적화: 구현 — `run_ini_opti.py --opti_mode xt --xt_loss eps_ref_mse` (UNet 1회 ⓑ + 양수 MSE
  minimize) · shell `shells/eval_web/sd14/ours/run_memo_eps_ref_mse.sh`
- **부호 복원 (2026-09-05)**: 09-01~09-04 사이 코드에 무문서화 음수(`−‖·‖²` → minimize가 실제로는
  MSE maximize)가 유입돼 있었음 → 양수(minimize)로 복원. **09-01 이후 ⑤ 최적화 결과는 전부 음수
  방향 기준** — 양수 재실험 전까지 참조 금지

### 수식

$$x_t \xrightarrow{\text{\textcircled{a}}} \epsilon_{\text{cfg}} \Rightarrow \mathcal{L} = \frac{\|\epsilon_{\text{ref}} - \epsilon_{\text{cfg}}(x_t, t)\|^2}{D}, \quad \epsilon_{\text{ref}} \sim \mathcal{N}(0, I)$$

- ⓐ: **UNet (Denoiser)** — $\epsilon_{\text{cfg}} \in \mathbb{R}^{1\times4\times64\times64}$,
  $\epsilon_{\text{cfg}}(x_t, t) = \epsilon_{\text{uc}} + w_{\text{cfg}} \cdot (\epsilon_c - \epsilon_{\text{uc}})$
  (①의 ⓑ와 동일 — 단 **ⓒ~ⓔ 재포워드(x̂₀ → x_s → ε_s) 없이 x_t 의 score 를 직접 사용**)
- (결과식) **Calculation** — $\mathcal{L} \in \mathbb{R}$, $\epsilon_{\text{ref}} \sim \mathcal{N}(0,I)$는
  x_T 와 독립 (① 의 reference 재사용), $D = 4 \times 64 \times 64$

**eps_ref MSE 의미**: x_t 에서 예측한 CFG score 가 표준정규 reference 에서 벗어난 정도

**eps_ref MSE 해석**: proxy가 클수록 $\epsilon_{\text{cfg}}(x_t,t)$가 white Gaussian
($\epsilon_{\text{ref}}$)에서 멀어져 memorized 방향 성분이 강함을 의미한다.

### grad 업데이트 대상 (Twd_gap 체계 — grad 를 latent x_t 에 흘려줌)

$$z_{t-1} \leftarrow \text{DDIM}(z_t) \qquad (\eta{=}0,\ \text{no grad})$$

$$z_{t-1} \leftarrow z_{t-1} - \mathrm{lr}\cdot\hat{\nabla}_{z_t} \qquad \hat{\nabla}_{z_t} = \frac{\nabla_{z_t}\,\|\epsilon_{\text{ref}} - \epsilon_{\text{cfg}}(z_t, t)\|^2 / D}{\|\nabla_{z_t}\,\mathcal{L}\|}$$

- grad 는 `torch.autograd.grad` 로 별도 추출 후 **단위 norm 정규화**해 **latent x_t 체계에
  가미** (1-1. [Twd_gap] v2 와 동일 방식 — update 적용점 $z_{t-1}$)

### 비교

- **①의 단순화**: ⓒⓓⓔ(x̂₀ → x_s → ε_s 재포워드)를 생략 — UNet 1회로 측정 (①은 2회)
- **③과의 관계**: 전개하면 $\|\epsilon_{\text{ref}} - \epsilon_{\text{cfg}}\|^2 = \|\epsilon_{\text{ref}}\|^2 - 2\langle\epsilon_{\text{ref}}, \epsilon_{\text{cfg}}\rangle + \|\epsilon_{\text{cfg}}\|^2$ —
  기댓값에서 $\|\epsilon_{\text{ref}}\|^2 \approx D$·상관항 $\approx 0$ → **③(‖ε_cfg‖²)와
  기댓값 동치**. 차이는 **grad의 상관항** $-2J^\top\epsilon_{\text{ref}}$ — ε_ref 성분이
  gradient 에 들어가 노이즈(또는 탈출 신호)로 작용하는지가 ③ 대비 차별점

---

## ⑥ [On-main] Compression into on-manifold — seed 인력 + Tweedie gap 결합  `[NEW 2026-09-01]`

### 상태

- **동기 (관측된 현상)**: 한 프롬pt에 seed를 다르게 했을 때 — **general text prompt**의
  seed별 x_t latent는 **collapse**(뭉침)되어 있고, **memorized text prompt**는 오히려
  seed별 x_t가 **심하게 벌어져** 있음 (seed-collapse 현상, 2026-09-01 발견)
- **가설**: memorized seed latent의 과잉 spread를 억눌러 뭉치게 하면(compression)
  general의 on-manifold 거동으로 회복 → memorization 완화
- 구현: `run_ini_opti.py --on_main --w_twd {w}` (`opti_mode=xt`, `batch_txt=1`,
  `num_seeds≥2` 필수) — 출력 `{output_dir}/on_main/`

### 수식

$$\mathcal{L} = \underbrace{\sum_{i<j}\frac{\|x^{(i)}_t - x^{(j)}_t\|^2}{D}}_{\text{loss_1: seed 간 인력(뭉침)}} \;+\; w_{\text{twd}} \cdot \underbrace{\frac{1}{B}\sum_b \frac{\|\epsilon_{\text{ref}} - \epsilon_s(x_s^{(b)}, s)\|^2}{D}}_{\text{loss_2: Tweedie gap (① memo_proxy head)}}$$

- **loss_1**: batch 행 = 한 프롬pt의 seed들 ($S=$ `num_seeds`, 기본 4) — pairwise
  제곱거리 쌍합을 **per-pair ÷D** 정규화 (proxy 스케일 정합). 최소화 → seed 뭉침
  - ※ 사용자 원식은 $-\sum\|x^i_t - x^j_t\|^2$ — 그대로 최소화하면 seed가 **분리**되어
    "뭉치는(compression)" 목적과 반대 → 기본 구현은 **+Σ(인력)**, 원식 리터럴(−Σ, 분리)은
    `--on_main_spread` 플래그로 전환 가능
- **loss_2**: ① memo_proxy의 ⓑ~ⓔ head 그대로 (z_{t-1} → ε_cfg → x̂₀ → x_s → ε_s)

### grad 업데이트 대상 (Twd_gap 체계 승계)

$$z_{t-1} \leftarrow \text{DDIM}(z_t)\ (\eta{=}0,\ \text{no grad}), \qquad z_{t-1} \leftarrow \text{Adam}(z_{t-1},\ \nabla_{z_{t-1}}\mathcal{L})$$

- update 변수: 중간 latent x_t (z_{t-1} leaf) — 기존 optimize_xt와 동일 구조
- grad 경로: loss_1은 z_{t-1} 직접(UNet 무경유), loss_2는 ⓑ~ⓔ UNet head 경유
- 관측 로그: `[opt-xt] ... spread={Σ pairwise} twd={loss_2}` — update마다 spread 감소 확인

### 비교

- **①(Twd_gap)과의 관계**: ①은 프롬pt 단일 trajectory의 score-기반 완화 — ⑥은 여기에
  **seed 축(trajectoory 분산)의 기하학적 제약**을 추가. w_twd→0이면 순수 seed 인력,
  w_twd→∞이면 ①에 수렴
- **Han(init_score_noise)과의 관계**: Han은 ‖ε_text−ε_uncond‖을 x_T에서 억누름 —
  ⑥의 loss_1은 유사한 'guidance 억제'를 **latent 공간의 seed 기하**로 실현
- sweep 축: lr (완화 강도), w_twd (두 loss의 균형)

### 셸

`shells/eval_web/sd14/ours/run_memo_on_main.sh` — sdv1_500(Wen) 세트, num_seeds=4,
lr sweep (LR_LIST/W_TWD 환경변수 오버라이드), 결과 `workdir/.../on_main/`

---

## ⑦ spectral score low freq energy — eps low-band sum_energy  `[PENDING 2026-09-05]`

### 상태

- `PENDING` (2026-09-05) — 측정: `eps_trajectory.py --proxy_trend --signal eps --proxy low
  --band_agg sum --lh_ratio 0.1` (④의 측정 체계 재사용, 집계만 low-band `sum` —
  신규 `--band_agg`·`--lh_ratio` 인자)
- 측정 결과 (2026-09-05): `eps_trajectory/results_proxy_trend/signal=eps/ddim/CFG=7.5_NFE=50/seed=42/batch=5/lh=0.1_band_agg=sum/plot_num=5`
  — 5/5셋 분리(1.366~2.063x, 곡선-mean pairwise AUC 1.000, text 41.84±1.38 / memo 70.54±6.52),
  개형: text 52→4 하강 vs memo 58→101 유지·상승 (②의 저주파 실패 패턴 미재현). 판정 대기
- 최적화: **구현 완료** (2026-09-05) — `run_ini_opti.py --opti_mode xt --xt_loss spec_low
  --lh_ratio 0.1 --block_size 16` (v4 체계 — no grad DDIM z_t→z_{t-1} 후 z_{t-1} leaf에서
  ε_cfg → rfft → low-band sum, UNet 1회 backward, minimize; 헬퍼 `spectral_low_band_energy`) ·
  shell `shells/eval_web/sd14/ours/run_memo_spec_low.sh` (Wen sdv1_500, lr sweep
  LR_LIST 기본 0.11/0.15/0.20, trd_path `ours/spec_low/.../lh=r/bs=B/`)
- lr sweep 1차 (2026-09-05, `ours/spec_low/.../lh=0.1/bs=16/batch=1/seed=42/trd/100`):
  lr 0.11/0.15/0.20 → SSCD 0.726/0.706/0.638 (memorized 8/8/6) · CLIP 34.75/34.51/33.01 ·
  Pick 19.42/19.34/18.93 · ImgR 0.62/0.60/0.20 — lr↑에 SSCD·T2I 동반 하락 (완화-품질
  trade-off 단조 개형, ImgR은 0.20에서 급락). 최적화 개형: text 프롬pt loss 54→5~14 하강,
  memo 프롬pt는 lr=0.11에서 60~80 유지 → lr=0.20에서 49.7→20.6 하강 (측정 개형과 정합)

### 수식

$$x_t \xrightarrow{\text{\textcircled{a}}} \epsilon_{\text{cfg}} \xrightarrow{\text{\textcircled{b}}} y \xrightarrow{\text{\textcircled{c}}} \{E_p\}_{p<L_{\text{lh}}} \Rightarrow \mathcal{L} = \text{sum}_{p<L_{\text{lh}}}(E_p), \quad L_{\text{lh}} = \lfloor P \cdot r_{\text{lh}} \rfloor$$

- ⓐ: **UNet (Denoiser)** — $\epsilon_{\text{cfg}} \in \mathbb{R}^{1\times4\times64\times64}$,
  $\epsilon_{\text{cfg}}(x_t, t) = \epsilon_{\text{uc}} + w_{\text{cfg}} \cdot (\epsilon_c - \epsilon_{\text{uc}})$
  (④의 ⓐ와 동일 노드)
- ⓑ: **Calculation (f_r_to_c)** — flatten($N = 4 \times 64 \times 64$) 후 rfft(ortho) →
  $y \in \mathbb{C}^{N/2}$ (④의 ⓑ와 동일 변환)
- ⓒ: **Calculation (Low-freq Block Sum)** — $y$ 를 $\frac{N}{2}/B \times B$ 로 reshape
  ($P = \frac{N}{2B} = 512$ blocks, $B = 16$), 앞 $L_{\text{lh}} = \lfloor P \cdot r_{\text{lh}} \rfloor$
  개 block(저주파)만 선택해 energy 총합 — $E_p = \|y[pB:(p+1)B]\|^2$,
  $\mathcal{L} = \frac{1}{B}\sum_{p<L_{\text{lh}}} E_p$ ($r_{\text{lh}}$ = LH_ratio,
  코드 `--lh_ratio`, 미지정 시 `--freq_ratio` (0.1) 사용 → $L_{\text{lh}} = 51$;
  측정 체계(④ 재사용)가 block 당 $\div B$ 저장이므로 $1/B$ 포함 —
  리터럴 $\sum E_p$ 와는 $B$ 배 스케일 차이, 개형 동일)
- (결과식) **Calculation (sum_energy)** — $\mathcal{L} \in \mathbb{R}$ (스칼라).
  정규화 기준: white Gaussian 이면 block 당 $E_p \approx B$ →
  $\mathcal{L} \approx L_{\text{lh}} = 51$ ($r_{\text{lh}} = 0.1$).
  ④의 $\text{mean}_p(E_p)/B$ (1 = white Gaussian) 와 달리 **sum 은 선택 block 수 비례 스케일**

**spectral score low freq energy 의미**: score(ε_cfg) 스펙트럼의 저주파 block 대역 총 energy

**spectral score low freq energy 해석**: 값이 클수록 ε_cfg 의 저주파 대역에 구조적 성분이
집중됨을 의미한다.

### grad 업데이트 대상

$$\boldsymbol{x}_t \xrightarrow[\text{UNet}]{\text{a}} \boldsymbol{\epsilon}_{\text{cfg}} \xrightarrow[\text{Calculation}]{\text{b}} \boldsymbol{y} \xrightarrow[\text{Calculation}]{\text{c}} \{\boldsymbol{E}_p\}_{p<L_{\text{lh}}} \Rightarrow \mathcal{L}$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_t - \nabla_{\boldsymbol{x}_t} \mathcal{L}$$

### 비교

- **④의 low-band 리터럴 sum 버전** — 체인(ⓐ~ⓑ)·block 분해(ⓒ)까지 ④와 동일,
  집계만 $\text{mean}_p(E_p)/B$ → $\text{sum}_{p<L_{\text{lh}}}(E_p)$ 로 변경
  (sum = mean × $L_{\text{lh}}$ 이므로 곡선 개형은 ④ low 와 상동, 스케일만 절대값)
- **②의 REJECT 사유 재검토**: ②(x_t 직접 spectral)는 "저주파 집중 = 일반 이미지 형성 현상"으로
  반려 — ④가 입증한 ε 공간의 판별력이 **저주파 대역만 남겼을 때** 유지되는지가 평가 포인트
  (②의 실패 요인이 ε 공간 저주파에서 재현될 위험 존재)

---

## ⑧ [Anchor: Twd_gap] initial noise 앵커 — s_Δ + Tweedie gap 결합  `[NEW 2026-09-06]`

### 상태

- **개념**: initial noise x_T 자체에 두 신호를 동시에 억눌러 anchor 시킴 —
  (a) **s_Δ**: x_T에서의 unconditional/conditional score 차이 (guidance 세기),
  (b) **Tweedie gap**(① memo_proxy): x_T에서 시작하는 ⓑ~ⓔ 헤드 재평가
- **구조적 특징**: loss가 x_T에서 **직접 계산**되는 얕은 헤드(UNet 2회) — DDIM 연쇄(ⓐ)를
  거치지 않으므로 **AdjointDPM(optimize_xT_adj) 불필요**, grad가 x_T로 곧장 흐름.
  `num_steps` = x_T 갱신 횟수 (init/gap_steps 무의미)
- **검증**: proxy 검증 완료 (사용자 확인, 2026-09-06) — 최적화·sweep 단계부터 진행
- 구현: `run_ini_opti.py --anchor --w_twd {w}` (`opti_mode=xT`, minimization 전용) —
  출력 `{output_dir}/anchor/`, 로그 `[opt-anchor] s_delta= gap= |g1| |g2|`

### 수식

$$\mathcal{L} = \underbrace{\frac{1}{B}\sum_b \|\epsilon_{\text{uc}}(x_T^{(b)}, T_{\max}) - \epsilon_c(x_T^{(b)}, T_{\max})\|_2}_{\text{s_Δ: guidance 세기 (per-sample L2 norm)}} \;+\; w_{\text{twd}} \cdot \underbrace{\frac{1}{B}\sum_b \frac{\|\epsilon_{\text{noise}} - \epsilon_s(x_s^{(b)}, s)\|^2}{D}}_{\text{Tweedie gap (① memo_proxy 헤드)}}$$

- **s_Δ 체인**: $x_T \Rightarrow \epsilon_{\text{uc}}, \epsilon_c$ (UNet #1, $T_{\max}$) —
  CFG 결합 이전의 두 score 자체의 차이 norm. 최소화 → conditional guidance 약화
  (Han의 $\|\epsilon_{\text{text}} - \epsilon_{\text{uncond}}\|$ 신호와 동종이나
  최적화 변수·체인이 다름 — 아래 비교)
- **Tweedie gap 체인**: $x_T \xrightarrow{\text{Tweedie}} \hat{x}_{0|T}
  \xrightarrow{\text{forward noise}} x_s \Rightarrow \epsilon_s$ (UNet #2) —
  ①의 ⓑ~ⓔ 헤드를 x_t가 아닌 x_T에 적용:
  $\hat{x}_{0|T} = (x_T\sigma_{\text{init}} - \sqrt{1-\bar\alpha_T}\,\epsilon_{\text{ref}})/\sqrt{\bar\alpha_T}$,
  $x_s = \sqrt{\alpha_s}\,\hat{x}_{0|T} + \sqrt{1-\alpha_s}\,\epsilon_{\text{noise}}$,
  gap $= \|\epsilon_{\text{noise}} - \epsilon_s\|^2/D$
- $\epsilon_{\text{ref}}$ = UNet #1의 CFG 결합 ε (staged CFG `cfg_eff_at` step 0 적용),
  $\epsilon_{\text{noise}}$ = x_T와 독립인 fresh noise (noising + gap 기준, update 동안 고정)
- $T_{\max}$의 $\hat{x}_{0|T}$는 $1/\sqrt{\bar\alpha_T}$ 증폭(~12x)이 크나 체인 자체는 문제없음
  (Adam은 방향 기반 — 스케일 증폭에 강인)

### grad 업데이트 대상

$$x_T \leftarrow \text{Adam}(x_T,\ \nabla_{x_T}\mathcal{L}), \qquad \nabla_{x_T}\mathcal{L} = \nabla s_\Delta + w_{\text{twd}}\nabla(\text{gap})$$

- update 변수: **initial noise x_T** (단일 leaf, num_steps회 갱신) — xt 모드(중간 latent)와
  xT-adjoint 모드(trajectory 전파) 모두와 다른 제3경로: **x_T 직접 + 얕은 헤드**
- grad 경로: s_Δ는 UNet #1 경유, gap은 UNet #1(ε_ref) + #2(ε_s) 경유 — 역방향 UNet 2회
- 관측 로그: `[opt-anchor i/N] s_delta= gap= |g1| |g2|` (⑥ on_main의 분리 로깅 패턴 승계)

### 비교

- **Han(init_score_noise)과의 관계**: Han도 ‖ε_text−ε_uncond‖을 x_T에서 최적화하나
  (per_sample, target_loss 도달까지 lr=0.01×1000 iter) — ⑧은 이 신호(s_Δ)에
  **Tweedie gap(①)을 결합**해 score-기반 완화 방향을 동시에 제약. 가중치 w_twd로
  균형 조절 (w_twd→0이면 순수 s_Δ(Han 신호 방향), →∞이면 ①-x_T에 수렴)
- **①(Twd_gap, opti_mode=xt)과의 관계**: 같은 gap 헤드를 **최적화 지점만 다르게** 적용 —
  ①은 중간 x_t를 순차 갱신(gap 전진 구조), ⑧은 x_T 하나를 처음부터 끝까지 갱신
  (trajectory 불변, adjoint 불필요)
- sweep 축: lr (완화 강도), w_twd (s_Δ vs gap 균형)

### 셸

`shells/eval_web/sd14/ours/run_memo_twd_anchor.sh` — sdv1_500(Wen) 세트, ipp=1,
lr sweep (LR_LIST/W_TWD 환경변수 오버라이드), 결과 `workdir/.../ours/twd_anchor/`

---

## ⑨ [Btw_anchor: batch-traction] initial noise 배치 인력 — s_Δ + batch 인력 결합  `[NEW 2026-09-07]`

### 상태

- **동기 (관측된 현상)**: 실험 결과에서 **batch 끼리의 std가 memorization에서 압도적으로
  컸음** (2026-09-07) — 같은 프롬pt의 batch 샘플(서로 다른 초기 seed의 x_T)이 memorized
  프롬pt에서 크게 벌어져 있음. ⑥의 seed-collapse 관측(2026-09-01)과 동일 계열 —
  batch 축의 잉여 spread를 인력으로 누르면 memorization 완화 기대
- **개념**: initial noise x_T를 batch 단위로 최적화 — (a) **s_Δ**(⑧과 동일, guidance
  세기), (b) **batch pairwise 인력** $\sum\|x_i - x_j\|$ — 두 신호를 동시에 억눌러 anchor
- **구조적 특징**: ⑧ anchor 대비 헤드가 더 얕음 — **역방향 UNet 1회** (⑧은 2회),
  btw 항은 x_T 직접(UNet 무경유, ⑥ loss_1과 동종). AdjointDPM 불필요,
  `num_steps` = x_T 갱신 횟수
- 구현: `run_ini_opti.py --btw_anchor --w_btw {w}` (`opti_mode=xT`, minimization 전용,
  `num_seeds≥2` 필수, **batch_txt 호환**) — 출력 `{output_dir}/btw_anchor/`,
  로그 `[opt-btw i/N] s_delta= btw= |g1| |g2|`

### 수식

$$\mathcal{L} = \underbrace{\frac{1}{B}\sum_b \|\epsilon_{\text{uc}}(x_T^{(b)}, T_{\max}) - \epsilon_c(x_T^{(b)}, T_{\max})\|_2}_{\text{s_Δ: guidance 세기 (⑧ 과 동일)}} \;+\; w_{\text{btw}} \cdot \underbrace{\sum_{i<j} \|x_T^{(i)} - x_T^{(j)}\|_2}_{\text{batch pairwise 인력 (norm)}}$$

- **s_Δ 체인**: $x_T \Rightarrow \epsilon_{\text{uc}}, \epsilon_c$ (UNet 1회, $T_{\max}$) —
  ⑧의 첫 항 그대로. 최소화 → conditional guidance 약화
- **btw 체인**: $x_T$ 직접 (Calculation, UNet 무경유) — batch 행 = prompt-major
  $[p_0$의 $S$개, $p_1$의 $S$개, $\ldots]$ ($B = S \times$ batch_txt), **프롬pt 블록
  안쪽 쌍만 인력** (⑥의 블록 pairwise 패턴 승계 — batch_txt>1에서 프롬pt 간 오염 없음),
  pairwise **L2 norm** 리터럴 sum (⑥ loss_1의 squared ÷D 와 달리 non-squared norm —
  s_Δ도 norm이므로 $O(\sqrt{D})$로 스케일 정합). 최소화 → batch 샘플 인력(뭉침) → 잉여 std 억제
- pair 수 = 프롬pt당 $\binom{S}{2}$ × batch_txt ($S$ = num_seeds)

### grad 업데이트 대상

$$x_T \leftarrow \text{Adam}(x_T,\ \nabla_{x_T}\mathcal{L}), \qquad \nabla_{x_T}\mathcal{L} = \nabla s_\Delta + w_{\text{btw}}\nabla(\text{btw})$$

- update 변수: **initial noise x_T** ($B$개 동시 갱신, num_steps회) — ⑧ anchor 체계 승계
- grad 경로: s_Δ는 UNet 1회 경유, btw는 x_T 직접 — **역방향 UNet 1회** (⑧의 2회보다 얕음)
- 관측 로그: `[opt-btw i/N] s_delta= btw= |g1| |g2|` (⑧ 분리 로깅 패턴 승계)

### 비교

- **⑧(Anchor: Twd_gap)과의 관계**: s_Δ 첫 항 공유, 둘째 항이 Tweedie gap(score 기반) →
  batch pairwise 인력(기하 기반)으로 교체 — w_btw→0이면 순수 s_Δ(Han 신호 방향),
  →∞이면 순수 batch collapse
- **⑥(On-main)과의 관계**: ⑥ loss_1 = Σ‖x^i_t−x^j_t‖²/D (중간 x_t, seed 축, squared) —
  ⑨는 같은 인력 아이디어를 **x_T 레벨·batch 축·norm**으로 실현. ⑥는 gap(loss_2)과
  결합, ⑨는 s_Δ와 결합
- sweep 축: lr (완화 강도), w_btw (s_Δ vs 인력 균형), B=num_seeds (인력 쌍 수)

### 셸

`shells/eval_web/sd14/ours/run_memo_btw_anchor.sh` — sdv1_500(Wen) 세트,
lr sweep (LR_LIST/W_BTW 환경변수 오버라이드), 결과 `workdir/.../ours/btw_anchor/`

---

## ⑩ spectral score high freq energy — eps high-band sum_energy  `[NEW 2026-09-07]`

### 상태

- **개념**: ⑦의 **고주파 대칭 변형** — 체인(ⓐ~ⓒ)·block 분해까지 ⑦과 완전 동일,
  선택 대역만 low(앞 $L_{\text{lh}}$개 block) → **high(뒤 $L_{\text{lh}}$개 block)**
- **동기**: ⑦이 입증한 ε 공간 대역별 판별력의 축 완성 — ②의 반려 사유("저주파 집중 =
  일반 이미지 형성 현상")가 고주파 대역에서 어떻게 나타나는지, 그리고 memorized의
  세밀 구조(고주파) 성분이 저주파와 다른 완화 거동을 보이는지가 평가 포인트
- 최적화: **구현 완료** (2026-09-07) — `run_ini_opti.py --opti_mode xt --xt_loss spec_high
  --lh_ratio 0.1 --block_size 16` (⑦ v4 체계 그대로 — no grad DDIM z_t→z_{t-1} 후
  z_{t-1} leaf에서 ε_cfg → rfft → **high-band** sum, UNet 1회 backward, minimize;
  헬퍼 `spectral_low_band_energy(..., band="high")`로 일반화)
- 셸: `shells/eval_web/sd14/ours/run_memo_spec_high.sh` (Wen sdv1_500, lr sweep,
  **NUM_EVAL 3차 프로토콜** — N=프롬pt 종류 수), trd_path `ours/spec_high/.../lh=r/bs=B/`

### 수식

$$x_t \xrightarrow{\text{\textcircled{a}}} \epsilon_{\text{cfg}} \xrightarrow{\text{\textcircled{b}}} y \xrightarrow{\text{\textcircled{c}}} \{E_p\}_{p\geq P-L_{\text{lh}}} \Rightarrow \mathcal{L} = \frac{1}{B}\sum_{p\geq P-L_{\text{lh}}} E_p, \quad L_{\text{lh}} = \lfloor P \cdot r_{\text{lh}} \rfloor$$

- ⓐ: **UNet (Denoiser)** — $\epsilon_{\text{cfg}}(x_t, t) = \epsilon_{\text{uc}} + w_{\text{cfg}} \cdot (\epsilon_c - \epsilon_{\text{uc}})$ (⑦의 ⓐ와 동일 노드)
- ⓑ: **Calculation (f_r_to_c)** — flatten($N = 4 \times 64 \times 64$) 후 rfft(ortho) → $y \in \mathbb{C}^{N/2}$ (⑦과 동일 변환)
- ⓒ: **Calculation (High-freq Block Sum)** — $P = \frac{N}{2B} = 512$ blocks 중
  **뒤 $L_{\text{lh}} = \lfloor P \cdot r_{\text{lh}} \rfloor$개 block(고주파)** 만 선택해
  $E_p$ 총합 ÷ $B$ (⑦과 집계 방식 동일, 선택 슬라이스만 `[:, :L_lh]` → `[:, P-L_lh:]`)
- (결과식) white Gaussian 기준 $\mathcal{L} \approx L_{\text{lh}}$ ($r_{\text{lh}} = 0.1$ → ≈51) — ⑦과 동일 스케일 기준

### grad 업데이트 대상

$$\boldsymbol{x}_t \xrightarrow[\text{UNet}]{\text{a}} \boldsymbol{\epsilon}_{\text{cfg}} \xrightarrow[\text{Calculation}]{\text{b}} \boldsymbol{y} \xrightarrow[\text{Calculation}]{\text{c}} \{\boldsymbol{E}_p\}_{p\geq P-L_{\text{lh}}} \Rightarrow \mathcal{L}$$

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_t - \nabla_{\boldsymbol{x}_t} \mathcal{L}$$

- ⑦ v4 체계 승계: no grad DDIM $z_t \to z_{t-1}$ 후 $z_{t-1}$ leaf에서 계산 — UNet 1회 backward

### 비교

- **⑦의 대칭**: 체인·block 분해·÷B 정규화·WG 기준(≈$L_{\text{lh}}$)까지 동일,
  대역 선택만 반대 — low/high 결과 대조로 memorized 신호의 주파수 로컬리티 판별 가능
- **②의 반려 사유 재검토**: ②(x_t 직접 spectral)는 저주파 집중이 일반 형성 현상이라
  반려 — 고주파 대역은 일반 형성에서 상대적으로 평평하므로 ②의 실패 패턴이 재현될
  위험이 낮을 것으로 예상하나, 반대로 판별 신호 자체가 약할 수 있음(양방향 리스크)

### 셸

`shells/eval_web/sd14/ours/run_memo_spec_high.sh` — sdv1_500(Wen) 세트,
lr sweep (LR_LIST/LH_RATIO 환경변수 오버라이드), 결과 `workdir/.../ours/spec_high/`
