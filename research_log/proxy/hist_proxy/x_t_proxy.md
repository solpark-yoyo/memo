# x_t Proxy Registry — 논문별 memorization proxy 요약

**중간 latent x_t로 유도되는** memorization proxy를 제시한 논문을 registry-entry 형식으로 축적
(x_t proxy survey — 스크리닝 조건: `memo/.claude/skills/proxy-his/rules/x_t_proxy.md`).
**①은 중간 x_t 기준** — x_T·첫 step(t≈T) 측정은 No (유의사항: rules 참조).

---

## 판정표 — ① x_t proxy 제시 = Yes 논문만 (2026-08-24 재판정)

| 논문 | ② opt→중간 x_t | ③ 연관 논리 | ④ 메인트랙 | ① proxy (짧은 설명) |
|---|---|---|---|---|
| **Huang et al. 2026** (KDD 2026) | **Yes** — z_t 직접 보정(projection) | Yes — stability region 이탈 | Yes | latent update norm $\|\delta_t\|$ z-score |
| Wen et al. 2025 (ICLR 2025) | No — update 대상: prompt embedding | Yes — 경험적(AUC 0.994) | Yes | trajectory 전체 $\|\epsilon_{cond}-\epsilon_{uc}\|$ 집계 |
| Zhang et al. 2026 (arXiv) | No — update 대상: representation | Yes — spiky vs balanced 이론 | No — arXiv-only | 중간 t bottleneck feature std, prompt-free |
| Kim et al. 2025 (arXiv) | No — 완화 미제시 (update 대상 없음) | Yes — overestimation 이론 | No — arXiv-only | x_t 분해 계수 $w_0^{(t)}/w_T^{(t)}$ |

**표 외 (① No — x_T 기반 측정)**: Han 2025(x_T 단발), Jeon 2025(첫 step t≈T·SAIL), Asthana
2026(x_T 4회), Chen BE-PRSS 2025(첫 step 트리거), Ma InvMM 2025·ICCV(inversion x_T 분포
추정 — 아래 항목) — 상세는 아래 표 외 항목

**표 외 (① No/△ — 기타 공간)**: Jiang IIP 2025·ICLR(image 재생성 유사도), Denoising-Centric
2026·arXiv(x₀ 데이터 공간·training-time 완화), CAPTAIN 2025(△ 감지 proxy 미명시 — ②만 Yes),
Kim Repulsive 2026(△·비메인트랙), Dodson 2026 ICML(△·training-time), AMG CVPR 2024(✗ ε
guidance), CFG-attraction-basin CVPR 2025(✗), Ren 2024 ECCV(△ attention)

**시사점**: 중간 x_t 자체를 측정 대상으로 하는 proxy는 Huang(latent 동역학)·Kim(latent 분해)
단 2개 + trajectory 집계(Wen)·feature(Zhang) — literature 대부분은 x_T 기반 감지이거나
prompt/ε-guidance 완화. ①+② 모두 충족은 Huang뿐.

---

## 1. [Huang et al., 2026] Broken Memories — latent update norm (stability region)  `[SUMMARIZED]`

### 스크리닝 (yes/no)

1. **x_t proxy 제시: Yes** — latent update norm $\|\delta_t\|$ — **trajectory 전체의 중간 latent
   공간 자체** (ε·score 아님)
2. **optimization → 중간 x_t: Yes** — 완화가 **중간 latent $z_t$를 직접 보정** (gradient 아닌
   projection/rescaling 방식 — 별도 표기)
3. **memorization 연관 논리: Yes** — empirical stability region — 정상 생성의 $\|\delta_t\|$
   통계영역 $[\mu_t-\gamma\sigma_t,\ \mu_t+\gamma\sigma_t]$ 이탈 = memorization 진입
   (수치해석 안정성 분석에서 영감, 경험적)
4. **탑티어 컨퍼 메인 트랙 게재: Yes** — KDD 2026 (arXiv 2605.22050)

### 설명

$$x_T \xrightarrow{\text{ⓐ}} x_t \xrightarrow{\text{ⓑ}} \delta_t \Rightarrow s_t = \frac{\big|\|\delta_t\| - \mu_t\big|}{\sigma_t},\quad S_{\text{mem}} = \max_t s_t$$

- ⓐ: **Multi UNet (Denoising Step)** — 표준 생성 trajectory (추가 UNet 호출 0회)
- ⓑ: **Calculation** — $\delta_t := z_t - z_{t+1}$ (PNDM AB4 update의 latent 변화량) z-score.
  방향: **$S_{\text{mem}}$ 크면 memorized** · 검증 AUC >0.999
- **완화 (adaptive 2단계)**: mild/strong 심도별로 ① $\hat z_0^{(t)}$ norm을 $\mu_0^{(t)}$로
  rescaling ② $\|z_t\|$·$\|\delta_t\|$를 stability region 내로 clamping — 재생성 없이 on-the-fly,
  0.01s/image, 완화 후 memorization rate 0.0%
- **아키텍처**: UNet (SD 1.4/1.5/2.1, PNDM/DDIM)

### 비교

$\|\delta_t\|$는 우리 ①·③(ε 공간)·②(spectrum)와 달리 **trajectory 동역학** 신호 —
eps_trajectory에 latent step 변화량 컬럼 추가만으로 재현 가능(추정). 중간 $z_t$ 보정은
`optimize_xt_spectral.py`와 동일 update 변수 — 직접 비교 대상

---

## 2. [Wen et al., 2025] memorization detection — text-conditional noise norm  `[SUMMARIZED]`

### 스크리닝 (yes/no)

1. **x_t proxy 제시: Yes** — **trajectory 전체 $x_t$에서 집계** (중간 x_t 포함.
   측정 공간: **ε** — 별도 표기)
2. **optimization → 중간 x_t: No** — update 대상: **prompt embedding e** (x_T 고정)
3. **memorization 연관 논리: Yes** — "지배적 text guidance" 관점 (경험적 검증 중심, 수학적
   프레임은 약함)
4. **탑티어 컨퍼 메인 트랙 게재: Yes** — ICLR 2025 (arXiv 2407.21720; repo bibtex "ICLR 2024"는 저자 오기)

### 설명

$$d = \frac{1}{T}\sum_t \|\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset)\|_2$$

- ⓐⓑ: **Multi UNet** 표준 CFG DDIM trajectory — 감지 추가 UNet 호출 **0회** (CFG 부산물)
- 방향: **d 크면 memorized** · 검증: SDv1.4 memorized 500 vs normal 500 — AUC 0.994 /
  첫 step만으로 AUC 0.960·0.2초 (첫 step 변형은 x_T 기반 — 본 metric은 trajectory 집계)
- **완화**: $e \leftarrow e - \nabla_e \|\epsilon_\theta(x_{t_0}, e) - \epsilon_\theta(x_{t_0}, e_\emptyset)\|_2$
  (77×768, AdamW lr 0.05, ≤10 iter)
- 평가 프로토콜만 Parksol 사용: `shells/eval_wen/`

### 비교

우리 ③(‖ε_cfg‖²/D)·①(‖ε_ref−ε_s‖²/D)와 같은 ε 공간, 참조가 $e_\emptyset$ 차이라는 점이 다름 —
개형 비교 가치 (eps_trajectory에 cond−uncond norm 추가만 필요, 추정)

---

## 3. [Zhang et al., 2026] Balanced Representation — 중간 feature std  `[SUMMARIZED · 비메인트랙]`

### 스크리닝 (yes/no)

1. **x_t proxy 제시: Yes** — **중간 timestep**(SD1.4 t=50)의 $x_t$를 UNet에 통과시켜 얻은
   bottleneck representation $h_\theta(x_t,t,c)$의 std 측정 (측정 공간: **feature** — 별도 표기)
2. **optimization → 중간 x_t: No** — update 대상: **representation** ($h_\theta + a\cdot v$
   steering) — 중간 x_t 아님
3. **memorization 연관 논리: Yes** — memorized = **spiky(고std)** / generalized =
   **balanced(저std)** representation — 2-layer ReLU DAE 이론 유도 + Jacobian SVD로 실모델 검증
4. **탑티어 컨퍼 메인 트랙 게재: No** — arXiv-only (2512.20963, UMich·Georgia Tech / Qing Qu 그룹)

### 설명

$$d_{\text{std}} = \mathrm{std}_{\text{dim}}\big[h_\theta(x_t, t^*, c)\big]\quad(t^*=\text{중간 step 고정})$$

- **UNet (Denoiser)** 1회 forward에서 bottleneck feature 추출 · **Calculation** — 차원별 std.
  방향: **std 크면 memorized**
- 검증: AUROC SD1.4-LAION **0.987**(0.067s) · DiT-ImageNet **0.995**(0.015s) ·
  CIFAR10-EDM **0.998**(0.020s) — landscape 기반(Jeon 계열) 대비 **8~36× 빠름**, **prompt-free**
- **steering (완화)**: $f_{\text{steered}} = g_\theta(h_\theta(x_t,t,c) + a\cdot v)$ — reference
  평균 표현 $v$로 편집. generalized는 부드러운 monotonic 반응, memorized는 threshold성 취약 반응

### 비교

내부 표현의 "spikiness" 관점은 우리 jepa_scoring(JEPA Jacobian-SVD)와 관측 축 유사 — 측정은
UNet 내부 훅 필요 (eps_trajectory 직접 확장 불가, 추정). prompt-free·저비용(1회 forward)이
벤치마크 관점에서 강점

---

## 4. [Kim et al., 2025] How Diffusion Models Memorize — x_t decomposition (overestimation)  `[SUMMARIZED · 비메인트랙]`

### 스크리닝 (yes/no)

1. **x_t proxy 제시: Yes** — **중간 x_t 자체를 최소제곱 분해** $x_t = w_0^{(t)} x + w_T^{(t)} x_T$,
   계수의 스케줄 이탈 측정 (측정 공간: **latent 자체**)
2. **optimization → 중간 x_t: No** — 신규 완화 미제시 (기존 Jain 2025 CFG 지연 완화의 작동
   원리를 이론적으로 설명하는 데 그침)
3. **memorization 연관 논리: Yes** — memorization = **early overestimation**: memorized sample
   $x$의 기여 $w_0^{(t)}$가 이론 스케줄 $\sqrt{\bar\alpha_t}$를 초과해 증가 + 초기 noise $x_T$
   기여 $w_T^{(t)}$가 $\sqrt{1-\bar\alpha_t}$보다 조기 억제 (이론 + 실측 Pearson 0.92)
4. **탑티어 컨퍼 메인 트랙 게재: No** — arXiv-only (2509.25705, Yonsei, 2025.09)

### 설명

$$D_1 = \sum_t\big(\mathbb{E}[w_0^{(t)}] - \sqrt{\bar\alpha_t}\big),\qquad D_2 = -\sum_t\big(\mathbb{E}[w_T^{(t)}] - \sqrt{1-\bar\alpha_t}\big)$$

- **Multi UNet (Denoising Step)** trajectory 전체에서 per-step least-squares 분해 (다중 seed의
  $\mathbb{E}[w]$ 필요 — 측정 비용 ≈ 생성 비용) · **Calculation**
- 방향: $D_1$·$D_2$ 클수록 memorized · SSCD(외부 GT)와 Pearson 0.92/0.92/0.70 상관
  (판정 라벨 자체는 SSCD ≥ 0.75 — decomposition metric은 상관 분석으로 제시)
- 실험: SD1.4/2.1/RealisticVision, DDIM T=50, g=7.5, Webster 436 prompts

### 비교

$w_T^{(t)}$(초기 noise 잔여 기여)는 우리 ② compact spectrum과 같은 **"x_t 직접 분석" family** —
spectral(주파수 축) 대비 signal-basis($x, x_T$ 혼합 계수 축)가 다름. trajectory least-squares라
prompt별 측정 비용이 생성과 동일

---

## 표 외 (① No — x_T 기반 측정)

### [Han et al., 2025] init_score_noise — guidance direction norm  `[SUMMARIZED · LOCAL]`

1. **x_t proxy 제시: No** — **x_T(initial noise)에서 단발 측정** (중간 x_t 아님) (측정 공간: ε)
2. **optimization → 중간 x_t: No** — update 대상: **x_T(initial noise)**
3. **연관 논리: Yes** — attraction basin 가설
4. **메인트랙: Yes** — NeurIPS 2025 (arXiv 2510.08625)

$$\mathcal{L} = \frac{1}{B}\sum_b \|\epsilon_\theta(x_T, t_1, c_b) - \epsilon_\theta(x_T, t_1, \emptyset)\|_2$$

- $[x_T;x_T]$ 2B 배치 1회 · 완화 $x_T \leftarrow x_T - \nabla_{x_T}\mathcal{L}$ (AdamW lr 0.01,
  ≤1000 iter, `target_loss` break) → 표준 DDIM
- **LOCAL**: `baselines/init_score_noise/` · UNet (SDv1.4, bf16)
- 비교: run_ini_opti와 동일 update 변수 — SOTA 정면 baseline

### [Jeon et al., 2025] Sharpness (SAIL) — score Hessian metric  `[SUMMARIZED]`

1. **x_t proxy 제시: No** — **첫 step $t{=}T{-}1$ = x_T 단계** 측정 (중간 x_t 아님)
   (측정 공간: score)
2. **optimization → 중간 x_t: No** — update 대상: **x_T(initial noise)** (SAIL)
3. **연관 논리: Yes** — sharp log-density (이론 유도)
4. **메인트랙: Yes** — ICML 2025 Spotlight (arXiv 2412.04140)

$$d_{\text{sharp}} = \|H_\theta^\Delta(x_t)\, s_\theta^\Delta(x_t)\|^2,\quad s^\Delta = s(x,c)-s(x)$$

- cond/uncond 2회 + autodiff HVP · 첫 step만으로 AUC 0.998 · 완화 SAIL(x_T 최적화, Adam 0.05)
- 비교: score 곡률 신호 — ①·③(norm)과 직교 · 상세는 `score_proxy.md` 항목 1

### [Asthana & Belagiannis, 2026] — guidance score 정렬+norm  `[SUMMARIZED]`

1. **x_t proxy 제시: No** — **x_T에서 4회 측정** (중간 x_t 아님) (측정 공간: score)
2. **optimization → 중간 x_t: No** — update 대상: **prompt embedding c**
3. **연관 논리: Yes** — 이론 명시 (aniso 정렬 t≈0 + iso norm t≈T)
4. **메인트랙: Yes** — ICLR 2026 (arXiv 2601.20642)

$$M(x_T, c) = \gamma_1 \cos\big(s^\Delta|_{t\approx0},\, s|_{t\approx0}\big) + \gamma_2 \|s^\Delta|_{t\approx T}\|$$

- 동일 $x_T$ 2 timestep × cond/uncond = 4회 · logistic regressor 판정
- 비교: cos(방향) 항은 우리에 없는 차원 · 상세는 `score_proxy.md` 항목 2

### [Chen et al., 2025] BE-PRSS — magnitude 트리거 + BE attention  `[SUMMARIZED · LOCAL]`

1. **x_t proxy 제시: No** — **첫 step 1회 트리거** (중간 x_t 아님)
   (측정 공간: ε + attention(BE))
2. **optimization → 중간 x_t: No** — 규칙 기반(최적화 아님): LLM 대안 prompt 탐색 + uncond anchor 교체
3. **연관 논리: Yes** — Wen 계승 + BE local 확장
4. **메인트랙: Yes** — CVPR 2025 (arXiv 2504.18032)

$$m_t = \|\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset)\|_2 \cdot 1_{m_{T-1}>\lambda}$$

- BE mask: 마지막 step cross-attention EOT 집중 patch Otsu 이진화 · masked magnitude Eq. 8
- **LOCAL**: `be_attention_mask.py`, `run_be_inference.py`, `compute_local_mem.py`,
  `compute_sscd_gt.py` + eval_chen 프로토콜 · UNet (SDv1.4)
- 비교: eval_chen·GT 원천 — BE mask(local memorization)는 우리에 없는 차원

### [Ma et al., 2025] InvMM — inversion 기반 memorization 측정  `[SUMMARIZED · 표 외]`

1. **x_t proxy 제시: No** — inversion으로 추정하는 "sensitive latent noise"가 **x_T(initial
   noise) 분포** (중간 x_t 아님) · 최종 판정은 재생성 성공(image)
2. **optimization → 중간 x_t: No** — 완화 미제시 (update 대상 없음 — 감지·auditing 전용)
3. **memorization 연관 논리: Yes** — membership(in/out 판정)과 memorization(재현 가능성)을
   구분하는 공격자 관점 정량화 (이론적)
4. **탑티어 컨퍼 메인 트랙 게재: Yes** — ICCV 2025 (arXiv 2405.05846, 코드 Maryeon/InvMM)

- sensitive noise 추정: normality(정상 노이즈 분포 정합)와 sensitivity(재생 성공)의 균형을
  맞추는 adaptive 알고리즘 · 검증 SSCD · DDPM(CIFAR-10) / LDM(CelebAHQ·FFHQ) / SD1.4·2.1·3.5
- 비교: "어떤 x_T가 memorized 재생을 trigger하는가"를 뒤집어 보는 관점 — x_T 최적화(Han)의
  쌍대 구조. 후속으로 Jiang IIP(ICLR 2025)의 inversion-perturbation 감지 계열

---

## 표 외 (① No/△ — 기타 공간)

### [Zhang et al., 2025] CAPTAIN — latent semantic feature injection  `[SUMMARIZED · 비메인트랙]`

1. **x_t proxy 제시: No (△)** — memorized region localize하나 감지 proxy 수식 미명시 (초록 기준)
2. **optimization → 중간 x_t: Yes** — denoising 중 **latent 직접 수정** (feature injection)
3. **연관 논리: △** — 초기 denoising 복제 경향 관찰 (경험적)
4. **메인트랙: No** — arXiv-only (2512.10655, KAUST) — ② 충족·최신성으로 조사 포함

$$x_T \xrightarrow{\text{ⓐ 주파수 교란}} x_t \xrightarrow{\text{ⓑ region localize}} m_t \Rightarrow x_t' = x_t \oplus \phi_{\text{ref}} \circ m_t$$

- ⓐ frequency-based noise initialization · ⓑ non-memorized reference feature $\phi_{\text{ref}}$
  injection (외부 reference 필요)
- 비교: reference 의존 — 우리 ①·③(reference-free)와 전제 상이. frequency init만 ② compact
  spectrum과 주파수 도메인 공유

---

## 관리 규칙

- 논문 1개 = 항목 1개, 제목: `## [저자 et al., 연도] 축약명 — proxy명 [상태]`
- **각 항목 맨 처음 4조건 yes/no 판정** → 이후 설명 (수식 체인+ⓛ라벨+UNet/Calculation →
  완화/사용 방식 → 우리 registry와 비교 1~2줄)
- **판정표에는 ①이 Yes인 논문만 등록** — **①은 중간 x_t 기준** (x_T·첫 step 측정은 No,
  유의사항: `skills/proxy-his/rules/x_t_proxy.md`)
- **연도·venue는 제목과 상태 첫 줄에 모두 표기**
- 상태: `SUMMARIZED` / `PENDING` / `LOCAL` / `비메인트랙`·`초록 수준` 등 한정자 명시
- 탑티어 컨퍼 메인 트랙만 등재 (repo 사용 방법은 예외 + 한정자 명시)
- 사실만 기록, 추정은 "추정" 명시
