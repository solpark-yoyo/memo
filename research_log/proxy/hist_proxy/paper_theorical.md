# Paper Theoretical Survey — 이론적 배경 중심

inference-time memorization proxy 논문들의 **이론적 배경** 정리 (조사일: 2026-08-22).
논문 1개 = 항목 1개. 구성: 핵심 가정 → proxy 유도 → 완화-이론 연결 → 이론적 지위 → 우리 체계와의 관계.
이론적 지위 등급: **명시적 수학 프레임 / 반(서술적) / 경험적 관찰**.

---

## [Wen et al., 2025] 지배적 text guidance — ICLR 2025

1. **핵심 가정**: memorized prompt에서는 text 조건이 denoising을 지배 — seed와 무관하게 항상
   memorized 해로 trajectory를 끌어당김. 정상은 text가 온화하게 조향, 출력이 seed 따라 다양
2. **proxy 유도**: 지배적 조향 = cond/uncond score 차이가 큼 → $\|\epsilon_{cond}-\epsilon_{uc}\|_2$
   를 trajectory 평균 → 조향 강도의 적분
3. **완화 연결**: t=0에서 prompt embedding을 최적화해 조향 강도를 낮춤 → 이후 step 인력도
   간접 약화 (생성은 표준 CFG, CLIP 정합성 유지 주장)
4. **이론적 지위**: **반(서술적)** — 관점은 정합하나 유도·증명 없음. 검증은 경험적(AUC 0.994)
5. **우리와의 관계**: ①(tweedie gap)·③(‖ε_cfg‖²/D)와 같은 "text 인력 세기" family일 가능성(추정)

---

## [Han et al., 2025] attraction basin — NeurIPS 2025 (LOCAL: baselines/init_score_noise/)

1. **핵심 가정**: memorization = CFG가 만드는 **attraction basin**에 x_T가 갇히는 현상.
   basin 내 x_T는 어떤 seed든 memorized output으로 수렴
2. **proxy 유도**: basin 소속도의 proxy로 x_T 단발의 gap norm ‖ε_text−ε_uncond‖₂
3. **완화 연결**: x_T를 AdamW로 최적화해 gap을 target_loss 이하로 낮춤 = basin 밖 이동
   (early escape) — "CFG가 아니라 x_T 위치가 원인"이라는 가정이 완화에 직결
4. **이론적 지위**: **서술적 가설** — basin 프레임은 명확하나 형태·크기의 수학적 정의 없음
5. **우리와의 관계**: 우리 run_ini_opti와 동일 완화 구조(x_T 직접 최적화) — 차이는 loss.
   "어느 loss가 basin 탈출을 더 잘 유도하나"가 정면 비교 질문

---

## [Chen et al., 2025] magnitude 트리거 + BE local — CVPR 2025 (LOCAL: be_attention_mask.py 등)

1. **핵심 가정**: (global) Wen의 gap 관점 계승 — 첫 step 1회로 트리거 가능.
   (local) memorization은 이미지 전역이 아닌 **국소 영역**에서 발생 — 마지막 step cross-attention의
   EOT 토큰 집중 patch("bright ending")가 복제 local region
2. **proxy 유도**: global $m_{T-1}=\|\epsilon_{cond}-\epsilon_{uc}\|_2$ (단발) ·
   local $m'_t=\|(\epsilon_{cond}-\epsilon_{uc})\circ m\|_2/\text{mean}(m)$ (BE mask m)
3. **완화 연결**: PRSS(LLM prompt 대안 + CFG uncond anchor 교체) — 감지 신호와 완화가
   **느슨하게 연결**된 공학적 대응 (최적화 아님)
4. **이론적 지위**: global은 반이론 · BE는 **경험적 관찰** · 완화는 규칙 기반
5. **우리와의 관계**: eval_chen·GT의 원천. **local 관점은 우리에 없는 차원** — ①·③(전역 신호)의
   보완 축 후보

---

## [Asthana & Belagiannis, 2026] score 정렬 이론 — ICLR 2026

1. **핵심 가정** (조사 대상 중 최명시): guidance 벡터 $s^\Delta=s_\theta(x,t,c)-s_\theta(x,t)$와
   uncond score의 기하학적 관계로 memorization 특징화 —
   t≈0(저노즈 anisotropic): memorized면 $s^\Delta$가 uncond score와 강히 **정렬**(같은 mode로 이끎) ·
   t≈T(고노즈 isotropic): $\|s^\Delta\|$가 큼(특정 해로 급격히 sharp)
2. **proxy 유도**: 두 성분 직접 측정 — $M=\gamma_1\cos(s^\Delta,s_\theta)+\gamma_2\|s^\Delta\|$
3. **완화 연결**: M을 loss로 **prompt embedding c 최적화** (x_T 고정) — 정렬·유인 동시 감소
4. **이론적 지위**: **명시적 수학 프레임** (Eq. 14-15). 임계는 logistic regression(경험적 요소 병존)
5. **우리와의 관계**: cos(정렬) 항은 우리에 없는 신호 — ①의 거리 기반과 **직교 보완** 가능성.
   ①·③·M으로 "x_T 단발 관측량 3종 비교" 확장 여지

---

## [Ren et al., 2024] cross-attention entropy — ECCV 2024 (MemAttn)

1. **핵심 가정**: memorized 생성은 cross-attention이 trigger 토큰들로 분산 → entropy 높음.
   정상은 BOS/시작 토큰 집중 → entropy 낮음
2. **proxy 유도**: 토큰 평균 attention의 entropy $E_t=\sum_i -\bar a_i\log\bar a_i$ — 후반 1/5 평균
   (+summary 토큰 보정)
3. **완화 연결**: attention logit 편집(첫 토큰 증폭 + summary masking) — 규칙 기반, 최적화 없음
4. **이론적 지위**: **경험적 관찰** — 상관 발견, 수학 프레임 없음
5. **우리와의 관계**: ε 공간이 아닌 **attention 공간** — 독립 관측. 비판적 참고: attention 기반
   완화는 기억을 지우지 않고 출력만 숨길 수 있다는 후속 지적 존재

---

## [Zhao et al., 2026] CA spike z-score — ICML 2026 (GUARD, 순위 외 참고)

1. **핵심 가정**: memorization 유발 토큰의 attention에 spike — **verbatim**: EOT 집중,
   **template**: 분산 (두 형태 구분이 고정 규칙 완화 실패의 설명)
2. **proxy 유도**: 토큰별 최대 attention mass의 z-score — 매 step 재계산(dynamic)
3. **완화 연결**: spike 토큰 logit 감쇠 cond를 쓰는 contrastive guidance — 규칙 기반
4. **이론적 지위**: **경험적** — spike 관찰 + 형태 구분
5. **우리와의 관계**: attention 계열 정밀화(Ren의 한계 개선) — ε 계열과 직접 비교 제한
