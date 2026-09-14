# Proxy Evaluation Journal

## 2026-08-24 — ⑤ eps_ref MSE — proxy-check
- 판정: **GOOD** — 20/20셋 방향 일치 (ratio 1.01~1.46x), 곡선-mean AUC 0.9958
  (text 1.846±0.021 / memo 2.156±0.261). step 분리 17/20셋 94~100% (예외: plot06 60%·plot02 76%)
- 구현: `--measure eps_ref_mse` 신규 (①의 ε_ref·③의 eps_l2_sig 재사용, alias 경로로
  저장·플롯 자동) — 측정 1회 실행 (results_eps_ref_mse_trajectory, 20 plot)
- 근거: `evi/proxy_report_eps_ref_mse_260824.md` (figure: `figs/eps_ref_mse_curves_260824.png`)
- 상태: NEXT = ③과 곡선 직접 비교(상관항 기여 분리) · Twd_gap 체계 loss 이식 검토
- 비고: twd_gap v2 sweep이 lr=0.12 도중 [killed] — 재개 대기 (lr 0.12 잔여 3프롬pt + 0.16 + trd)

## 2026-08-24 — ④ eps spectrum energy — proxy-check
- 판정: **GOOD** — 5/5셋 memo>text_max 100% step 분리, 곡선-mean pairwise AUC 1.000
  (text 0.0529±0.0012 / memo 0.0748±0.0180, ratio 1.16~2.11x). 기존 결과 재사용
  (results_proxy_trend/signal=eps, 재측정 없음)
- 근거: `evi/proxy_report_eps_spectral_260824.md` (figure: `figs/eps_spectral_curves_260824.png`)
- 교훈: 구현 여부는 eps_trajectory.py 전체 갈래 기준 — measure choices만 보고 미구현
  오판 (실제론 proxy_trend signal=eps로 완전 구현+데이터 존재). SKILL.md 3번 유의사항 추가
- 상태: NEXT = ④ 완화 최적화 연결 — optimize_xt_spectral eps 갈래는 존재하나 update
  구조가 Twd_gap v2와 상이 → 이식 여부 사용자 결정. wen2024 교차검증·low/high 대역 비교 후보

## 2026-08-24 — 1-1 twd_gap 첫 lr sweep — 완화 효과 미확보·중단 (사용자)
- config: cfgsr=0.10(+cond)/init=2/nsteps=9/gap=4, lr {0.00,0.04,0.08,0.12}, chen 프로토콜
  (webster GT — SSCD는 매칭 6프롬pt만 유효), batch=5, seed=42
- 결과: lr 0→0.08 SSCD **상승**(0.115→0.120) — minimize가 완화로 연결 안 됨. lr 0.12에서만
  SSCD 0.0985 감소하나 ImgR −0.31·CLIP −1.4 붕괴 → trade-off 우측 하단 미확보
- 결론: 현재 하이퍼파라미터에서 1-1 완화 효과 미확보 — 튜닝 중단 (사용자 판단)
- 재개 시 가설: init_steps 상향(2→10/15) — 초반 몰림 해소·중간 구간 최적화, 또는 threshold 모드
- 산출: `workdir/memorization/sd14_base/twd_gap/CFG=7.5_NFE=50/cfgsr=0.10/init=2/nsteps=9/gap=4/trd/`
  (csv + plot 3종: CLIP/Pick/ImgR × SSCD)

## 2026-08-24 — 1-1 구현 완료·smoke 통과 (tune 1단계)
- 구현: `run_ini_opti.py` — `optimize_xt()`(순차 전진 x_t 최적화, ⓑ~ⓔ terminal head) +
  `ddim_inference_with_proxy_from()`(resume inference) + `--opti_mode {xT,xt}` 분기 (기존 xT 무손상)
- smoke 통과 (init=2/nsteps=3/gap=1/lr=0.04, batch=5, prompt 5): `|g_xt|` 0.018~0.082 비-0,
  `|dxt|`≈11.4, resume@step 6 정상, result 25장 + record/comp 생성. 6.17s/sample, peak 15.65GB
- 출력 루트: `workdir/memorization/sd14_base/twd_gap/CFG=7.5_NFE=50/{config}/seed/batch/`
- 상태: NEXT = tune 2단계 — 기존 인벤토리 감사 → lr 우선 sweep list 제시 (eval_chen/wen shell 패턴)
- 구현 명세: `memo/.claude/rules/impl-xt-memo-proxy.md`

## 2026-08-24 — 1-1 memo_proxy (optimization: latent x_t) — proxy-check
- 판정: **GOOD** — 측정 체인 ①과 동일(loss·chain 공유), 기존 `results_eps_trajectory`
  재사용 (재측정 없음). ① 개형 근거 승계: 19/20셋 분리, step 비율 평균 94%, 비율 5~20x
- 근거: `evi/proxy_report_memo_proxy_xt_260824.md` (figure: `figs/memo_proxy_curves_260822.png` 재사용)
- 상태: NEXT = 1-1 완화 경로는 /tune 영역 — grad가 중간 x_t에만 흘러야 함
  (`optimize_xT_adj`는 x_T update이므로 중간 x_t update 구현 스펙 필요) → 완료
