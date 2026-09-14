# twd_gap config 상태

루트: `workdir/memorization/sd14_base/twd_gap/CFG=7.5_NFE=50/` · 디버깅 절차: `rules/debug/twd_gap.md`

## config 상태 table (inference)

**Trade-off Curve 개형 옳은가 (yes/no)** — SSCD와 T2I metric들이 trade-off 관계를 띄는가:
- **yes**: 완화 강도(lr)↑에 따라 **SSCD 감소**(memorization 완화)와 **T2I metric 감소**(품질
  저하)가 함께 매끄럽게 진행 — 완화-품질 trade-off 곡선이 형성됨
- **no**: SSCD가 완화 방향과 역행(완화 강도↑에 SSCD 상승·반전)하거나, SSCD 감소가 품질
  붕괴(ImgR 음수 등)로만 달성되는 경우 — 곡선으로서 형태가 없음

| config | 개형 옳은가 |
|---|---|
| `cfgsr=0.05/init=2/nsteps=9/gap=4/lr={0.00,0.04,0.08,0.12}` (4 config, 50장 each + eval) | **no** — lr 0.04→0.08 SSCD 재상승, 0.12 급락은 ImgR 붕괴 동반 |
| `cfgsr=0.10/init=2/nsteps=9/gap=4/lr={0.00,0.04,0.08,0.12}` (4 config, 50장 each + eval) | **no** — lr 0→0.08 SSCD 단조 상승(0.115→0.120), 0.12 급락은 ImgR 붕괴 동반 |
| `cfgsr=0.15/init=2/nsteps=9/gap=4/lr={0.00,0.04,0.08,0.12}` (4 config, 50장 each + eval) | **no** — lr 0.12에서 SSCD 반전 상승(비단조) |

## 부가 table (inference 외)

| config | 종류 | 내용 |
|---|---|---|
| `init=2/nsteps=3/gap=1/lr=0.04/seed=42/batch=5` (coco_v2 5프롬pt) | smoke | 동작 검증 — `\|g_xt\|` 비-0·resume 정상 (2026-08-24). 단 `\|dxt\| ≈ 11.44` (Adam 1-step 상한 lr·√D=5.12 초과) |
| `trd/` | trade-off curve | `csv/cfgsr_sweep_metrics.csv` + `tradeoff_{clipscore,pickscore,imagereward}_cfgsr.png` (cfgsr 3곡선 오버레이) — 형태 미형성 |

## 비고

- chen 프로토콜: cvpr2025 memo 10프롬pt, webster GT (SSCD는 매칭 6프롬pt만 유효), seed=42, batch=5
