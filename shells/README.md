# Memorization 실험 파이프라인 (eval_wen / eval_chen)

`ours(init_opti)` 의 memorization 완화 성능을 baseline(Han et al., DDIM 등)과 비교하는 실험 파이프라인.
**eval_wen**(Wen et al. memorized SD)과 **eval_chen**(Chen/Webster 일반 SD1.4)이 동일 구조(대칭).

## 환경
- conda env: `div_DM` (실행 전 `conda activate div_DM`)
- 실행 위치: `ori_memo/`
- 공통 스크립트(루트): `run_ini_opti.py`, `collect_trd.py`, `plot_trd.py`, `plot_compare.py`, `collect_tradeoff.py`, `plot_tradeoff.py`, `merge_benchmark.py`, `compute_sscd_gt.py`, `compute_t2i_metrics.py`

## Chen vs Wen 설정 차이
| 항목 | eval_wen | eval_chen |
|---|---|---|
| model | `sd14_memor_LAION2B_40k` (memorized) | `stable-diffusion-v1-4` (비-memorized) |
| prompt | `new_memorized_text_prompt.txt` | `cvpr2025_memo_prompt.txt` |
| GT ref | `datasets/wen2024_memorized` | `datasets/cvpr2025_webster_gt` |
| base_dir | `workdir/memorization/sd14_memor_LAION2B_40k` | `workdir/memorization/sd14_base` |

## 실험 파이프라인 (공통 단계)

```
code checking    → run_memo_opti_nsteps.sh      (nsteps × lr sweep, inference+eval)
trend 확인       → run_memo_opti_cfgsr.sh       (cfgsr(deferring ratio) × lr)
                 → run_memo_opti_thrsh.sh       (threshold(eps) × lr)
모델별 curve     → trd/run_ours_lr.sh           (ours lr sweep → trd)
                 → trd/run_Han_lr.sh            (Han lr sweep → trd)
최종 SOTA 비교   → trd/run_compare.sh           (Han vs ours overlay)
```

### 각 단계 상세

| 단계 | shell | 역할 | sweep |
|---|---|---|---|
| **code checking** | `run_memo_opti_nsteps.sh` | ours inference + eval, nsteps 영향 확인 | nsteps × lr |
| **trend (cfgsr)** | `run_memo_opti_cfgsr.sh` | staged CFG(deferring ratio) 효과 | cfgsr × lr |
| **trend (threshold)** | `run_memo_opti_thrsh.sh` | memo_proxy threshold(eps) 효과 | memo_threshold × lr |
| **ours curve** | `trd/run_ours_lr.sh` | ours lr sweep → trade-off curve | lr |
| **Han curve** | `trd/run_Han_lr.sh` | Han(init_score_noise) lr sweep → curve | lr |
| **SOTA 비교** | `trd/run_compare.sh` | Han vs ours overlay (T2I vs SSCD) | — |

## 실행 순서 (권장)

```bash
cd ori_memo
conda activate div_DM

# 1) code checking — nsteps 가 미치는 영향 (유의미한 nsteps 선정)
bash shells/eval_wen/run_memo_opti_nsteps.sh    # (또는 eval_chen/)

# 2) trend — cfgsr(deferring ratio) / threshold(eps) 효과
bash shells/eval_wen/run_memo_opti_cfgsr.sh
bash shells/eval_wen/run_memo_opti_thrsh.sh

# 3) 모델별 trade-off curve (eval 끝난 path 에서)
bash shells/eval_wen/trd/run_ours_lr.sh
bash shells/eval_wen/trd/run_Han_lr.sh

# 4) 최종 SOTA 비교 (Han vs ours overlay)
HAN_PATH=workdir/.../init_score_noise/NFE=50/per_sample/CFG=7.5 \
OURS_PATH=workdir/.../init_opti/.../gap=3 \
    bash shells/eval_wen/trd/run_compare.sh
```
→ `{base}/trd/compare_{clipscore,pickscore,imagereward}.png`

## 산출물 구조

### 실험 결과 (각 조합)
```
{base}/init_opti/CFG=7.5_NFE=50/cfgsr=X/base_s_ratio=0.5_lambda_align=0.00/
       memoloss=Y/[memothr=Z/]init=5/nsteps=N/gap=3/lr=L/seed=42/batch=5/
├── result/img_*.png            # init_opti inference 결과
├── eval/
│   ├── chen_sscd_gt_metrics.csv
│   ├── chen_t2i_metrics.csv
│   └── total_metrics.csv       # SSCD + CLIP/Pick/IR mean·std (merge_benchmark)
└── record/img_*/               # x_T opti 과정 기록 (grid, loss, memo_proxy plot)
```

### trd (trade-off)
```
{lr부모}/trd/
├── csv/
│   ├── total_log.csv           # label, CLIP/Pick/ImgR/SSCD mean·std·max (전부)
│   └── total_metrics.csv       # mean 만
└── plot/
    ├── tradeoff_clipscore.png  # x: CLIPScore, y: SSCD
    ├── tradeoff_pickscore.png
    └── tradeoff_imagereward.png
```

### 최종 비교
```
{base}/trd/
├── compare_clipscore.png       # Han vs ours overlay
├── compare_pickscore.png
└── compare_imagereward.png
```

## 핵심 인자
- `init_steps`: DDIM 시작 step (기본 5)
- `num_opt_steps` (nsteps): optimization update 횟수 — 완화 깊이의 핵심 factor
- `gap_steps`: update 간격
- `cfg_start_ratio` (cfgsr): staged CFG deferring ratio (0.0=항상 CFG, 0.3=초반 30% null)
- `type_memo_loss`: `minimization`(전체 완화) | `threshold`(proxy > memo_threshold 만 gradient)
- `memo_threshold` (eps): threshold 모드의 proxy 임계값 (`||ε-ε_s||²/D`)
- `lr`: optimization learning rate

## memo_proxy 정의
`memo_proxy(t) = ||ε_ref - ε_s(x_s(t), s)||² / D`
- `ε_ref` = 최적화된 x_T (DDIM 입력 noise)
- `x_s = √ᾱ_s·x̂₀|t + √(1-ᾱ_s)·ε_ref` (s = 고정 mid-noise)
- denoising 진행 → `x̂₀|t → x₀` → `ε_s → ε_ref` → proxy ↓ = memorization 완화
