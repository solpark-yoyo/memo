# baselines workdir 표준 폴더 구조

## 목적

`workdir/memorization/sd14_base/baselines/` 의 **표준 저장 레이아웃** 정의.

**모든 baseline shell(`shells/eval_web/sd14/baselines/`)과 eval 스크립트는 이 경로를
기준으로 조립·참조해야 한다.** 새 실험 추가·eval 재실행·trd 수집 시 아래 갈래를 그대로
따르고, 다른 모양의 경로를 새로 만들지 않는다.

기준 시점: 2026-09-05 (4차 레이아웃 + wen·han batch=4 500 prompt 풀셋 정렬 완료)

---

## 공통 템플릿

```text
baselines/{method}/{model_tag}/CFG=…_NFE=50/{고정 h.p.}/batch=B/seed=S/
├─ {knob=X}/
│   ├─ result/img_XXXX_YY.png        # 이미지 (XXXX=prompt idx, YY=sample idx)
│   │                                 # ※ han(init_score_noise)만 예외 — 아래 참조
│   └─ eval/<N>/chen_sscd_gt_metrics.csv
│              chen_t2i_metrics.csv
│              total_metrics.csv
└─ trd/<N>/
    ├─ csv/total_metrics.csv          # label,CLIP_mean,Pick_mean,ImgR_mean,SSCD_mean
    ├─ csv/total_log.csv
    └─ plot/tradeoff_{clipscore,pickscore,imagereward}.png
```

규칙:

1. **고정 하이퍼파라미터(thres/ms/it/lr·oi)는 바깥, sweep knob(lr/c1/tl)은
   `batch=B/seed=S/` 안쪽** — 같은 batch/seed 안에서 knob끼리만 비교
2. **trd는 knob 폴더의 부모** = `batch=B/seed=S/trd/<N>` (knob 폴더와 같은 깊이의 형제)
3. **eval·trd 폴더명 `<N>` = clamp 후 실제 평가 장수**
   (`NUM_EVAL` → `num_eval_prompts = N/ipp` (≤ num_samples) → `N = num_eval_prompts × ipp`)
   예: 요청 500, ipp=4 → 125 prompt → `eval/500`, `trd/500`
4. 이미지는 `result/` 안에 저장 (han 제외) — eval은 `--gen_dir <knob>/result`,
   csv 출력은 `--output_csv <knob>/eval/<N>/…`
5. ddim(완화 없음 기준선)은 knob·trd 없음 — 단일 점 eval만 존재

---

## method별 갈래 예시 (전부 실존 경로, `baselines/` 기준 상대)

### jeon — knob=lr

```text
jeon/stable-diffusion-v1-4/CFG=7.5_NFE=50/thres=8.2/ms=10/batch=1/seed=42/
├─ lr=0.01/result/img_0000_00.png
├─ lr=0.01/eval/sdv1_500/total_metrics.csv        (+ chen_sscd_gt, chen_t2i)
└─ trd/sdv1_500/csv/total_metrics.csv             (+ plot/tradeoff_clipscore.png)
```

※ 디스크는 아직 batch=1 소규모(옛 run). shell은 이미 표준(batch=ipp) 조립이므로
신규 실행 시 자동으로 표준 경로에 쌓임.

### ren — knob=c1

```text
ren/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=4/seed=42/
├─ c1=1.25/result/img_0000_00.png
├─ c1=1.25/eval/100/total_metrics.csv
└─ trd/100/csv/total_metrics.csv
```

### wen — knob=tl(λ), 고정 it=10

```text
wen/stable-diffusion-v1-4/CFG=7.5_NFE=50/it=10/batch=4/seed=42/
├─ tl=0.7/result/img_0000_00.png
├─ tl=0.7/eval/500/total_metrics.csv              (500p 풀 평가, 2026-09-05)
└─ trd/500/csv/total_metrics.csv
```

### han(init_score_noise) — knob=tl, 고정 lr·oi / **이미지 result/ 없이 직접**

```text
init_score_noise/NFE=50/per_sample/CFG=7.0/lr=0.01/oi=1000/batch=4/seed=42/
├─ tl=1.1/img_0000_00.png                         (+ config.json)
├─ tl=1.1/eval/500/total_metrics.csv
└─ trd/500/csv/total_metrics.csv
```

※ generator `setup_output_path` 사양: 이미지를 knob 폴더에 직접 저장
(2026-09-05 이 순서로 4차 정렬 완료). eval은 `--gen_dir <knob>` 그 자체.

### ddim — knob 없음(기준선), trd 없음(단일 점)

```text
ddim/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=1/seed=42/
├─ result/img_0000_00.png
└─ eval/sdv1_500/total_metrics.csv                (+ comp/, record/)
```

### 병합 — merge_trd_methods.py 출력

```text
trd/sdv1_500/
├─ csv/total_metrics.csv                           # method,lr,clipscore,pickscore,imagereward,sscd
└─ plot/tradeoff_{clipscore,pickscore,imagereward}.png
```

---

## shell·eval 참조 규칙 (필수)

**이 문서의 경로가 단일 source of truth다.**

1. baseline shell(`shells/eval_web/sd14/baselines/run_memo_chen_*.sh`)은
   `gen_dir`·`eval_dir`·`trd_dir` 를 **반드시 공통 템플릿대로** 조립한다.
2. eval 스크립트(`compute_sscd_gt.py`·`compute_t2i_metrics`·`merge_benchmark.py`)는
   shell이 조립한 `gen_dir`(이미지)·`eval_dir`(csv 출력) 경로를 그대로 받아 쓴다 —
   **eval 쪽에서 경로를 재조립하지 않는다.**
3. `collect_trd.py`·`plot_trd.py`도 shell이 지정한 표준 `--trd_dir` 만 사용한다.
4. 새 baseline 추가 시: `memo/.claude/rules/code.md`의 shell 패턴 + 이 문서의
   갈래를 함께 따른다.

### 정합 현황 (2026-09-05 확인)

| shell | gen_dir 조립 | trd_dir 조합 | 상태 |
|---|---|---|---|
| `run_memo_chen_jeon.sh` | `thres/ms/batch/seed/lr` | `thres/ms/batch/seed/trd/<N>` | ✅ 표준 |
| `run_memo_chen_ren.sh` | `batch/seed/c1` | `batch/seed/trd/<N>` | ✅ 표준 |
| `run_memo_chen_wen.sh` | `it=10/batch/seed/tl` | `it/batch/seed/trd/<N>` | ✅ 표준 |
| `run_memo_chen_score.sh` (han) | `per_sample/CFG/lr/oi/batch/seed/tl` | `lr/oi/batch/seed/trd/<N>` | ✅ 표준 |
| `run_memo_chen_ddim.sh` | `batch/seed` (+result) | trd 없음 | ✅ 표준 |

---

## 부가 — 디스크 레거시 (표준 아님, 신규 생성 금지)

- **`batch=1/` 갈래** (jeon 전체, ren·wen·han 각각) — 옛 소규모 run. 참조용으로만
  남기고 신규 실행 결과를 여기에 쌓지 않는다.
- **eval 폴더명 = 세트명**(`cvpr_2025`/`sdv1_500`) — 옛 방식. 신규는 전부 장수명
  (`<N>`). collect_trd.py `_knob_dir`/`_sb_dir` 후보 탐색이 양쪽을 모두 지원.
- `_bridge/` — 프롬pt 세트 브리지 csv(`cvpr2025_webster_bridge.csv`,
  `sdv1_500_rev_bridge.csv`).
- `others/` — 구 레이아웃 아카이브. **검사·수정 금지** (사용자 지시).
