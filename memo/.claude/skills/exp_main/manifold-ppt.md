# manifold-ppt — Memorization Trajectory Manifold Protocol

## 개요

diffusion memorization을 latent manifold 관점에서 분석하는 4가지 실험을 자동화합니다.

**주요 실험:**
1. **off-manifold**: on-manifold reference vs memo/general 거리 (8가지 거리 방법)
2. **fixation**: seed/prompt 고정 시 latent magnitude std
3. **trajstd**: 소스별 프롬프트 간 latent std 곡선 비교
4. **seedstd**: 프롬프트별 seed 축 std → prompt 평균 곡선 (권장)

---

## seedstd 워크플로우 (권장)

### Step 1: 첫 번째 실행 (latent pool 생성)

```bash
cd memo/ori_memo
conda activate div_DM
bash shells/exp_main/manifold/run_seedstd.sh
```

**수행:**
- DDIM rollout (prompt × seed 조합) → latent pool 저장
- 최종 이미지 저장 (img_XXXX_YY.png)
- Seed 축 std 곡선 생성 + 시각화

**소요 시간:** 5~10분 (GPU RTX 4090)

**출력:**
```
workdir/exp_main/manifold/latent_std/save/sd14_base/ddim/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=5/seed=42/
├── general/{gen_eval,for_diverse,drawbench}/
│   ├── results/img_0000_00.png ~ img_XXXX_YY.png
│   └── record/latents/latent.npz (캐시)
└── memo/webster/
    ├── results/img_0000_00.png ~ img_XXXX_YY.png
    └── record/latents/latent.npz
```

### Step 2: 재실행 (캐시 로드 - 즉시 완료)

동일 명령 재실행:
```bash
bash shells/exp_main/manifold/run_seedstd.sh
```

→ 모든 latent.npz 존재 확인 → 모델 로드 생략 (GPU 0GB) → 30초 완료

### Step 3: 결과 분석

**곡선 파일:**
```
workdir/exp_main/manifold/stable-diffusion-v1-4/seedstd_general_memo/
├── csv/std_curve.csv          # step별 std 값
└── plot/std_curve.png         # 오버레이 곡선
```

**해석:**
- y축: mean(std) of ‖z_t‖ across seeds (prompt 평균)
- x축: Time (t=50 노이즈 → t=0 이미지)
- **Memo 곡선이 general보다 아래** → memorization 신호 (seed 민감도 낮음)

---

## 파라미터 커스터마이징

### 환경 변수로 override

```bash
# 모델 변경
MODEL=stable-diffusion-v1-5 bash shells/exp_main/manifold/run_seedstd.sh

# CFG/NFE 변경
CFG=15.0 NFE=100 bash shells/exp_main/manifold/run_seedstd.sh

# GPU 지정
GPU=2 bash shells/exp_main/manifold/run_seedstd.sh

# 조합
CFG=10.0 BATCH=3 N_PROMPTS=50 bash shells/exp_main/manifold/run_seedstd.sh
```

### Python 직접 호출

```bash
python manifold.py --seedstd general_memo \
    --sources "general:gen_eval=examples/assets/gen_eval.txt" \
               "general:drawbench=examples/assets/drawbench.txt" \
               "memo:webster=examples/assets/sdv1_500_mem.txt" \
    --model_key ckpt/stable-diffusion-v1-5 \
    --cfg 10.0 --NFE 100 --batch 3 --n_prompts 50 --seed 42 --device cuda:0
```

---

## off-manifold 실험 (거리 기반 분석)

### 실행

```bash
python manifold.py --manifold centroid_l2 \
    --ref_prompt examples/assets/coco_v2.txt \
    --general_prompt examples/assets/geneval_prompts.txt \
    --memo_prompt examples/assets/cvpr2025_memo_prompt.txt \
    --model_key ckpt/stable-diffusion-v1-4
```

### 거리 방법 비교

```bash
# 빠른 방법 (초 단위)
for method in centroid_l2 cosine hausdorff; do
  python manifold.py --manifold $method ...
done

# 정교한 방법 (분 단위)
for method in knn mmd emd spectral; do
  python manifold.py --manifold $method ...
done
```

### 결과 해석

- **d(ref, memo) > d(ref, general)** → off-manifold 지지 (memorization)
- **곡선이 crossover** → memorization 강도가 step에 따라 변함

---

## fixation 실험 (고정 분석)

### Seed 고정 (한 프롬프트에 여러 seed)

```bash
python manifold.py --fixation seed \
    --fix_general_idx 0 --fix_memo_idx 0 \
    --seed_list "5 10 50 100" \
    --model_key ckpt/stable-diffusion-v1-4
```

→ 같은 prompt에 seed 개수를 5→10→50→100으로 증가 시 std 추이 (std 수렴 확인)

### Prompt 고정 (seed 고정, prompt 수 증가)

```bash
python manifold.py --fixation prompt \
    --np_list "1 5 10 20" \
    --seed 42 --batch 5 \
    --model_key ckpt/stable-diffusion-v1-4
```

→ prompt 수를 1→5→10→20으로 증가 시 std 추이 (diversity 추이 확인)

---

## trajstd 실험 (prompt 간 비교)

```bash
python manifold.py --trajstd general_memo \
    --sources "general:gen_eval=examples/assets/gen_eval.txt" \
               "general:drawbench=examples/assets/drawbench.txt" \
               "memo:webster=examples/assets/sdv1_500_mem.txt" \
    --model_key ckpt/stable-diffusion-v1-4
```

### seedstd vs trajstd 차이

| 차원 | seedstd | trajstd |
|------|---------|---------|
| **평균 축** | seed → prompt | prompt → (없음) |
| **std 대상** | 한 prompt의 seed들 | 모든 prompt들 |
| **물리적 의미** | seed 민감도 | prompt 다양성 |
| **Memo 신호** | std ↓ | std ↓ |

---

## 문제 해결

### 1. OOM (Out of Memory)

```bash
# VRAM 부족 시 batch/prompt 수 감소
BATCH=3 N_PROMPTS=50 bash shells/exp_main/manifold/run_seedstd.sh
```

### 2. 곡선이 flat/이상

- CFG 값 재확인 (shell vs Python)
- 프롬프트 파일 존재 및 내용 확인
- `--n_prompts 10`으로 smoke test 후 확대

### 3. 캐시 무효화

```bash
# 특정 config의 latent 재생성
rm -r workdir/exp_main/manifold/latent_std/save/sd14_base/ddim/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=5/seed=42/
bash shells/exp_main/manifold/run_seedstd.sh
```

### 4. 모델 로드 강제 (캐시 무시)

```bash
# pool이 있어도 rollout 수행하고 싶을 때
rm -r workdir/exp_main/manifold/latent_std/save/...
python manifold.py --seedstd ... --model_key ckpt/...
```

---

## 저장 경로 상세

### seedstd 저장 구조

```
workdir/exp_main/manifold/latent_std/save/
├── {model_base}/             # sd14_base, sd15_base, sdxl_base
│   └── ddim/                 # sampler (기본)
│       └── {model_key}/      # stable-diffusion-v1-4
│           └── CFG={c}_NFE={n}/
│               └── batch={b}/
│                   └── seed={s}/
│                       ├── general/
│                       │   ├── gen_eval/
│                       │   │   ├── results/         # img_0000_00.png, ...
│                       │   │   └── record/latents/  # latent.npz
│                       │   ├── for_diverse/
│                       │   └── drawbench/
│                       └── memo/
│                           └── webster/
│                               ├── results/
│                               └── record/latents/
└── plot 출력:
    workdir/exp_main/manifold/{model_key}/seedstd_{tag}/{csv,plot}/
```

### latent.npz 포맷

```python
np.load("latent.npz")
# 'x_t':       (T, n*B, 4, 64, 64) fp16
# 'n_prompts': int
# 'n_seeds':   int
```

---

## 성능 팁

| 상황 | 최적화 |
|------|-------|
| 빠른 탐사 | `--n_prompts 20 --batch 3 --NFE 25` |
| 표준 실험 | `--n_prompts 100 --batch 5 --NFE 50` |
| 고정밀 분석 | `--n_prompts 500 --batch 10 --NFE 100` |
| GPU 1개 | `--batch 5` (RTX 4090) or `--batch 3` (RTX 3090) |
| Multi-GPU | 현재 미지원 (sequential) |

**캐시 이점:**
- 첫 실행: 5~10분
- 재실행: 30초 (모델 로드 생략)
- 곡선 재생성만: 2초

---

## 참고 자료

- **메인 코드**: `manifold.py` (실험 1-4 구현)
- **Shell 자동화**: `shells/exp_main/manifold/run_seedstd.sh` (권장 실행 방식)
- **프롬프트 자료**: `examples/assets/*.txt` (general: gen_eval, drawbench, for_diverse / memo: webster, ...)
- **상세 문서**: `manifold.md`
