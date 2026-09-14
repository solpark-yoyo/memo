# manifold.py — Memorization Trajectory Manifold 실험

## 개요

diffusion latent space에서 memorization이 일어나는 방식을 분석하기 위한 4가지 실험을 수행:

| 실험 | 키워드 | 목적 |
|------|--------|------|
| **1. off-manifold** | `--manifold {centroid_l2,knn,...}` | on-manifold reference와의 거리 측정 (memo vs general) |
| **2. fixation** | `--fixation {seed,prompt}` | seed/prompt 고정 시 latent std 테이블 생성 |
| **3. trajstd** | `--trajstd <tag>` | 소스별 프롬프트 간 std(‖z_t‖) 곡선 오버레이 |
| **4. seedstd** | `--seedstd <tag>` | 소스별 프롬프트마다 seed 축 std → prompt 평균 곡선 |

---

## 실험 4: seedstd (Seed Sensitivity Analysis)

### 개념

한 text prompt에 여러 seed를 적용했을 때 latent가 얼마나 흔들리는가를 측정.

- **작은 std**: 서로 다른 seed가 동일한 trajectory로 수렴 → **memorization 신호**
- **큰 std**: 다양한 trajectory 생성 → 정상 작동

### 실행

```bash
cd memo/ori_memo
conda activate div_DM
bash shells/exp_main/manifold/run_seedstd.sh
```

### 저장 구조

```
workdir/exp_main/manifold/latent_std/save/
└── {model_base}/
    └── {sampler}/                    # 기본: ddim
        └── {model_key}/              # 예: stable-diffusion-v1-4
            └── CFG={cfg}_NFE={nfe}/
                └── batch={batch}/
                    └── seed={seed}/
                        ├── general/
                        │   ├── gen_eval/
                        │   │   ├── results/         # img_0000_00.png ~ img_XXXX_YY.png
                        │   │   └── record/latents/  # latent.npz (캐시)
                        │   ├── for_diverse/
                        │   └── drawbench/
                        └── memo/
                            └── webster/
                                ├── results/
                                └── record/latents/
```

**세부:**
- `{model_base}`: sd14_base, sd15_base, sdxl_base 등 (model_key로부터 자동 유도)
- `results/`: 생성 이미지 (512x512)
  - `img_XXXX_YY.png`: prompt_idx=XXXX, seed_idx=YY
  - 예) img_0025_03.png = 25번째 프롬프트 + 3번째 seed
- `record/latents/latent.npz`: 캐시된 latent pool
  - `x_t`: (T, n*B, 4, 64, 64) fp16 (T=50 steps, n=prompt 수, B=seed 수)
  - 메타데이터: `n_prompts`, `n_seeds`

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--seedstd <tag>` | - | 출력 폴더 태그 (예: general_memo) |
| `--sources` | - | "group:name=path" 리스트 (필수) |
| `--model_key` | ckpt/stable-diffusion-v1-4 | 모델 경로 |
| `--cfg` | 7.5 | CFG guidance scale |
| `--NFE` | 50 | DDIM step 수 |
| `--batch` | 5 | 프롬프트당 seed 수 |
| `--n_prompts` | 100 | 소스별 사용할 프롬프트 수 |
| `--seed` | 42 | 기준 seed |
| `--device` | cuda:0 | GPU 장치 |

### 출력

```
workdir/exp_main/manifold/
├── latent_std/save/                          # 실험 데이터
│   └── sd14_base/ddim/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=5/seed=42/
│       ├── general/gen_eval/
│       │   ├── results/img_XXXX_YY.png
│       │   └── record/latents/latent.npz
│       └── memo/webster/
│           ├── results/img_XXXX_YY.png
│           └── record/latents/latent.npz
└── stable-diffusion-v1-4/seedstd_general_memo/  # 시각화 (old style, 호환성)
    ├── csv/std_curve.csv
    └── plot/std_curve.png
```

**CSV 형식** (`csv/std_curve.csv`):
```
step,time,[general]gen_eval,[general]for_diverse,[general]drawbench,[memo]webster
0,50,0.123456,0.125000,0.120000,0.050000
1,49,0.125000,0.127000,0.122000,0.048000
...
49,1,0.250000,0.260000,0.240000,0.010000
```

---

## 캐싱 & 재사용

### 첫 번째 실행 (pool 생성)

```bash
bash shells/exp_main/manifold/run_seedstd.sh
```

→ rollout 수행 → latent & 이미지 저장 → 곡선 생성 (5~10분)

### 두 번째 실행 (캐시 로드)

동일한 명령 재실행:
```bash
bash shells/exp_main/manifold/run_seedstd.sh
```

→ 모든 소스의 `latent.npz` 존재 확인
→ **모델 로드 생략** (GPU 0GB)
→ pool 로드 → 곡선 계산 → 즉시 완료 (30초)

### 캐시 무효화

latent를 다시 생성하려면 삭제:
```bash
rm -r workdir/exp_main/manifold/latent_std/save/sd14_base/ddim/stable-diffusion-v1-4/CFG=7.5_NFE=50/batch=5/seed=42/
```

---

## 실험 1: off-manifold (거리 기반)

### 개념

on-manifold reference(실제 데이터 forward noising)를 기준으로, memo와 general 풀 각각과의 거리 측정.

### 실행

```bash
python manifold.py --manifold centroid_l2 \
    --ref_prompt examples/assets/coco_v2.txt \
    --general_prompt examples/assets/geneval_prompts.txt \
    --memo_prompt examples/assets/cvpr2025_memo_prompt.txt \
    --model_key ckpt/stable-diffusion-v1-4 \
    --cfg 7.5 --NFE 50 --seed 42 --batch 5 --n_prompts 100
```

### 거리 방법

| 방법 | 특징 |
|-----|------|
| `centroid_l2` | 중심 간 L2 거리 (가장 빠름) |
| `knn` | k-nearest neighbor 평균 (기본 k=5) |
| `cosine` | 코사인 유사도 |
| `mmd` | Maximum Mean Discrepancy (분포 비교) |
| `spectral` | Compact spectral block energy (주파수 기반) |
| `hausdorff` | Hausdorff distance (최대 최소 거리) |
| `emd` | Earth Mover's Distance (Sinkhorn) |

---

## 실험 2 & 3: fixation, trajstd

### fixation (seed/prompt 고정)

```bash
python manifold.py --fixation seed \
    --fix_general_idx 0 --fix_memo_idx 0 \
    --seed_list "5 10 50 100" \
    --model_key ckpt/stable-diffusion-v1-4
```

### trajstd (prompt 간 std)

```bash
python manifold.py --trajstd general_memo \
    --sources "general:gen_eval=examples/assets/gen_eval.txt" \
               "general:drawbench=examples/assets/drawbench.txt" \
               "memo:webster=examples/assets/sdv1_500_mem.txt" \
    --model_key ckpt/stable-diffusion-v1-4
```

---

## 모니터링 (tqdm 진행률)

rollout 진행 중 자동으로 progress bar 표시:

```
[rollout] prompts: 45%|████▌     | 45/100 [02:30<03:00, 3.00s/p]
[forward-noise] steps: 100%|██████████| 50/50 [00:15<00:00, 0.30s/s]
```

---

## 문제 해결

### 메모리 부족 (OOM)

- `--batch` 감소: `--batch 3` (기본 5)
- `--n_prompts` 감소: `--n_prompts 50`
- VAE decode 배치 크기는 자동 최적화 (코드 내 배치 8)

### 느린 첫 실행

- 정상 (rollout + 이미지 저장)
- 두 번째부터는 캐시로 빠름

### 곡선이 이상한 경우

1. **cfg 값 재확인**: shell script와 Python 호출 값 일치 확인
2. **프롬프트 파일 확인**: 경로 존재 및 내용 검증
3. **seed 고정**: `--seed 42` 사용하여 재현성 확보

---

## 참고

- 모든 코드 실행: `memo/ori_memo/` 디렉토리에서 `conda activate div_DM` 후
- Shell script 예제: `shells/exp_main/manifold/run_seedstd.sh`
- 프롬프트 자료: `examples/assets/*.txt`
