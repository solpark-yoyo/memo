# 초기 최적화(init_step)가 Memorized Trajectory에 미치는 영향 분석

## 📋 실험 개요

초기에 noise MSE를 줄일 때, memorization 완화 trajectory가 어떻게 변하는지 실증적으로 분석.

### 실험 1: `init_step` 깊이 비교 (같은 위치, 다른 opti 횟수)
```
x_T → [init_step 번 최적화] → DDIM denoising → trajectory 기록

init_step = 1, 2, 3, 4, 5, 10
각 경우의 noise MSE 및 memorized trajectory 비교
```

**가설**: init_step ↑ → noise MSE ↓ → memorization 완화 개선?

### 실험 2: 초기 스텝별 최적화 효과 (다른 위치, 같은 opti 횟수)
```
Denoising step 0, 1, 2, 3, 4, 5까지만 최적화
나머지는 표준 DDIM step

어느 단계까지 최적화하는 것이 memorized trajectory에 가장 효과적?
```

**가설**: 초기 몇 step 최적화 → 전체 trajectory에 미치는 영향 추적

---

## 🚀 실행 방법

### 환경 설정
```bash
cd /home/geonsoo/Desktop/Datasets/Parksol/memo/ori_memo
conda activate div_DM
```

### Exp1: init_step sweep
```bash
bash shells/exp_main/trajectory/run_init_steps_sweep.sh exp1
```

**출력**: `workdir/exp_main/trajectory/exp1_init_steps_sweep/`
- `memorized_init_sweep.png` — memorized prompt trajectory 곡선 (init_step별 오버레이)
- `normal_init_sweep.png` — normal prompt trajectory 곡선 비교
- `summary.txt` — init_step별 noise MSE 테이블

### Exp2: step-by-step optimization
```bash
bash shells/exp_main/trajectory/run_init_steps_sweep.sh exp2
```

**출력**: `workdir/exp_main/trajectory/exp2_step_by_step/`
- `memorized_step_by_step.png` — 각 단계까지 최적화했을 때의 trajectory
- `normal_step_by_step.png` — normal 대조
- step별 최적화 효과 정량화

### 모든 실험 실행
```bash
bash shells/exp_main/trajectory/run_init_steps_sweep.sh all
```

---

## 📊 결과 해석 가이드

### Exp1 결과
```
Exp1이 성공하면:
  1. init_step별 trajectory 곡선이 나뉨
  2. 초기 noise MSE가 init_step 증가에 따라 감소 → confirm
  3. trajectory의 "memorized" 구간이 변하는지 확인
     - 변화 ✓ → init_step이 memorization 완화에 효과 있음
     - 불변 ✗ → init_step만으로는 부족 (gap_steps, num_steps 등 필요)
```

### Exp2 결과
```
Exp2가 성공하면:
  1. 초기 몇 step만 최적화해도 전체 trajectory에 영향?
  2. 최적화 효과의 "depth" 확인:
     - opti_until_step=1 vs opti_until_step=5
     - 어느 시점에서 plateau?
  3. 초기 최적화의 "critical window" 식별
```

---

## 🔧 코드 구조

### `trajectory_init_study.py`
- `parse_args()`: CLI 인자 처리
- `exp1_init_steps_sweep()`: Exp1 메인 로직
  - 각 init_step마다 trajectory 수집
  - step별 latent norm 추적
  - 곡선 오버레이 플롯
- `exp2_step_by_step_optimization()`: Exp2 메인 로직
  - denoising step마다 조건부 최적화
  - trajectory 단계별 기록

### `run_init_steps_sweep.sh`
- 하이퍼파라미터 정의
- Exp1/Exp2 wrapper 함수
- 결과 저장 및 요약

---

## 📝 관련 참고 파일

| 파일 | 역할 | 참고점 |
|-----|------|---------|
| `run_ini_opti.py` | x_T 최적화 메인 스크립트 | init_step, num_steps, gap_steps 의미 |
| `eps_trajectory.py` | trajectory 측정 유틸 | `compact_spectral_block_energies`, `kl_to_standard_normal` |
| `manifold.py` | trajectory manifold 분석 | seedstd, trajstd 측정 방식 |
| `latent_diffusion.py` | SD 모델 래퍼 | `predict_noise()`, scheduler API |

---

## ⚙️ 커스터마이징

### init_step 값 변경 (Exp1)
```bash
bash run_init_steps_sweep.sh exp1
# 내부에서 INIT_STEPS_LIST 수정
```

### 최적화까지의 step 범위 변경 (Exp2)
```bash
# run_init_steps_sweep.sh 내
OPTI_STEPS_LIST=(0 1 2 3 4 5 10)  # step 10까지도 포함
```

### 프롬프트 변경
```bash
python trajectory_init_study.py \
    --exp exp1 \
    --prompt_memo "Custom memorized prompt" \
    --prompt_normal "Custom normal prompt" \
    ...
```

### 샘플 수 및 시드 조정
```bash
python trajectory_init_study.py \
    --exp exp1 \
    --num_samples_per_init 5 \
    --seed 123 \
    ...
```

---

## 📈 성능 & 실행 시간 예상

| 실험 | 프롬프트 | init_steps | samples/step | 총 시간 |
|-----|--------|-----------|-------------|---------|
| Exp1 | 2 | 6개 | 3 | ~2-3시간 |
| Exp2 | 2 | 6개 | 3 | ~2-3시간 |
| 둘다 | 2 | - | - | ~5-6시간 |

**참고**: RTX 4090 기준, FP16 사용

---

## 💡 다음 단계

1. **Exp1 결과 분석**
   - init_step과 memorization 개선도 상관관계 정량화
   - optimal init_step 찾기

2. **Exp2 결과 분석**
   - 초기 최적화의 "critical window" 특정
   - step별 trajectory divergence 측정

3. **결합 실험**
   - Exp1의 최적 init_step + Exp2의 최적 opti_range 조합
   - num_steps, gap_steps와의 interaction 분석

4. **이론 분석**
   - Tweedie 추정량 관점에서 초기 최적화의 의미
   - noise schedule 관점에서 optimal init_steps 도출

---

## 📧 문제 해결

### 메모리 부족
```bash
# 샘플 수 감소
python trajectory_init_study.py --num_samples_per_init 1
```

### 느린 실행
```bash
# step 수 감소
python trajectory_init_study.py --num_inference_steps 25
```

### 플롯 저장 실패
```bash
# 출력 디렉토리 확인
mkdir -p workdir/exp_main/trajectory/exp1_init_steps_sweep
```

---

**생성**: 2026-09-18  
**마지막 수정**: 2026-09-18
