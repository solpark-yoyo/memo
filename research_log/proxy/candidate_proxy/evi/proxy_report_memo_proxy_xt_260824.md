# memo_proxy 1-1 (optimization: latent x_t) 판별력 평가 — 개형 기준 (2026-08-24)

## 판정: **GOOD**

- **1-1의 측정 체인(ⓑ~ⓔ + 결과식)은 ① memo_proxy와 완전히 동일** — loss·측정 코드·곡선을
  공유. 차이는 **grad 업데이트 대상뿐** (①: x_T / 1-1: 중간 latent $x_t$, ⓐ 구간 제외)
- 감지력(개형 판별)은 측정 체인으로 결정되므로 ① 근거 승계 — **기존 측정 재사용** (재측정 금지):
  **20개 비교셋 중 19개에서 memorized 곡선 분리, step 비율 평균 94%, 비율 5~20x**
- grad 도달점($x_t$)은 proxy 측정값과 무관 → 판별력에 영향 없음

## 데이터 (① 측정 재사용)

- 측정: `results_eps_trajectory/ddim/CFG=7.5_NFE=50/seed=42/batch=5/plot00-19`
  (`run_eps_trajectory.sh`, MEASURE=memo_proxy)
- 모델: vanilla SD1.4, CFG=7.5, NFE=50, seed=42, batch=5 (seed 평균 곡선)
- 구성: plot당 coco_v2 3개 (general) + memorized_prompts_membench 1개 (memorized)
- plot별 수치표: [proxy_report_memo_proxy_260822.md](proxy_report_memo_proxy_260822.md)

## 곡선 개형

![memo_proxy 개형 비교](../figs/memo_proxy_curves_260822.png)

*①과 동일 측정 — General 60곡선(회색 밴드, mean±std) 대비 memorized 20곡선 중 19개(적색)가
밴드 위로 명확히 분리, 점선(plot12)만 예외*

## 1-1 고유 평가 축 — grad 도달점

| 항목 | ① (update: x_T) | 1-1 (update: x_t) |
|---|---|---|
| grad chain | ⓐ+ⓑ~ⓔ (전체) | **ⓑ~ⓔ만** (ⓐ 제외) |
| AdjointDPM | 필요 (ⓐ 구간 vanishing 대응) | **불필요** |
| 최적화 시작점 | x_T (trajectory 처음) | 중간 x_t (init_steps 이후) |
| 감지력(개형) | GOOD | **GOOD (동일 — 측정 체인 공유)** |

- 1-1의 실질 차이는 **완화 최적화의 비용 구조**: ⓐ 연쇄 backprop 제거로 최적화 1회 비용
  절감 + AdjointDPM 의존 제거. 단 $x_T$ 자체는 불변 → 동일 seed의 초기 trajectory 유지
- 이 축은 감지력이 아닌 **완화 효율·구조** 문제 — 판정(GOOD)과 독립

## 한계

- 1-1의 완화 성능($x_t$ 보정이 memorization rate·품질에 미치는 효과)은 이 평가 범위 밖 —
  proxy-check는 감지력(개형) 판정만 수행
- ① 평가의 한계 승계: plot12 예외(약기억 프롬pt 추정), 개형 감지 기준 임시 정의,
  SSCD 강도 상관 미측정

## 다음 액션

1. **완화 최적화 연결 (/tune)** — grad를 $x_t$에만 흘리는 최적화 경로 확인:
   `run_ini_opti.py`의 `optimize_xT_adj`는 x_T update → 1-1 실현에는 중간 $x_t$ update
   구조 필요 (`optimize_xt_spectral.py`류와 비교·구현 스펙 작성)
2. ① 보고서의 후속(plot12 예외 확인, 개형 기준 명문화)과 공유 진행
