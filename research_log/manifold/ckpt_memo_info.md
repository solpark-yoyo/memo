# CKPT_MEMO_INFO

각 체크포인트(diffusion)별 memorized / general text prompt 소스 등록 — manifold-dist 실험용.
(경로는 `/memo/ori_memo/` 기준)

## 1. sd14 (`ckpt/stable-diffusion-v1-4`) — vanilla SD1.4에서 발견된 memorized prompt

### sd14 평가(실험) 정합 매핑 — prompt ↔ GT (2026-09-01 확정)

| text prompt | GT (SSCD-to-GT ref) | 비고 |
|---|---|---|
| `examples/assets/cvpr2025_memo_prompt.txt` (500) | `datasets/memo/eval_sscd/sd14/cvpr2025_webster_gt` (prompt_idx 행 i = 프롬pt 행 i 대응) | Webster 세트 — 88개 GT는 MISSING 마킹(자동 skip) |
| `examples/assets/sdv1_500_mem.txt` (500) | `datasets/memo/eval_sscd/sd14/sdv1_500_mem_groundtruth` | Wen sdv1_500 세트 |
| 세트 관계 | 캡션 교집합 **291/500, 순서는 전부 다름** | 교차 평가 시 **캡션 브리지** 필수 (생성 idx→캡션→상대 GT idx) |

### 1-a. [Webster] Webster et al. 2023 (arXiv [2305.08694](https://arxiv.org/abs/2305.08694))

| 파일 | 개수 | 설명 |
|---|---|---|
| `examples/assets/cvpr2025_memo_prompt.txt` | **500** | 전체 세트 — GT: `datasets/cvpr2025_webster_gt/` ✓ |
| `examples/assets/cvpr2025_rv_prompts.txt` | ~250 | verbatim 분류 (이미지 거의 동일복제) |
| `examples/assets/cvpr2025_tv_prompts.txt` | ~250 | template 분류 (구조·스타일 복제) |
| 유형표 | | `cvpr2025_memo_type.csv` |

- 발굴: vanilla SD1.4로 175K prompt 생성 → SSCD 매칭
- latent: `save/memo/cvpr2025_memo/` ✓ (shared noise)

### 1-b. [Han] Han et al. NeurIPS 2025 (arXiv [2510.08625](https://arxiv.org/abs/2510.08625))

| 파일 | 개수 | 설명 |
|---|---|---|
| `examples/assets/han_memorized.txt` | **500** | 전체 세트 (orig에서 추출) |
| `baselines/init_score_noise/prompts/memorized_laion_prompts_orig.csv` | 500 | 원본 CSV |
| `baselines/init_score_noise/prompts/memorized_laion_prompts.csv` | 15 | test subset |

- 발굴: LAION 중복 기반 → vanilla SD1.4
- **Webster와 독립 발굴**: 교집합 347 / Han만 153 / Webster만 153 (69% overlap)
- latent: `save/memo/han/` ← 생성 중

### 1-c. [MemBench] Hong et al. 2024 (arXiv [2407.17095](https://arxiv.org/abs/2407.17095))

| 파일 | 개수 | 설명 |
|---|---|---|
| `examples/assets/memorized_prompts_membench.txt` | **3,000** | 전체 세트 — GT: `datasets/membench_ref/` ✓ (1,741개 확보) |
| `baselines/init_score_noise/SD1_final.csv` | 3,000 | 원본 CSV (대상 = SD1 확인) |

- 발굴: MCMC 기반 memorized trigger prompt 탐색 (SD1)
- **대상 모델 = SD1** (baselines/init_score_noise/SD1_final.csv 파일명으로 확인)
- latent: `save/memo/membench/` ✓ (shared noise)

### vanilla SD1.4 memorized 소스 요약

| 소스 | 개수 | GT | latent | 비고 |
|---|---|---|---|---|
| **Webster** | 500 | ✓ (500) | ✓ | 주력 benchmark |
| ~~Han~~ | ~~500~~ | ✗ | ✗ 제외 | **Webster와 69% overlap + 결과 동일** → 중복, 제외 |
| **MemBench** | 3,000 | ✓ (1,741) | ✓ | SD1.4 확인, 대규모 확장 |

[파인튠 SD1.4 — vanilla 아님]
- a. (eval_chen) cvpr2025_memo_prompt.txt : /memo/ori_memo/examples/assets/cvpr2025_memo_prompt.txt
  — webster GT 정합 (`datasets/cvpr2025_webster_gt`), verbatim/template 구분: cvpr2025_rv_prompts.txt / cvpr2025_tv_prompts.txt, 유형표 cvpr2025_memo_type.csv
- b. (membench) memorized_prompts_membench.txt : /memo/ori_memo/examples/assets/memorized_prompts_membench.txt
  — `datasets/membench_ref` 정합 (compute_sscd_gt 기본 ref)
- c. (wen 원본) sdv1_500_memorized.jsonl : /memo/ori_memo/examples/assets/sdv1_500_memorized.jsonl
  — Wen et al. 공개 500 (SDv1.4 발굴)
- d. (wen2024) wen2024_memorized_prompts.txt : /memo/ori_memo/examples/assets/wen2024_memorized_prompts.txt
  — `datasets/wen2024_memorized` GT 정합
- e. (Han) memorized_laion_prompts.csv : /memo/ori_memo/baselines/init_score_noise/prompts/memorized_laion_prompts.csv
  — LAION 중복 기반 (SD1.4, init_score_noise 벤치마크)
- f. (기타) memorized_prompts.txt / new_memorized_text_prompt.txt : /memo/ori_memo/examples/assets/ —
  초기 세트 (출처 미표기, 사용 시 주의)

## 2. sd15 (`ckpt/stable-diffusion-v1-5`)

[memorized text prompt]
- a. sd14 소스(1-a~f) 공용 사용 — 동일 계열 모델이나 **SD1.5 전용 발굴 세트는 미확보**
  (사용 시 GT 정합 재확인 필요)

## 2-S. sd20 (`ckpt/stable-diffusion-2-base`) — 2026-08-30 등록

### 2S-a. [Webster] Webster et al. 2023 (arXiv [2305.08694](https://arxiv.org/abs/2305.08694))

| 파일 | 개수 | 설명 |
|---|---|---|
| `examples/assets/sd2_mem219.txt` | **219** | **SD2.0 전용 memorized 세트** ★ — SD2.0의 유일 공개 release |

- 발굴: vanilla SD2.0에서 SSCD 매칭 → 219개 memorized 발견
- **GT: 확보 (2026-08-30)** ★ — `datasets/sd2_mem_gt/` **195/219장** + `prompt_to_ref.csv`.
  원본: `baselines/jeon_sail/prompts/sd2_mem.jsonl`의 학습 이미지 URL에서 직접 다운로드
  (Jeon ICML 2025 공식 방식 — "training images extracted from sdX_mem.jsonl" 승계.
  웹 dead link 24장 제외, 89% 성공). **SSCD-to-GT 평가 가능** — 기존 "미확보" 결론 갱신
- 원본 소스: `baselines/ren_memattn/prompt/sd2_mem219.txt` (MemBench_code repo 클론으로도 확인)
- jeon_sail 대응 세트: `baselines/jeon_sail/prompts/sd2_mem.txt` + `sd2_mem.jsonl` (동일 219,
  jsonl과 txt는 219 전체 순서·caption 100% 정합 — URL 매칭에 사용)

### 2S-b. [MemBench] Hong et al. 2024 (arXiv [2407.17095](https://arxiv.org/abs/2407.17095))

| 파일 | 개수 | 설명 |
|---|---|---|
| (미release) | ~1,500 (논문 기준) | **SD2 확장 셋은 코드 repo에 release되지 않음** — SD1_final.csv(3,000)만 공개 |

- 논문에서 SD2 1,500개 언급하나 `chunsanHong/MemBench_code` repo에 SD2 CSV/URL 없음
- 확인 방법: repo 클론(`/tmp/membench_code`) → 파일 목록에서 SD1_final.csv + prompts_url.pickle만 존재
- Asthana(ICLR 2026)도 SD2에 MemBench 대신 Webster 219 사용 — release 부재가 업계 표준 행동

### SD2.0 memorized 소스 요약

| 소스 | 개수 | GT | 비고 |
|---|---|---|---|
| **Webster** | **219** | ✗ (미release) | **유일한 공개 SD2 세트** — detection 전용 |
| ~~MemBench~~ | ~~1,500~~ | ✗ | 논문 언급만, 코드 미release |

## 3. sd14_memor_LAION2B_40k (`ckpt/sd14_memor_LAION2B_40k`)

[memorized text prompt]
- a. LAION-2B 40k 파인튠 전용 (memorization 유발 학습) — 대응 prompt 세트
  **경로 미확인** (`examples/text_to_mscoco.py --model sd14_memor_*` 사용 시 확인)
- b. fallback: 1-e (Han laion csv) — 동일 LAION 중복 계열

## 4. wen2024_finetuned (`ckpt/wen2024_finetuned`)

[memorized text prompt]
- a. (전용) wen2024_memorized_prompts.txt : /memo/ori_memo/examples/assets/wen2024_memorized_prompts.txt
  — 이 ckpt의 파인튠 대상 프롬pt·GT (`datasets/wen2024_memorized`)

## 5. sd21 (`ckpt/stable-diffusion-2-1-base`) / sdxl (`ckpt/stable-diffusion-xl-base-1.0`) / sd3 (`ckpt/stable-diffusion-3-medium-diffusers`) / flux (`ckpt/FLUX.1-dev`, `ckpt/FLUX.1-schnell`)

[memorized text prompt]
- a. sd21 → **2-S 참조** (sd2_mem219.txt 공용 가능 — 동일 SD2 계열)
- b. sdxl/sd3/flux → **전용 memorized 세트 미확보** — sd14 소스(1-a~f) 전용 여부는 모델별 사전 검증 필요
  (타 모델에서 해당 프롬pt가 memorized로 발현하는지 SSCD로 확인 후 사용)

---

## OTHERS

[general text prompt]
- a. [mscoco] coco_v2.txt : /memo/ori_memo/examples/assets/coco_v2.txt — 기본 general (eval 표준)
- b. [mscoco-person] coco_v2_5k_person.txt / coco_v2_10k_person.txt (+ `_idx` 버전) : /memo/ori_memo/examples/assets/
- c. [complex] complex_val.txt / complex_val_200_300.txt : /memo/ori_memo/examples/assets/
- d. [기타] gen_eval.txt / for_diverse.txt : /memo/ori_memo/examples/assets/
- e. [bias 계열] laion_aesthetic-v2_prompts_male-biased(_extended).txt ·
  occupation_prompts_female/male-biased(_v2)(_extended).txt · minor_gender_per_occupations.txt : /memo/ori_memo/examples/assets/

[Manifold 실험 소스 정리 — experiment.md 설문 2·3번 대응]

**[On-manifold Reference DB]** (on-manifold trajectory 기준 풀 — 설문 2번)
- a. [mscoco] coco_v2.txt : /memo/ori_memo/examples/assets/coco_v2.txt — **표준**
- b. [laion] LAION-Aesthetics v2 5+ : `manifold/.../batch=5/save/ref/laion_Aes_v2/512x512/`
  — **SD1.4 training DB** ★ (2026-08-30 구축. 원본 19,835장 중 5,000장 512x512 리사이즈 +
  latent 사전계산 (50, 5000, 4, 64, 64) fp16 — `precompute_ref_latent.py`.
  원본: `datasets/sd14_training_db/LAION_Aes_v2_5+/ori/`, 30K 샘플 중 66% 다운로드 성공)

**[general_text_prompt]** (측정 대상 general 풀 — 설문 3번, ref과 별개 소스 권장)
- a. [geneval] geneval_prompts.txt : /memo/ori_memo/examples/assets/geneval_prompts.txt
  — **공식 원본** (djghosh13/geneval, 553개, 2026-08-27 확보.
  기존 gen_eval.txt와 동일 세트 검증 완료 — 마지막 줄 newline 차이만)
- b. [parti] parti_prompts.txt : /memo/ori_memo/examples/assets/parti_prompts.txt
  — **PartiPrompts P2** (google-research/parti, **1,632개**, 2026-08-27 확보.
  Google Parti 논문 — 다양성·복잡성 평가 표준 ★★★)
- c. [drawbench] drawbench.txt : /memo/ori_memo/examples/assets/drawbench.txt
  — **DrawBench** (kennymckormick/DrawBench-Glance, **200개**, 2026-08-27 확보.
  Imagen 논문 — 도전적 프롬프트 stress-test ★★☆)
- d. [complex] complex_val.txt 등 기타 → OTHERS 참조

**[memorized_text_prompt]** (측정 대상 memo 풀 — 설문 3번, selected ckpt 대응)
- sd14 → 1-a~f (chen cvpr2025 / membench / wen 원본·wen2024 / Han laion / 기타)
- sd15 → 2-a (sd14 공용, GT 정합 재확인)
- **sd20 → 2S-a** (`examples/assets/sd2_mem219.txt` — Webster 219, detection 전용)
- sd14_memor_LAION2B_40k → 3-a/b · wen2024_finetuned → 4-a
- sd21 → 5-a (sd2_mem219 공용 가능) · sdxl/sd3/flux → 5-b (전용 미확보 — 발현 검증 후 사용)

[Manifold Distance 방법]
- 정의·수식·선택 가이드: `skills/manifold-dist/rules/dist-measure.md` 참조
  (centroid_l2 | knn | cosine | mmd | spectral | hausdorff | **emd★** —
  전역 분포 거리 표준 emd, 국소 이탈 knn, worst-case 보조 hausdorff)
