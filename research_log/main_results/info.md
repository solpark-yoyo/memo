# Main Results — Baseline · 평가 프로토콜 정보

Ours(**Twd_Gap**) vs baselines 의 CLIPScore/PickScore/ImageReward–SSCD trade-off curve
비교실험(`skills/main_results`)의 대상·측정 정보 정리.

## Baseline 구성 출처

Baseline 4종(Wen, Ren, Jeon, Han)은
**Asthana & Belagiannis, "Detecting and Mitigating Memorization in Diffusion Models
through Anisotropy of the Log-Probability" (ICLR 2026)** — [arXiv 2601.20642](https://arxiv.org/abs/2601.20642) ·
[코드](https://github.com/rohanasthana/memorization-anisotropy) 에서 비교한
mitigation baselines(Wen·Ren·Jeon + no mitigation)에 Han et al. 을 포함한 구성.
이 논문의 평가 방식(SSCD–CLIP/Aesthetic trade-off, 5개 하이퍼파라미터 config,
MemBench 프로토콜)이 우리 main_results 실험 설계의 참조.

## 비교 대상 (methods)

| method | 논문 | venue | arXiv | 완화 방식 | 구현 상태 |
|---|---|---|---|---|---|
| **ours — Twd_Gap** | Tweedie domain gap proxy 기반 x_t 최적화 (자체) | — | — | grad DDIM 1-step 포함 x_t 최적화 | `run_ini_opti.py` (tune 스킬로 lr sweep) |
| `wen` | Wen et al., *Detecting, Explaining, and Mitigating Memorization in Diffusion Models* | ICLR 2024 (Oral) | [2407.21720](https://arxiv.org/abs/2407.21720) · [코드](https://github.com/YuxinWenRick/diffusion_memorization) | prompt augmentation — cond/uncond score 차 norm 으로 text embedding 최적화 | **클론 완료** — `baselines/wen_prompt_aug/` (`detect_mem.py`, `inference_mem.py`, `local_sd_pipeline.py`) |
| `ren` | Ren et al., *Unveiling and Mitigating Memorization in Text-to-Image Diffusion Models Through Cross Attention* | ECCV 2024 | [2403.11052](https://arxiv.org/abs/2403.11052) · [코드(MemAttn)](https://github.com/renjie3/MemAttn) | cross-attention score 조정 (text embedding attention 축소) | **클론 완료** — `baselines/ren_memattn/` (`detect.py`, `local_cmd_inference_time_mitigation.sh`) |
| `jeon` | Jeon, Kim, No et al., *Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes* (SAIL) | ICML 2025 | [2412.04140](https://arxiv.org/abs/2412.04140) · [PMLR](https://proceedings.mlr.press/v267/jeon25a.html) · [코드](https://github.com/Dongjae0324/sharpness_memorization_diffusion) | log-prob sharpness(Hessian trace) 기반 완화 | **클론 완료** — `baselines/jeon_sail/` (`detect_eval.py`, `mitigate_eval.py`) |
| `han` | Han et al., *Adjusting Initial Noise to Mitigate Memorization in Text-to-Image Diffusion Models* | NeurIPS 2025 | [2510.08625](https://arxiv.org/abs/2510.08625) | initial noise 조정 (batch-wise / per-sample) | **구현됨** — `baselines/init_score_noise/` (method명 `init_score_noise`) |
| `ddim` | (완화 없음 — no mitigation 기준점) | — | — | — | `text_to_mscoco.py --method ddim` |

비고:
- Wen(2024)이 cond/uncond score 차 norm 기반 탐지·완화의 원형 — Jeon(2025)이 여기에
  Hessian(−∇²log p)을 결합해 sharpness 관점으로 일반화, Asthana(2026)가 anisotropy
  alignment 까지 확장. Han(2025)은 initial noise 축에서 접근.
- 2601.20642 의 저자 실험 프로토콜: SD v1.4 — MemBench 3000 memorized prompts /
  SD v2.0 — Webster 219 prompts, 5개 하이퍼파라미터 config 으로
  SSCD↓ vs CLIP↑/Aesthetic↑ trade-off 비교 (Fig. 4).

## SSCD 측정 방법 (memorization 지표)

- 측정 스크립트: `compute_sscd_gt.py` — 생성 이미지 vs GT 원본의 **SSCD(Self-Supervised Copy Detection) 유사도**
- 판정: per-prompt mean SSCD > 0.5 → memorized (Webster 2023 기준 threshold)
- ref 디렉토리 공통 규칙: `prompt_to_ref.csv`(prompt_idx, ref_file) 필수

## 평가 프로토콜 3종 (GT 데이터 위치)

| 프로토콜 | 논문 | arXiv | GT 데이터 | 프롬프트 소스 |
|---|---|---|---|---|
| **Chen/Webster** (CVPR 2025) | Chen, Liu, Shah, Xu — *Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models* | [2504.18032](https://arxiv.org/abs/2504.18032) · [코드](https://github.com/chenchen-usyd/BE-PRSS) | `datasets/cvpr2025_webster_gt/` (NNNNN.jpg + prompt_to_ref.csv) | `examples/assets/cvpr2025_memo_prompt.txt` — Webster(2023) memorized prompts 계승 |
| **Wen** (ICLR 2024, Oral) | Wen, Liu, Chen, Lyu — *Detecting, Explaining, and Mitigating Memorization in Diffusion Models* | [2407.21720](https://arxiv.org/abs/2407.21720) · [코드](https://github.com/YuxinWenRick/diffusion_memorization) | `datasets/wen2024_memorized/` (prompt/이미지 + prompt_to_ref.csv) | 동일 폴더 프롬프트 (sd14 fine-tune 유발) |
| **MemBench** | Hong, Oh, Sung — *MemBench: Memorized Image Trigger Prompt Dataset for Diffusion Models* | [2507.06088](https://arxiv.org/abs/2507.06088) | `datasets/membench_ref/` (compute_sscd_gt 기본 ref) | MemBench 프롬프트 (SD v1.4 3000개) |

계보 비고: Webster et al. 2023(*A Reproducible Extraction of Training Images from
Diffusion Models*, arXiv 2305.08694)가 SSCD>0.5 기준 memorized prompt 셋을 정립 →
Wen(ICLR 2024)이 탐지·완화로 확장 → Chen/Webster(CVPR 2025)가 privacy-utility
trade-off 평가로 발전, MemBench(TMLR 2025)가 표준 완화 벤치마크화.
로컬 `cvpr2025_webster_gt` 명칭은 이 계보를 따름.

## T2I 품질 지표

`compute_t2i_metrics.py` — **CLIPScore** / **PickScore** / **ImageReward**
(ckpt: `clip-vit-base-patch16`, `PickScore_v1`, `ImageReward`)

## 결과물 경로 규약

- 스윕 개별: `workdir/memorization/{model}/{method}/CFG=X_NFE=Y/{hp}/seed=S/batch=B/` (`result/`, `eval/`)
- 통합 CSV: `collect_tradeoff.py → {base_dir}/tradeoff.csv` (method/lr 컬럼 자동 추출)
- Plot: `plot_tradeoff.py --x_metric {clipscore,pickscore,imagereward} --y_metric sscd`
  → `{base_dir}/main_tradeoff_{x_metric}.png`
