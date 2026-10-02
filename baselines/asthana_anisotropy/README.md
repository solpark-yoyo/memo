# asthana_anisotropy — Memorization Anisotropy Baseline

**Paper:** "Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability"  
**Authors:** Rohan Asthana, Vasileios Belagiannis  
**Venue:** ICLR 2026  
**arXiv:** https://arxiv.org/abs/2601.20642

## 핵심 아이디어

memorized 프롬프트는 저노이즈 타임스텝에서 guidance vector(`c_eps - uc_eps`)와 unconditional score 사이에 강한 angular alignment(anisotropy)를 보임.  
이를 norm 기반 지표(기존 방법, 고/중간 노이즈에서 유효)와 결합해 detection metric 구성:

```
combined = gamma_1 * norm_metric + gamma_2 * cosine_metric
```

- SD1: `gamma_1=1.0, gamma_2=2.0`
- SD2: `gamma_1=1.0, gamma_2=0.1`

detection은 순수 Gaussian noise 입력에서 **forward pass 2회**만 필요 (denoising rollout 불필요).

## 초기 설정

```bash
# pipe.py + prompt 파일 다운로드 (curl 필요)
bash baselines/asthana_anisotropy/setup.sh
```

또는 직접 clone:
```bash
git clone https://github.com/rohanasthana/memorization-anisotropy /tmp/aniso
cp /tmp/aniso/local_model/pipe.py baselines/asthana_anisotropy/local_model/
cp /tmp/aniso/prompts/*.txt       baselines/asthana_anisotropy/prompts/
```

## 실행

```bash
cd baselines/asthana_anisotropy

# detection
bash run_detection.sh

# evaluation
python detect_eval.py \
    --path   det_outputs/sd1_mem_gen1_modex,c|x_seed51.pt \
    --npath  det_outputs/sd1_nmem_gen1_modex,c|x_seed51.pt \
    --path_cosine  det_outputs/sd1_mem_cosine_gen1_modex,c|x_seed51.pt \
    --npath_cosine det_outputs/sd1_nmem_cosine_gen1_modex,c|x_seed51.pt \
    --sd_ver 1
```

## 파일 구조

```
asthana_anisotropy/
├── detect_mem.py       # detection 메인 스크립트
├── detect_eval.py      # AUC / TPR@1%FPR / TPR@3%FPR 계산
├── utils.py            # arnoldi_iteration_jvp, measure_CLIP/SSCD_similarity
├── run_detection.sh    # sweep 오케스트레이션
├── setup.sh            # pipe.py + prompts 다운로드
├── local_model/
│   ├── __init__.py
│   └── pipe.py         # LocalStableDiffusionPipeline (setup.sh로 받아야 함)
└── prompts/
    ├── sd1_mem.txt     # SD1.4 memorized prompts (~500)
    ├── sd1_nmem.txt    # SD1.4 non-memorized prompts (COCO captions)
    ├── sd2_mem.txt
    └── sd2_nmem.txt
```

## Citation

```bibtex
@inproceedings{asthana2026detecting,
  title={Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability},
  author={Rohan Asthana and Vasileios Belagiannis},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=HTPGy5ydAY}
}
```
