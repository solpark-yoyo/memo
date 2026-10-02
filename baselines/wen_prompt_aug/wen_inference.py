#!/usr/bin/env python
"""Wen et al. prompt augmentation 러너 — inference_mem.py 저장 보강판.

원본 inference_mem.py 는 생성 후 내부 평가만 하고 PNG 를 디스크에 저장하지
않는다(지표 print/wandb 만). 본 run.py 는 동일한 공식 사용 경로로 생성하여
img_XXXX_YY.png 로 저장하는 것만 추가한다 (core 로직은 기존 모듈 재사용).

흐름 (inference_mem.py:85-120 재현):
  1) pipe.aug_prompt(prompt, target_loss=λ, lr, optim_iters) → 최적화 cond 임베딩
     - 내부: x_T 고정 프로브에서 ‖ε_cond−ε_uncond‖₂ 를 λ 이하로 GD (조기 종료)
  2) pipe(prompt_embeds=auged, ...) — 같은 seed 로 처음부터 NFE 생성
  3) output_dir/img_{prompt:04d}_{sample:02d}.png 저장

seed 정책: 프롬pt별 gen_seed + i (프롬pt 간 재현성, 샘플 간 다양성).
사용 (ori_memo/ 에서):
    python baselines/wen_prompt_aug/run.py \
        --dataset examples/assets/cvpr2025_memo_prompt_wen.jsonl \
        --model_id ckpt/stable-diffusion-v1-4 \
        --optim_target_loss 3.0 --gen_seed 42 \
        --output_dir {gen_dir}/result
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo 내부 모듈

import torch
from diffusers import DDIMScheduler

from local_sd_pipeline import LocalStableDiffusionPipeline
from optim_utils import set_random_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_name", default="chen")
    ap.add_argument("--dataset", required=True, help="jsonl (caption 필드)")
    ap.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--num_images_per_prompt", type=int, default=5)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--gen_seed", type=int, default=0)
    # mitigation (aug_prompt 전달) — optim_target_loss=None 이면 완화 없음
    ap.add_argument("--optim_lr", type=float, default=0.05)
    ap.add_argument("--optim_iters", type=int, default=10)
    ap.add_argument("--optim_target_steps", type=int, default=0)
    ap.add_argument("--optim_target_loss", type=float, default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda", type=str, help="연산 장치 (예: cuda, cuda:3)")
    ap.add_argument("--batch_txt", type=int, default=1,
                    help='한 번의 최적화+생성에 묶을 프롬pt 수 (1=기존 per-prompt 경로). '
                         '예: --batch_txt 5 --num_images_per_prompt 1')
    ap.add_argument("--num_prompts", type=int, default=0,
                    help='사용할 프롬pt 상한 (0=전체) — jsonl 앞 N개만 실행')
    args = ap.parse_args()

    # ---- pipe 로드 (inference_mem.py 와 동일: bf16 + DDIM) ----
    device = args.device
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset) as f:
        prompts = [json.loads(l)["caption"] for l in f if l.strip()]
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]           # --num_prompts 상한 (0=전체)
    print(f"[wen] prompts={len(prompts)} λ={args.optim_target_loss} "
          f"lr={args.optim_lr} iters={args.optim_iters} batch={args.num_images_per_prompt}")

    # ---- 프롬pt 배치 청킹 (batch_txt=1이면 크기 1 청크 = 기존 per-prompt 경로 그대로) ----
    nipp = args.num_images_per_prompt
    # record 루트: output_dir(=gen_dir/result)의 부모/record/per_prompt/img_{pid:04d}/
    _record_root = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), "record", "per_prompt")
    for bstart in range(0, len(prompts), max(args.batch_txt, 1)):
        chunk = prompts[bstart:bstart + max(args.batch_txt, 1)]
        # seed: 배치 첫 pid 기준 — probe와 생성이 같은 x_T를 공유하는 기존 성질 유지
        seed = args.gen_seed + bstart

        twd_record_dir = os.path.join(_record_root, f"img_{bstart:04d}")

        if args.optim_target_loss is not None:
            if len(chunk) == 1:
                # 기존 단일 경로 (회귀 없음)
                set_random_seed(seed)
                auged = pipe.aug_prompt(
                    chunk[0],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=nipp,
                    target_steps=[args.optim_target_steps],
                    lr=args.optim_lr,
                    optim_iters=args.optim_iters,
                    target_loss=args.optim_target_loss,
                )
            else:
                set_random_seed(seed)
                auged = pipe.aug_prompt_batch(
                    list(chunk),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=nipp,
                    target_steps=[args.optim_target_steps],
                    lr=args.optim_lr,
                    optim_iters=args.optim_iters,
                    target_loss=args.optim_target_loss,
                )
            # 최적화 임베딩으로 생성 (같은 seed → 같은 x_T)
            set_random_seed(seed)
            outputs = pipe(
                prompt_embeds=auged,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=nipp,
                twd_gap_record_dir=twd_record_dir,
            )
        else:
            set_random_seed(seed)
            outputs = pipe(
                prompt=list(chunk) if len(chunk) > 1 else chunk[0],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=nipp,
                twd_gap_record_dir=twd_record_dir,
            )

        # ours 규격 저장 — pid는 전체 목록 기준 전역 인덱스 (bstart+j)
        for j in range(len(chunk)):
            for k in range(nipp):
                idx = j * nipp + k
                outputs.images[idx].save(
                    os.path.join(args.output_dir, f"img_{bstart + j:04d}_{k:02d}.png"))
        print(f"[wen {bstart + len(chunk)}/{len(prompts)}] "
              f"img_{bstart:04d}~{bstart + len(chunk) - 1:04d} × {nipp}")


if __name__ == "__main__":
    main()
