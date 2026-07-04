"""BE attention mask batch inference — Webster TV(local) prompts 로 BE mask + 생성 이미지 저장.

local memorization eval 파이프라인:
  ① 본 스크립트: TV prompts → SD1.4 inference + BE mask 저장 (mask/, result/)
  ② compute_sscd_gt (global SSCD, run_memo_chen)
  ③ compute_local_mem.py (global SSCD + mask → local metric)

Usage:
    python run_be_inference.py \\
        --model_key ckpt/stable-diffusion-v1-4 \\
        --prompt_dir examples/assets/cvpr2025_tv_prompts.txt \\
        --output_dir workdir/memorization/sd14_base/be_local \\
        --num_prompts 229 --nfe 50 --cfg 7.5 --seed 42
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from munch import munchify
from latent_diffusion import StableDiffusion
from be_attention_mask import generate_be_mask


def main():
    ap = argparse.ArgumentParser(description="BE attention mask batch inference (local memorization)")
    ap.add_argument("--model_key", default="ckpt/stable-diffusion-v1-4")
    ap.add_argument("--prompt_dir", default="examples/assets/cvpr2025_tv_prompts.txt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_prompts", type=int, default=229)
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn_res", type=int, nargs=2, default=[16, 16])
    ap.add_argument("--final_token_idx", type=int, default=-2)
    args = ap.parse_args()

    # 출력 디렉토리
    result_dir = os.path.join(args.output_dir, "result")
    mask_dir = os.path.join(args.output_dir, "mask")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # prompts 로드
    prompts = []
    with open(args.prompt_dir) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    prompts = prompts[: args.num_prompts]
    print(f"[INFO] {len(prompts)} prompts from {args.prompt_dir}")

    # prompts.txt 저장 (pairing 용)
    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        for p in prompts:
            f.write(p + "\n")

    # SD 로드
    sd = StableDiffusion(
        solver_config=munchify({"num_sampling": args.nfe}),
        model_key=args.model_key, device=args.device, seed=args.seed,
    )

    # batch (per prompt): BE mask + 이미지
    for i, prompt in enumerate(prompts):
        try:
            mask, img = generate_be_mask(
                sd, prompt, cfg=args.cfg, device=args.device,
                final_token_idx=args.final_token_idx,
                image_size=512, attn_res=tuple(args.attn_res),
            )
            # 이미지 저장 [1,3,512,512] → img_{pid:04d}_00.png
            arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(result_dir, f"img_{i:04d}_00.png"))
            # mask 저장 mask_{pid:04d}.npy
            np.save(os.path.join(mask_dir, f"mask_{i:04d}.npy"), mask.cpu().numpy().astype(np.float32))
            print(f"[{i+1}/{len(prompts)}] {prompt[:40]}... → img + mask (mem px {int(mask.sum().item())})")
        except Exception as e:
            print(f"[{i+1}/{len(prompts)}] FAIL {prompt[:40]}... : {e}")
            continue

    print(f"[Done] result → {result_dir}/  mask → {mask_dir}/")


if __name__ == "__main__":
    main()
