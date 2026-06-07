"""
Generate baseline images for both memorized and normal text prompts.

Folder structure:
  workdir/
  ├── red_car_rainy_night/baseline/       ← 일반 prompt DDIM 결과
  ├── sandcastle_beach/baseline/
  └── ...
  └── ini_opti/
      └── memorized/                      ← memorized prompt 실험
          ├── astronaut_on_the_moon/baseline/
          └── ...
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import torch
from tqdm import tqdm
from munch import munchify

from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
from torchvision.utils import save_image


# ── Prompt definitions ──
MEMORIZED_PROMPTS = {
    "astronaut_on_the_moon":       "An astronaut on the moon",
    "vangogh_cafe_terrasse":       "VAN GOGH CAFE TERASSE copy.jpg",
    "captain_marvel":              "Captain Marvel Exclusive Ccxp Poster Released Online By Marvel",
    "tiger_portrait":              "Portrait of Tiger in black and white by Lukas Holas",
    "duck_glass_sculpture":        "A transparent sculpture of a duck made out of glass",
    "unicorn_snowy":               "A unicorn in a snowy",
}

NORMAL_PROMPTS = {
    "red_car_rainy_night":         "A red car parked on a rainy street at night with neon reflections",
    "sandcastle_beach":            "A child building a sandcastle on a cloudy beach with a small bucket",
    "cats_windowsill":             "Two cats sitting on a windowsill watching birds outside on a sunny morning",
    "mountain_lake_sunset":        "A mountain lake reflecting the sunset with pine trees in the foreground",
    "jazz_musician_stage":         "A jazz musician playing saxophone on a dimly lit stage",
    "robot_flower_garden":         "A small robot tending a flower garden in spring",
}


def generate_prompt_group(sd, prompts_dict, output_root, cfg_guidance, null_prompt,
                          base_seed, num_seeds, device):
    """Generate images for a group of prompts."""

    for idx, (folder_name, prompt) in enumerate(prompts_dict.items()):
        prompt_dir = os.path.join(output_root, folder_name, "baseline")
        os.makedirs(prompt_dir, exist_ok=True)

        # prompt.txt
        ptxt_path = os.path.join(output_root, folder_name, "prompt.txt")
        with open(ptxt_path, "w") as f:
            f.write(prompt + "\n")

        print(f"\n  [{idx+1}/{len(prompts_dict)}] \"{prompt}\"")

        # Text embedding (fixed)
        uc, c = sd.get_text_embed(null_prompt=null_prompt, prompt=prompt)

        for seed_offset in tqdm(range(num_seeds), desc=f"    Seeds"):
            seed = base_seed + seed_offset * 100
            set_seed(seed)
            x_T = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float32)
            zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

            # DDIM sampling
            for step_idx, t in enumerate(sd.scheduler.timesteps):
                at = sd.alpha(t)
                at_prev = sd.alpha(t - sd.skip)

                with torch.no_grad():
                    noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
                    eps_theta = noise_uc + cfg_guidance * (noise_c - noise_uc)

                x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
                zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

            img = sd.decode(x0_hat)
            img = (img / 2 + 0.5).clamp(0, 1)

            save_path = os.path.join(prompt_dir, f"seed_{seed:04d}.png")
            save_image(img, save_path)

        print(f"    -> {num_seeds} images -> {prompt_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_guidance", type=float, default=7.5)
    p.add_argument("--null_prompt", type=str, default="")
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--num_seeds", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "workdir", "ini_opti"))
    args = p.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("Baseline Image Generation")
    print(f"  model      : {args.model_key}")
    print(f"  device     : {device}")
    print(f"  NFE        : {args.num_inference_steps}")
    print(f"  CFG        : {args.cfg_guidance}")
    print(f"  num_seeds  : {args.num_seeds}")
    print(f"  memorized  : {len(MEMORIZED_PROMPTS)} prompts")
    print(f"  normal     : {len(NORMAL_PROMPTS)} prompts")
    print("=" * 60)

    # Load model
    solver_config = munchify({"num_sampling": args.num_inference_steps})
    sd = StableDiffusion(
        solver_config=solver_config,
        model_key=args.model_key,
        device=device,
        seed=args.base_seed,
    )

    # ── memorized ──
    print("\n" + "=" * 60)
    print("  [MEMORIZED PROMPTS]")
    print("=" * 60)
    generate_prompt_group(
        sd, MEMORIZED_PROMPTS,
        os.path.join(args.output_dir, "memorized"),
        args.cfg_guidance, args.null_prompt,
        args.base_seed, args.num_seeds, device,
    )

    # ── text_prompt (normal) → workdir/ 바로 아래 ──
    print("\n" + "=" * 60)
    print("  [NORMAL TEXT PROMPTS]")
    print("=" * 60)
    generate_prompt_group(
        sd, NORMAL_PROMPTS,
        os.path.join(args.output_dir, "DDIM", "text_prompt"),
        args.cfg_guidance, args.null_prompt,
        args.base_seed, args.num_seeds, device,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  memorized  : {args.output_dir}/memorized/")
    print(f"  normal     : {args.output_dir}/text_prompt/")
    print("=" * 60)


if __name__ == "__main__":
    main()
