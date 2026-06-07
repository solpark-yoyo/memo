"""
Training-Free Memorization Mitigation via x_T Optimization.

Phase 1: Optimize x_T to minimize ||ε - ε_s||² (epsilon consistency)
Phase 2: Generate image with standard DDIM from optimized x_T

Algorithm:
    1. Start from x_T ~ N(0,I)
    2. Run DDIM to step t (with gradient)
    3. x̂_0|t = Tweedie(x_t)
    4. x_s = √(ᾱ_s) x̂_0|t + √(1-ᾱ_s) ε   (ε = x_T)
    5. ε_s = ε_θ(x_s, s)
    6. L = ||ε - ε_s||²
    7. x_T ← x_T - lr · ∇_{x_T} L
    8. Repeat N iters → get x_T*
    9. Standard DDIM from x_T* → diverse image

Usage:
    python optimize_xT.py \
        --prompt "An astronaut on the moon" \
        --num_opt_steps 20 \
        --target_step_ratio 0.5 \
        --lr 0.01 \
        --cfg_guidance 7.5 \
        --device cuda:0
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import munchify

from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed


# ===================================================================
#  x_T Optimizer
# ===================================================================
class xTOptimizer(StableDiffusion):
    """
    Optimizes initial noise x_T to minimize epsilon consistency loss,
    then generates an image with standard DDIM.
    """

    def optimize_xT(
        self,
        prompt: str,
        cfg_guidance: float = 7.5,
        null_prompt: str = "",
        num_opt_steps: int = 20,
        lr: float = 0.01,
        target_step_ratio: float = 0.5,
        target_s_ratio: float = 0.5,
        log_every: int = 5,
    ) -> dict:
        """
        Optimize x_T via epsilon consistency loss.

        Args:
            num_opt_steps: number of optimization iterations
            lr: learning rate for x_T update
            target_step_ratio: which denoising step to compute loss at (0~1)
            target_s_ratio: which timestep s for re-forward (0~1)
            log_every: log frequency
        """
        # --- text embeddings (fixed) ---
        uc, c = self.get_text_embed(null_prompt=null_prompt, prompt=prompt)

        # --- Enable gradient checkpointing to save memory ---
        self.unet.enable_gradient_checkpointing()

        # --- initial x_T ---
        x_T_init = torch.randn(1, 4, 64, 64, device=self.device, dtype=torch.float32)
        x_T = x_T_init.clone().requires_grad_(True)

        # --- fixed target step indices ---
        timesteps_list = list(self.scheduler.timesteps)
        target_t_idx = int(len(timesteps_list) * target_step_ratio)
        target_s_idx = int(len(timesteps_list) * target_s_ratio)
        t_target = timesteps_list[target_t_idx]
        s_target = timesteps_list[target_s_idx]
        alpha_s = self.alpha(s_target)

        print(f"  t_target = {t_target.item()}, s_target = {s_target.item()}, ᾱ_s = {alpha_s.item():.4f}")

        optimizer = torch.optim.Adam([x_T], lr=lr)

        # --- optimization loop ---
        losses = []
        print(f"  Optimizing x_T ({num_opt_steps} steps, lr={lr}) ...")

        for opt_iter in range(num_opt_steps):
            optimizer.zero_grad()

            # ε = x_T (original noise IS x_T)
            epsilon = x_T

            # Run DDIM denoising from x_T to target step t (with gradient)
            zt = x_T.to(self.dtype) * self.scheduler.init_noise_sigma

            for step_idx, t in enumerate(self.scheduler.timesteps):
                at = self.alpha(t)
                at_prev = self.alpha(t - self.skip)

                # UNet forward (need gradient through this)
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                eps_theta = noise_uc + cfg_guidance * (noise_c - noise_uc)

                # Tweedie
                x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

                # DDIM step
                zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

                # Stop at target step
                if step_idx == target_t_idx:
                    break

            # x̂_0|t is x0_hat from the last iteration
            # Re-forward x̂_0|t to fixed s using ε = x_T
            x_s = alpha_s.sqrt().to(self.dtype) * x0_hat + (1 - alpha_s).sqrt().to(self.dtype) * epsilon.to(self.dtype)

            # Model prediction at forwarded point
            noise_uc_s, noise_c_s = self.predict_noise(x_s, s_target, uc, c)
            eps_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s)

            # Loss = ||ε - ε_s||²
            loss = (epsilon.to(self.dtype) - eps_s).reshape(1, -1).pow(2).sum()
            loss.backward()

            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if (opt_iter + 1) % log_every == 0 or opt_iter == 0:
                print(f"    iter {opt_iter+1:3d}/{num_opt_steps}  loss = {loss_val:.1f}")

        # --- optimized x_T ---
        x_T_opt = x_T.detach().clone()

        return {
            "x_T_init": x_T_init.cpu(),
            "x_T_opt": x_T_opt.cpu(),
            "losses": losses,
            "prompt": prompt,
        }

    @torch.no_grad()
    def generate_from_xT(
        self,
        x_T: torch.Tensor,
        prompt: str,
        cfg_guidance: float = 7.5,
        null_prompt: str = "",
    ) -> torch.Tensor:
        """
        Standard DDIM generation from a given x_T.
        """
        uc, c = self.get_text_embed(null_prompt=null_prompt, prompt=prompt)

        zt = x_T.to(self.device).to(self.dtype) * self.scheduler.init_noise_sigma

        for step_idx, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM sampling")):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
            eps_theta = noise_uc + cfg_guidance * (noise_c - noise_uc)

            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

        # Decode
        img = self.decode(x0_hat)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


# ===================================================================
#  Plotting
# ===================================================================
def plot_optimization(losses_dict, output_dir, filename="optimization_loss.png"):
    """Plot optimization loss curves for each prompt."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for result in losses_dict:
        label = result["prompt"][:50]
        ax.plot(range(1, len(result["losses"]) + 1), result["losses"],
                linewidth=2, label=label, alpha=0.85)

    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel(r"$\|\, \epsilon - \epsilon_s \,\|^2$", fontsize=14)
    ax.set_title("x_T Optimization: Epsilon Consistency Loss", fontsize=14)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {path}")
    plt.close()


# ===================================================================
#  Main
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="x_T optimization for memorization mitigation")
    p.add_argument("--prompts", nargs="+", type=str, default=None,
                   help="Prompts (default: 3 normal + 1 memorized)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_guidance", type=float, default=7.5)
    p.add_argument("--null_prompt", type=str, default="")
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "results_xT_opt"))
    # optimization
    p.add_argument("--num_opt_steps", type=int, default=20,
                   help="Number of optimization iterations")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate for x_T update")
    p.add_argument("--target_step_ratio", type=float, default=0.5,
                   help="Which denoising step to compute loss (0=start, 1=end)")
    p.add_argument("--target_s_ratio", type=float, default=0.5,
                   help="Which timestep s for re-forward (0=start, 1=end)")
    p.add_argument("--num_gen_samples", type=int, default=3,
                   help="Number of images to generate per prompt (with different seeds)")
    return p.parse_args()


# Default prompts
DEFAULT_PROMPTS = [
    "A red car parked on a rainy street at night with neon reflections",
    "A child building a sandcastle on a cloudy beach with a small bucket",
    "Two cats sitting on a windowsill watching birds outside on a sunny morning",
    "An astronaut on the moon",  # memorized
]


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print("=" * 60)
    print("x_T Optimization for Memorization Mitigation")
    print(f"  model     : {args.model_key}")
    print(f"  device    : {device}")
    print(f"  NFE       : {args.num_inference_steps}")
    print(f"  CFG       : {args.cfg_guidance}")
    print(f"  opt_steps : {args.num_opt_steps}")
    print(f"  lr        : {args.lr}")
    print(f"  prompts   : {len(prompts)}")
    print("=" * 60)

    # Load model
    solver_config = munchify({"num_sampling": args.num_inference_steps})
    optimizer = xTOptimizer(
        solver_config=solver_config,
        model_key=args.model_key,
        device=device,
        seed=args.seed,
    )

    # ---- Phase 1: Optimize x_T for each prompt ----
    print("\n" + "=" * 60)
    print("Phase 1: x_T Optimization")
    print("=" * 60)

    all_results = []
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] {prompt}")
        result = optimizer.optimize_xT(
            prompt=prompt,
            cfg_guidance=args.cfg_guidance,
            null_prompt=args.null_prompt,
            num_opt_steps=args.num_opt_steps,
            lr=args.lr,
            target_step_ratio=args.target_step_ratio,
            target_s_ratio=args.target_s_ratio,
        )
        all_results.append(result)

    # Plot optimization curves
    plot_optimization(all_results, args.output_dir)

    # ---- Phase 2: Generate images with standard DDIM ----
    print("\n" + "=" * 60)
    print("Phase 2: Standard DDIM Generation")
    print("=" * 60)

    for idx, result in enumerate(all_results):
        prompt = result["prompt"]
        safe_name = "".join(c if c.isalnum() else "_" for c in prompt[:30])
        prompt_dir = os.path.join(args.output_dir, f"{idx:02d}_{safe_name}")
        os.makedirs(prompt_dir, exist_ok=True)

        print(f"\n[{idx+1}/{len(all_results)}] {prompt}")

        for sample_idx in range(args.num_gen_samples):
            seed = args.seed + sample_idx * 100

            # Generate with ORIGINAL x_T
            torch.manual_seed(seed)
            x_T_orig = torch.randn(1, 4, 64, 64)
            img_orig = optimizer.generate_from_xT(
                x_T_orig, prompt, cfg_guidance=args.cfg_guidance)
            save_path = os.path.join(prompt_dir, f"orig_seed{seed}.png")
            from torchvision.utils import save_image
            save_image(img_orig, save_path)
            print(f"  [orig seed={seed}] saved → {save_path}")

            # Generate with OPTIMIZED x_T
            torch.manual_seed(seed)
            img_opt = optimizer.generate_from_xT(
                result["x_T_opt"], prompt, cfg_guidance=args.cfg_guidance)
            save_path = os.path.join(prompt_dir, f"optimized_seed{seed}.png")
            save_image(img_opt, save_path)
            print(f"  [optimized] saved → {save_path}")

        # Save x_T tensors
        np.savez(
            os.path.join(prompt_dir, "xT.npz"),
            x_T_init=result["x_T_init"].numpy(),
            x_T_opt=result["x_T_opt"].numpy(),
            losses=np.array(result["losses"]),
            prompt=prompt,
        )

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
