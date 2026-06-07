import sys
import os

# ---------------------------------------------------------------------------
# SCRIPT_DIR = ori_memo/  (this script lives inside ori_memo)
# ---------------------------------------------------------------------------
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
#  Default prompts for memorization comparison
# ===================================================================
#  3 normal prompts + 1 memorized prompt
DEFAULT_PROMPTS = [
    # --- 3 normal (non-memorized) ---
    "A red car parked on a rainy street at night with neon reflections",
    "A child building a sandcastle on a cloudy beach with a small bucket",
    "Two cats sitting on a windowsill watching birds outside on a sunny morning",
    # --- 1 memorized ---
    "An astronaut on the moon",
]
DEFAULT_MEMO_INDICES = [3]   # index of memorized prompt(s) in DEFAULT_PROMPTS


# ===================================================================
#  Analyzer class (extends StableDiffusion)
# ===================================================================
class MemorizationAnalyzer(StableDiffusion):
    """
    Extends StableDiffusion with epsilon trajectory analysis for memorization.

    At every denoising step t we record:
        - ε_s     = ε_θ(x_t, t)           (model noise prediction)
        - x̂_0    = Tweedie estimate of x_0
        - ||ε - ε_s||²                     (distance to original noise)
    """

    def analyze_single(
        self,
        prompt: str,
        cfg_guidance: float = 7.5,
        null_prompt: str = "",
        target_step_ratio: float = 0.5,
    ) -> dict:
        """
        Run one full DDIM denoising pass and collect epsilon trajectory.

        Pipeline at each step t:
            1. x_t → Tweedie → x̂_0|t
            2. x̂_0|t → forward with ε → x_s  (s = fixed mid-noise level)
            3. ε_s = ε_θ(x_s, s)
            4. ||ε - ε_s||²

        As x̂_0|t → x_0, x_s → true forward x_s, so ε_s → ε.
        Therefore ||ε - ε_s||² should DECREASE with denoising progress.
        """
        # --- text embeddings ---
        uc, c = self.get_text_embed(null_prompt=null_prompt, prompt=prompt)

        # --- initial noise ε ---
        zt = self.initialize_latent().to(self.dtype)   # match model dtype (fp16)
        epsilon_original = zt.clone().detach()          # save the original noise

        # --- Fixed target step s (mid-noise level) ---
        #     Use the timestep at 50% of the schedule as the re-forward target
        timesteps_list = list(self.scheduler.timesteps)
        target_idx = int(len(timesteps_list) * target_step_ratio)
        s = timesteps_list[target_idx]
        as_ = self.alpha(s)   # ᾱ_s (fixed)
        print(f"  [target s = {s.item()}, ᾱ_s = {as_.item():.4f}]")

        # --- storage ---
        step_indices  = []
        timesteps_rec = []
        eps_diff_sq   = []
        tweedie_x0    = []
        eps_s_list    = []

        # --- DDIM denoising loop ---
        pbar = tqdm(self.scheduler.timesteps, desc=f"[{prompt[:45]}]")
        for step_idx, t in enumerate(pbar):
            at     = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # 1) Model noise prediction at current DDIM point: ε_θ(x_t, t)
            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                eps_theta = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # 2) Tweedie estimate  x̂_0|t = (x_t - √(1-ᾱ_t) ε_θ) / √(ᾱ_t)
            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

            # 3) Forward x̂_0|t to FIXED step s using ORIGINAL noise ε:
            #    x_s = √(ᾱ_s) x̂_0|t + √(1-ᾱ_s) ε
            x_s = as_.sqrt() * x0_hat + (1 - as_).sqrt() * epsilon_original

            # 4) Model prediction at forwarded point: ε_θ(x_s, s)
            with torch.no_grad():
                noise_uc_s, noise_c_s = self.predict_noise(x_s, s, uc, c)
                eps_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s)

            # 5) ||ε - ε_s||²  (flattened L2 squared)
            diff = (epsilon_original - eps_s).reshape(eps_s.shape[0], -1)
            diff_norm_sq = (diff ** 2).sum(dim=-1).item()

            # record
            step_indices.append(step_idx)
            timesteps_rec.append(t.item() if torch.is_tensor(t) else int(t))
            eps_diff_sq.append(diff_norm_sq)
            tweedie_x0.append(x0_hat.detach().cpu())
            eps_s_list.append(eps_s.detach().cpu())

            # 6) DDIM step (deterministic, η = 0) — use eps_theta, NOT eps_s
            #    x_{t-1} = √(ᾱ_{t-1}) x̂_0 + √(1-ᾱ_{t-1}) ε_θ(x_t, t)
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

        return {
            "prompt":           prompt,
            "step_indices":     step_indices,
            "timesteps":        timesteps_rec,
            "eps_diff_sq":      eps_diff_sq,
            "tweedie_x0":       tweedie_x0,
            "eps_s_list":       eps_s_list,
            "epsilon_original": epsilon_original.cpu(),
        }

    def analyze_multi_sample(
        self,
        prompt: str,
        cfg_guidance: float = 7.5,
        null_prompt: str = "",
        num_samples: int = 5,
        base_seed: int = 42,
    ) -> dict:
        """
        Run analyze_single multiple times with different seeds.

        For *memorized* prompts the ||ε - ε_s||² curve should look
        nearly identical regardless of the initial noise ε.

        Returns:
            dict with keys:
                prompt  : str
                samples : list[dict]  (one per seed)
        """
        samples = []
        for i in range(num_samples):
            seed = base_seed + i * 1000
            self.generator.manual_seed(seed)
            # also set torch seed for randn
            torch.manual_seed(seed)
            result = self.analyze_single(
                prompt=prompt,
                cfg_guidance=cfg_guidance,
                null_prompt=null_prompt,
            )
            result["seed"] = seed
            samples.append(result)
            print(f"  sample {i+1}/{num_samples}  seed={seed}  done")

        return {"prompt": prompt, "samples": samples}


# ===================================================================
#  Plotting helpers
# ===================================================================
def plot_single_trajectory(results_list, output_dir,
                          memo_indices=None, filename="eps_trajectory.png"):
    """
    Plot ||ε - ε_s||² vs denoising step — one curve per prompt.

    Args:
        results_list : list[dict]   one dict per prompt
        memo_indices : set[int]     indices of memorized prompts
                                   (drawn in red, thicker, dashed)
    """
    if memo_indices is None:
        memo_indices = set()

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, use_log in [(ax_lin, False), (ax_log, True)]:
        for idx, result in enumerate(results_list):
            prompt = result["prompt"]
            is_memo = idx in memo_indices

            if len(prompt) > 45:
                label = prompt[:45] + "..."
            else:
                label = prompt

            if is_memo:
                label = f"[MEMO] {label}"
                ax.plot(result["step_indices"], result["eps_diff_sq"],
                        linewidth=2.5, linestyle="--", color="red",
                        marker="o", markersize=3, label=label, alpha=0.95, zorder=10)
            else:
                ax.plot(result["step_indices"], result["eps_diff_sq"],
                        linewidth=1.5, linestyle="-",
                        marker="o", markersize=2, label=label, alpha=0.8)

        ax.set_xlabel("Denoising Step", fontsize=12)
        if use_log:
            ax.set_yscale("log")
            ax.set_ylabel(r"$\|\, \epsilon - \epsilon_s \,\|^2$  (log scale)", fontsize=13)
            ax.set_title("Log Scale", fontsize=13)
        else:
            ax.set_ylabel(r"$\|\, \epsilon - \epsilon_s \,\|^2$", fontsize=13)
            ax.set_title("Linear Scale", fontsize=13)
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Memorization:  $\|\, \epsilon - \epsilon_s \,\|^2$"
        "  (red = memorized)",
        fontsize=14,
    )
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {path}")
    plt.close()


def plot_multi_sample(multi_results_list, output_dir, filename="eps_trajectory_multi.png"):
    """
    Plot multi-sample analysis with mean ± std shading.
    One subplot-group per prompt.
    """
    n = len(multi_results_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, mr in zip(axes, multi_results_list):
        prompt = mr["prompt"]
        label = (prompt[:35] + "...") if len(prompt) > 35 else prompt

        all_curves = np.array([s["eps_diff_sq"] for s in mr["samples"]])
        steps = mr["samples"][0]["step_indices"]
        mean = all_curves.mean(axis=0)
        std  = all_curves.std(axis=0)

        ax.plot(steps, mean, linewidth=2, color="tab:blue")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="tab:blue")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Denoising Step", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\|\, \epsilon - \epsilon_s \,\|^2$", fontsize=13)
    fig.suptitle(
        r"Memorization: $\|\, \epsilon - \epsilon_s \,\|^2$  (mean ± std, "
        f"{len(multi_results_list[0]['samples'])} seeds)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {path}")
    plt.close()


# ===================================================================
#  Main
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Epsilon trajectory analysis for T2I memorization")
    p.add_argument("--prompts", nargs="+", type=str, default=None,
                   help="Prompts to analyze (default: 3 normal + 1 memorized)")
    p.add_argument("--memo_indices", nargs="+", type=int, default=None,
                   help="0-based indices of memorized prompts (default: [3])")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--cfg_guidance", type=float, default=7.5)
    p.add_argument("--null_prompt", type=str, default="")
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"),
                   help="Path or HF hub id for SD 1.5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "results_eps_trajectory"))
    # multi-sample
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of different seeds per prompt (1 = single run)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    memo_indices = set(args.memo_indices) if args.memo_indices else set(DEFAULT_MEMO_INDICES)

    # ---- Load model ----
    print("=" * 60)
    print("Loading Stable Diffusion 1.5 ...")
    print(f"  model_key : {args.model_key}")
    print(f"  device    : {device}")
    print(f"  NFE       : {args.num_inference_steps}")
    print(f"  CFG       : {args.cfg_guidance}")
    print(f"  prompts   : {len(prompts)}")
    print("=" * 60)

    solver_config = munchify({"num_sampling": args.num_inference_steps})
    analyzer = MemorizationAnalyzer(
        solver_config=solver_config,
        model_key=args.model_key,
        device=device,
        seed=args.seed,
    )

    # ---- Single-sample mode ----
    if args.num_samples == 1:
        results = []
        for idx, prompt in enumerate(prompts):
            print(f"\n[{idx+1}/{len(prompts)}] {prompt}")
            analyzer.generator.manual_seed(args.seed)
            torch.manual_seed(args.seed)
            res = analyzer.analyze_single(
                prompt=prompt,
                cfg_guidance=args.cfg_guidance,
                null_prompt=args.null_prompt,
            )
            results.append(res)

            # save per-prompt numpy data
            np.savez(
                os.path.join(args.output_dir, f"eps_traj_{idx:02d}.npz"),
                step_indices=np.array(res["step_indices"]),
                timesteps=np.array(res["timesteps"]),
                eps_diff_sq=np.array(res["eps_diff_sq"]),
                prompt=prompt,
            )

        plot_single_trajectory(results, args.output_dir, memo_indices=memo_indices)

    # ---- Multi-sample mode ----
    else:
        multi_results = []
        for idx, prompt in enumerate(prompts):
            print(f"\n[{idx+1}/{len(prompts)}] {prompt}  ({args.num_samples} seeds)")
            mr = analyzer.analyze_multi_sample(
                prompt=prompt,
                cfg_guidance=args.cfg_guidance,
                null_prompt=args.null_prompt,
                num_samples=args.num_samples,
                base_seed=args.seed,
            )
            multi_results.append(mr)

            # save numpy data
            all_curves = np.array([s["eps_diff_sq"] for s in mr["samples"]])
            np.savez(
                os.path.join(args.output_dir, f"eps_traj_multi_{idx:02d}.npz"),
                step_indices=np.array(mr["samples"][0]["step_indices"]),
                all_eps_diff_sq=all_curves,   # (num_samples, num_steps)
                prompt=prompt,
            )

        plot_multi_sample(multi_results, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
