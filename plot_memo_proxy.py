"""
Plot memorization proxy vs denoising step for a few memorized prompts.

For each prompt: fix x_T (seed), run DDIM forward step by step, and at each
step s compute the normalized Tweedie-gap proxy

    proxy(s) = || eps_ref - eps_s ||^2 / D   (D = noise dim = 4*64*64)

where
    eps_ref = x_T   (fixed injected noise, detached)
    eps_s   = eps_theta(x_s, s)   with CFG
    x_s     = sqrt(alpha_s) * x0_hat(s) + sqrt(1-alpha_s) * x_T
    x0_hat(s) = Tweedie estimate of x0 at step s

x-axis: denoising step (0..NFE-1)
y-axis: normalized memorization proxy (0 = perfect noise recovery, larger = gap)

Usage:
    python plot_memo_proxy.py --num_prompts 3 --gpu 0
"""
import sys, os, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from munch import munchify
from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed

DEFAULT_PROMPT = os.path.join(SCRIPT_DIR, "examples", "assets", "memorized_prompts_membench.txt")


def predict_noise_cfg_batch(sd, zt, t, uc, c):
    """CFG noise prediction for a batch (B prompts at once).

    zt: (B, 4, 64, 64) latent
    uc, c: (B, 77, 768) text embeddings (null / conditional)
    t: scalar timestep tensor

    Returns (noise_uc, noise_c), each (B, 4, 64, 64).
    (sd.predict_noise is hardcoded for batch=1 CFG timestep, so we call the UNet directly.)
    """
    c_embed = torch.cat([uc, c], dim=0)                       # (2B, 77, 768)
    z_in = torch.cat([zt, zt], dim=0)                         # (2B, 4, 64, 64)
    t_in = t.view(1).to(z_in.device).expand(z_in.shape[0])    # (2B,)
    noise_pred = sd.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
    noise_uc, noise_c = noise_pred.chunk(2)
    return noise_uc, noise_c


def load_prompts(path, n):
    with open(path) as f:
        ps = [line.strip() for line in f if line.strip()]
    return ps[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_prompts", type=int, default=3)
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_dir", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--model_key", type=str, default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-5"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output", type=str, default=os.path.join(SCRIPT_DIR, "workdir", "memo_proxy_vs_step.png"))
    args = p.parse_args()
    device = torch.device(args.device)

    sd = StableDiffusion(solver_config=munchify({"num_sampling": args.NFE}),
                         model_key=args.model_key, device=device, seed=args.seed)
    sd.unet.enable_gradient_checkpointing()
    timesteps = list(sd.scheduler.timesteps)
    prompts = load_prompts(args.prompt_dir, args.num_prompts)

    print(f"prompts ({len(prompts)}):")
    for i, pr in enumerate(prompts):
        print(f"  [{i}] {pr}")

    cfg = args.cfg
    init_noise_sigma = sd.scheduler.init_noise_sigma
    # ---- batch over prompts (batch size = num_prompts) ----
    B = len(prompts)
    set_seed(args.seed)

    # text embeddings: null is shared across batch, conditional per prompt
    uc_list, c_list = [], []
    for pr in prompts:
        ue, te = sd.get_text_embed(null_prompt="", prompt=pr)
        uc_list.append(ue)
        c_list.append(te)
    uc = uc_list[0].repeat(B, 1, 1).to(device)          # (B, 77, 768), null shared
    c = torch.cat(c_list, 0).to(device)                 # (B, 77, 768)

    # one initial noise per prompt
    x_T = torch.randn(B, 4, 64, 64, device=device, dtype=torch.float32)
    eps_ref = x_T
    D = eps_ref[0].numel()  # noise dim = 4 * 64 * 64

    zt = x_T.to(sd.dtype) * init_noise_sigma
    all_curves = [[] for _ in range(B)]

    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            at = sd.alpha(t)
            at_prev = sd.alpha(t - sd.skip)
            noise_uc, noise_c = predict_noise_cfg_batch(sd,zt, t, uc, c)
            eps_theta = noise_uc + cfg * (noise_c - noise_uc)
            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

            # proxy at THIS step: reconstruct x_s using the current step's alpha
            alpha_s = at
            x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * x_T.to(sd.dtype)
            nuc_s, nc_s = predict_noise_cfg_batch(sd,x_s, t, uc, c)
            eps_s = nuc_s + cfg * (nc_s - nuc_s)
            # normalized by noise dim (mean over D = 4*64*64 latent elements)
            proxy_b = (eps_ref.float() - eps_s.float()).reshape(B, -1).pow(2).mean(dim=1)  # (B,)
            for b in range(B):
                all_curves[b].append(proxy_b[b].item())

            # advance DDIM trajectory
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

    # batch proxy -> mean / std per step
    import numpy as np
    curve_arr = np.array([all_curves[b] for b in range(B)])  # (B, NFE)
    mean_curve = curve_arr.mean(axis=0)
    std_curve = curve_arr.std(axis=0)

    mid = len(timesteps) // 2
    print(f"  batch (n={B}): proxy[0]={mean_curve[0]:.3f}±{std_curve[0]:.3f}  "
          f"proxy[mid={mid}]={mean_curve[mid]:.3f}±{std_curve[mid]:.3f}  "
          f"proxy[-1]={mean_curve[-1]:.3f}±{std_curve[-1]:.3f}")

    # plot: batch mean ± std (error bar)
    plt.figure(figsize=(10, 6))
    steps = list(range(len(timesteps)))
    plt.errorbar(steps, mean_curve, yerr=std_curve, marker='o', linewidth=1.5,
                 markersize=4, capsize=3, color='tab:blue', ecolor='tab:blue',
                 alpha=0.85, label=f"batch mean ± std (n={B})")

    # mark SNR=1 / SNR=2 reference steps
    snr_of = lambda t: (sd.alpha(t) / (1 - sd.alpha(t))).item()
    s_snr1 = next((i for i, t in enumerate(timesteps) if snr_of(t) >= 1.0), None)
    s_snr2 = next((i for i, t in enumerate(timesteps) if snr_of(t) >= 2.0), None)
    if s_snr1 is not None:
        plt.axvline(s_snr1, color='gray', linestyle='--', alpha=0.6, label=f"SNR=1 (step {s_snr1})")
    if s_snr2 is not None:
        plt.axvline(s_snr2, color='gray', linestyle=':', alpha=0.6, label=f"SNR=2 (step {s_snr2})")

    plt.xlabel("denoising step", fontsize=12)
    plt.ylabel(r"$||\epsilon_{ref} - \epsilon_s||^2\ /\ D$  (D = noise dim)", fontsize=12)
    plt.title("Memorization proxy vs denoising step (memorized prompts)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
