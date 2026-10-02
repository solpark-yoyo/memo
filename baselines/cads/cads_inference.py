#!/usr/bin/env python
"""CADS (Condition-Annealed Diffusion Sampler) — memorization 완화 baseline.

논문: "CADS: Unleashing the Diversity of Diffusion Models through
       Condition-Annealed Sampling" (arXiv 2310.17347)

핵심 아이디어:
  DDIM 각 step에서 conditioning(text embed)에 스케줄에 따라 Gaussian noise 주입.
    ĉ = √γ(t) · c  +  s · √(1−γ(t)) · n,   n ~ N(0,I)
  γ(t) = annealing schedule (초반=0 → 후반=1): 초반에 conditioning 교란 → 다양성↑
  후반 step (세부 묘사)에서는 γ=1 → 원래 conditioning 복원 → 품질 유지.

하이퍼파라미터:
  --tau1       annealing 시작점 (정규화 t_norm ≤ tau1: γ=1, 교란 없음). 기본 0.6
  --tau2       annealing 종료점 (t_norm ≥ tau2: γ=0, 최대 교란).         기본 0.9
  --noise_scale 노이즈 스케일 s (교란 강도). 기본 0.25
  --psi        rescaling 혼합 계수. 기본 1.0
  --rescale    conditioning 정규화 여부 (default: 활성)

t_norm 정의: DDIM step_idx=0(최고 noise)→1, step_idx=NFE-1(data)→0

사용 (ori_memo/ 에서):
    python baselines/cads/cads_inference.py \
        --prompt_dir examples/assets/sdv1_500_mem.txt \
        --model_key ckpt/stable-diffusion-v1-4 \
        --num_samples 10 --num_images_per_prompt 4 \
        --NFE 50 --cfg 7.5 \
        --tau1 0.6 --tau2 0.9 --noise_scale 0.25 \
        --base_seed 42 --device cuda:0 \
        --output_dir workdir/memorization/sd14_base/baselines/cads/.../result
"""

import argparse
import csv
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from munch import munchify
from torchvision.utils import save_image

# ori_memo/ 를 sys.path에 추가 (latent_diffusion import용)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from latent_diffusion import StableDiffusion


# ─────────────────────── CADS core ───────────────────────

def cads_gamma(t_norm: float, tau1: float, tau2: float) -> float:
    """γ(t_norm): piecewise linear annealing schedule.
    t_norm=1 → noise 시작, t_norm=0 → data 종료.
    """
    if t_norm <= tau1:
        return 1.0
    if t_norm >= tau2:
        return 0.0
    return (tau2 - t_norm) / (tau2 - tau1)


def cads_perturb(c: torch.Tensor, gamma: float, noise_scale: float,
                 psi: float = 1.0, rescale: bool = True) -> torch.Tensor:
    """conditioning c에 CADS 노이즈 주입.
    ĉ = √γ · c  +  s · √(1−γ) · n
    """
    if gamma >= 1.0:
        return c  # 교란 없음

    c_mean = c.mean()
    c_std  = c.std().clamp(min=1e-8)
    n = torch.randn_like(c)
    c_noisy = (math.sqrt(gamma) * c
               + noise_scale * math.sqrt(1.0 - gamma) * n)

    if rescale:
        std_noisy = c_noisy.std().clamp(min=1e-8)
        c_scaled = (c_noisy - c_noisy.mean()) / std_noisy * c_std + c_mean
        if not torch.isnan(c_scaled).any():
            c_noisy = psi * c_scaled + (1.0 - psi) * c_noisy

    return c_noisy


# ─────────────────────── twd_gap record ───────────────────────

def _save_twd_gap_record(proxy_traj: list, record_dir: str, p_idx: int) -> None:
    """per-prompt twd_gap mean/std plot + CSV 저장.

    proxy_traj: list of (B,) float32 CPU tensors, length = N_steps
    """
    arr = torch.stack(proxy_traj, dim=0).numpy()   # (N, B)
    N, B = arr.shape
    steps = np.arange(N)
    mean  = arr.mean(axis=1)
    std   = arr.std(axis=1)

    prompt_dir = os.path.join(record_dir, f"img_{p_idx:04d}")
    os.makedirs(prompt_dir, exist_ok=True)

    # ---- plot 1: mean ± std (faint individual lines) ----
    fig, ax = plt.subplots(figsize=(8, 4))
    for b in range(B):
        ax.plot(steps, arr[:, b], alpha=0.25, linewidth=0.8, color="steelblue")
    ax.plot(steps, mean, linewidth=1.8, color="steelblue", label="mean")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="steelblue",
                    label="±std")
    ax.set_xlabel("DDIM step")
    ax.set_ylabel("twd_gap  ‖ε_ref − ε_s‖²/D")
    ax.set_title(f"twd_gap — prompt {p_idx:04d}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(prompt_dir, "twd_gap.png"), dpi=120)
    plt.close(fig)

    # ---- plot 2: per-batch (each sample distinctly colored) ----
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 4))
    for b in range(B):
        label = f"img_{p_idx:04d}_{b:02d}"
        ax.plot(steps, arr[:, b], linewidth=1.5,
                color=cmap(b % 10), label=label)
    ax.set_xlabel("DDIM step")
    ax.set_ylabel("twd_gap  ‖ε_ref − ε_s‖²/D")
    ax.set_title(f"twd_gap per sample — prompt {p_idx:04d}")
    ax.legend(fontsize=8, ncol=max(1, B // 4))
    plt.tight_layout()
    plt.savefig(os.path.join(prompt_dir, "twd_gap_per_sample.png"), dpi=120)
    plt.close(fig)

    # ---- csv ----
    with open(os.path.join(prompt_dir, "twd_gap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean", "std"] + [f"sample_{b:02d}" for b in range(B)])
        for i in range(N):
            w.writerow([i, float(mean[i]), float(std[i])] + arr[i].tolist())


# ─────────────────────── CADS DDIM sampler ───────────────────────

@torch.no_grad()
def cads_sample(sd: StableDiffusion,
                x_T: torch.Tensor,
                uc: torch.Tensor,
                c: torch.Tensor,
                cfg: float,
                tau1: float,
                tau2: float,
                noise_scale: float,
                psi: float,
                rescale: bool,
                base_s_ratio: float = 0.5,
                record_dir: str | None = None,
                p_idx: int = 0) -> torch.Tensor:
    """CADS conditioning perturbation 을 적용한 DDIM sampling.

    base_s_ratio : twd_gap proxy 계산에 사용할 s_target 위치 (0~1).
    record_dir   : 지정 시 record/per_prompt/img_XXXX/ 에 twd_gap 저장.
    Returns decoded image tensor (B, 3, H, W) in [0, 1].
    """
    timesteps = list(sd.scheduler.timesteps)
    N = len(timesteps)
    B = x_T.shape[0]
    zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma

    # twd_gap setup: 고정 reference noise + s_target
    eps_ref = torch.randn_like(x_T)                              # (B,4,64,64) float32
    s_idx    = min(int(N * base_s_ratio), N - 1)
    s_target = timesteps[s_idx]
    at_s     = sd.alpha(s_target)
    proxy_traj: list[torch.Tensor] = []

    for step_idx, t in enumerate(timesteps):
        # t_norm: 1 at first step (noise), 0 at last step (data)
        t_norm = 1.0 - step_idx / max(N - 1, 1)
        gamma  = cads_gamma(t_norm, tau1, tau2)
        c_cads = cads_perturb(c, gamma, noise_scale, psi, rescale)

        at      = sd.alpha(t)
        at_prev = sd.alpha(t - sd.skip)

        noise_uc, noise_c = sd.predict_noise(zt, t, uc, c_cads)
        eps    = noise_uc + cfg * (noise_c - noise_uc)
        x0_hat = (zt - (1 - at).sqrt() * eps) / at.sqrt()
        zt     = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps

        # ── twd_gap proxy (original c, not perturbed c_cads) ──
        x_s = at_s.sqrt() * x0_hat + (1 - at_s).sqrt() * eps_ref.to(sd.dtype)
        _, eps_s = sd.predict_noise(x_s, s_target, uc, c)
        proxy = ((eps_ref.float() - eps_s.float())
                 .reshape(B, -1).pow(2).mean(dim=-1))            # (B,)
        proxy_traj.append(proxy.cpu())

    if record_dir is not None:
        _save_twd_gap_record(proxy_traj, record_dir, p_idx)

    img = (sd.decode(zt) / 2 + 0.5).clamp(0, 1).cpu()
    return img


# ─────────────────────── main ───────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # model / inference
    ap.add_argument("--model_key",  default="ckpt/stable-diffusion-v1-4")
    ap.add_argument("--NFE",        type=int,   default=50)
    ap.add_argument("--cfg",        type=float, default=7.5)
    ap.add_argument("--base_seed",  type=int,   default=42)
    ap.add_argument("--device",     default="cuda:0")
    # data
    ap.add_argument("--prompt_dir",             required=True,
                    help="프롬pt 파일 (.txt, 한 줄 = 한 프롬pt)")
    ap.add_argument("--num_samples",            type=int, default=10,
                    help="사용할 프롬pt 수 (앞에서 N개)")
    ap.add_argument("--num_images_per_prompt",  type=int, default=4)
    # CADS hyperparams
    ap.add_argument("--tau1",        type=float, default=0.6,
                    help="annealing 시작점 (t_norm ≤ tau1: γ=1). 기본 0.6")
    ap.add_argument("--tau2",        type=float, default=0.9,
                    help="annealing 종료점 (t_norm ≥ tau2: γ=0). 기본 0.9")
    ap.add_argument("--noise_scale", type=float, default=0.25,
                    help="노이즈 스케일 s. 기본 0.25")
    ap.add_argument("--psi",         type=float, default=1.0,
                    help="rescaling 혼합 계수. 기본 1.0")
    ap.add_argument("--no_rescale",  action="store_true",
                    help="conditioning rescaling 비활성화")
    # twd_gap record
    ap.add_argument("--base_s_ratio", type=float, default=0.5,
                    help="twd_gap s_target 위치 (0~1). 기본 0.5")
    ap.add_argument("--no_record",   action="store_true",
                    help="twd_gap record 저장 비활성화")
    # output
    ap.add_argument("--output_dir",  required=True)
    args = ap.parse_args()

    device   = torch.device(args.device)
    rescale  = not args.no_rescale
    S        = args.num_images_per_prompt
    record_base = (None if args.no_record
                   else os.path.join(args.output_dir, "record", "per_prompt"))

    # ---- 모델 로드 ----
    sd = StableDiffusion(
        solver_config=munchify({"num_sampling": args.NFE}),
        model_key=args.model_key,
        device=device,
        seed=args.base_seed,
    )
    sd.unet.eval()

    # ---- 프롬pt 로드 ----
    with open(args.prompt_dir, "r") as f:
        prompts = [l.strip() for l in f if l.strip()]
    prompts = prompts[:args.num_samples]

    result_dir = os.path.join(args.output_dir, "result")
    comp_dir   = os.path.join(args.output_dir, "comp")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(comp_dir,   exist_ok=True)

    # ---- 벤치마크 변수 ----
    comp_rows   = []
    total_time  = 0.0
    total_peak  = 0.0

    print(f"[CADS] tau1={args.tau1} tau2={args.tau2} "
          f"noise_scale={args.noise_scale} psi={args.psi} rescale={rescale}")
    print(f"  NFE={args.NFE} cfg={args.cfg} seed={args.base_seed}")
    print(f"  prompts={len(prompts)} × seeds={S} = {len(prompts)*S} images")

    for p_idx, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        # text embed (uc 공용, c per-prompt)
        uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
        uc_b  = uc.repeat(S, 1, 1)   # (S, 77, 768)
        c_b   = c.repeat(S, 1, 1)    # (S, 77, 768)

        # initial noise (seed 결정론적: prompt_idx × S + seed_offset)
        x_T_list = []
        for s_idx in range(S):
            seed = args.base_seed + p_idx * S + s_idx
            gen  = torch.Generator(device=device).manual_seed(seed)
            x_T_list.append(
                torch.randn(1, 4, 64, 64, device=device,
                            dtype=torch.float32, generator=gen)
            )
        x_T = torch.cat(x_T_list, dim=0)   # (S, 4, 64, 64)

        # ---- CADS sampling ----
        imgs = cads_sample(
            sd, x_T, uc_b, c_b, args.cfg,
            args.tau1, args.tau2, args.noise_scale, args.psi, rescale,
            base_s_ratio=args.base_s_ratio,
            record_dir=record_base,
            p_idx=p_idx,
        )   # (S, 3, H, W)

        torch.cuda.synchronize()
        elapsed   = time.perf_counter() - t0
        peak_gb   = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        per_sample = elapsed / S
        total_time += elapsed
        total_peak  = max(total_peak, peak_gb)

        for s_idx in range(S):
            fname = f"img_{p_idx:04d}_{s_idx:02d}.png"
            save_image(imgs[s_idx], os.path.join(result_dir, fname))
            comp_rows.append((p_idx * S + s_idx, per_sample, peak_gb))
            print(f"  [{p_idx:04d}_{s_idx:02d}] → {fname}")

    # ---- comp_metrics.csv ----
    import statistics as _st
    times = [r[1] for r in comp_rows]
    with open(os.path.join(comp_dir, "comp_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "time_per_sample_sec", "peak_vram_GB"])
        w.writerows(comp_rows)
        w.writerow(["mean", _st.mean(times), total_peak])
        w.writerow(["std",  _st.stdev(times) if len(times) > 1 else 0.0, ""])

    print(f"\n[Done] {len(prompts)*S} images → {result_dir}/")
    print(f"  total_time={total_time:.1f}s  peak_vram={total_peak:.2f}GB")


if __name__ == "__main__":
    main()
