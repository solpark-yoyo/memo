"""
optimize_xt_spectral: x_t at init_steps 에서 DDIM chain 을 통해
spectral L2² loss 를 backward 하여 x_t 를 최적화.

loss 갈래 (--loss_type):
  xt  (기존) = x_t 의 compact spectral L2²           (FFT only, UNet backward 없음)
  eps (신규) = x_t → ε_θ(x_t,t) (CFG 결합) 의 compact spectral L2²
               → UNet forward 가 grad chain 에 포함 (x_t 까지 backward)

파이프라인 (run_ini_opti.py 의 optimize_xT 와 동일한 패턴, but):
  - 최적화 대상: x_T → x_t@init_steps (x_T 최적화 아님)
  - loss: memo_proxy → compact spectral L2² (gaussianity)
  - grad: DDIM chain 을 통해 backward (no_grad 아님)

  1. x_T → DDIM forward 0→init_steps (no grad) → x_t 획득
  2. x_t 최적화: update_indices 마다
     a. x_t → DDIM forward (WITH grad) init_steps→t_idx
     b. spectral L2² loss 계산 (xt 또는 eps 갈래)
     c. backward → DDIM chain 타고 x_t 까지 grad
     d. optimizer.step()
  3. x_t(opt) → DDIM continue (no grad) → x_0 → image

Usage:
    python optimize_xt_spectral.py \
        --init_steps 2 --num_steps 3 --gap_steps 1 --lr 0.01 \
        --model_key ckpt/stable-diffusion-v1-4 \
        --num_samples 10 --num_seeds 5 \
        --output_dir workdir/spectral_opt/init=2
    # eps 갈래 (x_t → ε_θ → spectrum energy 최소화):
    python optimize_xt_spectral.py --loss_type eps --lr 0.04 ...
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import csv
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import munchify

from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
from torchvision.utils import save_image


# ===================================================================
#  Staged CFG (run_ini_opti.py 와 동일)
# ===================================================================
def cfg_eff_at(sd, step_idx, cfg):
    """Staged CFG: step_idx < cfg_start_ratio*NFE 구간은 conditional만.
    0.0 = 항상 정상 CFG. cfgsr_cond=True 시 early 구간 cfg=1 (noise_c).
    """
    ratio = getattr(sd, "cfg_start_ratio", 0.0) or 0.0
    if ratio <= 0:
        return cfg
    total = len(sd.scheduler.timesteps)
    if step_idx < int(total * ratio):
        return 1.0 if getattr(sd, "cfgsr_cond", False) else 0.0
    return cfg


# ===================================================================
#  Batch-safe compact spectral L2² (differentiable)
# ===================================================================
def spectral_l2_loss(x_t, block_size=16):
    """x_t (B, C, H, W) → compact spectral → block energy → loss.

    differentiable: FFT → block → energy → mean / B
    proxy = mean_energy / B  (1 = white Gaussian, < 1 = energy 감소)
    """
    SQRT2 = 2.0 ** 0.5
    B_bs = x_t.shape[0]           # batch size
    B = block_size
    xf = x_t.reshape(B_bs, -1).to(torch.float64)    # (B_bs, N)
    f = torch.fft.rfft(xf, norm="ortho")             # (B_bs, N/2+1) complex
    f[:, 0] = (f[:, 0] + 1j * f[:, -1]) / SQRT2     # compact spectral (per sample)
    y = f[:, :-1]                                    # (B_bs, N/2)
    pad_len = (-y.shape[1]) % B
    if pad_len:
        y = torch.nn.functional.pad(y, (0, pad_len))
    y = y.reshape(B_bs, -1, B)                       # (B_bs, P, B)
    energies = y.abs().square().sum(dim=2).real       # (B_bs, P) per-sample block energy
    # print(f"energies: {energies.shape}")
    # print(f"energies: {energies}")
    mean_e = energies.mean()                          # scalar
    return mean_e / float(B)                          # proxy = mean_energy / B


def compact_spectral_l2_stats(x, block_size=16):
    """측정용 (no grad). proxy, std 반환."""
    with torch.no_grad():
        loss = spectral_l2_loss(x, block_size)
    SQRT2 = 2.0 ** 0.5
    B_bs = x.shape[0]; B = block_size
    xf = x.reshape(B_bs, -1).to(torch.float64)
    f = torch.fft.rfft(xf, norm="ortho")
    f[:, 0] = (f[:, 0] + 1j * f[:, -1]) / SQRT2
    y = f[:, :-1]
    pad_len = (-y.shape[1]) % B
    if pad_len:
        y = torch.nn.functional.pad(y, (0, pad_len))
    y = y.reshape(B_bs, -1, B)
    energies = y.abs().square().sum(dim=2).real
    std_e = energies.std().item()
    return loss.item(), std_e


# ===================================================================
#  eps 갈래 loss: x_t → ε_θ(x_t,t) → compact spectral L2²
#  (UNet forward 가 grad chain 에 포함 → x_t 까지 gradient 흐름)
# ===================================================================
def eps_spectral_l2_loss(sd, zt, t, step_idx, uc, c, cfg, block_size=16):
    """cfg_epsilon = ε_uc + cfg_eff·(ε_c − ε_uc) 의 compact spectral L2² loss.

    x_t (fp32 leaf) → fp16 cast → UNet → cfg_epsilon → FFT → block energy → /B.
    differentiable w.r.t. zt: backward 가 UNet chain 을 타고 x_t 까지 전파됨.
    """
    noise_uc, noise_c = sd.predict_noise(zt.to(sd.dtype), t, uc, c)
    cfg_epsilon = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)
    return spectral_l2_loss(cfg_epsilon, block_size)


# ===================================================================
#  DDIM step helper (grad 추적 가능)
# ===================================================================
def ddim_step(sd, zt, t, step_idx, uc, c, cfg):
    """한 DDIM denoising step. zt → x0_hat → zt_next. grad 추적 가능.
    cfg_eff_at 적용: early step 은 conditional만 (staged CFG).
    """
    at = sd.alpha(t)
    at_prev = sd.alpha(t - sd.skip)
    noise_uc, noise_c = sd.predict_noise(zt, t, uc, c)
    eps_theta = noise_uc + cfg_eff_at(sd, step_idx, cfg) * (noise_c - noise_uc)
    x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()
    zt_next = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta
    return zt_next, x0_hat


# ===================================================================
#  Main optimization
# ===================================================================
def run_spectral_opt(sd, prompt, cfg, device,
                      init_steps, num_steps, gap_steps, lr,
                      block_size, num_seeds, seed, loss_type="xt"):
    """prompt 1개에 대해 spectral optimization 수행.

    loss_type:
      'xt'  = spectral L2² on x_t               (FFT only, UNet backward 없음)
      'eps' = spectral L2² on cfg_epsilon(x_t,t)  (x_t → UNet+CFG → ε → FFT)
              → UNet forward 가 grad chain 에 포함 (x_t 까지 backward)

    1. DDIM forward 0→init_steps (no grad) → x_t 획득
    2. x_t 최적화: update_indices 마다 선택 loss minimize
       → detach → DDIM gap step 전진 (no grad)
    3. DDIM continue (no grad) → image

    xt 갈래는 fp32 강제 안 함 (UNet backward 없음 → fp16 유지).
    eps 갈래는 zt_leaf(fp32) → fp16 cast → UNet (fp16) 로 forward/backward.
    spectral loss (FFT) 는 내부적으로 float64 사용 → dtype 무관.
    """
    timesteps = list(sd.scheduler.timesteps)
    uc, c = sd.get_text_embed(null_prompt="", prompt=prompt)
    uc_batch = uc.repeat(num_seeds, 1, 1).to(sd.dtype)
    c_batch = c.repeat(num_seeds, 1, 1).to(sd.dtype)

    # ---- 1. x_T → DDIM forward 0→init_steps (no grad) ----
    x_T = torch.randn(num_seeds, 4, 64, 64, device=device, dtype=torch.float32)
    zt = x_T.to(sd.dtype) * sd.scheduler.init_noise_sigma   # ★ fp16 변환 (DDIM과 동일)

    print(f"  [forward] DDIM 0 → step {init_steps} (no grad)")
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            zt, _ = ddim_step(sd, zt, t, step_idx, uc_batch, c_batch, cfg)
            if step_idx == init_steps:
                break

    # 최적화 전 proxy (선택 loss 갈래 기준)
    with torch.no_grad():
        if loss_type == "eps":
            t_meas = timesteps[min(init_steps, len(timesteps) - 1)]
            proxy_before = eps_spectral_l2_loss(sd, zt, t_meas, init_steps,
                                                uc_batch, c_batch, cfg,
                                                block_size).item()
        else:
            proxy_before, _ = compact_spectral_l2_stats(zt, block_size)
    print(f"    reached step {init_steps}, proxy[{loss_type}] = {proxy_before:.4f}")

    # ---- 2. x_t 최적화 (init_steps ~ init_steps+(num_steps-1)*gap) ----
    update_indices = [init_steps + i * gap_steps for i in range(num_steps)]
    update_indices = [i for i in update_indices if i < len(timesteps)]

    proxy_curve = []

    print(f"  [optimize] loss={loss_type}  updates={update_indices}  "
          f"lr={lr}  (batch={num_seeds})")
    for ui, t_idx in enumerate(update_indices):
        # ① zt 직접 optimize (update 시점의 x_t 에서 loss 1회)
        #    zt_leaf 를 fp32 로: grad 정밀도 + optimizer 안정성
        zt_leaf = zt.detach().to(torch.float32).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([zt_leaf], lr=lr)
        optimizer.zero_grad()

        if loss_type == "eps":
            t_upd = timesteps[t_idx]
            loss = eps_spectral_l2_loss(sd, zt_leaf, t_upd, t_idx,
                                        uc_batch, c_batch, cfg, block_size)
        else:
            loss = spectral_l2_loss(zt_leaf, block_size)
        loss.backward()
        optimizer.step()

        proxy_curve.append(loss.item())
        print(f"    [{ui+1}/{len(update_indices)}] t_idx={t_idx}: "
              f"proxy={loss.item():.6f}")

        # ★ 최적화된 zt_leaf → fp16 변환 → 다음 DDIM forward 시작점
        zt = zt_leaf.detach().to(sd.dtype)
        del zt_leaf, loss
        torch.cuda.empty_cache()

        # ② DDIM step gap_steps 만큼 전진 (no grad) → 다음 update 위치로
        with torch.no_grad():
            for step_idx in range(t_idx + 1, min(t_idx + 1 + gap_steps, len(timesteps))):
                t = timesteps[step_idx]
                zt, x0_hat = ddim_step(sd, zt, t, step_idx,
                                        uc_batch, c_batch, cfg)

    x_t_opt = zt.detach()

    # ---- 3. DDIM continue from current position to end (no grad, fp16 유지) ----
    # zt 는 optimization loop 마지막 gap_steps 까지 이미 전진했음
    # 마지막으로 DDIM step 한 위치 = min(last_t_idx + gap_steps, len(timesteps)-1)
    last_forwarded = min(update_indices[-1] + gap_steps, len(timesteps) - 1)

    zt_final = zt.detach().to(sd.dtype)
    print(f"  [forward] DDIM step {last_forwarded+1} → {len(timesteps)-1} (no grad)")
    with torch.no_grad():
        for step_idx in range(last_forwarded + 1, len(timesteps)):
            t = timesteps[step_idx]
            zt_final, x0_hat = ddim_step(sd, zt_final, t, step_idx,
                                         uc_batch, c_batch, cfg)

    # ---- 4. decode → image ----
    img = (sd.decode(x0_hat) / 2 + 0.5).clamp(0, 1)
    return img, x_t_opt, proxy_curve


# ===================================================================
#  Main
# ===================================================================
def main():
    p = argparse.ArgumentParser(description="x_t spectral optimization (DDIM chain with grad)")
    p.add_argument("--NFE", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--init_steps", type=int, default=2,
                   help="x_t optimization 시작 DDIM step")
    p.add_argument("--num_steps", type=int, default=3,
                   help="gradient update 횟수 (= update_indices 개수)")
    p.add_argument("--gap_steps", type=int, default=1,
                   help="update 간격 (DDIM chain 길이)")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--loss_type", type=str, default="xt", choices=["xt", "eps"],
                   help="compact spectral L2² loss 를 계산할 신호: "
                        "xt = x_t 직접 (기존, FFT only) | "
                        "eps = x_t → ε_θ(x_t,t) (CFG 결합) → spectrum. "
                        "eps 갈래는 UNet forward 가 grad chain 에 포함. default xt")
    p.add_argument("--cfg_start_ratio", type=float, default=0.0,
                   help="staged CFG: early step (step < ratio*NFE) 에 conditional 만 사용")
    p.add_argument("--cfgsr_cond", action="store_true",
                   help="cfgsr 구간을 null(cfg=0) 대신 conditional(cfg=1) 로")
    p.add_argument("--block_size", type=int, default=16)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--prompt_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "cvpr2025_memo_prompt.txt"))
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--model_key", type=str,
                   default=os.path.join(SCRIPT_DIR, "ckpt", "stable-diffusion-v1-4"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "workdir", "spectral_opt"))
    args = p.parse_args()

    device = torch.device(args.device)

    # ---- update_indices 미리 출력 ----
    update_indices = [args.init_steps + i * args.gap_steps for i in range(args.num_steps)]
    update_indices = [i for i in update_indices if i < args.NFE]

    print("=" * 60)
    print("optimize_xt_spectral")
    print(f"  model     : {args.model_key}")
    print(f"  NFE={args.NFE}  CFG={args.cfg}  seed={args.base_seed}")
    print(f"  loss_type = {args.loss_type}  "
          f"({'x_t 직접 (FFT only)' if args.loss_type == 'xt' else 'x_t → ε_θ → spectrum'})")
    print(f"  init_steps={args.init_steps}  num_steps={args.num_steps}  gap={args.gap_steps}")
    print(f"  update_indices = {update_indices}")
    print(f"  lr={args.lr}  block_size={args.block_size}  batch={args.num_seeds}")
    print(f"  prompt_dir={args.prompt_dir}  num_samples={args.num_samples}")
    print("=" * 60)

    solver_config = munchify({"num_sampling": args.NFE})
    sd = StableDiffusion(solver_config=solver_config, model_key=args.model_key,
                         device=device, seed=args.base_seed)
    sd.cfg_start_ratio = args.cfg_start_ratio   # staged CFG
    sd.cfgsr_cond = args.cfgsr_cond             # conditional in early steps

    # ---- Load prompts ----
    with open(args.prompt_dir) as f:
        prompts = [line.strip() for line in f if line.strip()]
    prompts = prompts[:args.num_samples]

    # ★ set_seed AFTER model loading (run_ini_opti.py 와 동일 패턴)
    # → 같은 seed 로 같은 x_T 생성 → DDIM/init_opti/spectral_opt 간 metric 비교 가능
    set_seed(args.base_seed)

    # ---- Output dirs ----
    result_dir = os.path.join(args.output_dir, "result")
    proxy_dir = os.path.join(args.output_dir, "proxy")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(proxy_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        for pr in prompts:
            f.write(pr + "\n")

    # ---- Per-prompt loop ----
    import time
    _t0 = time.perf_counter()

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] \"{prompt[:60]}\"")

        img, _, proxy_curve = run_spectral_opt(
            sd, prompt, args.cfg, device,
            args.init_steps, args.num_steps, args.gap_steps, args.lr,
            args.block_size, args.num_seeds, args.base_seed,
            loss_type=args.loss_type)

        for j in range(args.num_seeds):
            save_image(img[j].float(), os.path.join(result_dir, f"img_{i:04d}_{j:02d}.png"))

        with open(os.path.join(proxy_dir, f"proxy_{i:04d}.csv"), "w") as f:
            f.write("update_step,t_idx,proxy\n")
            for si, v in enumerate(proxy_curve):
                f.write(f"{si},{update_indices[min(si, len(update_indices)-1)]},{v}\n")

        print(f"  → result/img_{i:04d}_*.png  proxy/proxy_{i:04d}.csv")

    _t1 = time.perf_counter()
    print(f"\n[Done] {len(prompts)} prompts | {_t1-_t0:.1f}s")
    print(f"  result → {result_dir}/")
    print(f"  proxy  → {proxy_dir}/")


if __name__ == "__main__":
    main()
