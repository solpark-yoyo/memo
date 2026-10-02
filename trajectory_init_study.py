"""
trajectory_init_study.py — 초기 최적화가 memorized trajectory에 미치는 영향 분석

실험 1 [같은 위치, 다른 opti 횟수]:
  --exp1: x_T에서 init_step을 [1,2,3,4,5,10] 등 다양하게 설정 → noise MSE 변화 → memorized trajectory 추적

실험 2 [다른 위치, 같은 opti 횟수]:
  --exp2: 초기 몇 step(s)만 최적화, 나머지는 표준 denoising → step별 최적화 효과 분석

출력: workdir/exp_main/trajectory/
  - exp1_init_steps_sweep/: init_step별 trajectory curve
  - exp2_step_by_step/: 각 step에서의 최적화 효과

참고: run_ini_opti.py, eps_trajectory.py, manifold.py
환경: conda div_DM, memo/ori_memo/에서 실행
"""

import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import munchify
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image

from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
from eps_trajectory import (
    compact_spectral_block_energies,
    compact_spectral_l2_stats,
    kl_to_standard_normal,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory init_step study")

    # 공통 인자
    parser.add_argument("--model_key", type=str, default="stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./workdir/exp_main/trajectory")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    # 실험 선택
    parser.add_argument("--exp", type=str, choices=["exp1", "exp2"], required=True,
                        help="exp1: init_steps sweep | exp2: step-by-step optimization")

    # Prompt sources (파일에서 읽기)
    parser.add_argument("--prompt_sources", type=str, nargs="+",
                        default=None,
                        help="Prompt sources in format 'type:name=path' (e.g., 'memo:webster=examples/assets/sdv1_500_mem.txt')")
    parser.add_argument("--prompt_memo", type=str, default="An astronaut on the moon",
                        help="memorized prompt (--prompt_sources 없을 때만 사용)")
    parser.add_argument("--prompt_normal", type=str, default="A red car parked on a rainy street",
                        help="normal prompt (--prompt_sources 없을 때만 사용)")

    # Exp1 인자: init_step 수 변화
    parser.add_argument("--init_steps_list", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 10],
                        help="시도할 init_step 값 리스트 (Exp1용)")
    parser.add_argument("--gap_steps", type=int, default=0,
                        help="gap between optimization steps (Exp1용, usually 0)")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="number of optimization steps (Exp1용, usually 1)")
    parser.add_argument("--num_samples_per_init", type=int, default=3,
                        help="각 init_step마다 샘플 수 (Exp1용)")
    parser.add_argument("--num_opt_steps", type=int, default=20,
                        help="각 step에서의 최적화 iteration 수 (Exp1용)")

    # Exp2 인자: n_opti sweep (init_step=0 fixed)
    parser.add_argument("--init_step", type=int, default=0,
                        help="fixed init step index (Exp2용, default 0)")
    parser.add_argument("--nopt_list", type=int, nargs="+",
                        default=[1, 2, 5, 10, 20],
                        help="sweep할 num_opt_steps 리스트 (Exp2용)")
    parser.add_argument("--opti_steps", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5],
                        help="[deprecated] gap_step과 num_step 조합 리스트 (역호환용)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="초기 최적화 learning rate")
    parser.add_argument("--num_opt_steps_per_denoise", type=int, default=20,
                        help="각 denoising step에서의 최적화 횟수 (역호환용)")
    parser.add_argument("--base_s_ratio", type=float, default=0.5,
                        help="memo_proxy에서 s target timestep 위치: s_idx = int(NFE * base_s_ratio)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="기존 결과가 있어도 재실행 (기본은 skip)")

    parser.add_argument("--memo_threshold", type=float, default=0.3,
                        help="memorization threshold (memo_proxy 기준)")
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--save_csv", action="store_true", default=True)

    return parser.parse_args()


def load_memo_prompt_from_sources(prompt_sources):
    """
    Parse prompt_sources and load first memorized (memo) prompt.
    Format: 'type:name=path' (e.g., 'memo:webster=examples/assets/sdv1_500_mem.txt')
    Returns: (source_name, prompt) tuple or (None, None)
    """
    if prompt_sources is None:
        return None, None

    for source in prompt_sources:
        if '=' not in source:
            print(f"Warning: Invalid source format '{source}', expected 'type:name=path'")
            continue

        type_name, path = source.split('=', 1)
        if ':' not in type_name:
            print(f"Warning: Invalid source format '{source}', expected 'type:name=path'")
            continue

        ptype, name = type_name.split(':', 1)

        if ptype != 'memo':
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                prompt = f.readline().strip()
                if prompt:
                    print(f"Loaded memo prompt from {name} ({path}): {prompt[:50]}...")
                    return name, prompt
        except FileNotFoundError:
            print(f"Warning: Prompt file not found: {path}")
            continue

    return None, None


def check_config_complete(cfg_dir, num_prompts, B):
    """
    Check if all output files (latents, twd_gap, images) exist for this config.
    Returns True if config is fully complete → can skip inference.
    """
    cfg_dir = Path(cfg_dir)
    twd_csv = cfg_dir / "record" / "twd_gap" / "twd_gap_raw.csv"
    if not twd_csv.exists():
        return False
    for p_idx in range(num_prompts):
        latents_npz = cfg_dir / "record" / "latents" / f"latents_{p_idx:04d}.npz"
        if not latents_npz.exists():
            return False
        for b in range(B):
            img_path = cfg_dir / "imgs" / f"img_{p_idx:04d}_{b:02d}.png"
            if not img_path.exists():
                return False
    return True


def load_twd_gap_from_csv(cfg_dir, num_prompts, B, NFE):
    """
    Load twd_gap values from saved twd_gap_raw.csv.
    Returns numpy array of shape (num_prompts, NFE, B).
    """
    cfg_dir = Path(cfg_dir)
    twd_csv = cfg_dir / "record" / "twd_gap" / "twd_gap_raw.csv"
    twd_gaps = np.zeros((num_prompts, NFE, B), dtype=np.float32)
    with open(twd_csv, "r") as f:
        _ = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2 + NFE:
                continue
            p_idx = int(parts[0])
            s_idx = int(parts[1])
            values = [float(x) for x in parts[2:2 + NFE]]
            if p_idx < num_prompts and s_idx < B:
                twd_gaps[p_idx, :, s_idx] = np.array(values, dtype=np.float32)
    return twd_gaps


def compute_memo_proxy(sd, zt, t, s_target, alpha_s, epsilon_ref, uc_b, c_b, cfg,
                       s_idx=None, step_idx=None):
    """
    memo_proxy chain: x_t → x_0|t (Tweedie) → x_s → eps_s → ||eps_ref - eps_s||^2

    Args:
        sd: StableDiffusion instance
        zt: latent x_t (B, 4, 64, 64)
        t: current timestep
        s_target: target timestep s
        alpha_s: sd.alpha(s_target)
        epsilon_ref: reference noise (B, 4, 64, 64) — x_T와 독립
        uc_b, c_b: batched embeddings
        cfg: guidance scale
        s_idx, step_idx: optional (for cfg_eff_at 계산에 필요할 경우)

    Returns:
        memo_proxy: (B,) per-sample proxy
        x0_hat: (B, 4, 64, 64) Tweedie estimation (for align loss if needed)
    """
    at = sd.alpha(t)

    # UNet at t
    noise_uc, noise_c = sd.predict_noise(zt, t, uc_b, c_b)
    eps_theta = noise_uc + cfg * (noise_c - noise_uc)

    # Tweedie x_0|t
    x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

    # Forward to x_s
    x_s = alpha_s.sqrt().to(sd.dtype) * x0_hat + (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype)

    # UNet at s
    noise_uc_s, noise_c_s = sd.predict_noise(x_s, s_target, uc_b, c_b)
    eps_s = noise_uc_s + cfg * (noise_c_s - noise_uc_s)

    # memo_proxy = ||eps_ref - eps_s||^2 / D (per-sample)
    B = eps_s.shape[0]
    memo_proxy = (epsilon_ref.to(sd.dtype) - eps_s).reshape(B, -1).pow(2).mean(-1)

    return memo_proxy, x0_hat


# ===================================================================
#  Experiment 1: init_step sweep
# ===================================================================

def exp1_init_steps_sweep(args):
    """
    Exp1: init_step sweep (gap_steps/num_steps 고정, init_step 다양하게 설정)
    각 init_step에서의 memorized trajectory 비교
    """

    print("\n" + "="*70)
    print("EXPERIMENT 1: init_steps sweep")
    print("="*70)

    set_seed(args.seed)
    device = torch.device(args.device)

    # Base path: trajectory/{model}/ddim/CFG={cfg}_NFE={NFE}/batch={B}/seed={seed}/
    # Structure: base_path/{curve, save, exp/exp1}
    model_name = args.model_key.split("/")[-1]
    B = args.num_samples_per_init

    base_path = (
        Path(args.output_dir) /
        model_name / "ddim" /
        f"CFG={args.cfg_guidance}_NFE={args.num_inference_steps}" /
        f"batch={B}" /
        f"seed={args.seed}"
    )
    base_path.mkdir(parents=True, exist_ok=True)

    # Shared curve/, save/ (across experiments — config로 자동 구분)
    curve_dir = base_path / "curve"
    save_root = base_path / "save"
    curve_dir.mkdir(exist_ok=True)
    save_root.mkdir(exist_ok=True)

    # Exp-specific metadata dir: exp/exp1/
    exp_dir = base_path / "exp" / "exp1"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # SD 모델 로드
    solver_config = munchify({"num_sampling": args.num_inference_steps})
    sd = StableDiffusion(
        solver_config=solver_config,
        model_key=args.model_key,
        device=device,
        pipe_dtype=torch.float16,
    )
    # VRAM 절약: gradient checkpointing (matches run_ini_opti.py:1933)
    sd.unet.enable_gradient_checkpointing()

    # Load memo prompt (source_name, prompt_text)
    if args.prompt_sources:
        print(f"\nLoading memo prompt from sources...")
        source_name, memo_prompt = load_memo_prompt_from_sources(args.prompt_sources)
        if memo_prompt is None:
            source_name = "default"
            memo_prompt = args.prompt_memo
    else:
        source_name = "default"
        memo_prompt = args.prompt_memo

    prompts_to_test = [
        # (prompt_type, source_name, prompt_text, prompt_idx)
        ("memo", source_name, memo_prompt, 0),
    ]

    results_by_config = {}
    index_entries = []

    gap_steps = args.gap_steps
    num_steps = args.num_steps

    for init_step in args.init_steps_list:
        print(f"\n  init_step={init_step}...")

        # Config dir per (prompt_type, source_name, gap, nstep, init, nopt, lr)
        # NOTE: We collect image/latent/twd_gap for ALL prompts under same config
        # For Exp1, only 1 prompt currently, but structure supports multiple
        config_dir_per_source = {}
        for _, sname, _, _ in prompts_to_test:
            if sname not in config_dir_per_source:
                cfg_dir = (
                    save_root / "memo" / sname /
                    f"gap={gap_steps}" /
                    f"nstep={num_steps}" /
                    f"init={init_step}" /
                    f"nopt={args.num_opt_steps}" /
                    f"lr={args.lr}"
                )
                (cfg_dir / "imgs").mkdir(parents=True, exist_ok=True)
                (cfg_dir / "record" / "latents").mkdir(parents=True, exist_ok=True)
                (cfg_dir / "record" / "twd_gap").mkdir(parents=True, exist_ok=True)
                (cfg_dir / "prompts").mkdir(parents=True, exist_ok=True)
                config_dir_per_source[sname] = cfg_dir

        # Aggregate storage across all prompts in this config
        all_trajs = []  # for curve plot
        # per-source aggregation: {source: {prompt_idx: (latents, twd_gaps, prompt_text)}}
        source_agg = {sname: [] for sname in config_dir_per_source.keys()}

        # ---- Check if this config is already complete (manifold.py style reuse) ----
        num_prompts_here = len(prompts_to_test)
        all_complete = all(
            check_config_complete(config_dir_per_source[sname], num_prompts_here, B)
            for sname in config_dir_per_source
        )
        if all_complete and not args.overwrite:
            print(f"  [skip] init={init_step}: all files exist, loading from cache...")
            for sname in config_dir_per_source:
                cfg_dir = config_dir_per_source[sname]
                twd_arr = load_twd_gap_from_csv(
                    cfg_dir, num_prompts_here, B, args.num_inference_steps
                )  # (num_prompts, NFE, B)
                # Use last prompt's twd_gap for curve (single-prompt case → prompt 0)
                twd_gaps_per_sample = twd_arr[-1]  # (NFE, B)
                all_trajs.append((init_step, twd_gaps_per_sample.copy()))
                index_entries.append({
                    "init": init_step,
                    "prompt": "cache",
                    "src": sname,
                    "gap": gap_steps,
                    "nstep": num_steps,
                    "nopt": args.num_opt_steps,
                    "lr": args.lr,
                })
            # Aggregate for curve (skip inference block below)
            if all_trajs:
                _, latest = all_trajs[-1]
                traj_mean = latest.mean(axis=1)
                traj_std = latest.std(axis=1)
                config_key = f"init={init_step}"
                results_by_config[config_key] = {
                    "traj_mean": traj_mean,
                    "traj_std": traj_std,
                    "noise_mse_mean": float(traj_mean.mean()),
                }
            continue

        for prompt_type, sname, prompt_text, prompt_idx in prompts_to_test:
            print(f"    [prompt_idx={prompt_idx}] {prompt_text[:50]}...")

            set_seed(args.seed)

            # Get embeddings and repeat for batch
            uc, c = sd.get_text_embed(null_prompt="", prompt=prompt_text)
            uc = uc.to(sd.dtype)
            c = c.to(sd.dtype)
            uc_b = uc.repeat(B, 1, 1)
            c_b = c.repeat(B, 1, 1)

            # Initialize x_T batch (different seeds per sample)
            x_T_list = []
            for sample_idx in range(B):
                set_seed(args.seed + sample_idx)
                x_T_i = torch.randn(1, 4, 64, 64, device=device, dtype=sd.dtype)
                x_T_list.append(x_T_i)
            x_T_batch = torch.cat(x_T_list, dim=0)

            # Update indices: [init + i*gap for i in range(nstep)]
            timesteps = list(sd.scheduler.timesteps)
            update_indices = [init_step + i * gap_steps for i in range(num_steps)]
            update_indices = [i for i in update_indices if i < len(timesteps)]

            # memo_proxy target: s_idx = int(NFE * base_s_ratio)
            s_idx = int(len(timesteps) * args.base_s_ratio)
            s_target = timesteps[s_idx]
            alpha_s = sd.alpha(s_target)

            # eps_ref: fresh random noise, x_T와 독립
            set_seed(args.seed + 10000)
            epsilon_ref = torch.randn(B, 4, 64, 64, device=device, dtype=sd.dtype)

            # DDIM denoising with in-loop memo_proxy optimization at update_indices
            latents_array = np.zeros((args.num_inference_steps, B, 4, 64, 64), dtype=np.float16)
            twd_gaps_per_sample = np.zeros((args.num_inference_steps, B), dtype=np.float32)
            trajs = [[] for _ in range(B)]

            # KEY: initial latent = x_T * init_noise_sigma (matches run_ini_opti.py:1489)
            zt = x_T_batch.to(sd.dtype) * sd.scheduler.init_noise_sigma
            z0t = None  # Tweedie x_0 estimation, decoded at the end

            # DIAGNOSTIC: initial zt stats
            print(f"[DEBUG init={init_step}] init_noise_sigma={sd.scheduler.init_noise_sigma}, "
                  f"x_T_batch dtype={x_T_batch.dtype}, "
                  f"initial zt: dtype={zt.dtype}, min={zt.min().item():.4f}, "
                  f"max={zt.max().item():.4f}, mean={zt.mean().item():.4f}, "
                  f"has_nan={torch.isnan(zt).any().item()}")
            print(f"[DEBUG init={init_step}] update_indices={update_indices}, "
                  f"s_idx={s_idx}, s_target={s_target}, alpha_s={alpha_s.item() if hasattr(alpha_s, 'item') else alpha_s}, "
                  f"num_opt_steps={args.num_opt_steps}")

            pbar_ddim = tqdm(enumerate(timesteps), total=len(timesteps),
                             desc=f"DDIM (init={init_step}, gap={gap_steps}, nstep={num_steps})",
                             leave=False)
            for step_idx, t in pbar_ddim:
                # Record latents BEFORE optimization
                latents_array[step_idx] = zt.cpu().numpy().astype(np.float16)

                latent_norms = zt.view(B, -1).norm(dim=1)
                for b in range(B):
                    trajs[b].append(latent_norms[b].item())

                # In-loop optimization at update indices (fp32 for stability)
                # optimize the noisy state x_t directly using memo_proxy chain
                if step_idx in update_indices and args.num_opt_steps > 0:
                    # ---- fp32 UNet for stable gradient (matches run_ini_opti.py) ----
                    _orig_sd_dtype = sd.dtype
                    sd.unet.float()
                    sd.dtype = torch.float32
                    uc_b_fp = uc_b.float()
                    c_b_fp = c_b.float()
                    ep_ref_fp = epsilon_ref.float()

                    zt_opt = zt.clone().detach().float().requires_grad_(True)
                    opt_inner = torch.optim.Adam([zt_opt], lr=args.lr)
                    pbar_inner = tqdm(range(args.num_opt_steps),
                                      desc=f"step {step_idx} memo_proxy opt (fp32)",
                                      leave=False)
                    for _ in pbar_inner:
                        with torch.enable_grad():
                            mp, _ = compute_memo_proxy(
                                sd, zt_opt, t, s_target, alpha_s,
                                ep_ref_fp, uc_b_fp, c_b_fp, args.cfg_guidance
                            )
                            loss = mp.mean()
                            opt_inner.zero_grad()
                            loss.backward()
                            opt_inner.step()
                        pbar_inner.set_postfix({"memo_proxy": f"{mp.mean().item():.6f}"})

                    # ---- restore fp16 ----
                    sd.unet.half()
                    sd.dtype = _orig_sd_dtype
                    zt = zt_opt.detach().to(sd.dtype)

                # DDIM step (manual, matches run_ini_opti.py:1490-1496)
                with torch.no_grad():
                    at = sd.alpha(t)
                    at_prev = sd.alpha(t - sd.skip)

                    noise_uc, noise_c = sd.predict_noise(zt, t, uc_b, c_b)
                    noise_pred = noise_uc + args.cfg_guidance * (noise_c - noise_uc)

                    # Tweedie x_0|t
                    z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

                    # Metric: memo_proxy (using current zt, reuse noise_pred)
                    x_s_metric = alpha_s.sqrt().to(sd.dtype) * z0t + \
                                 (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype)
                    n_uc_s, n_c_s = sd.predict_noise(x_s_metric, s_target, uc_b, c_b)
                    eps_s_metric = n_uc_s + args.cfg_guidance * (n_c_s - n_uc_s)
                    mp_metric = (epsilon_ref.to(sd.dtype) - eps_s_metric).reshape(B, -1).pow(2).mean(-1)
                    twd_gaps_per_sample[step_idx] = mp_metric.cpu().numpy()

                    # DDIM update: zt = sqrt(at_prev) * z0t + sqrt(1-at_prev) * noise_pred
                    zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

                pbar_ddim.set_postfix({
                    "memo_proxy": f"{mp_metric.mean().item():.6f}"
                })

            # x_T_opt for align loss reference (kept for compatibility)
            x_T_opt = x_T_batch.detach()

            # Decode final latent using SD's built-in decode_latents (dtype-safe)
            # DIAGNOSTIC: check z0t stats
            print(f"[DEBUG init={init_step}] z0t stats: shape={z0t.shape}, dtype={z0t.dtype}, "
                  f"min={z0t.min().item():.4f}, max={z0t.max().item():.4f}, "
                  f"mean={z0t.mean().item():.4f}, has_nan={torch.isnan(z0t).any().item()}, "
                  f"has_inf={torch.isinf(z0t).any().item()}")

            # For the last step, do not add noise → decode z0t (Tweedie x_0), NOT zt
            # (matches run_ini_opti.py:1622-1623: img = sd.decode(x0_hat); (img/2+0.5).clamp(0,1))
            with torch.no_grad():
                imgs = sd.decode(z0t)
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)

            # DIAGNOSTIC: check imgs stats
            print(f"[DEBUG init={init_step}] imgs stats: shape={imgs.shape}, dtype={imgs.dtype}, "
                  f"min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, "
                  f"mean={imgs.mean().item():.4f}")

            cfg_dir = config_dir_per_source[sname]

            # Save images using torchvision.save_image (matches run_ini_opti.py)
            for b in range(B):
                img_path = cfg_dir / "imgs" / f"img_{prompt_idx:04d}_{b:02d}.png"
                save_image(imgs[b].float().cpu(), str(img_path))

            # Save prompts file (append if multi-prompt)
            prompts_file = cfg_dir / "prompts" / "prompts.txt"
            with open(prompts_file, "a") as f:
                f.write(f"{prompt_idx}\t{prompt_text}\n")

            # Accumulate for combined save
            source_agg[sname].append((prompt_idx, latents_array, twd_gaps_per_sample))

            # Store twd_gap per sample for curve plot (shape: (NFE, B))
            all_trajs.append((init_step, twd_gaps_per_sample.copy()))
            index_entries.append({
                "init": init_step,
                "prompt": prompt_idx,
                "src": sname,
                "gap": gap_steps,
                "nstep": num_steps,
                "nopt": args.num_opt_steps,
                "lr": args.lr,
            })

        # Save aggregated latents and twd_gap CSV per source
        for sname, entries in source_agg.items():
            if not entries:
                continue
            cfg_dir = config_dir_per_source[sname]

            # Save latents.npz per prompt: shape (NFE, B, 4, 64, 64)
            for p_idx, lat_arr, _ in entries:
                latents_npz = cfg_dir / "record" / "latents" / f"latents_{p_idx:04d}.npz"
                np.savez_compressed(latents_npz, latents=lat_arr)

            # Save twd_gap_raw.csv
            # Format: prompt_idx,seed_idx,step_0,step_1,...,step_{NFE-1}
            twd_csv = cfg_dir / "record" / "twd_gap" / "twd_gap_raw.csv"
            NFE = args.num_inference_steps
            with open(twd_csv, "w") as f:
                header = "prompt_idx,seed_idx," + ",".join([f"step_{i}" for i in range(NFE)])
                f.write(header + "\n")
                for p_idx, _, gaps in entries:
                    for b in range(B):
                        row_vals = ",".join([f"{gaps[s, b]:.6f}" for s in range(NFE)])
                        f.write(f"{p_idx},{b},{row_vals}\n")

        # Aggregate twd_gap for curve: latest shape (NFE, B) → batch mean/std per step
        if all_trajs:
            _, latest = all_trajs[-1]  # (NFE, B)
            traj_mean = latest.mean(axis=1)  # (NFE,) — batch mean
            traj_std = latest.std(axis=1)    # (NFE,) — batch std
            noise_mse_mean = float(traj_mean.mean())

            config_key = f"init={init_step}"
            results_by_config[config_key] = {
                "traj_mean": traj_mean,
                "traj_std": traj_std,
                "noise_mse_mean": noise_mse_mean,
            }

    # Plot: init_step별 twd_gap(noise MSE) trajectory 오버레이 (batch mean ± std)
    fig, ax = plt.subplots(figsize=(12, 6))
    for config_key, data in sorted(results_by_config.items()):
        traj = data["traj_mean"]
        std = data["traj_std"]
        steps = np.arange(len(traj))
        ax.plot(steps, traj, label=config_key, marker="o", markersize=3)
        ax.fill_between(steps, traj - std, traj + std, alpha=0.15)

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel(r"Tweedie Gap (noise MSE)  $\|\epsilon_{ref} - \epsilon_s\|^2 / D$")
    ax.set_title(f"Exp1: init_step sweep (gap={gap_steps}, nstep={num_steps}, batch n={B})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = curve_dir / "exp1_init_sweep.png"
    plt.savefig(fig_path, dpi=150)
    print(f"  Saved curve: {fig_path}")
    plt.close()

    # Save index CSV (exp-specific metadata)
    index_path = exp_dir / "index.csv"
    if index_entries:
        fieldnames = ["init", "prompt", "src", "gap", "nstep", "nopt", "lr"]
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in index_entries:
                writer.writerow(entry)
        print(f"  Saved index: {index_path}")

    # Summary CSV (exp-specific metadata)
    summary_path = exp_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "noise_MSE"])
        for config_key in sorted(results_by_config.keys()):
            noise_mse = results_by_config[config_key]["noise_mse_mean"]
            writer.writerow([config_key, f"{noise_mse:.6f}"])

    print(f"\n[Done] Exp1 complete. Output: {exp_dir}")
    return exp_dir


# ===================================================================
#  Experiment 2: step-by-step optimization
# ===================================================================

def exp2_step_by_step_optimization(args):
    """
    Exp2: n_opti sweep (init_step, gap_steps, num_steps 고정)
    각 nopt에서의 memorized trajectory 비교
    """

    print("\n" + "="*70)
    print("EXPERIMENT 2: n_opti sweep (init_step fixed)")
    print("="*70)

    set_seed(args.seed)
    device = torch.device(args.device)

    # Base path: trajectory/{model}/ddim/CFG={cfg}_NFE={NFE}/batch={B}/seed={seed}/
    # Structure: base_path/{curve, save, exp/exp2}
    model_name = args.model_key.split("/")[-1]
    B = args.num_samples_per_init

    base_path = (
        Path(args.output_dir) /
        model_name / "ddim" /
        f"CFG={args.cfg_guidance}_NFE={args.num_inference_steps}" /
        f"batch={B}" /
        f"seed={args.seed}"
    )
    base_path.mkdir(parents=True, exist_ok=True)

    # Shared curve/, save/ (across experiments — config로 자동 구분)
    curve_dir = base_path / "curve"
    save_root = base_path / "save"
    curve_dir.mkdir(exist_ok=True)
    save_root.mkdir(exist_ok=True)

    # Exp-specific metadata dir: exp/exp2/
    exp_dir = base_path / "exp" / "exp2"
    exp_dir.mkdir(parents=True, exist_ok=True)

    solver_config = munchify({"num_sampling": args.num_inference_steps})
    sd = StableDiffusion(
        solver_config=solver_config,
        model_key=args.model_key,
        device=device,
        pipe_dtype=torch.float16,
    )
    # VRAM 절약: gradient checkpointing (matches run_ini_opti.py:1933)
    sd.unet.enable_gradient_checkpointing()

    # Load memo prompt (source_name, prompt_text)
    if args.prompt_sources:
        print(f"\nLoading memo prompt from sources...")
        source_name, memo_prompt = load_memo_prompt_from_sources(args.prompt_sources)
        if memo_prompt is None:
            source_name = "default"
            memo_prompt = args.prompt_memo
    else:
        source_name = "default"
        memo_prompt = args.prompt_memo

    prompts_to_test = [
        ("memo", source_name, memo_prompt, 0),
    ]

    results_by_config = {}
    index_entries = []

    init_step = args.init_step
    gap_steps = args.gap_steps
    num_steps = args.num_steps
    lr = args.lr

    for nopt in args.nopt_list:
        print(f"\n  nopt={nopt}...")

        # Update indices: [init + i*gap for i in range(nstep)]
        timesteps = list(sd.scheduler.timesteps)
        update_indices = [init_step + i * gap_steps for i in range(num_steps)]
        update_indices = [i for i in update_indices if i < len(timesteps)]

        # memo_proxy target: s_idx = int(NFE * base_s_ratio)
        s_idx = int(len(timesteps) * args.base_s_ratio)
        s_target = timesteps[s_idx]
        alpha_s = sd.alpha(s_target)

        # Config dir per source
        config_dir_per_source = {}
        for _, sname, _, _ in prompts_to_test:
            if sname not in config_dir_per_source:
                cfg_dir = (
                    save_root / "memo" / sname /
                    f"gap={gap_steps}" /
                    f"nstep={num_steps}" /
                    f"init={init_step}" /
                    f"nopt={nopt}" /
                    f"lr={lr}"
                )
                (cfg_dir / "imgs").mkdir(parents=True, exist_ok=True)
                (cfg_dir / "record" / "latents").mkdir(parents=True, exist_ok=True)
                (cfg_dir / "record" / "twd_gap").mkdir(parents=True, exist_ok=True)
                (cfg_dir / "prompts").mkdir(parents=True, exist_ok=True)
                config_dir_per_source[sname] = cfg_dir

        source_agg = {sname: [] for sname in config_dir_per_source.keys()}
        all_trajs = []

        # ---- Check if this config is already complete (manifold.py style reuse) ----
        num_prompts_here = len(prompts_to_test)
        all_complete = all(
            check_config_complete(config_dir_per_source[sname], num_prompts_here, B)
            for sname in config_dir_per_source
        )
        if all_complete and not args.overwrite:
            print(f"  [skip] nopt={nopt}: all files exist, loading from cache...")
            for sname in config_dir_per_source:
                cfg_dir = config_dir_per_source[sname]
                twd_arr = load_twd_gap_from_csv(
                    cfg_dir, num_prompts_here, B, args.num_inference_steps
                )  # (num_prompts, NFE, B)
                twd_gaps_per_sample = twd_arr[-1]  # (NFE, B)
                all_trajs.append((nopt, twd_gaps_per_sample.copy()))
                index_entries.append({
                    "nopt": nopt,
                    "init": init_step,
                    "gap": gap_steps,
                    "nstep": num_steps,
                    "prompt": "cache",
                    "src": sname,
                    "lr": lr,
                })
            if all_trajs:
                _, latest = all_trajs[-1]
                traj_mean = latest.mean(axis=1)
                traj_std = latest.std(axis=1)
                config_key = f"nopt={nopt}"
                results_by_config[config_key] = {
                    "traj_mean": traj_mean,
                    "traj_std": traj_std,
                }
            continue

        for prompt_type, sname, prompt_text, prompt_idx in prompts_to_test:
            print(f"    [prompt_idx={prompt_idx}] {prompt_text[:50]}...")

            set_seed(args.seed)

            # Embeddings
            uc, c = sd.get_text_embed(null_prompt="", prompt=prompt_text)
            uc = uc.to(sd.dtype)
            c = c.to(sd.dtype)
            uc_b = uc.repeat(B, 1, 1)
            c_b = c.repeat(B, 1, 1)

            # Initialize x_T batch
            x_T_list = []
            for sample_idx in range(B):
                set_seed(args.seed + sample_idx)
                x_T_i = torch.randn(1, 4, 64, 64, device=device, dtype=sd.dtype)
                x_T_list.append(x_T_i)
            x_T_batch = torch.cat(x_T_list, dim=0).detach()

            # eps_ref: fresh random noise, x_T와 독립
            set_seed(args.seed + 10000)
            epsilon_ref = torch.randn(B, 4, 64, 64, device=device, dtype=sd.dtype)

            # DDIM denoising with in-loop memo_proxy optimization at update_indices
            latents_array = np.zeros((args.num_inference_steps, B, 4, 64, 64), dtype=np.float16)
            twd_gaps_per_sample = np.zeros((args.num_inference_steps, B), dtype=np.float32)
            trajs = [[] for _ in range(B)]

            # KEY: initial latent = x_T * init_noise_sigma (matches run_ini_opti.py:1489)
            zt = x_T_batch.to(sd.dtype) * sd.scheduler.init_noise_sigma
            z0t = None  # Tweedie x_0 for final decode

            pbar_ddim = tqdm(enumerate(timesteps), total=len(timesteps),
                             desc=f"DDIM (nopt={nopt}, init={init_step})",
                             leave=False)
            for step_idx, t in pbar_ddim:
                # Record latents BEFORE optimization
                latents_array[step_idx] = zt.cpu().numpy().astype(np.float16)

                latent_norms = zt.view(B, -1).norm(dim=1)
                for b in range(B):
                    trajs[b].append(latent_norms[b].item())

                # In-loop optimization at update indices (fp32 for stability)
                if step_idx in update_indices and nopt > 0:
                    # ---- fp32 UNet for stable gradient ----
                    _orig_sd_dtype = sd.dtype
                    sd.unet.float()
                    sd.dtype = torch.float32
                    uc_b_fp = uc_b.float()
                    c_b_fp = c_b.float()
                    ep_ref_fp = epsilon_ref.float()

                    zt_opt = zt.clone().detach().float().requires_grad_(True)
                    opt_inner = torch.optim.Adam([zt_opt], lr=lr)
                    pbar_inner = tqdm(range(nopt),
                                      desc=f"step {step_idx} memo_proxy opt (fp32, nopt={nopt})",
                                      leave=False)
                    for _ in pbar_inner:
                        with torch.enable_grad():
                            mp, _ = compute_memo_proxy(
                                sd, zt_opt, t, s_target, alpha_s,
                                ep_ref_fp, uc_b_fp, c_b_fp, args.cfg_guidance
                            )
                            loss = mp.mean()
                            opt_inner.zero_grad()
                            loss.backward()
                            opt_inner.step()
                        pbar_inner.set_postfix({"memo_proxy": f"{mp.mean().item():.6f}"})

                    # ---- restore fp16 ----
                    sd.unet.half()
                    sd.dtype = _orig_sd_dtype
                    zt = zt_opt.detach().to(sd.dtype)

                # DDIM step (manual, matches run_ini_opti.py:1490-1496)
                with torch.no_grad():
                    at = sd.alpha(t)
                    at_prev = sd.alpha(t - sd.skip)

                    noise_uc, noise_c = sd.predict_noise(zt, t, uc_b, c_b)
                    noise_pred = noise_uc + args.cfg_guidance * (noise_c - noise_uc)

                    # Tweedie x_0|t
                    z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()

                    # Metric: memo_proxy (using current zt)
                    x_s_metric = alpha_s.sqrt().to(sd.dtype) * z0t + \
                                 (1 - alpha_s).sqrt().to(sd.dtype) * epsilon_ref.to(sd.dtype)
                    n_uc_s, n_c_s = sd.predict_noise(x_s_metric, s_target, uc_b, c_b)
                    eps_s_metric = n_uc_s + args.cfg_guidance * (n_c_s - n_uc_s)
                    mp_metric = (epsilon_ref.to(sd.dtype) - eps_s_metric).reshape(B, -1).pow(2).mean(-1)
                    twd_gaps_per_sample[step_idx] = mp_metric.cpu().numpy()

                    # DDIM update
                    zt = at_prev.sqrt() * z0t + (1 - at_prev).sqrt() * noise_pred

                pbar_ddim.set_postfix({
                    "memo_proxy": f"{mp_metric.mean().item():.6f}"
                })

            x_T_opt = x_T_batch.detach()

            # DIAGNOSTIC: check z0t stats
            print(f"[DEBUG nopt={nopt}] z0t stats: "
                  f"shape={z0t.shape}, dtype={z0t.dtype}, "
                  f"min={z0t.min().item():.4f}, max={z0t.max().item():.4f}, "
                  f"mean={z0t.mean().item():.4f}, "
                  f"has_nan={torch.isnan(z0t).any().item()}, "
                  f"has_inf={torch.isinf(z0t).any().item()}")

            # Decode final z0t (fully denoised x_0 estimation), NOT zt
            # (matches run_ini_opti.py:1622-1623: img = sd.decode(x0_hat); (img/2+0.5).clamp(0,1))
            with torch.no_grad():
                imgs = sd.decode(z0t)
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)

            # DIAGNOSTIC: check imgs stats
            print(f"[DEBUG nopt={nopt}] imgs stats: "
                  f"shape={imgs.shape}, dtype={imgs.dtype}, "
                  f"min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, "
                  f"mean={imgs.mean().item():.4f}")

            cfg_dir = config_dir_per_source[sname]

            # Save images using torchvision.save_image (matches run_ini_opti.py)
            for b in range(B):
                img_path = cfg_dir / "imgs" / f"img_{prompt_idx:04d}_{b:02d}.png"
                save_image(imgs[b].float().cpu(), str(img_path))

            # Save prompts file
            prompts_file = cfg_dir / "prompts" / "prompts.txt"
            with open(prompts_file, "a") as f:
                f.write(f"{prompt_idx}\t{prompt_text}\n")

            source_agg[sname].append((prompt_idx, latents_array, twd_gaps_per_sample))
            # Store twd_gap per sample for curve plot (shape: (NFE, B))
            all_trajs.append((nopt, twd_gaps_per_sample.copy()))
            index_entries.append({
                "nopt": nopt,
                "init": init_step,
                "gap": gap_steps,
                "nstep": num_steps,
                "prompt": prompt_idx,
                "src": sname,
                "lr": lr,
            })

        # Save aggregated latents and twd_gap CSV per source
        for sname, entries in source_agg.items():
            if not entries:
                continue
            cfg_dir = config_dir_per_source[sname]

            for p_idx, lat_arr, _ in entries:
                latents_npz = cfg_dir / "record" / "latents" / f"latents_{p_idx:04d}.npz"
                np.savez_compressed(latents_npz, latents=lat_arr)

            twd_csv = cfg_dir / "record" / "twd_gap" / "twd_gap_raw.csv"
            NFE = args.num_inference_steps
            with open(twd_csv, "w") as f:
                header = "prompt_idx,seed_idx," + ",".join([f"step_{i}" for i in range(NFE)])
                f.write(header + "\n")
                for p_idx, _, gaps in entries:
                    for b in range(B):
                        row_vals = ",".join([f"{gaps[s, b]:.6f}" for s in range(NFE)])
                        f.write(f"{p_idx},{b},{row_vals}\n")

        # Aggregate twd_gap for curve: latest shape (NFE, B) → batch mean/std per step
        if all_trajs:
            _, latest = all_trajs[-1]  # (NFE, B)
            traj_mean = latest.mean(axis=1)  # (NFE,) — batch mean
            traj_std = latest.std(axis=1)    # (NFE,) — batch std

            config_key = f"nopt={nopt}"
            results_by_config[config_key] = {
                "traj_mean": traj_mean,
                "traj_std": traj_std,
            }

    # Plot: nopt별 twd_gap(noise MSE) trajectory 오버레이 (batch mean ± std)
    fig, ax = plt.subplots(figsize=(12, 6))
    for config_key, data in sorted(results_by_config.items()):
        traj = data["traj_mean"]
        std = data["traj_std"]
        steps = np.arange(len(traj))
        ax.plot(steps, traj, label=config_key, marker="o", markersize=3)
        ax.fill_between(steps, traj - std, traj + std, alpha=0.15)

    ax.set_xlabel("Denoising Step")
    ax.set_ylabel(r"Tweedie Gap (noise MSE)  $\|\epsilon_{ref} - \epsilon_s\|^2 / D$")
    ax.set_title(f"Exp2: n_opti sweep (init={init_step}, gap={gap_steps}, nstep={num_steps}, batch n={B})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = curve_dir / "exp2_nopt_sweep.png"
    plt.savefig(fig_path, dpi=150)
    print(f"  Saved curve: {fig_path}")
    plt.close()

    # Save index CSV (exp-specific metadata)
    index_path = exp_dir / "index.csv"
    if index_entries:
        fieldnames = ["nopt", "init", "gap", "nstep", "prompt", "src", "lr"]
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in index_entries:
                writer.writerow(entry)
        print(f"  Saved index: {index_path}")

    # Summary CSV
    summary_path = exp_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "noise_MSE_mean"])
        for config_key in sorted(results_by_config.keys()):
            traj_mean = results_by_config[config_key]["traj_mean"]
            noise_mse_mean = float(traj_mean.mean())
            writer.writerow([config_key, f"{noise_mse_mean:.6f}"])

    print(f"\n[Done] Exp2 complete. Output: {exp_dir}")
    return exp_dir


# ===================================================================
#  Main
# ===================================================================

if __name__ == "__main__":
    args = parse_args()

    if args.exp == "exp1":
        exp1_init_steps_sweep(args)
    elif args.exp == "exp2":
        exp2_step_by_step_optimization(args)
