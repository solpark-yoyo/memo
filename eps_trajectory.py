import sys
import os

# ---------------------------------------------------------------------------
# SCRIPT_DIR = ori_memo/  (this script lives inside ori_memo)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from munch import munchify

from latent_diffusion import StableDiffusion
from utils_local.log_util import set_seed
from torchvision.utils import save_image


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

            # 5) ||ε - ε_s||² / D  (normalized by epsilon dim D = 4*64*64)
            diff = (epsilon_original - eps_s).reshape(eps_s.shape[0], -1)
            diff_norm_sq = (diff ** 2).mean(dim=-1).item()

            # record
            step_indices.append(step_idx)
            timesteps_rec.append(t.item() if torch.is_tensor(t) else int(t))
            eps_diff_sq.append(diff_norm_sq)
            tweedie_x0.append(x0_hat.detach().cpu())
            eps_s_list.append(eps_s.detach().cpu())

            # 6) DDIM step (deterministic, η = 0) — use eps_theta, NOT eps_s
            #    x_{t-1} = √(ᾱ_{t-1}) x̂_0 + √(1-ᾱ_{t-1}) ε_θ(x_t, t)
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

        # decode final latent -> image (real inference result for this prompt)
        img = (self.decode(x0_hat) / 2 + 0.5).clamp(0, 1)

        return {
            "prompt":           prompt,
            "step_indices":     step_indices,
            "timesteps":        timesteps_rec,
            "eps_diff_sq":      eps_diff_sq,
            "tweedie_x0":       tweedie_x0,
            "eps_s_list":       eps_s_list,
            "epsilon_original": epsilon_original.cpu(),
            "img":              img,
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


def plot_multi_sample(multi_results_list, output_dir, filename="eps_trajectory_multi.png",
                     memo_indices=None):
    """Plot ALL prompts overlaid in ONE figure (lin / log 2 axes).

    Each curve = batch mean ± std (shading). Memorized prompts in red dashed.
    multi_results_list: one entry per prompt (text + memo).
    """
    if memo_indices is None:
        memo_indices = set()

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 7))
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    for idx, mr in enumerate(multi_results_list):
        prompt = mr["prompt"]
        all_curves = np.array([s["eps_diff_sq"] for s in mr["samples"]])  # (batch, steps)
        steps = mr["samples"][0]["step_indices"]
        mean = all_curves.mean(axis=0)
        std = all_curves.std(axis=0)

        is_memo = idx in memo_indices
        _short = (prompt[:25] + "...") if len(prompt) > 25 else prompt
        label = ("[Memo] " if is_memo else "") + _short
        color = "red" if is_memo else palette[idx % len(palette)]
        ls = "--" if is_memo else "-"
        lw = 2.4 if is_memo else 1.6

        for ax in (ax_lin, ax_log):
            ax.plot(steps, mean, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    ax_log.set_yscale("log")
    for ax, t in [(ax_lin, "linear"), (ax_log, "log")]:
        ax.set_xlabel("Denoising Step", fontsize=12)
        ax.set_ylabel(r"$||\epsilon - \epsilon_s||^2\ /\ D$", fontsize=12)
        ax.set_title(f"{t} scale", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(
        r"Memorization proxy  $||\epsilon - \epsilon_s||^2\ /\ D$"
        f"  (batch mean ± std, n={len(multi_results_list[0]['samples'])}, red = memorized)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved -> {path}")
    plt.close()


# ===================================================================
#  Main
# ===================================================================
def plot_combined(all_plot_data, output_dir, filename="eps_trajectory_combined.png"):
    """Combine all plots into ONE figure (lin / log 2 axes), each plot = one curve.

    Each plot's per-prompt curves are averaged into one representative curve,
    so num_plot plots become num_plot curves overlaid in lin & log axes.
    """
    n = len(all_plot_data)
    if n == 0:
        return
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 2)))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 7))

    for c, item in zip(colors, all_plot_data):
        mode, data, memo_indices, title = item
        if mode == "single":
            curves = np.array([res["eps_diff_sq"] for res in data])      # (P, T)
            steps = data[0]["step_indices"]
        else:  # multi: mean over seeds per prompt, then curves (P, T)
            curves = np.array([np.mean([s["eps_diff_sq"] for s in mr["samples"]], axis=0)
                               for mr in data])
            steps = data[0]["samples"][0]["step_indices"]
        mean_curve = curves.mean(axis=0)
        ax_lin.plot(steps, mean_curve, color=c, linewidth=1.8, marker='o', markersize=3, label=title)
        ax_log.plot(steps, mean_curve, color=c, linewidth=1.8, marker='o', markersize=3, label=title)

    for ax, use_log, lbl in [(ax_lin, False, "linear"), (ax_log, True, "log")]:
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("Denoising Step", fontsize=12)
        ax.set_ylabel(r"$||\epsilon - \epsilon_s||^2\ /\ D$  (" + lbl + ")", fontsize=12)
        ax.set_title(f"{lbl} scale", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)

    fig.suptitle("Memorization proxy — all plots overlaid", fontsize=14)
    plt.tight_layout()
    out = os.path.join(output_dir, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] combined saved -> {out}")


def read_proxy_csv(path):
    """proxy_*.csv 에서 (steps, mean, std) 읽기. 형식: step,proxy_mean,proxy_std"""
    steps, means, stds = [], [], []
    with open(path) as f:
        next(f, None)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0]:
                try:
                    steps.append(int(float(parts[0])))
                    means.append(float(parts[1]) if parts[1] else float("nan"))
                    stds.append(float(parts[2]) if len(parts) > 2 and parts[2] else 0.0)
                except ValueError:
                    continue
    return np.array(steps), np.array(means), np.array(stds)


def plot_overall(plot_root, num_tp, num_plot, out_name="memo_proxy_overall.png"):
    """plot_num=N/ 아래 모든 plot 의 proxy curve 를 모아 overall mean ± std plot.
    각 plot 의 csv/proxy_*.csv 에서 idx < num_tp → general(text), idx >= num_tp → memorized."""
    import glob
    text_curves, memo_curves = [], []
    steps_ref = None
    for k in range(num_plot):
        csv_dir = os.path.join(plot_root, f"plot{k:02d}", "csv")
        csvs = sorted(glob.glob(os.path.join(csv_dir, "proxy_*.csv")))
        for idx, c in enumerate(csvs):
            steps, mean, _ = read_proxy_csv(c)
            if len(mean) == 0:
                continue
            if steps_ref is None:
                steps_ref = steps
            (text_curves if idx < num_tp else memo_curves).append(mean)

    if steps_ref is None:
        print("[plot] overall: 수집된 curve 없음 — skip")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    stats = {}
    for label, curves, color in [("general", text_curves, "tab:blue"),
                                 ("memorized", memo_curves, "tab:red")]:
        if not curves:
            stats[label] = None
            continue
        arr = np.array(curves)                  # (num_curves, steps)
        m, s = arr.mean(axis=0), arr.std(axis=0)
        stats[label] = (m, s)
        ax.plot(steps_ref, m, color=color, linewidth=2.2, marker="o", markersize=3,
                label=f"{label} mean (n={len(curves)})")
        ax.fill_between(steps_ref, m - s, m + s, color=color, alpha=0.2,
                        label=f"{label} ±1 std")
    ax.set_xlabel("Denoising Step", fontsize=12)
    ax.set_ylabel(r"memo_proxy  $\|\,\epsilon - \epsilon_s\,\|^2 / D$", fontsize=12)
    ax.set_title(f"Overall memo_proxy (mean ± std across {num_plot} plots)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _total_dir = os.path.join(plot_root, "total")
    os.makedirs(_total_dir, exist_ok=True)
    out_path = os.path.join(_total_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] overall mean±std -> {out_path}")

    # 전체 평균/std csv → plot_num=N/total_csv/{general,memo}.csv
    import csv as _csv
    total_csv_dir = os.path.join(plot_root, "total", "total_csv")
    os.makedirs(total_csv_dir, exist_ok=True)
    _nan = np.full(len(steps_ref), np.nan)
    for _name, _key in [("general.csv", "general"), ("memo.csv", "memorized")]:
        _st = stats.get(_key)
        _m, _s = _st if _st is not None else (_nan, _nan)
        _cp = os.path.join(total_csv_dir, _name)
        with open(_cp, "w", newline="") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["step", "mean", "std"])
            for _si, _a, _b in zip(steps_ref, _m, _s):
                _w.writerow([int(_si), f"{_a:.6f}", f"{_b:.6f}"])
        print(f"[csv]  overall -> {_cp}")


def parse_args():
    p = argparse.ArgumentParser(description="Epsilon trajectory analysis for T2I memorization")
    # prompt sources
    p.add_argument("--text_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "coco_v2.txt"),
                   help="normal text prompt file")
    p.add_argument("--memo_dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "examples", "assets", "memorized_prompts_membench.txt"),
                   help="memorized text prompt file")
    p.add_argument("--num_tp", type=int, default=3, help="number of text prompts per plot")
    p.add_argument("--num_mtp", type=int, default=1, help="number of memorized prompts per plot")
    p.add_argument("--num_plot", type=int, default=1,
                   help="number of (num_tp text + num_mtp memo) comparison plots")
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


def load_prompts(path, n, seed=42):
    """파일에서 n 개 prompt 를 random sample (seed 로 재현 가능).
    n 이 전체 줄 수 이상이면 전체를 그대로 반환."""
    with open(path) as f:
        ps = [line.strip() for line in f if line.strip()]
    if n >= len(ps):
        return ps
    return random.Random(seed).sample(ps, n)


def plot_snr(sd, output_dir, filename="snr_proxy.png"):
    """Plot Tweedie SNR(t) = alpha_t / (1 - alpha_t) vs denoising step.

    SNR depends only on the schedule (alpha_t), so it is identical for
    text and memorized prompts — this plot visualizes that schedule.
    """
    ts = list(sd.scheduler.timesteps)
    steps = list(range(len(ts)))
    snr = [(sd.alpha(t) / (1 - sd.alpha(t))).item() for t in ts]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, snr, marker='o', linewidth=1.8, color='tab:blue', label="SNR (schedule)")
    ax.set_yscale("log")
    ax.set_xlabel("Denoising Step", fontsize=12)
    ax.set_ylabel(r"SNR(t) = $\alpha_t\, /\, (1-\alpha_t)$", fontsize=12)
    ax.set_title("Tweedie SNR vs denoising step  (schedule; identical for text & memo prompts)",
                 fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved -> {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # load prompts: num_tp text + num_mtp memo, repeated for num_plot plots
    # random sample (재현 가능) — text/memo 각각 다른 seed 로 독립 추출
    text_all = load_prompts(args.text_dir, args.num_tp * args.num_plot, seed=args.seed)
    memo_all = load_prompts(args.memo_dir, args.num_mtp * args.num_plot, seed=args.seed + 1)

    # ---- Load model ----
    print("=" * 60)
    print(f"Loading model: {os.path.basename(os.path.normpath(args.model_key))} ...")
    print(f"  model_key : {args.model_key}")
    print(f"  device    : {device}")
    print(f"  NFE       : {args.num_inference_steps}")
    print(f"  CFG       : {args.cfg_guidance}")
    print(f"  text_dir  : {args.text_dir}  (num_tp={args.num_tp})")
    print(f"  memo_dir  : {args.memo_dir}  (num_mtp={args.num_mtp})")
    print(f"  num_plot  : {args.num_plot}")
    print("=" * 60)

    solver_config = munchify({"num_sampling": args.num_inference_steps})
    analyzer = MemorizationAnalyzer(
        solver_config=solver_config,
        model_key=args.model_key,
        device=device,
        seed=args.seed,
    )

    # plot_num=N 폴더 아래에 plot00, plot01 ... 저장
    plot_root = os.path.join(args.output_dir, f"plot_num={args.num_plot}")
    os.makedirs(plot_root, exist_ok=True)

    # 이번 실행에 실제 사용된 prompt 목록 저장 → total/txt/
    _total_txt = os.path.join(plot_root, "total", "txt")
    os.makedirs(_total_txt, exist_ok=True)
    with open(os.path.join(_total_txt, "gen_text.txt"), "w") as _f:
        _f.write("\n".join(text_all) + "\n")
    with open(os.path.join(_total_txt, "memo_text.txt"), "w") as _f:
        _f.write("\n".join(memo_all) + "\n")
    print(f"[prompt] 사용 general {len(text_all)}개, memo {len(memo_all)}개 "
          f"-> {os.path.join(plot_root, 'total', 'txt')}/{{gen,memo}}_text.txt")

    for k in range(args.num_plot):
        # each plot = one folder plot{k}/ under plot_num=N/ with imgs/, npz/, csv/
        plot_dir = os.path.join(plot_root, f"plot{k:02d}")
        npz_dir = os.path.join(plot_dir, "npz")
        img_dir = os.path.join(plot_dir, "imgs")
        csv_dir = os.path.join(plot_dir, "csv")
        os.makedirs(npz_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)

        cur_text = text_all[k * args.num_tp:(k + 1) * args.num_tp]
        cur_memo = memo_all[k * args.num_mtp:(k + 1) * args.num_mtp]
        prompts = cur_text + cur_memo
        memo_indices = set(range(len(cur_text), len(prompts)))  # memo are the tail
        print(f"\n##### plot{k:02d}: {len(cur_text)} text + {len(cur_memo)} memo -> {plot_dir} #####")

        if args.num_samples == 1:
            results = []
            for idx, prompt in enumerate(prompts):
                print(f"[{idx+1}/{len(prompts)}] {prompt}")
                _is_memo = idx in memo_indices
                _prefix = "memo" if _is_memo else "gen"
                _gidx = idx - len(cur_text) if _is_memo else idx
                analyzer.generator.manual_seed(args.seed)
                torch.manual_seed(args.seed)
                res = analyzer.analyze_single(
                    prompt=prompt,
                    cfg_guidance=args.cfg_guidance,
                    null_prompt=args.null_prompt,
                )
                results.append(res)
                np.savez(
                    os.path.join(npz_dir, f"eps_traj_{idx:02d}.npz"),
                    step_indices=np.array(res["step_indices"]),
                    timesteps=np.array(res["timesteps"]),
                    eps_diff_sq=np.array(res["eps_diff_sq"]),
                    prompt=prompt,
                )
                save_image(res["img"].float(), os.path.join(img_dir, f"{_prefix}_{_gidx:02d}_00.png"))
                with open(os.path.join(csv_dir, f"proxy_{idx:02d}.csv"), "w") as _f:
                    _f.write("step,proxy_mean,proxy_std\n")
                    for _s, _v in zip(res["step_indices"], res["eps_diff_sq"]):
                        _f.write(f"{_s},{_v},\n")
            plot_single_trajectory(results, plot_dir, memo_indices=memo_indices,
                                   filename="eps_trajectory.png")
        else:
            multi_results = []
            for idx, prompt in enumerate(prompts):
                print(f"[{idx+1}/{len(prompts)}] {prompt}  ({args.num_samples} seeds)")
                _is_memo = idx in memo_indices
                _prefix = "memo" if _is_memo else "gen"
                _gidx = idx - len(cur_text) if _is_memo else idx
                mr = analyzer.analyze_multi_sample(
                    prompt=prompt,
                    cfg_guidance=args.cfg_guidance,
                    null_prompt=args.null_prompt,
                    num_samples=args.num_samples,
                    base_seed=args.seed,
                )
                multi_results.append(mr)
                all_curves = np.array([s["eps_diff_sq"] for s in mr["samples"]])
                np.savez(
                    os.path.join(npz_dir, f"eps_traj_multi_{idx:02d}.npz"),
                    step_indices=np.array(mr["samples"][0]["step_indices"]),
                    all_eps_diff_sq=all_curves,
                    prompt=prompt,
                )
                for s_idx, s in enumerate(mr["samples"]):
                    save_image(s["img"].float(), os.path.join(img_dir, f"{_prefix}_{_gidx:02d}_{s_idx:02d}.png"))
                with open(os.path.join(csv_dir, f"proxy_{idx:02d}.csv"), "w") as _f:
                    _f.write("step,proxy_mean,proxy_std\n")
                    for _s, _m, _sd in zip(mr["samples"][0]["step_indices"],
                                           all_curves.mean(axis=0), all_curves.std(axis=0)):
                        _f.write(f"{_s},{_m},{_sd}\n")
            plot_multi_sample(multi_results, plot_dir, filename="memo_proxy.png",
                              memo_indices=memo_indices)
        # SNR schedule plot (text & memo share the same schedule)
        plot_snr(analyzer, plot_dir)

    # 전체 plot 통합: text(general) vs memo overall mean ± std → plot_num=N/memo_proxy_overall.png
    plot_overall(plot_root, args.num_tp, args.num_plot)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
