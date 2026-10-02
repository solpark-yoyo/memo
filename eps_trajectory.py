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
#  Gaussianity metric: KL to N(0,1) via moment matching
# ===================================================================
def kl_to_standard_normal(x):
    """x 의 원소 분포를 N(μ, σ²) 로 근사 → KL(N(μ,σ²) ‖ N(0,1)).
    = 0.5·(μ² + σ² − log σ² − 1).  0 에 가까우면 N(0,1)."""
    xf = x.flatten().float()
    mu = xf.mean()
    var = xf.var(unbiased=False)
    return (0.5 * (mu ** 2 + var - torch.log(var + 1e-8) - 1)).item()


def compact_spectral_l2_stats(x, block_size=16):
    """x_t → compact spectral (wgnc.py f_r_to_c) → block 분할 →
    각 block 의 energy(‖y^(p)‖_2²) 를 계산 → mean, std.

    proxy = mean_energy / B    (dim 정규화, B=block_size)
      White Gaussian: mean_energy ≈ B → proxy ≈ 1
      Signal-bearing: energy 변화 → proxy ≠ 1

    Returns: (proxy, std)
      proxy = mean_energy / B   ← y-axis (1 = white Gaussian)
      std = 각 block energy 의 std
    """
    SQRT2 = 2.0 ** 0.5
    xf = x.flatten().to(torch.float64)
    B = block_size
    # compact spectral (wgnc f_r_to_c)
    f = torch.fft.rfft(xf, norm="ortho")
    f[0] = (f[0] + 1j * f[-1]) / SQRT2
    y = f[:-1]                        # N/2 complex (Hermitian 제거)
    # block 분할
    pad = (-y.shape[0]) % B
    if pad:
        y = torch.nn.functional.pad(y, (0, pad))
    y = y.reshape(-1, B)
    energies = y.abs().square().sum(dim=1).real    # (P,) 각 block energy
    mean_e = energies.mean()
    std_e = energies.std()
    proxy = (mean_e / float(B)).item()              # dim 정규화: 1 = white Gaussian
    return proxy, std_e.item()


def compact_spectral_block_energies(x, block_size=16, normalize=False):
    """x_t → compact spectral (wgnc.py f_r_to_c) → block 분할 →
    각 block 의 energy(‖y^(p)‖_2²) 를 **full vector** 로 반환.

    compact_spectral_l2_stats 가 mean/std 만 반환한 것과 달리,
    모든 block 의 energy (P,) vector 를 그대로 돌려준다.
    → denoising step 마다 이 vector 를 한 curve 씩 overlay 하면
      주파수 block 간 energy 분포가 denoising 에 따라 어떻게 변하는지 볼 수 있다.

    White Gaussian x  : 모든 block energy ≈ block_size  (flat 곡선)
    Signal-bearing x  : 저주파/특정 block 에 energy 집중 → 곡선 모양 변화

    Args:
        x          : latent tensor (임의 shape, flatten 후 사용)
        block_size : 한 block 의 주파수 bin 수 B (default 16)
        normalize  : True 면 energy / B  (1 = white Gaussian proxy)
    Returns:
        energies   : np.ndarray (P,)   각 block 의 energy
    """
    SQRT2 = 2.0 ** 0.5
    xf = x.flatten().to(torch.float64)
    B = block_size
    # compact spectral (wgnc f_r_to_c)
    f = torch.fft.rfft(xf, norm="ortho")
    f[0] = (f[0] + 1j * f[-1]) / SQRT2
    y = f[:-1]                              # N/2 complex (Hermitian 제거)
    pad = (-y.shape[0]) % B
    if pad:
        y = torch.nn.functional.pad(y, (0, pad))
    y = y.reshape(-1, B)
    energies = y.abs().square().sum(dim=1).real    # (P,) 각 block energy
    if normalize:
        energies = energies / float(B)
    # zt 가 requires_grad 이므로 detach (통계 용도, grad 추적 불필요)
    return energies.detach().cpu().numpy()


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
        loss_cfg: float = None,
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
        epsilon_original = zt.clone().detach()          # x_s forward noise (실제 주입 noise)
        epsilon_ref = epsilon_original                  # proxy reference (self-referential = x_T)

        # --- Fixed target step s (mid-noise level) ---
        #     Use the timestep at 50% of the schedule as the re-forward target
        timesteps_list = list(self.scheduler.timesteps)
        target_idx = int(len(timesteps_list) * target_step_ratio)
        s = timesteps_list[target_idx]
        as_ = self.alpha(s)   # ᾱ_s (fixed)
        print(f"  [target s = {s.item()}, ᾱ_s = {as_.item():.4f}]")

        # --- eps_l2 measure 용 CFG 가중치 (None → cfg_guidance 와 동일) ---
        loss_w = cfg_guidance if loss_cfg is None else loss_cfg

        # --- storage ---
        step_indices  = []
        timesteps_rec = []
        eps_diff_sq   = []
        kl_curve      = []    # KL(x_t ‖ N(0,1)) per step
        cmp_l2_curve  = []    # compact spectral L2² mean MSE per step
        cmp_l2_std_curve = [] # compact spectral block energy std per step
        tweedie_x0    = []
        eps_s_list    = []
        eps_l2_curve  = []    # ‖ε_uc + w·(ε_c−ε_uc)‖² per step (measure=eps_l2, w=loss_w)
        eps_ref_mse_curve = []  # ⑤ ‖ε_ref − ε_cfg(x_t,t)‖²/D per step (measure=eps_ref_mse)

        # --- DDIM denoising loop ---
        pbar = tqdm(self.scheduler.timesteps, desc=f"[{prompt[:45]}]")
        for step_idx, t in enumerate(pbar):
            at     = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            # 1) Model noise prediction at current DDIM point: ε_θ(x_t, t)
            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                eps_theta = noise_uc + cfg_guidance * (noise_c - noise_uc)
                eps_l2_sig = noise_uc + loss_w * (noise_c - noise_uc)  # eps_l2 신호 (w=loss_cfg)

            # 2) Tweedie estimate  x̂_0|t = (x_t - √(1-ᾱ_t) ε_θ) / √(ᾱ_t)
            x0_hat = (zt - (1 - at).sqrt() * eps_theta) / at.sqrt()

            # 3) Forward x̂_0|t to FIXED step s using ORIGINAL noise ε:
            #    x_s = √(ᾱ_s) x̂_0|t + √(1-ᾱ_s) ε
            x_s = as_.sqrt() * x0_hat + (1 - as_).sqrt() * epsilon_original

            # 4) Model prediction at forwarded point: ε_θ(x_s, s)
            with torch.no_grad():
                noise_uc_s, noise_c_s = self.predict_noise(x_s, s, uc, c)
                eps_s = noise_uc_s + cfg_guidance * (noise_c_s - noise_uc_s) # shape: [1,4,64,64]

            # 5) ||ε_ref - ε_s||² / D  (ε_ref = fresh random, ε_s = ε_θ(x_s, s))
            diff = (epsilon_ref - eps_s).reshape(eps_s.shape[0], -1) # shape: [1,4*64*64]
            diff_norm_sq = (diff ** 2).mean(dim=-1).item()

            # record
            step_indices.append(step_idx)  # DDIM step [0,1,2 , ... , 49]
            timesteps_rec.append(t.item() if torch.is_tensor(t) else int(t)) # 981, 961, 941, ..., 1
            eps_diff_sq.append(diff_norm_sq)
            kl_curve.append(kl_to_standard_normal(zt))
            _mse, _std_e = compact_spectral_l2_stats(zt)
            cmp_l2_curve.append(_mse)
            cmp_l2_std_curve.append(_std_e)
            eps_l2_curve.append((eps_l2_sig.float() ** 2).mean().item())
            # ⑤ eps_ref_mse: ‖ε_ref − ε_cfg(x_t,t)‖²/D — ③의 eps_l2_sig, ①의 epsilon_ref 재사용
            eps_ref_mse_curve.append(
                ((epsilon_ref.float() - eps_l2_sig.float()) ** 2).mean().item())
            tweedie_x0.append(x0_hat.detach().cpu())
            eps_s_list.append(eps_s.detach().cpu())

            # 6) DDIM step (deterministic, η = 0) — use eps_theta, NOT eps_s
            #    x_{t-1} = √(ᾱ_{t-1}) x̂_0 + √(1-ᾱ_{t-1}) ε_θ(x_t, t)
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * eps_theta

        # decode final latent -> image (real inference result for this prompt)
        img = (self.decode(x0_hat) / 2 + 0.5).clamp(0, 1)
        # print(f"tweedie_x0: {len(tweedie_x0)}")
        return {
            "prompt":           prompt,
            "step_indices":     step_indices,
            "timesteps":        timesteps_rec,
            "eps_diff_sq":      eps_diff_sq,
            "kl":               kl_curve,
            "cmp_l2":           cmp_l2_curve,
            "cmp_l2_std":       cmp_l2_std_curve,
            "eps_l2":           eps_l2_curve,
            "eps_ref_mse":      eps_ref_mse_curve,
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
        loss_cfg: float = None,
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
        # Batch inference (various seed)
        # num_samples: Gen num img per prompts
        for i in range(num_samples):
            seed = base_seed + i * 1000
            self.generator.manual_seed(seed)
            # also set torch seed for randn
            torch.manual_seed(seed)
            result = self.analyze_single(
                prompt=prompt,
                cfg_guidance=cfg_guidance,
                null_prompt=null_prompt,
                loss_cfg=loss_cfg,
            )
            result["seed"] = seed
            samples.append(result)
            print(f"  sample {i+1}/{num_samples}  seed={seed}  done")

        return {"prompt": prompt, "samples": samples}

    # ------------------------------------------------------------------
    #  Block-energy trajectory (주파수 block 별 energy 의 denoising 추이)
    # ------------------------------------------------------------------
    def analyze_block_energy(
        self,
        prompt: str,
        cfg_guidance: float = 7.5,
        null_prompt: str = "",
        block_size: int = 16,
        normalize: bool = False,
        signal: str = "xt",
    ) -> dict:
        """DDIM denoising 한 번 돌며, 각 step 에 signal 의 주파수 block 별
        energy vector (P,) 를 기록.

        signal:
          'xt'  → 현재 latent x_t 의 spectrum energy
          'eps' → ε_θ(x_t, t) (CFG 결합 noise prediction) 의 spectrum energy

        analyze_single 의 무거운 ε_s 재추론은 생략하고 DDIM update 만 수행 →
        step 당 UNet 호출 1 회로 가볍게 trajectory 를 확보.

        Returns:
            step_indices : list[int]      DDIM step [0, 1, ..., NFE-1]
            timesteps    : list[int]      981, 961, ..., 1
            block_energy : np.ndarray (T, P)   각 step 의 block energy vector
            block_size   : int
            normalize    : bool
            signal       : str
        """
        uc, c = self.get_text_embed(null_prompt=null_prompt, prompt=prompt)
        zt = self.initialize_latent().to(self.dtype)

        step_indices, timesteps_rec, block_energy_list = [], [], []
        pbar = tqdm(self.scheduler.timesteps, desc=f"[{signal}-E {prompt[:30]}]")
        for step_idx, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                # cfg_epsilon: DDIM update 에 실제로 쓰이는 CFG 결합 noise prediction
                cfg_epsilon = noise_uc + cfg_guidance * (noise_c - noise_uc)

            # signal 선택: 'xt' → 현재 latent x_t, 'eps' → cfg_epsilon(x_t, t)
            sig = zt if signal == "xt" else cfg_epsilon
            energies = compact_spectral_block_energies(
                sig, block_size=block_size, normalize=normalize)
            step_indices.append(step_idx)
            timesteps_rec.append(t.item() if torch.is_tensor(t) else int(t))
            block_energy_list.append(energies)

            # DDIM step (deterministic, η = 0) — cfg_epsilon 사용
            x0_hat = (zt - (1 - at).sqrt() * cfg_epsilon) / at.sqrt()
            zt = at_prev.sqrt() * x0_hat + (1 - at_prev).sqrt() * cfg_epsilon

        # 최종 x0_hat → image (해당 seed 의 생성 결과, imgs/ 저장용)
        img = (self.decode(x0_hat) / 2 + 0.5).clamp(0, 1)

        return {
            "prompt":       prompt,
            "step_indices": step_indices,
            "timesteps":    timesteps_rec,
            "block_energy": np.stack(block_energy_list),   # (T, P)
            "block_size":   block_size,
            "normalize":    normalize,
            "signal":       signal,
            "img":          img,
        }

    def analyze_block_energy_multi_sample(
        self,
        prompt: str,
        cfg_guidance: float = 7.5,
        null_prompt: str = "",
        num_samples: int = 5,
        base_seed: int = 42,
        block_size: int = 16,
        normalize: bool = False,
        signal: str = "xt",
    ) -> dict:
        """여러 seed 로 analyze_block_energy 반복 → seed 평균 (T, P) trajectory.

        memorized prompt 는 seed 무관하게 비슷한 trajectory 를 가지므로
        평균을 내면 안정적인 block-energy curve 가 된다.

        Returns:
            prompt            : str
            step_indices      : list[int]
            timesteps         : list[int]
            block_energy_mean : np.ndarray (T, P)
            block_energy_std  : np.ndarray (T, P)
            block_energy_raw  : np.ndarray (S, T, P)   per-seed 원본
            block_size        : int
            normalize         : bool
        """
        raw = []
        samples_ = []
        step_indices = timesteps_rec = None
        for i in range(num_samples):
            seed = base_seed + i * 1000
            self.generator.manual_seed(seed)
            torch.manual_seed(seed)
            res = self.analyze_block_energy(
                prompt=prompt,
                cfg_guidance=cfg_guidance,
                null_prompt=null_prompt,
                block_size=block_size,
                normalize=normalize,
                signal=signal,
            )
            step_indices = res["step_indices"]
            timesteps_rec = res["timesteps"]
            raw.append(res["block_energy"])
            samples_.append(res)
            print(f"  [{signal}-E] sample {i+1}/{num_samples}  seed={seed}  done")
        raw = np.stack(raw)                              # (S, T, P)
        return {
            "prompt":            prompt,
            "step_indices":      step_indices,
            "timesteps":         timesteps_rec,
            "block_energy_mean": raw.mean(axis=0),        # (T, P)
            "block_energy_std":  raw.std(axis=0),         # (T, P)
            "block_energy_raw":  raw,                     # (S, T, P)
            "block_size":        block_size,
            "normalize":         normalize,
            "signal":           signal,
            "imgs":             [s["img"] for s in samples_],
        }


# ===================================================================
#  Plotting helpers
# ===================================================================
def plot_single_trajectory(results_list, output_dir,
                          memo_indices=None, filename="eps_trajectory.png",
                          ylabel=None):
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
            ax.set_ylabel((ylabel or r"$\|\, \epsilon - \epsilon_s \,\|^2$") + "  (log scale)", fontsize=13)
            ax.set_title("Log Scale", fontsize=13)
        else:
            ax.set_ylabel(ylabel or r"$\|\, \epsilon - \epsilon_s \,\|^2$", fontsize=13)
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
                     memo_indices=None, ylabel=None):
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
        ax.set_ylabel(ylabel or r"$||\epsilon - \epsilon_s||^2\ /\ D$", fontsize=12)
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
#  Block-energy trajectory plot + csv
# ===================================================================
def plot_block_energy_combined(mr, output_dir, freq_ratio=0.1, curve_steps=None,
                               filename="block_energy.png", tag="memo"):
    """전체 / Low / High 주파수 block energy curve 를 한 figure 에 함께 (모두 linear, log 없음).

    3 panel: [전체] [Low freq] [High freq], 모두 linear y-scale.
    DC(block 0) 는 linear 가독성을 위해 전체/Low panel 에서 제외 (High 는 DC 미포함).
    freq_ratio → n_band=int(P*ratio), low=[0,n_band-1], high=[P-n_band,P-1].
    """
    import matplotlib as mpl

    be = mr["block_energy_mean"]                    # (T, P)
    T, P = be.shape
    n_seeds = mr["block_energy_raw"].shape[0]
    if curve_steps is None:
        curve_steps = list(range(T))
    curve_steps = [i for i in curve_steps if 0 <= i < T]

    n_band = max(1, min(int(P * freq_ratio), P // 2))
    low = (0, n_band - 1)
    high = (P - n_band, P - 1)

    norm_flag = mr.get("normalize", False)
    y_label = (r"block energy $\,/\,B$   (1 = white Gaussian)"
               if norm_flag else "block energy")

    cmap = plt.cm.plasma
    cnorm = mpl.colors.Normalize(vmin=0, vmax=max(T - 1, 1))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (panel 이름, x array, y (T, P')) — 전체/Low 는 DC(block 0) 제외, High 는 DC 없음
    panels = [
        ("All",       np.arange(1, P),                 be[:, 1:]),
        ("Low freq",  np.arange(1, low[1] + 1),        be[:, 1:low[1] + 1]),
        ("High freq", np.arange(high[0], high[1] + 1), be[:, high[0]:high[1] + 1]),
    ]
    rng_str = {
        "All":       f"[1:{P - 1}]  (DC excl.)",
        "Low freq":  f"[1:{low[1]}]  (of [0:{low[1]}], DC excl.)",
        "High freq": f"[{high[0]}:{high[1]}]",
    }

    for ax, (name, x, y) in zip(axes, panels):
        for i in curve_steps:
            c = cmap(cnorm(i))
            ax.plot(x, y[i], color=c, linewidth=1.8, marker="o", markersize=3,
                    alpha=0.95, label=f"step {i} (t={mr['timesteps'][i]})", zorder=3)
        ax.set_xlabel(f"Block index  {rng_str[name]}", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="best", framealpha=0.9)
        ax.tick_params(labelsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    cbar = fig.colorbar(sm, ax=axes[2], pad=0.02)
    cbar.set_label("Denoising step  (0 = most noisy)", fontsize=10)

    _short = (mr["prompt"][:36] + "...") if len(mr["prompt"]) > 36 else mr["prompt"]
    fig.suptitle(
        f"[{tag}] Per-block energy (All / Low / High) over DDIM steps  —  \"{_short}\"\n"
        f"freq_ratio={freq_ratio} (n_band={n_band}), linear scale, "
        f"mean over {n_seeds} seed{'s' if n_seeds > 1 else ''}, "
        f"{len(curve_steps)} step{'s' if len(curve_steps) > 1 else ''}",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] combined (all/low/high, linear) saved -> {path}")
    plt.close()


def plot_band_compare(mr, output_dir, freq_ratio=0.1,
                      filename="block_energy_band.png", tag="memo"):
    """Low / High / All 주파수 밴드의 평균 energy 를 denoising step 에 따라 비교.

    freq_ratio 로 정의된 low/high 밴드와 전체(all) 의 평균 block energy 를
    한 plot 에 그려, denoising 진행 시 energy 가 어느 주파수로 집중/이동하는지 보여줌.
      low  = [0, n_band-1],  high = [P-n_band, P-1],  n_band = int(P*freq_ratio)
    """
    be = mr["block_energy_mean"]                    # (T, P)
    T, P = be.shape
    steps = mr["step_indices"]
    n_band = max(1, min(int(P * freq_ratio), P // 2))
    low = (0, n_band - 1)
    high = (P - n_band, P - 1)

    mean_all = be.mean(axis=1)
    mean_low = be[:, low[0]:low[1] + 1].mean(axis=1)
    mean_high = be[:, high[0]:high[1] + 1].mean(axis=1)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, mean_low, color="tab:red", linewidth=2.2, marker="o",
            markersize=4.5, label=f"Low freq  {low}   ({n_band} blocks)")
    ax.plot(steps, mean_high, color="tab:blue", linewidth=2.2, marker="s",
            markersize=4.5, label=f"High freq {high}   ({n_band} blocks)")
    ax.plot(steps, mean_all, color="tab:gray", linewidth=1.6, marker="^",
            markersize=4, linestyle="--",
            label=f"All [0:{P - 1}]   ({P} blocks)")
    ax.set_xlabel("Denoising step  (0 = most noisy, high t)", fontsize=12)
    ax.set_ylabel(
        (r"mean block energy $\,/\,B$ in band   (1 = white Gaussian)"
         if mr.get("normalize", False) else "mean block energy in band"),
        fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    _short = (mr["prompt"][:40] + "...") if len(mr["prompt"]) > 40 else mr["prompt"]
    n_seeds = mr["block_energy_raw"].shape[0]
    fig.suptitle(
        f"[{tag}] Low vs High freq band energy  —  \"{_short}\"\n"
        f"freq_ratio={freq_ratio}  (n_band={n_band}),   "
        f"mean over {n_seeds} seed{'s' if n_seeds > 1 else ''}",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] band compare (low/high/all) saved -> {path}")
    plt.close()


def plot_proxy_trend(text_mrs, memo_mrs, output_dir, proxy="all", freq_ratio=0.1,
                     signal="xt", filename=None, num_bins=8, band_agg="mean"):
    """text(general) vs memorized 그룹의 spectrum-energy proxy 를
    denoising step 에 따라 비교 (x=step, y=proxy).

    각 prompt 그룹을 mean ± std band 로 표시 → 그룹 간 차이(trend) 가 한눈에.
    proxy: 'all' | 'low' | 'high'  (freq_ratio 로 low/high 밴드 산출)
    signal: 'xt' (x_t) | 'eps' (ε_θ) — 라벨 표기용
    band_agg: 'mean' (④ 체계, 밴드 block 평균) | 'sum' (⑦ 선택 block energy 총합)
    """
    _agg_sfx = "" if band_agg == "mean" else f"_{band_agg}"
    if filename is None:
        filename = f"proxy_trend_{signal}_{proxy}{_agg_sfx}.png"
    be_t0 = text_mrs[0]["block_energy_mean"]
    T, P = be_t0.shape
    n_band = max(1, min(int(P * freq_ratio), P // 2))
    low = (0, n_band - 1)
    high = (P - n_band, P - 1)

    def _band_mean(mr):
        be = mr["block_energy_mean"]                     # (T, P)
        if proxy == "low":
            band = be[:, low[0]:low[1] + 1]
        elif proxy == "high":
            band = be[:, high[0]:high[1] + 1]
        else:
            band = be                                    # 'all'
        return band.sum(axis=1) if band_agg == "sum" else band.mean(axis=1)

    steps = text_mrs[0]["step_indices"]
    norm_flag = text_mrs[0].get("normalize", False)
    B = text_mrs[0].get("block_size", 16)
    if band_agg == "sum":
        y_label = {
            "all":  f"SUM block energy ÷B  ([0:{P - 1}], {P} blocks)",
            "low":  f"low-freq band SUM energy ÷B  ([0:{n_band - 1}], {n_band} blocks, "
                    f"white Gaussian ≈ {n_band})",
            "high": f"high-freq band SUM energy ÷B  ([{P - n_band}:{P - 1}], {n_band} blocks)",
        }[proxy]
    else:
        y_label = {
            ("all", True):  r"mean block energy $\,/\,B$  (1 = white Gaussian)",
            ("all", False): "mean block energy (all blocks)",
            ("low", True):  r"low-freq band energy $\,/\,B$  ($[0:%d]$)" % (n_band - 1),
            ("low", False): f"low-freq band energy  ([0:{n_band - 1}])",
            ("high", True): r"high-freq band energy $\,/\,B$  ($[%d:%d]$)" % (P - n_band, P - 1),
            ("high", False): f"high-freq band energy  ([{P - n_band}:{P - 1}])",
        }[(proxy, norm_flag)]
    sig_label = {"xt": r"$x_t$", "eps": r"$\epsilon_\theta(x_t, t)$"}[signal]

    fig, ax = plt.subplots(figsize=(11, 6))
    stats = {}
    for label, mrs, color in [("text (general)", text_mrs, "tab:blue"),
                              ("memorized", memo_mrs, "tab:red")]:
        curves = np.stack([_band_mean(mr) for mr in mrs])   # (n_prompts, T)
        m, s = curves.mean(axis=0), curves.std(axis=0)
        stats[label] = (m, s)
        ax.plot(steps, m, color=color, linewidth=2.2, marker="o", markersize=4,
                label=f"{label}  (n={len(mrs)})")
        ax.fill_between(steps, m - s, m + s, color=color, alpha=0.2)

    ax.set_xlabel("Denoising step  (0 = most noisy, high t)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.suptitle(
        f"Spectrum-energy proxy trend:  {sig_label} → spectrum energy\n"
        f"text vs memorized  (proxy={proxy}, band_agg={band_agg}, freq_ratio={freq_ratio}, "
        f"n_band={n_band}, mean ± std over prompts)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] proxy trend saved -> {path}")
    plt.close()
    return stats


def save_proxy_trend_csv(text_mrs, memo_mrs, output_dir, proxy="all", freq_ratio=0.1,
                         signal="xt", filename=None, band_agg="mean"):
    """text vs memo 그룹의 proxy trajectory 를 CSV 로: step,timestep,text_mean,text_std,memo_mean,memo_std."""
    import csv as _csv
    _agg_sfx = "" if band_agg == "mean" else f"_{band_agg}"
    if filename is None:
        filename = f"proxy_trend_{signal}_{proxy}{_agg_sfx}.csv"
    be_t0 = text_mrs[0]["block_energy_mean"]
    T, P = be_t0.shape
    n_band = max(1, min(int(P * freq_ratio), P // 2))
    low = (0, n_band - 1)
    high = (P - n_band, P - 1)

    def _band_mean(mr):
        be = mr["block_energy_mean"]
        if proxy == "low":
            band = be[:, low[0]:low[1] + 1]
        elif proxy == "high":
            band = be[:, high[0]:high[1] + 1]
        else:
            band = be                                    # 'all'
        return band.sum(axis=1) if band_agg == "sum" else band.mean(axis=1)

    steps = text_mrs[0]["step_indices"]
    ts = text_mrs[0]["timesteps"]
    tc = np.stack([_band_mean(mr) for mr in text_mrs])      # (n_text, T)
    mc = np.stack([_band_mean(mr) for mr in memo_mrs])      # (n_memo, T)
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["step", "timestep", "text_mean", "text_std",
                    "memo_mean", "memo_std"])
        for i in range(T):
            w.writerow([int(steps[i]), int(ts[i]),
                        f"{tc[:, i].mean():.6f}", f"{tc[:, i].std():.6f}",
                        f"{mc[:, i].mean():.6f}", f"{mc[:, i].std():.6f}"])
    print(f"[csv]  proxy trend saved -> {path}")


def plot_block_energy_heatmap(mr, output_dir,
                              filename="block_energy_heatmap.png", tag="memo"):
    """2D heatmap: y=denoising step, x=block index, color=log10(block energy).

    50 개 curve overlay 보다 '추이' 를 한눈에 파악하기 좋은 표준 시각화.
    step(행) × block(열) matrix 의 energy 분포 변화를 연속적인 색으로 표현.
    """
    be = mr["block_energy_mean"]                   # (T, P)
    T, P = be.shape
    B = mr["block_size"]
    log_be = np.log10(be + 1e-8)                   # 다이나믹 레인지 큼 → log

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(log_be, aspect="auto", origin="lower",
                   cmap="magma", interpolation="nearest",
                   extent=[0, P, 0, T])
    ax.set_xlabel(f"Block index   (block_size={B},  #blocks={P})", fontsize=12)
    ax.set_ylabel("Denoising step  (0 = most noisy, high t)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}$  block energy  $\|\,y^{(p)}\,\|_2^2$", fontsize=11)

    _short = (mr["prompt"][:40] + "...") if len(mr["prompt"]) > 40 else mr["prompt"]
    fig.suptitle(
        f"[{tag}] Block-energy heatmap over DDIM steps  —  \"{_short}\"\n"
        f"y = denoising step,  x = block index,  color = log10(energy)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] block-energy heatmap saved -> {path}")
    plt.close()


def save_block_energy_csv(mr, output_dir, filename="block_energy.csv",
                          freq_ratio=0.1):
    """Block-energy matrix → CSV.  row = denoising step, col = block_0..block_{P-1}.
    freq_ratio 로 정의된 all/low/high 밴드의 sum·mean 컬럼을 끝에 추가."""
    import csv as _csv
    be = mr["block_energy_mean"]                   # (T, P)
    steps = mr["step_indices"]
    ts = mr["timesteps"]
    T, P = be.shape
    n_band = max(1, min(int(P * freq_ratio), P // 2))
    low = (0, n_band - 1)
    high = (P - n_band, P - 1)
    band_low = be[:, low[0]:low[1] + 1]            # inclusive
    band_high = be[:, high[0]:high[1] + 1]
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            ["step", "timestep"]
            + [f"block_{j:04d}" for j in range(P)]
            + ["band_all_sum", "band_all_mean",
               f"band_low_sum_{low[0]}_{low[1]}", f"band_low_mean_{low[0]}_{low[1]}",
               f"band_high_sum_{high[0]}_{high[1]}", f"band_high_mean_{high[0]}_{high[1]}"]
        )
        for i in range(T):
            w.writerow(
                [int(steps[i]), int(ts[i])]
                + [f"{v:.6f}" for v in be[i]]
                + [f"{be[i].sum():.6f}", f"{be[i].mean():.6f}",
                   f"{band_low[i].sum():.6f}", f"{band_low[i].mean():.6f}",
                   f"{band_high[i].sum():.6f}", f"{band_high[i].mean():.6f}"]
            )
    print(f"[csv]  block-energy saved -> {path}  "
          f"(low={low}, high={high}, n_band={n_band})")


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


def plot_overall(plot_root, num_tp, num_plot, out_name="memo_proxy_overall.png",
                 ylabel=None):
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
    ax.set_ylabel(ylabel or r"memo_proxy  $\|\,\epsilon - \epsilon_s\,\|^2 / D$", fontsize=12)
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
    p.add_argument("--measure", type=str, default="memo_proxy",
                   choices=["memo_proxy", "kl_div", "cmp_l2", "eps_l2", "eps_ref_mse"],
                   help="측정 metric: memo_proxy=||eps_ref-eps_s||^2/D | "
                        "kl_div=KL(x_t||N(0,1)) | "
                        "cmp_l2=compact spectral blockwise L2^2 deviation MSE | "
                        "eps_l2=||eps_uc+w*(eps_c-eps_uc)||^2/D (CFG 결합 예측 에너지)")
    p.add_argument("--loss_cfg", type=float, default=None,
                   help="eps_l2 measure 의 CFG 가중치 w (loss 조절용). "
                        "default=None → cfg_guidance 와 동일. 1.0 → ε_uc 에너지")
    # block-energy trajectory (memorized prompt, 주파수 block 별 energy 추이)
    p.add_argument("--block_energy", action="store_true",
                   help="memorized prompt 의 주파수 block-energy trajectory "
                        "(x=block index, y=energy, step 당 1 curve) plot+csv 저장")
    p.add_argument("--block_size", type=int, default=16,
                   help="주파수 block 하나의 bin 수 B (block_energy 모드, default 16)")
    p.add_argument("--num_curve_steps", type=int, default=5,
                   help="block-energy curve 로 그릴 denoising step 개수. "
                        "전체 NFE step 중 균등 간격으로 선택 "
                        "(예: NFE=50, 5 → step 0,10,20,30,40). default 5")
    p.add_argument("--freq_ratio", type=float, default=0.1,
                   help="Low/High freq 밴드 비율 (전체 block 수 P 의 비율). "
                        "n_band = int(P*freq_ratio), "
                        "low = [0, n_band-1], high = [P-n_band, P-1]. "
                        "예: P=512, ratio=0.1 → low[0:50], high[461:511] (각 51개). "
                        "curve(all/low/high 3종) + Low vs High band 비교 plot + "
                        "csv 밴드 컬럼 생성. default 0.1")
    # ---- proxy trend mode: x_t → ε_θ → spectrum energy, text vs memo 비교 ----
    p.add_argument("--proxy_trend", action="store_true",
                   help="text(general) vs memorized 그룹의 spectrum-energy proxy "
                        "trend 를 denoising step 에 따라 비교 "
                        "(signal='xt' → x_t, 'eps' → ε_θ(x_t,t))")
    p.add_argument("--signal", type=str, default="xt", choices=["xt", "eps"],
                   help="proxy_trend 에서 spectrum energy 를 계산할 신호: "
                        "xt = latent x_t | eps = ε_θ(x_t,t) (CFG 결합). default xt")
    p.add_argument("--proxy", type=str, default="all",
                   choices=["all", "low", "high"],
                   help="proxy_trend 의 y-axis: all = 전체 block 평균 | "
                        "low = 저주파 밴드 평균 | high = 고주파 밴드 평균 "
                        "(밴드는 freq_ratio 로 산출). default all")
    p.add_argument("--band_agg", type=str, default="mean", choices=["mean", "sum"],
                   help="proxy_trend 밴드 집계 (⑦ spectral score low freq energy): "
                        "mean = 밴드 block 평균 (④ 체계) | "
                        "sum = 선택 block energy 총합 (⑦ sum_energy 리터럴, "
                        "white Gaussian 기준 ≈ n_band·B). default mean")
    p.add_argument("--lh_ratio", type=float, default=None,
                   help="⑦ spectral score low freq energy 의 low-band 비율 LH_ratio "
                        "(block 총 수 P 기준 — 앞 int(P·LH_ratio) 개 block 선택). "
                        "지정 시 proxy_trend 밴드를 freq_ratio 대신 이 값으로 산출 "
                        "(예: 0.1 → P=512 중 앞 51 blocks). "
                        "default None → --freq_ratio 값 사용")
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

    # ===============================================================
    #  Proxy-trend mode: x_t → ε_θ → spectrum energy, text vs memo 비교
    #  구조: plot_num=N/plot{k}/{csv,imgs,npz} (results_cmp_l2_trajectory 참고)
    #  각 plot = num_tp text + num_mtp memo, batch 개 seed → mean ± std plot
    # ===============================================================
    if args.proxy_trend:
        print("=" * 60)
        print("  [proxy_trend] spectrum-energy proxy: text vs memorized")
        print(f"  signal = {args.signal}   proxy = {args.proxy}   band_agg = {args.band_agg}")
        print(f"  freq_ratio = {args.freq_ratio}   lh_ratio = {args.lh_ratio}"
              + ("   (lh_ratio 우선)" if args.lh_ratio is not None else ""))
        print(f"  num_samples = {args.num_samples}")
        print(f"  output_dir = {args.output_dir}")
        print("=" * 60)

        text_all = load_prompts(args.text_dir, args.num_tp * args.num_plot,
                                seed=args.seed)
        memo_all = load_prompts(args.memo_dir, args.num_mtp * args.num_plot,
                                seed=args.seed + 1)
        print(f"[prompt] text {len(text_all)}개 / memo {len(memo_all)}개")

        # plot_num=N 아래 plot00, plot01 ... 각 plot = num_tp text + num_mtp memo
        plot_root = os.path.join(args.output_dir, f"plot_num={args.num_plot}")
        os.makedirs(plot_root, exist_ok=True)

        for k in range(args.num_plot):
            cur_text = text_all[k * args.num_tp:(k + 1) * args.num_tp]
            cur_memo = memo_all[k * args.num_mtp:(k + 1) * args.num_mtp]
            prompts = cur_text + cur_memo
            memo_indices = set(range(len(cur_text), len(prompts)))
            plot_dir_k = os.path.join(plot_root, f"plot{k:02d}")
            csv_dir_k = os.path.join(plot_dir_k, "csv")
            img_dir_k = os.path.join(plot_dir_k, "imgs")
            npz_dir_k = os.path.join(plot_dir_k, "npz")
            for _d in (csv_dir_k, img_dir_k, npz_dir_k):
                os.makedirs(_d, exist_ok=True)
            print(f"\n##### plot{k:02d}: {len(cur_text)} text + {len(cur_memo)} memo "
                  f"-> {plot_dir_k} #####")

            text_mrs, memo_mrs = [], []
            for idx, prompt in enumerate(prompts):
                is_memo = idx in memo_indices
                prefix = "memo" if is_memo else "text"
                print(f"[{idx+1}/{len(prompts)}] {prompt}  ({args.num_samples} seeds)")
                mr = analyzer.analyze_block_energy_multi_sample(
                    prompt=prompt,
                    cfg_guidance=args.cfg_guidance,
                    null_prompt=args.null_prompt,
                    num_samples=args.num_samples,
                    base_seed=args.seed,
                    block_size=args.block_size,
                    normalize=True,          # benchmark scale (energy/B, 1=WG)
                    signal=args.signal,
                )
                # npz: batch (seed) trajectory 원본
                np.savez(
                    os.path.join(npz_dir_k, f"{prefix}_{idx:02d}_{args.signal}.npz"),
                    step_indices=np.array(mr["step_indices"]),
                    timesteps=np.array(mr["timesteps"]),
                    block_energy_mean=mr["block_energy_mean"],
                    block_energy_std=mr["block_energy_std"],
                    block_energy_raw=mr["block_energy_raw"],
                    block_size=np.array(mr["block_size"]),
                    normalize=np.array(True),
                    signal=args.signal,
                    prompt=prompt,
                )
                # imgs: batch 개 생성 이미지 (prefix_idx_seed.png)
                for s_idx, img in enumerate(mr["imgs"]):
                    save_image(img.float(),
                               os.path.join(img_dir_k, f"{prefix}_{idx:02d}_{s_idx:02d}.png"))
                # csv: step,timestep,mean,std (batch 평균 proxy trajectory)
                _bmean = mr["block_energy_mean"].mean(axis=1)   # (T,) batch mean
                _bstd = mr["block_energy_mean"].std(axis=1)     # (T,) batch std
                with open(os.path.join(csv_dir_k,
                                       f"proxy_{prefix}_{idx:02d}.csv"), "w") as _f:
                    _f.write("step,timestep,proxy_mean,proxy_std\n")
                    for _s, _tt, _m, _sd in zip(mr["step_indices"],
                                                mr["timesteps"], _bmean, _bstd):
                        _f.write(f"{_s},{_tt},{_m:.6f},{_sd:.6f}\n")
                (memo_mrs if is_memo else text_mrs).append(mr)

            # plot png: text vs memo 그룹 mean ± std (batch 로 이미 평균난 mr 들)
            # 밴드 비율: ⑦ --lh_ratio 지정 시 우선, 아니면 --freq_ratio
            _band_ratio = args.lh_ratio if args.lh_ratio is not None else args.freq_ratio
            plot_proxy_trend(text_mrs, memo_mrs, plot_dir_k, proxy=args.proxy,
                             freq_ratio=_band_ratio, signal=args.signal,
                             band_agg=args.band_agg)
            save_proxy_trend_csv(text_mrs, memo_mrs, csv_dir_k, proxy=args.proxy,
                                 freq_ratio=_band_ratio, signal=args.signal,
                                 band_agg=args.band_agg)
            print(f"[plot{k:02d}] done")

        print(f"\n[proxy_trend] all done -> {plot_root}")
        return

    # ===============================================================
    #  Block-energy trajectory mode (memorized prompts only)
    #  x=block index, y=block energy, step 당 1 curve → NFE 개 curve overlay
    # ===============================================================
    if args.block_energy:
        print("=" * 60)
        print("  [block_energy] memorized prompt 주파수 block-energy trajectory")
        print(f"  block_size = {args.block_size}")
        print(f"  num_samples = {args.num_samples} (seed 평균)")
        print(f"  output_dir = {args.output_dir}")
        print("=" * 60)

        memo_all = load_prompts(args.memo_dir, args.num_mtp * args.num_plot,
                                seed=args.seed + 1)
        print(f"[prompt] {len(memo_all)} memorized prompt(s) loaded "
              f"from {args.memo_dir}")

        # 출력 정리: LH_energy/{csv, plot}
        csv_dir = os.path.join(args.output_dir, "LH_energy", "csv")
        plot_dir = os.path.join(args.output_dir, "LH_energy", "plot")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        print(f"  [freq_ratio] {args.freq_ratio}  → low/high 밴드는 "
              f"각 mr 의 block 수 P 에서 산출 (curve 3종 + band 비교 plot)")

        for idx, prompt in enumerate(memo_all):
            print(f"\n##### [block_energy {idx+1}/{len(memo_all)}]  {prompt} #####")
            # spectrum energy 기준 (run_memo_spectral.sh / optimize_xt_spectral.py
            # 의 spectral_l2_loss 와 동일 scale): energy / B, 1 = white Gaussian
            if args.num_samples > 1:
                mr = analyzer.analyze_block_energy_multi_sample(
                    prompt=prompt,
                    cfg_guidance=args.cfg_guidance,
                    null_prompt=args.null_prompt,
                    num_samples=args.num_samples,
                    base_seed=args.seed,
                    block_size=args.block_size,
                    normalize=True,
                )
            else:
                res = analyzer.analyze_block_energy(
                    prompt=prompt,
                    cfg_guidance=args.cfg_guidance,
                    null_prompt=args.null_prompt,
                    block_size=args.block_size,
                    normalize=True,
                )
                mr = {
                    "prompt":            prompt,
                    "step_indices":      res["step_indices"],
                    "timesteps":         res["timesteps"],
                    "block_energy_mean": res["block_energy"],
                    "block_energy_std":  np.zeros_like(res["block_energy"]),
                    "block_energy_raw":  res["block_energy"][None],
                    "block_size":        res["block_size"],
                    "normalize":         res["normalize"],
                }

            _tag = f"memo{idx:02d}"
            # curve 로 그릴 denoising step index (균등 간격 선택)
            # 예: NFE=50, num_curve_steps=5 → [0,10,20,30,40]
            _T = len(mr["step_indices"])
            _n = min(args.num_curve_steps, _T)
            _stride = max(_T // _n, 1)
            curve_steps = list(range(0, _T, _stride))[:_n]
            print(f"  [curve] plotting {len(curve_steps)} step(s): "
                  f"{curve_steps}  (t={[mr['timesteps'][s] for s in curve_steps]})")

            # freq_ratio → low / high 밴드 (inclusive). P = block 수
            _P = mr["block_energy_mean"].shape[1]
            _n_band = max(1, min(int(_P * args.freq_ratio), _P // 2))
            low_range = (0, _n_band - 1)
            high_range = (_P - _n_band, _P - 1)
            print(f"  [freq] ratio={args.freq_ratio}, P={_P}, n_band={_n_band} "
                  f"-> low={low_range}, high={high_range}")

            np.savez(
                os.path.join(csv_dir, f"block_energy_{_tag}.npz"),
                step_indices=np.array(mr["step_indices"]),
                timesteps=np.array(mr["timesteps"]),
                block_energy_mean=mr["block_energy_mean"],
                block_energy_std=mr["block_energy_std"],
                block_energy_raw=mr["block_energy_raw"],
                block_size=np.array(mr["block_size"]),
                freq_ratio=np.array(args.freq_ratio),
                low_range=np.array(low_range),
                high_range=np.array(high_range),
                normalize=np.array(mr["normalize"]),
                prompt=prompt,
            )
            # 전체/Low/High 를 한 figure 에 (linear, log 없음)
            plot_block_energy_combined(
                mr, plot_dir, freq_ratio=args.freq_ratio, curve_steps=curve_steps,
                filename=f"block_energy_{_tag}.png", tag=_tag)
            # Low vs High vs All band 비교 trajectory (linear)
            plot_band_compare(
                mr, plot_dir, freq_ratio=args.freq_ratio,
                filename=f"block_energy_{_tag}_band.png", tag=_tag)
            # 전체 step × block heatmap (개요)
            plot_block_energy_heatmap(
                mr, plot_dir,
                filename=f"block_energy_{_tag}_heatmap.png", tag=_tag)
            save_block_energy_csv(
                mr, csv_dir,
                filename=f"block_energy_{_tag}.csv", freq_ratio=args.freq_ratio)

        print(f"\n[block_energy] all done -> {os.path.join(args.output_dir, 'LH_energy')}")
        return

    # ---- measure key: --measure 에 따라 plot/csv/npz 의 y값 결정 ----
    measure_key = {"memo_proxy": "eps_diff_sq", "kl_div": "kl", "cmp_l2": "cmp_l2",
                   "eps_l2": "eps_l2", "eps_ref_mse": "eps_ref_mse"}[args.measure]
    measure_label = (r"$||\epsilon - \epsilon_s||^2\,/\,D$ (memo_proxy)"
                     if args.measure == "memo_proxy"
                     else (r"$||\epsilon_{\mathrm{cfg}}||^2\,/\,D$ (eps_l2)"
                           if args.measure == "eps_l2"
                           else (r"$||\epsilon_{\mathrm{ref}} - \epsilon_{\mathrm{cfg}}||^2\,/\,D$ (eps_ref_mse)"
                                 if args.measure == "eps_ref_mse"
                                 else r"$\mathrm{KL}(x_t\,\|\,N(0,1))$ (gaussianity)")))
    print(f"  measure   : {args.measure}  (key={measure_key})")

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
                    loss_cfg=args.loss_cfg,
                )
                res["eps_diff_sq"] = res[measure_key]   # --measure 값으로 alias (plot/csv/npz 호환)
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
                                   filename=f"{args.measure}.png",
                                   ylabel=measure_label)
        else:
            multi_results = []
            # num imgs per prompts 
            for idx, prompt in enumerate(prompts):
                print(f"[{idx+1}/{len(prompts)}] {prompt}  ({args.num_samples} seeds)")
                _is_memo = idx in memo_indices # Is the idx the memo index? yes: True
                _prefix = "memo" if _is_memo else "gen" 
                _gidx = idx - len(cur_text) if _is_memo else idx
                
                mr = analyzer.analyze_multi_sample(
                    prompt=prompt,
                    cfg_guidance=args.cfg_guidance,
                    null_prompt=args.null_prompt,
                    num_samples=args.num_samples,
                    base_seed=args.seed,
                    loss_cfg=args.loss_cfg,
                ) # mr: dict_keys(['prompt', 'samples'])
                # mr['samples'][0] : dict
                # --> dict_keys(['prompt', 'step_indices', 'timesteps', 'eps_diff_sq', 'kl', 'tweedie_x0', 'eps_s_list', 'epsilon_original', 'img', 'seed'])
                for _s in mr["samples"]:
                    _s["eps_diff_sq"] = _s[measure_key]   # --measure 값으로 alias
                multi_results.append(mr) # info about one prompt
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
            # Final Plot
            plot_multi_sample(multi_results, plot_dir, filename=f"{args.measure}.png",
                              memo_indices=memo_indices, ylabel=measure_label)
        # SNR schedule plot (text & memo share the same schedule)
        plot_snr(analyzer, plot_dir)

    # 전체 plot 통합: text(general) vs memo overall mean ± std → plot_num=N/memo_proxy_overall.png
    plot_overall(plot_root, args.num_tp, args.num_plot, ylabel=measure_label)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
